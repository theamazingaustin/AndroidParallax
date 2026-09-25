package com.example.depthpaper.core

import ai.onnxruntime.OnnxJavaType
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.content.Context
import android.graphics.Bitmap
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.max
import kotlin.math.min

/**
 * Robust ONNX Runtime Depth Anything V2 Engine.
 *
 * Implements Shubham0204's working Depth-Anything-Android architecture:
 * - Uses Microsoft ONNX Runtime Android (CPU execution with multi-threading)
 * - Model: fused_model_uint8_256.onnx (input [1, 256, 256, 3] uint8, output [1, 252, 252] uint8)
 * - Ultra-fast: ~50-80ms on mobile CPU
 * - Direct bilinear upsampling to source image dimensions
 * - Google Turbo colormap visualization
 * - Full Z-depth foreground mask slicing
 */
object OnnxDepthAnythingEngine {

    private const val TAG = "OnnxDepthAnything"
    const val MODEL_ASSET = "models/depth_anything_v2.onnx"
    private const val INPUT_DIM = 256
    private const val OUTPUT_DIM = 252

    @Volatile private var ortEnvironment: OrtEnvironment? = null
    @Volatile private var ortSession: OrtSession? = null
    @Volatile private var inputName: String? = null
    @Volatile var lastError: String? = null
        private set

    // Uses DepthAnythingEngine.DepthResult for unified pipeline compatibility
    @Synchronized
    fun initialize(context: Context) {
        if (ortSession != null) return
        try {
            val env = OrtEnvironment.getEnvironment()
            ortEnvironment = env
            val modelBytes = context.assets.open(MODEL_ASSET).use { it.readBytes() }
            val opts = OrtSession.SessionOptions().apply {
                setIntraOpNumThreads(4)
                setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
            }
            val session = env.createSession(modelBytes, opts)
            ortSession = session
            inputName = session.inputNames.iterator().next()
            lastError = null
            AppLogger.i(TAG, "ONNX Depth Anything V2 loaded successfully on CPU (4 threads)")
        } catch (t: Throwable) {
            lastError = "${t.javaClass.simpleName}: ${t.message}"
            AppLogger.e(TAG, "Failed to load ONNX Depth Anything: $lastError", t)
        }
    }

    fun estimateDepth(
        context: Context,
        inputBitmap: Bitmap,
        clockZDepth: Float = 0.50f
    ): DepthAnythingEngine.DepthResult? {
        if (ortSession == null) {
            initialize(context)
        }
        val session = ortSession ?: run {
            AppLogger.e(TAG, "Session is null: $lastError")
            return null
        }
        val env = ortEnvironment ?: return null
        val inName = inputName ?: session.inputNames.iterator().next()

        val srcW = inputBitmap.width
        val srcH = inputBitmap.height

        return try {
            // 1. Scale input bitmap to [256, 256]
            val scaledBmp = if (srcW == INPUT_DIM && srcH == INPUT_DIM) {
                inputBitmap
            } else {
                Bitmap.createScaledBitmap(inputBitmap, INPUT_DIM, INPUT_DIM, true)
            }

            // 2. Convert to [1, 256, 256, 3] UINT8 buffer
            val pixels = IntArray(INPUT_DIM * INPUT_DIM)
            scaledBmp.getPixels(pixels, 0, INPUT_DIM, 0, 0, INPUT_DIM, INPUT_DIM)
            if (scaledBmp !== inputBitmap && !scaledBmp.isRecycled) {
                scaledBmp.recycle()
            }

            val imgBuffer = ByteBuffer.allocateDirect(1 * INPUT_DIM * INPUT_DIM * 3).apply {
                order(ByteOrder.nativeOrder())
                for (i in 0 until INPUT_DIM * INPUT_DIM) {
                    val c = pixels[i]
                    put((c shr 16 and 0xFF).toByte()) // R
                    put((c shr 8 and 0xFF).toByte())  // G
                    put((c and 0xFF).toByte())        // B
                }
                rewind()
            }

            // 3. Create input tensor and run inference
            val inputTensor = OnnxTensor.createTensor(
                env,
                imgBuffer,
                longArrayOf(1, INPUT_DIM.toLong(), INPUT_DIM.toLong(), 3),
                OnnxJavaType.UINT8
            )

            val startTime = System.currentTimeMillis()
            val outputs = session.run(mapOf(inName to inputTensor))
            val durationMs = System.currentTimeMillis() - startTime
            AppLogger.i(TAG, "ONNX inference completed in ${durationMs}ms")

            // 4. Read output tensor [1, 252, 252] UINT8
            val outputTensor = outputs[0] as OnnxTensor
            val outBuffer = outputTensor.byteBuffer
            outBuffer.rewind()
            val rawDepthBytes = ByteArray(OUTPUT_DIM * OUTPUT_DIM)
            outBuffer.get(rawDepthBytes)
            inputTensor.close()
            outputs.close()

            // 5. Bilinear upsampling to original photo dimensions (srcW × srcH)
            val outDepth = FloatArray(srcW * srcH)
            val invSrcW = 1.0f / max(1, srcW - 1)
            val invSrcH = 1.0f / max(1, srcH - 1)

            var minD = 255f
            var maxD = 0f

            for (y in 0 until srcH) {
                val v = y * invSrcH
                val srcY = v * (OUTPUT_DIM - 1)
                val y0 = srcY.toInt().coerceIn(0, OUTPUT_DIM - 2)
                val y1 = y0 + 1
                val dy = srcY - y0

                val row0 = y0 * OUTPUT_DIM
                val row1 = y1 * OUTPUT_DIM
                val dstRow = y * srcW

                for (x in 0 until srcW) {
                    val u = x * invSrcW
                    val srcX = u * (OUTPUT_DIM - 1)
                    val x0 = srcX.toInt().coerceIn(0, OUTPUT_DIM - 2)
                    val x1 = x0 + 1
                    val dx = srcX - x0

                    val d00 = (rawDepthBytes[row0 + x0].toInt() and 0xFF).toFloat()
                    val d10 = (rawDepthBytes[row0 + x1].toInt() and 0xFF).toFloat()
                    val d01 = (rawDepthBytes[row1 + x0].toInt() and 0xFF).toFloat()
                    val d11 = (rawDepthBytes[row1 + x1].toInt() and 0xFF).toFloat()

                    val top = d00 * (1f - dx) + d10 * dx
                    val bot = d01 * (1f - dx) + d11 * dx
                    val d = top * (1f - dy) + bot * dy

                    outDepth[dstRow + x] = d
                    if (d < minD) minD = d
                    if (d > maxD) maxD = d
                }
            }

            // 6. Normalize depth [0.0, 1.0] where 1.0 = near (foreground), 0.0 = far (sky/background)
            val range = max(0.0001f, maxD - minD)
            val normalizedDepth = FloatArray(srcW * srcH)
            val depthPixels = IntArray(srcW * srcH)

            for (i in 0 until srcW * srcH) {
                val norm = ((outDepth[i] - minD) / range).coerceIn(0f, 1f)
                normalizedDepth[i] = norm
                depthPixels[i] = DepthAnythingEngine.turboColormap(norm)
            }

            val naturalGap = DepthAnythingEngine.computeNaturalDepthGap(normalizedDepth)

            // 7. Foreground mask from clock Z-depth
            val fgMask = FloatArray(srcW * srcH)
            val zCut = clockZDepth.coerceIn(0.0f, 1.0f)
            when {
                zCut <= 0.001f -> fgMask.fill(1.0f)
                zCut >= 0.999f -> fgMask.fill(0.0f)
                else -> {
                    val halfBand = 0.04f
                    val bandDenom = max(0.0001f, halfBand * 2f)
                    for (i in 0 until srcW * srcH) {
                        val d = normalizedDepth[i]
                        fgMask[i] = when {
                            d >= zCut + halfBand -> 1.0f
                            d <= zCut - halfBand -> 0.0f
                            else -> ((d - (zCut - halfBand)) / bandDenom).coerceIn(0f, 1f)
                        }
                    }
                }
            }

            val depthBitmap = Bitmap.createBitmap(srcW, srcH, Bitmap.Config.ARGB_8888)
            depthBitmap.setPixels(depthPixels, 0, srcW, 0, 0, srcW, srcH)
            lastError = null

            DepthAnythingEngine.DepthResult(
                depthBitmap = depthBitmap,
                normalizedDepth = normalizedDepth,
                depthWidth = srcW,
                depthHeight = srcH,
                foregroundConfidenceMask = fgMask,
                naturalDepthGap = naturalGap
            )
        } catch (t: Throwable) {
            lastError = "${t.javaClass.simpleName}: ${t.message}"
            AppLogger.e(TAG, "Inference error: $lastError", t)
            null
        }
    }

    @Synchronized
    fun close() {
        try { ortSession?.close() } catch (_: Exception) {}
        try { ortEnvironment?.close() } catch (_: Exception) {}
        ortSession = null
        ortEnvironment = null
        inputName = null
    }
}

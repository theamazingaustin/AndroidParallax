package com.example.depthpaper.core

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.gpu.CompatibilityList
import org.tensorflow.lite.gpu.GpuDelegate
import org.tensorflow.lite.nnapi.NnApiDelegate
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.channels.FileChannel
import kotlin.math.max
import kotlin.math.min

/**
 * On-Device Hardware-Accelerated Depth Anything V2 Inference Engine.
 *
 * Automatically detects the actual TFLite tensor shapes and input format (NCHW vs NHWC)
 * at model-load time, eliminating hardcoded dimension assumptions that cause garbage output.
 *
 * Features:
 * - Auto-detects input tensor shape: [1,3,H,W] NCHW or [1,H,W,3] NHWC
 * - Auto-detects output tensor shape: [1,H,W], [1,H,W,1], or [1,1,H,W]
 * - Jet colormap visualization matching the official Depth Anything V2 demo
 * - Aspect-Preserving Letterboxed Inference
 * - Automatic Depth Gap Detection
 */
object DepthAnythingEngine {

    private const val TAG = "DepthAnythingEngine"
    const val MODEL_SMALL_ASSET = "models/depth_anything_v2_small.tflite"
    const val MODEL_BASE_ASSET = "models/depth_anything_v2_base.tflite"

    // Detected from the actual model at load time — do NOT hardcode
    @Volatile private var inputW = 518
    @Volatile private var inputH = 518
    @Volatile private var isInputNCHW = false   // true=[1,3,H,W], false=[1,H,W,3]
    @Volatile private var outputW = 518
    @Volatile private var outputH = 518

    // ImageNet standard normalization constants expected by DINOv2 / Depth Anything V2
    private val MEAN = floatArrayOf(0.485f, 0.456f, 0.406f)
    private val STD  = floatArrayOf(0.229f, 0.224f, 0.225f)

    @Volatile private var interpreter: Interpreter? = null
    private var gpuDelegate: GpuDelegate? = null
    private var nnApiDelegate: NnApiDelegate? = null
    private var loadedModelAsset: String? = null

    data class DepthResult(
        val depthBitmap: Bitmap,
        val normalizedDepth: FloatArray,
        val depthWidth: Int,
        val depthHeight: Int,
        val foregroundConfidenceMask: FloatArray,
        val naturalDepthGap: Float
    )

    // ──────────────────────────────────────────────────────────────────────────────
    // Initialization
    // ──────────────────────────────────────────────────────────────────────────────

    @Synchronized
    fun initialize(context: Context, modelAsset: String = MODEL_SMALL_ASSET) {
        if (interpreter != null && loadedModelAsset == modelAsset) return
        close()

        val activeAsset = try {
            context.assets.openFd(modelAsset)
            modelAsset
        } catch (_: Exception) {
            AppLogger.i(TAG, "Asset $modelAsset not found; falling back to $MODEL_SMALL_ASSET")
            MODEL_SMALL_ASSET
        }

        val modelBuffer = try {
            loadModelFile(context, activeAsset)
        } catch (e: Exception) {
            AppLogger.e(TAG, "Failed to load $activeAsset: ${e.message}", e)
            return
        }

        loadedModelAsset = activeAsset
        val loaded = tryLoadInterpreter(modelBuffer) ?: return
        interpreter = loaded
        detectTensorShapes(loaded)
        AppLogger.i(TAG, "Model $activeAsset ready — input ${if (isInputNCHW) "NCHW" else "NHWC"} ${inputH}×${inputW}, output ${outputH}×${outputW}")
    }

    /**
     * Queries the loaded interpreter to detect actual input/output tensor shapes.
     * This is critical: TFLite models may be NHWC or NCHW, and may have dimensions
     * different from any hardcoded constants. Detecting at runtime makes the engine
     * robust to different model exports (518×518, 384×384, 256×256, etc.).
     */
    private fun detectTensorShapes(interp: Interpreter) {
        try {
            val inShape = interp.getInputTensor(0).shape()
            AppLogger.i(TAG, "Input tensor shape: ${inShape.toList()}")
            when {
                // [1, 3, H, W]  → NCHW
                inShape.size == 4 && inShape[1] == 3 -> {
                    isInputNCHW = true; inputH = inShape[2]; inputW = inShape[3]
                }
                // [1, H, W, 3] → NHWC
                inShape.size == 4 && inShape[3] == 3 -> {
                    isInputNCHW = false; inputH = inShape[1]; inputW = inShape[2]
                }
                // [3, H, W] unbatched NCHW
                inShape.size == 3 && inShape[0] == 3 -> {
                    isInputNCHW = true; inputH = inShape[1]; inputW = inShape[2]
                }
                // [H, W, 3] unbatched NHWC
                inShape.size == 3 && inShape[2] == 3 -> {
                    isInputNCHW = false; inputH = inShape[0]; inputW = inShape[1]
                }
                else -> AppLogger.w(TAG, "Unrecognized input shape ${inShape.toList()}, using defaults")
            }
        } catch (e: Exception) {
            AppLogger.w(TAG, "Could not detect input tensor shape: ${e.message}")
        }

        try {
            val outShape = interp.getOutputTensor(0).shape()
            AppLogger.i(TAG, "Output tensor shape: ${outShape.toList()}")
            when {
                // [1, H, W]
                outShape.size == 3 -> { outputH = outShape[1]; outputW = outShape[2] }
                // [1, H, W, 1]
                outShape.size == 4 && outShape[3] == 1 -> { outputH = outShape[1]; outputW = outShape[2] }
                // [1, 1, H, W]
                outShape.size == 4 && outShape[1] == 1 -> { outputH = outShape[2]; outputW = outShape[3] }
                // [H, W]
                outShape.size == 2 -> { outputH = outShape[0]; outputW = outShape[1] }
                else -> { outputH = inputH; outputW = inputW }
            }
        } catch (e: Exception) {
            AppLogger.w(TAG, "Could not detect output tensor shape: ${e.message}")
            outputH = inputH; outputW = inputW
        }
    }

    private fun tryLoadInterpreter(modelBuffer: ByteBuffer): Interpreter? {
        // Tier 1: GPU
        try {
            val compatList = CompatibilityList()
            if (compatList.isDelegateSupportedOnThisDevice) {
                val delegate = GpuDelegate(GpuDelegate.Options().apply { setPrecisionLossAllowed(true) })
                val opts = Interpreter.Options().apply { addDelegate(delegate); setNumThreads(4) }
                val interp = Interpreter(modelBuffer, opts)
                gpuDelegate = delegate
                AppLogger.i(TAG, "Depth Anything V2 loaded on GPU")
                return interp
            }
        } catch (e: Exception) {
            AppLogger.w(TAG, "GPU init failed: ${e.message}")
            gpuDelegate?.close(); gpuDelegate = null
        }

        // Tier 2: NNAPI (Hexagon NPU)
        try {
            val delegate = NnApiDelegate(NnApiDelegate.Options().apply {
                setAllowFp16(true)
                setExecutionPreference(NnApiDelegate.Options.EXECUTION_PREFERENCE_SUSTAINED_SPEED)
            })
            val opts = Interpreter.Options().apply { addDelegate(delegate); setNumThreads(4) }
            val interp = Interpreter(modelBuffer, opts)
            nnApiDelegate = delegate
            AppLogger.i(TAG, "Depth Anything V2 loaded on NNAPI")
            return interp
        } catch (e: Exception) {
            AppLogger.w(TAG, "NNAPI init failed: ${e.message}")
            nnApiDelegate?.close(); nnApiDelegate = null
        }

        // Tier 3: CPU
        return try {
            Interpreter(modelBuffer, Interpreter.Options().apply { setNumThreads(4) }).also {
                AppLogger.i(TAG, "Depth Anything V2 loaded on CPU (4 threads)")
            }
        } catch (e: Exception) {
            AppLogger.e(TAG, "All interpreter init failed: ${e.message}", e); null
        }
    }

    private fun initializeCpuOnly(context: Context, assetPath: String) {
        try {
            val modelBuffer = loadModelFile(context, assetPath)
            val interp = Interpreter(modelBuffer, Interpreter.Options().apply { setNumThreads(4) })
            interpreter = interp
            detectTensorShapes(interp)
            AppLogger.i(TAG, "Depth Anything V2 CPU fallback ready — ${inputH}×${inputW}")
        } catch (e: Exception) {
            AppLogger.e(TAG, "CPU fallback init failed: ${e.message}", e)
        }
    }

    private fun loadModelFile(context: Context, assetPath: String): ByteBuffer {
        val fd = context.assets.openFd(assetPath)
        return FileInputStream(fd.fileDescriptor).channel
            .map(FileChannel.MapMode.READ_ONLY, fd.startOffset, fd.declaredLength)
    }

    // ──────────────────────────────────────────────────────────────────────────────
    // Inference
    // ──────────────────────────────────────────────────────────────────────────────

    /**
     * Estimates dense continuous scene depth with Aspect-Preserving Letterboxing.
     *
     * @param inputBitmap Source photograph (any aspect ratio).
     * @param clockZDepth Clock Z-depth plane (0.0=behind everything, 1.0=in front of everything).
     */
    fun estimateDepth(
        context: Context,
        inputBitmap: Bitmap,
        clockZDepth: Float = 0.50f,
        modelAsset: String = MODEL_SMALL_ASSET
    ): DepthResult? {
        if (interpreter == null || loadedModelAsset != modelAsset) {
            initialize(context, modelAsset)
        }
        var interp = interpreter ?: run {
            AppLogger.e(TAG, "Interpreter not initialized.")
            return null
        }

        val inW = inputW
        val inH = inputH

        // 1. Aspect-Preserving Letterbox (prevents anatomy distortion)
        val srcW = inputBitmap.width
        val srcH = inputBitmap.height
        val scale = min(inW.toFloat() / srcW, inH.toFloat() / srcH)
        val scaledW = (srcW * scale).toInt().coerceIn(1, inW)
        val scaledH = (srcH * scale).toInt().coerceIn(1, inH)
        val padLeft = (inW - scaledW) / 2
        val padTop  = (inH - scaledH) / 2

        val letterboxBmp = Bitmap.createBitmap(inW, inH, Bitmap.Config.ARGB_8888)
        Canvas(letterboxBmp).apply {
            drawColor(Color.rgb(124, 116, 104))  // neutral gray padding (close to ImageNet mean)
            val scaledSrc = if (srcW == scaledW && srcH == scaledH) inputBitmap
                            else Bitmap.createScaledBitmap(inputBitmap, scaledW, scaledH, true)
            drawBitmap(scaledSrc, padLeft.toFloat(), padTop.toFloat(), Paint(Paint.FILTER_BITMAP_FLAG))
            if (scaledSrc !== inputBitmap && !scaledSrc.isRecycled) scaledSrc.recycle()
        }

        val pixels = IntArray(inW * inH)
        letterboxBmp.getPixels(pixels, 0, inW, 0, 0, inW, inH)
        if (!letterboxBmp.isRecycled) letterboxBmp.recycle()

        // 2. Build input buffer in detected format with ImageNet normalization
        val inv255 = 1.0f / 255.0f
        val inputBuffer: ByteBuffer = if (isInputNCHW) {
            // NCHW [1, 3, H, W] — write full R plane, then G plane, then B plane
            ByteBuffer.allocateDirect(1 * 3 * inH * inW * 4).apply {
                order(ByteOrder.nativeOrder())
                val mR = MEAN[0]; val sR = STD[0]
                for (i in 0 until inW * inH) putFloat(((pixels[i] shr 16 and 0xFF) * inv255 - mR) / sR)
                val mG = MEAN[1]; val sG = STD[1]
                for (i in 0 until inW * inH) putFloat(((pixels[i] shr 8 and 0xFF) * inv255 - mG) / sG)
                val mB = MEAN[2]; val sB = STD[2]
                for (i in 0 until inW * inH) putFloat(((pixels[i] and 0xFF) * inv255 - mB) / sB)
                rewind()
            }
        } else {
            // NHWC [1, H, W, 3] — write R,G,B interleaved per pixel (TFLite standard on Android)
            ByteBuffer.allocateDirect(1 * inH * inW * 3 * 4).apply {
                order(ByteOrder.nativeOrder())
                val mR = MEAN[0]; val sR = STD[0]
                val mG = MEAN[1]; val sG = STD[1]
                val mB = MEAN[2]; val sB = STD[2]
                for (i in 0 until inW * inH) {
                    val c = pixels[i]
                    putFloat(((c shr 16 and 0xFF) * inv255 - mR) / sR)
                    putFloat(((c shr 8  and 0xFF) * inv255 - mG) / sG)
                    putFloat(((c        and 0xFF) * inv255 - mB) / sB)
                }
                rewind()
            }
        }

        // 3. Allocate output buffer based on detected output shape
        val outputBuffer = ByteBuffer.allocateDirect(outputH * outputW * 4).apply {
            order(ByteOrder.nativeOrder())
        }

        // 4. Run inference with hardware fallback
        val startTime = System.currentTimeMillis()
        try {
            interp.run(inputBuffer, outputBuffer)
        } catch (e: Exception) {
            AppLogger.w(TAG, "Hardware inference error (${e.message}), recovering on CPU...")
            try {
                close()
                initializeCpuOnly(context, loadedModelAsset ?: MODEL_SMALL_ASSET)
                val cpuInterp = interpreter ?: return null
                inputBuffer.rewind(); outputBuffer.rewind()
                cpuInterp.run(inputBuffer, outputBuffer)
                interp = cpuInterp
            } catch (e2: Exception) {
                AppLogger.e(TAG, "CPU inference also failed: ${e2.message}", e2)
                return null
            }
        }
        AppLogger.i(TAG, "Depth inference in ${System.currentTimeMillis() - startTime}ms [${if (isInputNCHW) "NCHW" else "NHWC"} ${inH}×${inW} → ${outputH}×${outputW}]")

        // 5. Read raw outputs and compute global min/max for normalization
        outputBuffer.rewind()
        val totalOut = outputH * outputW
        val rawDepth = FloatArray(totalOut)
        var minVal = Float.MAX_VALUE
        var maxVal = -Float.MAX_VALUE
        for (i in 0 until totalOut) {
            val v = outputBuffer.float
            rawDepth[i] = v
            if (v < minVal) minVal = v
            if (v > maxVal) maxVal = v
        }
        val range = max(0.00001f, maxVal - minVal)
        AppLogger.i(TAG, "Raw depth range: min=$minVal max=$maxVal range=$range")

        // 6. Un-pad and crop to original image aspect ratio
        // Output may have different resolution than input (scale by output/input ratio)
        val depthScaleX = outputW.toFloat() / inW.toFloat()
        val depthScaleY = outputH.toFloat() / inH.toFloat()
        val dPadLeft = (padLeft * depthScaleX).toInt()
        val dPadTop  = (padTop  * depthScaleY).toInt()
        val outCropW = (scaledW * depthScaleX).toInt().coerceIn(1, outputW)
        val outCropH = (scaledH * depthScaleY).toInt().coerceIn(1, outputH)

        val unpaddedDepth = FloatArray(outCropW * outCropH)
        val depthPixels   = IntArray(outCropW * outCropH)

        for (y in 0 until outCropH) {
            val srcRow = (y + dPadTop) * outputW
            val dstRow = y * outCropW
            for (x in 0 until outCropW) {
                val rawVal = rawDepth[srcRow + (x + dPadLeft)]
                val norm = ((rawVal - minVal) / range).coerceIn(0f, 1f)
                val dstIdx = dstRow + x
                unpaddedDepth[dstIdx] = norm
                depthPixels[dstIdx] = jetColormap(norm)
            }
        }

        // 7. Natural depth gap (valley between foreground and background peaks)
        val naturalGap = computeNaturalDepthGap(unpaddedDepth)

        // 8. 2D foreground confidence mask based on clock Z-depth
        val fgMask = FloatArray(outCropW * outCropH)
        val zCut = clockZDepth.coerceIn(0.0f, 1.0f)
        when {
            zCut <= 0.001f -> fgMask.fill(1.0f)
            zCut >= 0.999f -> fgMask.fill(0.0f)
            else -> {
                val halfBand = 0.04f
                val bandDenom = max(0.0001f, halfBand * 2f)
                for (i in 0 until outCropW * outCropH) {
                    val d = unpaddedDepth[i]
                    fgMask[i] = when {
                        d >= zCut + halfBand -> 1.0f
                        d <= zCut - halfBand -> 0.0f
                        else -> ((d - (zCut - halfBand)) / bandDenom).coerceIn(0f, 1f)
                    }
                }
            }
        }

        val depthBitmap = Bitmap.createBitmap(outCropW, outCropH, Bitmap.Config.ARGB_8888)
        depthBitmap.setPixels(depthPixels, 0, outCropW, 0, 0, outCropW, outCropH)

        return DepthResult(
            depthBitmap = depthBitmap,
            normalizedDepth = unpaddedDepth,
            depthWidth = outCropW,
            depthHeight = outCropH,
            foregroundConfidenceMask = fgMask,
            naturalDepthGap = naturalGap
        )
    }

    // ──────────────────────────────────────────────────────────────────────────────
    // Colormap & Analytics
    // ──────────────────────────────────────────────────────────────────────────────

    /**
     * Jet colormap matching the official Depth Anything V2 online demo.
     * Maps normalized depth [0,1] to: purple/blue (far) → cyan → green → yellow → red/warm (near).
     * 1.0 = close to camera = warm (red/orange), 0.0 = far = cool (blue/purple).
     */
    private fun jetColormap(v: Float): Int {
        val r = when {
            v < 0.35f -> 0f
            v < 0.66f -> (v - 0.35f) / 0.31f
            v < 0.89f -> 1f
            else -> 1f - (v - 0.89f) / 0.11f
        }.coerceIn(0f, 1f)

        val g = when {
            v < 0.125f -> 0f
            v < 0.375f -> (v - 0.125f) / 0.25f
            v < 0.64f  -> 1f
            v < 0.91f  -> 1f - (v - 0.64f) / 0.27f
            else -> 0f
        }.coerceIn(0f, 1f)

        val b = when {
            v < 0.11f  -> 0.5f + v / 0.11f * 0.5f
            v < 0.375f -> 1f
            v < 0.625f -> 1f - (v - 0.375f) / 0.25f
            else -> 0f
        }.coerceIn(0f, 1f)

        return (0xFF shl 24) or ((r * 255).toInt() shl 16) or ((g * 255).toInt() shl 8) or (b * 255).toInt()
    }

    /**
     * Analyzes the depth histogram to find the natural valley separating
     * foreground subjects from the background.
     */
    fun computeNaturalDepthGap(depth: FloatArray): Float {
        if (depth.isEmpty()) return 0.50f
        val numBins = 32
        val bins = IntArray(numBins)
        for (v in depth) bins[(v * (numBins - 1)).toInt().coerceIn(0, numBins - 1)]++

        val smooth = FloatArray(numBins)
        for (i in 0 until numBins) {
            var sum = 0f; var count = 0
            for (di in -2..2) {
                val ni = i + di
                if (ni in 0 until numBins) { sum += bins[ni]; count++ }
            }
            smooth[i] = sum / count
        }

        var fgPeak = 22; var fgMax = 0f
        for (i in 14 until numBins) if (smooth[i] > fgMax) { fgMax = smooth[i]; fgPeak = i }

        var bgPeak = 6; var bgMax = 0f
        for (i in 0 until min(fgPeak, 16)) if (smooth[i] > bgMax) { bgMax = smooth[i]; bgPeak = i }

        var minVal = Float.MAX_VALUE; var valleyStart = -1; var valleyEnd = -1
        for (i in bgPeak..fgPeak) {
            if (smooth[i] < minVal) { minVal = smooth[i]; valleyStart = i; valleyEnd = i }
            else if (smooth[i] == minVal) valleyEnd = i
        }

        val valleyBin = if (valleyStart != -1) (valleyStart + valleyEnd) / 2
                        else (bgPeak + fgPeak) / 2
        return (valleyBin.toFloat() / (numBins - 1).toFloat()).coerceIn(0.18f, 0.82f)
    }

    // ──────────────────────────────────────────────────────────────────────────────
    // Cleanup
    // ──────────────────────────────────────────────────────────────────────────────

    @Synchronized
    fun close() {
        try { interpreter?.close(); gpuDelegate?.close(); nnApiDelegate?.close() } catch (_: Exception) {}
        interpreter = null; gpuDelegate = null; nnApiDelegate = null; loadedModelAsset = null
    }
}

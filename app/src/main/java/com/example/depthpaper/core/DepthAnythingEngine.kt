package com.example.depthpaper.core

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Color
import org.tensorflow.lite.Interpreter
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.channels.FileChannel
import kotlin.math.max
import kotlin.math.min

/**
 * On-Device Hardware-Accelerated Depth Anything V2 Inference Engine.
 *
 * Implements the official LiteRT / Hugging Face specification:
 * - DINOv2 ViT-Small backbone + DPT relative-depth decoder
 * - Pure FP32 multi-threaded CPU execution with XNNPACK (4 threads)
 *   (NOTE: GPU/NNAPI delegates use FP16 which corrupts Vision Transformer self-attention,
 *   causing attention collapse into high-pass edge-detection artifacts)
 * - Fixed input contract: [1, 3, 518, 686] NCHW float32 with ImageNet normalization
 * - Direct bicubic stretch to 686×518, then bilinear reconstruction to full image size
 * - Google Turbo colormap visualization matching the official online demo
 */
object DepthAnythingEngine {

    private const val TAG = "DepthAnythingEngine"
    const val MODEL_SMALL_ASSET = "models/depth_anything_v2_small.tflite"
    const val MODEL_BASE_ASSET = "models/depth_anything_v2_base.tflite"

    // Official input contract from LiteRT / Hugging Face model card
    const val INPUT_WIDTH = 686
    const val INPUT_HEIGHT = 518

    // ImageNet normalization constants expected by DINOv2
    private val MEAN = floatArrayOf(0.485f, 0.456f, 0.406f)
    private val STD  = floatArrayOf(0.229f, 0.224f, 0.225f)

    @Volatile private var interpreter: Interpreter? = null
    private var loadedModelAsset: String? = null

    data class DepthResult(
        val depthBitmap: Bitmap,
        val normalizedDepth: FloatArray,
        val depthWidth: Int,
        val depthHeight: Int,
        val foregroundConfidenceMask: FloatArray,
        val naturalDepthGap: Float
    )

    @Synchronized
    fun initialize(context: Context, modelAsset: String = MODEL_SMALL_ASSET) {
        if (interpreter != null && loadedModelAsset == modelAsset) return
        close()

        val activeAsset = try {
            context.assets.openFd(modelAsset)
            modelAsset
        } catch (_: Exception) {
            AppLogger.i(TAG, "Asset $modelAsset not found; using $MODEL_SMALL_ASSET")
            MODEL_SMALL_ASSET
        }

        val modelBuffer = try {
            loadModelFile(context, activeAsset)
        } catch (e: Exception) {
            AppLogger.e(TAG, "Failed to load $activeAsset: ${e.message}", e)
            return
        }

        // Depth Anything V2's DINOv2 ViT attention layers REQUIRE full-precision FP32
        // XNNPACK CPU execution. Mobile GPU delegates use FP16 which causes transformer
        // self-attention to collapse into high-pass edge outlines.
        try {
            val options = Interpreter.Options().apply {
                setNumThreads(4)
                setUseXNNPACK(true)
            }
            interpreter = Interpreter(modelBuffer, options)
            loadedModelAsset = activeAsset
            AppLogger.i(TAG, "Depth Anything V2 loaded successfully on XNNPACK CPU (4 threads)")
        } catch (e: Exception) {
            AppLogger.e(TAG, "Failed to initialize Depth Anything V2 interpreter: ${e.message}", e)
        }
    }

    private fun loadModelFile(context: Context, assetPath: String): ByteBuffer {
        val fd = context.assets.openFd(assetPath)
        return FileInputStream(fd.fileDescriptor).channel
            .map(FileChannel.MapMode.READ_ONLY, fd.startOffset, fd.declaredLength)
    }

    /**
     * Estimates dense, continuous 3D relative depth using Depth Anything V2.
     *
     * @param inputBitmap Source photograph (any aspect ratio).
     * @param clockZDepth Clock Z-depth plane (0.0 = behind everything, 1.0 = in front of everything).
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
        val interp = interpreter ?: run {
            AppLogger.e(TAG, "Interpreter not initialized.")
            return null
        }

        val srcW = inputBitmap.width
        val srcH = inputBitmap.height
        val inW = INPUT_WIDTH
        val inH = INPUT_HEIGHT

        // 1. Direct stretch resize to 686×518 (official model contract)
        val scaledBmp = if (srcW == inW && srcH == inH) {
            inputBitmap
        } else {
            Bitmap.createScaledBitmap(inputBitmap, inW, inH, true)
        }

        val totalPix = inW * inH
        val pixels = IntArray(totalPix)
        scaledBmp.getPixels(pixels, 0, inW, 0, 0, inW, inH)
        if (scaledBmp !== inputBitmap && !scaledBmp.isRecycled) {
            scaledBmp.recycle()
        }

        // 2. Preprocessing: Rescale 1/255 and ImageNet normalize in NCHW order [1, 3, 518, 686]
        val inputBuffer = ByteBuffer.allocateDirect(1 * 3 * inH * inW * 4).apply {
            order(ByteOrder.nativeOrder())
        }

        val inv255 = 1.0f / 255.0f
        val mR = MEAN[0]; val sR = STD[0]
        val mG = MEAN[1]; val sG = STD[1]
        val mB = MEAN[2]; val sB = STD[2]

        // Channel 0: Red plane
        for (i in 0 until totalPix) {
            val r = (pixels[i] shr 16 and 0xFF) * inv255
            inputBuffer.putFloat((r - mR) / sR)
        }
        // Channel 1: Green plane
        for (i in 0 until totalPix) {
            val g = (pixels[i] shr 8 and 0xFF) * inv255
            inputBuffer.putFloat((g - mG) / sG)
        }
        // Channel 2: Blue plane
        for (i in 0 until totalPix) {
            val b = (pixels[i] and 0xFF) * inv255
            inputBuffer.putFloat((b - mB) / sB)
        }
        inputBuffer.rewind()

        // 3. Output buffer: [1, 518, 686] float32
        val outputBuffer = ByteBuffer.allocateDirect(1 * inH * inW * 4).apply {
            order(ByteOrder.nativeOrder())
        }

        // 4. Run inference on CPU (XNNPACK 4 threads)
        val startTime = System.currentTimeMillis()
        try {
            interp.run(inputBuffer, outputBuffer)
        } catch (e: Exception) {
            AppLogger.e(TAG, "Depth Anything V2 inference error: ${e.message}", e)
            return null
        }
        val durationMs = System.currentTimeMillis() - startTime
        AppLogger.i(TAG, "Depth Anything V2 inference completed in ${durationMs}ms on CPU")

        // 5. Read raw depth outputs [518, 686]
        outputBuffer.rewind()
        val rawDepth = FloatArray(totalPix)
        for (i in 0 until totalPix) {
            rawDepth[i] = outputBuffer.float
        }

        // 6. Bilinear reconstruction: map 518×686 depth map to original image resolution (srcW × srcH)
        val outDepth = FloatArray(srcW * srcH)
        val invSrcW = 1.0f / max(1, srcW - 1)
        val invSrcH = 1.0f / max(1, srcH - 1)

        var minD = Float.MAX_VALUE
        var maxD = -Float.MAX_VALUE

        for (y in 0 until srcH) {
            val v = y * invSrcH
            val srcY = v * (inH - 1)
            val y0 = srcY.toInt().coerceIn(0, inH - 2)
            val y1 = y0 + 1
            val dy = srcY - y0

            val row0 = y0 * inW
            val row1 = y1 * inW
            val dstRow = y * srcW

            for (x in 0 until srcW) {
                val u = x * invSrcW
                val srcX = u * (inW - 1)
                val x0 = srcX.toInt().coerceIn(0, inW - 2)
                val x1 = x0 + 1
                val dx = srcX - x0

                val d00 = rawDepth[row0 + x0]
                val d10 = rawDepth[row0 + x1]
                val d01 = rawDepth[row1 + x0]
                val d11 = rawDepth[row1 + x1]

                val top = d00 * (1f - dx) + d10 * dx
                val bot = d01 * (1f - dx) + d11 * dx
                val d = top * (1f - dy) + bot * dy

                outDepth[dstRow + x] = d
                if (d < minD) minD = d
                if (d > maxD) maxD = d
            }
        }

        // 7. Normalize depth to [0.0, 1.0] where 1.0 = near (foreground), 0.0 = far (sky/background)
        val range = max(0.00001f, maxD - minD)
        val normalizedDepth = FloatArray(srcW * srcH)
        val depthPixels = IntArray(srcW * srcH)

        for (i in 0 until srcW * srcH) {
            val norm = ((outDepth[i] - minD) / range).coerceIn(0f, 1f)
            normalizedDepth[i] = norm
            depthPixels[i] = turboColormap(norm)
        }

        // 8. Compute natural depth gap for automatic clock placement
        val naturalGap = computeNaturalDepthGap(normalizedDepth)

        // 9. Generate 2D continuous foreground confidence mask from clock Z-depth
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

        return DepthResult(
            depthBitmap = depthBitmap,
            normalizedDepth = normalizedDepth,
            depthWidth = srcW,
            depthHeight = srcH,
            foregroundConfidenceMask = fgMask,
            naturalDepthGap = naturalGap
        )
    }

    /**
     * Google Turbo Colormap (exact match to the official Depth Anything V2 demo).
     * Maps normalized depth [0.0, 1.0]:
     * - 0.0 (Far / Sky): Deep blue / violet
     * - 0.5 (Midground): Cyan / green
     * - 1.0 (Near / Foreground): Bright yellow / orange / red
     */
    fun turboColormap(x: Float): Int {
        val v = x.coerceIn(0f, 1f)
        val v2 = v * v
        val v3 = v2 * v
        val v4 = v3 * v
        val v5 = v4 * v

        val r = (0.13572138f + 4.61539260f * v - 42.66032258f * v2 + 132.13108234f * v3 - 152.94239396f * v4 + 59.28637943f * v5).coerceIn(0f, 1f)
        val g = (0.09140261f + 2.19418839f * v + 4.84296658f * v2 - 14.18503333f * v3 + 4.27729857f * v4 + 2.82956604f * v5).coerceIn(0f, 1f)
        val b = (0.10667330f + 12.64194608f * v - 60.58204836f * v2 + 110.36276771f * v3 - 89.90310912f * v4 + 27.34824973f * v5).coerceIn(0f, 1f)

        return (0xFF shl 24) or ((r * 255f).toInt() shl 16) or ((g * 255f).toInt() shl 8) or (b * 255f).toInt()
    }

    /**
     * Analyzes the depth histogram to find the natural depth valley separating foreground from background.
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

    @Synchronized
    fun close() {
        try { interpreter?.close() } catch (_: Exception) {}
        interpreter = null
        loadedModelAsset = null
    }
}

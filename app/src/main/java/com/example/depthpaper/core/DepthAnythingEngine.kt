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
 * Runs the real Depth Anything V2 Vision Transformer (ViT-Small and ViT-Base)
 * on-device using Qualcomm Adreno GPU or Hexagon NPU (via NNAPI) with multi-threaded CPU fallback.
 * 
 * Features:
 * - Aspect-Preserving Letterboxed Inference: Eliminates aspect ratio distortion across 4:3, 16:9, and vertical photos.
 * - Automatic Depth Gap Detection: Identifies natural 3D scene boundaries for effortless clock placement.
 * - Continuous 2D Spatial Depth: Eliminates 1D scanline slicing artifacts.
 */
object DepthAnythingEngine {

    private const val TAG = "DepthAnythingEngine"
    const val MODEL_SMALL_ASSET = "models/depth_anything_v2_small.tflite"
    const val MODEL_BASE_ASSET = "models/depth_anything_v2_base.tflite"

    const val INPUT_WIDTH = 686
    const val INPUT_HEIGHT = 518

    // ImageNet standard normalization constants expected by DINOv2 / Depth Anything V2
    private val MEAN = floatArrayOf(0.485f, 0.456f, 0.406f)
    private val STD = floatArrayOf(0.229f, 0.224f, 0.225f)

    @Volatile
    private var interpreter: Interpreter? = null
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

    @Synchronized
    fun initialize(context: Context, modelAsset: String = MODEL_SMALL_ASSET) {
        if (interpreter != null && loadedModelAsset == modelAsset) return
        close()

        val activeAsset = try {
            context.assets.openFd(modelAsset)
            modelAsset
        } catch (_: Exception) {
            MODEL_SMALL_ASSET
        }

        val modelBuffer = try {
            loadModelFile(context, activeAsset)
        } catch (e: Exception) {
            AppLogger.e(TAG, "Failed to load $activeAsset: ${e.message}", e)
            return
        }

        loadedModelAsset = activeAsset

        // Tier 1: Hardware GPU Acceleration (OpenCL / Vulkan compute shader)
        try {
            val compatList = CompatibilityList()
            if (compatList.isDelegateSupportedOnThisDevice) {
                val gpuOptions = GpuDelegate.Options().apply {
                    setPrecisionLossAllowed(true) // FP16 on Adreno GPU for 3x speedup
                }
                val delegate = GpuDelegate(gpuOptions)
                val options = Interpreter.Options().apply {
                    addDelegate(delegate)
                    setNumThreads(4)
                }
                val interp = Interpreter(modelBuffer, options)
                gpuDelegate = delegate
                interpreter = interp
                AppLogger.i(TAG, "Depth Anything V2 loaded on GPU ($activeAsset)")
                return
            }
        } catch (e: Exception) {
            AppLogger.w(TAG, "GPU initialization failed: ${e.message}. Trying NNAPI/NPU.")
            gpuDelegate?.close()
            gpuDelegate = null
        }

        // Tier 2: Qualcomm Hexagon NPU Acceleration via Android NNAPI
        try {
            val nnApiOptions = NnApiDelegate.Options().apply {
                setAllowFp16(true)
                setExecutionPreference(NnApiDelegate.Options.EXECUTION_PREFERENCE_SUSTAINED_SPEED)
            }
            val delegate = NnApiDelegate(nnApiOptions)
            val options = Interpreter.Options().apply {
                addDelegate(delegate)
                setNumThreads(4)
            }
            val interp = Interpreter(modelBuffer, options)
            nnApiDelegate = delegate
            interpreter = interp
            AppLogger.i(TAG, "Depth Anything V2 loaded on NNAPI ($activeAsset)")
            return
        } catch (e: Exception) {
            AppLogger.w(TAG, "NNAPI initialization failed: ${e.message}. Falling back to CPU.")
            nnApiDelegate?.close()
            nnApiDelegate = null
        }

        // Tier 3: Multi-threaded CPU with ARM NEON SIMD acceleration
        initializeCpuOnly(context, activeAsset)
    }

    private fun initializeCpuOnly(context: Context, assetPath: String) {
        try {
            val modelBuffer = loadModelFile(context, assetPath)
            val options = Interpreter.Options().apply {
                setNumThreads(4)
            }
            interpreter = Interpreter(modelBuffer, options)
            AppLogger.i(TAG, "Depth Anything V2 initialized on CPU (4 threads, NEON SIMD)")
        } catch (e: Exception) {
            AppLogger.e(TAG, "Failed to initialize Depth Anything V2 on CPU: ${e.message}", e)
        }
    }

    private fun loadModelFile(context: Context, assetPath: String): ByteBuffer {
        val fileDescriptor = context.assets.openFd(assetPath)
        val inputStream = FileInputStream(fileDescriptor.fileDescriptor)
        val fileChannel = inputStream.channel
        val startOffset = fileDescriptor.startOffset
        val declaredLength = fileDescriptor.declaredLength
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength)
    }

    /**
     * Estimates dense, continuous 3D scene depth with Aspect-Preserving Letterboxing.
     * 
     * @param inputBitmap Source photograph (any aspect ratio: 4:3, 16:9, vertical phone 9:20).
     * @param clockZDepth Clock Z-depth plane (0.0 to 1.0, where 1.0 = camera surface, 0.0 = infinite horizon).
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

        val inW = INPUT_WIDTH
        val inH = INPUT_HEIGHT

        // 1. Aspect-Preserving Letterbox Scaling (Zero human anatomy or compositions warping!)
        val srcW = inputBitmap.width
        val srcH = inputBitmap.height
        val scale = min(inW.toFloat() / srcW.toFloat(), inH.toFloat() / srcH.toFloat())
        val scaledW = (srcW * scale).toInt().coerceIn(1, inW)
        val scaledH = (srcH * scale).toInt().coerceIn(1, inH)
        val padLeft = (inW - scaledW) / 2
        val padTop = (inH - scaledH) / 2

        val letterboxBmp = Bitmap.createBitmap(inW, inH, Bitmap.Config.ARGB_8888)
        val canvas = Canvas(letterboxBmp)
        // Neutral gray background for padding borders
        canvas.drawColor(Color.rgb(124, 116, 104))

        val scaledSrc = if (srcW == scaledW && srcH == scaledH) {
            inputBitmap
        } else {
            Bitmap.createScaledBitmap(inputBitmap, scaledW, scaledH, true)
        }
        canvas.drawBitmap(scaledSrc, padLeft.toFloat(), padTop.toFloat(), Paint(Paint.FILTER_BITMAP_FLAG))
        if (scaledSrc != inputBitmap && !scaledSrc.isRecycled) {
            scaledSrc.recycle()
        }

        val pixels = IntArray(inW * inH)
        letterboxBmp.getPixels(pixels, 0, inW, 0, 0, inW, inH)
        if (!letterboxBmp.isRecycled) {
            letterboxBmp.recycle()
        }

        // 2. Format input as NCHW [1, 3, 518, 686] float32 with ImageNet normalization
        val inputBuffer = ByteBuffer.allocateDirect(1 * 3 * inH * inW * 4).apply {
            order(ByteOrder.nativeOrder())
        }

        val inv255 = 1.0f / 255.0f
        val meanR = MEAN[0]
        val stdR = STD[0]
        for (y in 0 until inH) {
            val row = y * inW
            for (x in 0 until inW) {
                val r = (pixels[row + x] shr 16 and 0xFF) * inv255
                inputBuffer.putFloat((r - meanR) / stdR)
            }
        }

        val meanG = MEAN[1]
        val stdG = STD[1]
        for (y in 0 until inH) {
            val row = y * inW
            for (x in 0 until inW) {
                val g = (pixels[row + x] shr 8 and 0xFF) * inv255
                inputBuffer.putFloat((g - meanG) / stdG)
            }
        }

        val meanB = MEAN[2]
        val stdB = STD[2]
        for (y in 0 until inH) {
            val row = y * inW
            for (x in 0 until inW) {
                val b = (pixels[row + x] and 0xFF) * inv255
                inputBuffer.putFloat((b - meanB) / stdB)
            }
        }
        inputBuffer.rewind()

        // 3. Allocate output buffer [1, 518, 686] float32
        val outputBuffer = ByteBuffer.allocateDirect(1 * inH * inW * 4).apply {
            order(ByteOrder.nativeOrder())
        }

        // 4. Run hardware-accelerated inference with automatic CPU fallback
        val startTime = System.currentTimeMillis()
        try {
            interp.run(inputBuffer, outputBuffer)
        } catch (e: Exception) {
            AppLogger.w(TAG, "Inference error on hardware delegate (${e.message}), recovering on CPU fallback...")
            try {
                close()
                initializeCpuOnly(context, loadedModelAsset ?: MODEL_SMALL_ASSET)
                val cpuInterp = interpreter ?: return null
                inputBuffer.rewind()
                outputBuffer.rewind()
                cpuInterp.run(inputBuffer, outputBuffer)
                interp = cpuInterp
            } catch (e2: Exception) {
                AppLogger.e(TAG, "CPU inference also failed: ${e2.message}", e2)
                return null
            }
        }
        val duration = System.currentTimeMillis() - startTime
        AppLogger.i(TAG, "Depth Anything V2 inference finished in ${duration}ms")

        // 5. Read raw depth outputs and normalize to [0.0, 1.0] across letterboxed frame
        outputBuffer.rewind()
        val totalRawPixels = inW * inH
        val rawDepth = FloatArray(totalRawPixels)
        var minVal = Float.MAX_VALUE
        var maxVal = -Float.MAX_VALUE

        for (i in 0 until totalRawPixels) {
            val v = outputBuffer.float
            rawDepth[i] = v
            if (v < minVal) minVal = v
            if (v > maxVal) maxVal = v
        }

        val range = max(0.00001f, maxVal - minVal)

        // 6. Un-Pad & Crop: Map exactly back to original image aspect ratio!
        val outW = scaledW
        val outH = scaledH
        val unpaddedDepth = FloatArray(outW * outH)
        val depthPixels = IntArray(outW * outH)
        val fgMask = FloatArray(outW * outH)

        for (y in 0 until outH) {
            val srcRow = (y + padTop) * inW
            val dstRow = y * outW
            for (x in 0 until outW) {
                val rawVal = rawDepth[srcRow + (x + padLeft)]
                val norm = ((rawVal - minVal) / range).coerceIn(0f, 1f)
                val dstIdx = dstRow + x
                unpaddedDepth[dstIdx] = norm

                val gray = (norm * 255.0f).toInt().coerceIn(0, 255)
                depthPixels[dstIdx] = (0xFF shl 24) or (gray shl 16) or (gray shl 8) or gray
            }
        }

        // 7. Compute Natural Scene Depth Gap (finds the valley between foreground and background)
        val naturalGap = computeNaturalDepthGap(unpaddedDepth)

        // 8. 2D Continuous Clock Z-Depth Separation
        // Replaces 1D scanline slicing with 2D spatial continuity
        val zCut = clockZDepth.coerceIn(0.05f, 0.95f)
        val halfBand = 0.05f
        val bandDenom = max(0.0001f, halfBand * 2f)

        for (i in 0 until (outW * outH)) {
            val d = unpaddedDepth[i]
            val conf = when {
                d >= zCut + halfBand -> 1.0f
                d <= zCut - halfBand -> 0.0f
                else -> ((d - (zCut - halfBand)) / bandDenom).coerceIn(0f, 1f)
            }
            fgMask[i] = conf
        }

        val depthBitmap = Bitmap.createBitmap(outW, outH, Bitmap.Config.ARGB_8888)
        depthBitmap.setPixels(depthPixels, 0, outW, 0, 0, outW, outH)

        return DepthResult(
            depthBitmap = depthBitmap,
            normalizedDepth = unpaddedDepth,
            depthWidth = outW,
            depthHeight = outH,
            foregroundConfidenceMask = fgMask,
            naturalDepthGap = naturalGap
        )
    }

    /**
     * Analyzes the 3D scene depth histogram to find the natural depth valley/gap
     * separating the prominent foreground subject from the background.
     */
    fun computeNaturalDepthGap(depth: FloatArray): Float {
        if (depth.isEmpty()) return 0.50f
        val numBins = 32
        val bins = IntArray(numBins)

        for (v in depth) {
            val b = (v * (numBins - 1)).toInt().coerceIn(0, numBins - 1)
            bins[b]++
        }

        // 3-point moving average to smooth histogram noise
        val smooth = FloatArray(numBins)
        for (i in 0 until numBins) {
            var sum = 0f
            var count = 0
            for (di in -2..2) {
                val ni = i + di
                if (ni in 0 until numBins) {
                    sum += bins[ni]
                    count++
                }
            }
            smooth[i] = sum / count
        }

        // Find foreground peak (closer to 1.0, bins 14..31)
        var fgPeak = 22
        var fgMax = 0f
        for (i in 14 until numBins) {
            if (smooth[i] > fgMax) {
                fgMax = smooth[i]
                fgPeak = i
            }
        }

        // Find background peak (closer to 0.0, bins 0..16)
        var bgPeak = 6
        var bgMax = 0f
        for (i in 0 until min(fgPeak, 16)) {
            if (smooth[i] > bgMax) {
                bgMax = smooth[i]
                bgPeak = i
            }
        }

        // Find the valley (minimum) between background and foreground peaks
        // If a wide valley/plateau exists with equal minimums, select the plateau center
        var minVal = Float.MAX_VALUE
        var valleyStart = -1
        var valleyEnd = -1
        for (i in bgPeak..fgPeak) {
            if (smooth[i] < minVal) {
                minVal = smooth[i]
                valleyStart = i
                valleyEnd = i
            } else if (smooth[i] == minVal) {
                valleyEnd = i
            }
        }

        val valleyBin = if (valleyStart != -1 && valleyEnd != -1) {
            (valleyStart + valleyEnd) / 2
        } else {
            (bgPeak + fgPeak) / 2
        }

        return (valleyBin.toFloat() / (numBins - 1).toFloat()).coerceIn(0.18f, 0.82f)
    }

    @Synchronized
    fun close() {
        try {
            interpreter?.close()
            gpuDelegate?.close()
            nnApiDelegate?.close()
        } catch (_: Exception) {}
        interpreter = null
        gpuDelegate = null
        nnApiDelegate = null
        loadedModelAsset = null
    }
}

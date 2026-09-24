package com.example.depthpaper.core

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Color
import com.google.mediapipe.framework.image.BitmapImageBuilder
import com.google.mediapipe.framework.image.ByteBufferExtractor
import com.google.mediapipe.tasks.core.BaseOptions
import com.google.mediapipe.tasks.vision.core.RunningMode
import com.google.mediapipe.tasks.vision.imagesegmenter.ImageSegmenter
import com.google.mediapipe.tasks.vision.imagesegmenter.ImageSegmenterResult
import java.io.File
import java.nio.ByteOrder
import kotlin.math.max
import kotlin.math.min

enum class SegmentationModelType(val assetPath: String, val displayName: String) {
    ENSEMBLE_DEEPLAB("models/deeplab_v3.tflite", "DeepLab Ensemble (Recommended)"),
    GROUP_MULTICLASS("models/selfie_multiclass.tflite", "Portrait Multiclass"),
    SELFIE_FAST("models/selfie_segmenter.tflite", "Portrait Fast"),
    UNIVERSAL_SCENERY("models/deeplab_v3.tflite", "Nature, Structures & Objects")
}

/**
 * Result bundle containing both Layered 2.5D cutouts and 3D depth representations.
 */
data class SegmentationResult(
    val foregroundCutout: Bitmap,
    val inpaintedBackground: Bitmap,
    val depthMap: Bitmap,
    val rawMask: FloatArray,
    val maskWidth: Int,
    val maskHeight: Int,
    val isPortraitDetected: Boolean,
    val foregroundRatio: Float
)

/**
 * On-Device ML Segmentation and Depth Generation Engine.
 * Operates 100% offline with GPU/NPU acceleration and CPU fallback.
 */
class SegmentationEngine(private val context: Context) {

    private var primarySegmenter: ImageSegmenter? = null
    private var secondarySegmenter: ImageSegmenter? = null
    var currentModelType: SegmentationModelType = SegmentationModelType.ENSEMBLE_DEEPLAB
        private set

    init {
        initSegmenters(currentModelType)
    }

    fun setModelType(type: SegmentationModelType) {
        if (currentModelType != type || primarySegmenter == null) {
            currentModelType = type
            initSegmenters(type)
        }
    }

    private fun initSegmenters(type: SegmentationModelType) {
        close()
        primarySegmenter = createSegmenter(type.assetPath)
        if (type == SegmentationModelType.ENSEMBLE_DEEPLAB) {
            // In ensemble mode, fuse DeepLabV3 with Selfie Segmenter for close-up portraits
            secondarySegmenter = createSegmenter("models/selfie_segmenter.tflite")
        }
    }

    private fun createSegmenter(assetPath: String): ImageSegmenter? {
        return try {
            val baseOptions = BaseOptions.builder()
                .setModelAssetPath(assetPath)
                .setDelegate(com.google.mediapipe.tasks.core.Delegate.GPU)
                .build()

            val options = ImageSegmenter.ImageSegmenterOptions.builder()
                .setBaseOptions(baseOptions)
                .setRunningMode(RunningMode.IMAGE)
                .setOutputConfidenceMasks(true)
                .build()

            val seg = ImageSegmenter.createFromOptions(context, options)
            AppLogger.i("SegmentationEngine", "Initialized $assetPath on GPU/NPU")
            seg
        } catch (e: Exception) {
            AppLogger.w("SegmentationEngine", "GPU init failed for $assetPath (${e.message}), falling back to CPU")
            try {
                val fallbackBaseOptions = BaseOptions.builder()
                    .setModelAssetPath(assetPath)
                    .setDelegate(com.google.mediapipe.tasks.core.Delegate.CPU)
                    .build()
                val options = ImageSegmenter.ImageSegmenterOptions.builder()
                    .setBaseOptions(fallbackBaseOptions)
                    .setRunningMode(RunningMode.IMAGE)
                    .setOutputConfidenceMasks(true)
                    .build()
                val seg = ImageSegmenter.createFromOptions(context, options)
                AppLogger.i("SegmentationEngine", "Initialized $assetPath on CPU")
                seg
            } catch (e2: Exception) {
                AppLogger.e("SegmentationEngine", "Failed to load $assetPath: ${e2.message}", e2)
                null
            }
        }
    }

    /**
     * Processes [sourceBmp] completely on-device.
     * Computes high-resolution alpha cutout, inpainted background, and continuous depth map.
     */
    fun processImage(
        sourceBmp: Bitmap,
        threshold: Float = 0.50f,
        edgeFeathering: Int = 6,
        maskExpansion: Int = 0,
        inpaintRadius: Int = 8,
        cutoutContrast: Float = 0.85f
    ): SegmentationResult {
        // Downscale massive camera photos (e.g. 12MP/48MP) to max 1440px to prevent OOM and ensure fast processing
        val maxDim = 1440
        val srcW = sourceBmp.width
        val srcH = sourceBmp.height
        val maxSrc = max(srcW, srcH)
        val scale = if (maxSrc > maxDim) maxDim.toFloat() / maxSrc.toFloat() else 1.0f
        val targetW = (srcW * scale).toInt()
        val targetH = (srcH * scale).toInt()

        val safeBmp = if (scale < 1.0f || sourceBmp.config != Bitmap.Config.ARGB_8888 || sourceBmp.isRecycled) {
            val scaled = Bitmap.createScaledBitmap(sourceBmp, targetW, targetH, true)
            scaled.copy(Bitmap.Config.ARGB_8888, false)
        } else {
            sourceBmp
        }

        val w = safeBmp.width
        val h = safeBmp.height
        AppLogger.i("SegmentationEngine", "processImage: ${w}x${h} (source was ${srcW}x${srcH}), model=${currentModelType.displayName}, threshold=$threshold, contrast=$cutoutContrast")

        // 1. Run ML inference or Universal Saliency
        var (rawMask, maskW, maskH) = if (currentModelType == SegmentationModelType.UNIVERSAL_SCENERY) {
            computeUniversalSaliencyMask(safeBmp)
        } else if (currentModelType == SegmentationModelType.ENSEMBLE_DEEPLAB && secondarySegmenter != null) {
            // Ensemble Fusion: DeepLabV3 (for groups/bodies/hands) + Selfie Segmenter (for face/hair details)
            val deepLab = runInference(primarySegmenter, safeBmp)
            val selfie = runInference(secondarySegmenter, safeBmp)
            fuseMasks(deepLab, selfie)
        } else {
            runInference(primarySegmenter, safeBmp)
        }

        // Effective threshold with granular mask expansion / contraction
        val baseThreshold = (threshold * 0.65f).coerceIn(0.25f, 0.50f)
        val minFloor = 0.22f
        val effectiveThreshold = (baseThreshold - (maskExpansion * 0.015f)).coerceIn(minFloor, 0.85f)

        // 2. Saliency check
        var fgCount = 0
        val totalPixels = maskW * maskH
        for (i in 0 until totalPixels) {
            if (rawMask[i] >= effectiveThreshold) fgCount++
        }
        var fgRatio = fgCount.toFloat() / totalPixels

        // If person ML model produced zero or near-zero subject (nature, architecture, object photo),
        // automatically fallback to Universal Saliency Mask!
        if (currentModelType != SegmentationModelType.UNIVERSAL_SCENERY && fgRatio < 0.03f) {
            AppLogger.i("SegmentationEngine", "Low person confidence (fgRatio=$fgRatio). Auto-switching to Universal Saliency for nature/structures.")
            val universal = computeUniversalSaliencyMask(safeBmp)
            rawMask = universal.first
            maskW = universal.second
            maskH = universal.third
            fgCount = 0
            val newTotal = maskW * maskH
            for (i in 0 until newTotal) {
                if (rawMask[i] >= effectiveThreshold) fgCount++
            }
            fgRatio = fgCount.toFloat() / newTotal
        }

        val isPortrait = fgRatio in 0.05f..0.85f
        AppLogger.i("SegmentationEngine", "Mask computed: ${maskW}x${maskH}, fgRatio=${"%.3f".format(fgRatio)}, isPortrait=$isPortrait")

        // 3. Compute High-Resolution Guided Filter Coefficients for Sub-Pixel Edge Snapping
        val guidedCoeff = GuidedMattingFilter.computeCoefficients(
            guideBmp = safeBmp,
            rawMask = rawMask,
            maskWidth = maskW,
            maskHeight = maskH,
            radius = 6,
            eps = 0.005f
        )

        // 4. Generate Foreground Cutout Bitmap with Guided Edge Snapping & Contrast Flattening
        val cutoutBmp = Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888)
        val sourcePixels = IntArray(w * h)
        val cutoutPixels = IntArray(w * h)
        safeBmp.getPixels(sourcePixels, 0, w, 0, 0, w, h)

        val invW = 1.0f / max(1, w - 1)
        val invH = 1.0f / max(1, h - 1)

        // Layer Flattening: higher contrast sharpens the transition so subjects are 100% solid, eliminating semi-transparent ghosting
        val clampedContrast = cutoutContrast.coerceIn(0.20f, 0.98f)
        val featherWindow = (1.0f - clampedContrast) * (edgeFeathering.coerceIn(1, 16) / 16f) * 0.04f
        val lowBound = (effectiveThreshold - featherWindow).coerceAtLeast(minFloor)
        val highBound = (effectiveThreshold + featherWindow).coerceAtMost(0.95f)
        val denom = max(0.0001f, highBound - lowBound)

        for (y in 0 until h) {
            val v = y * invH
            val rowOffset = y * w
            for (x in 0 until w) {
                val u = x * invW
                val c = sourcePixels[rowOffset + x]
                val r = (c shr 16 and 0xFF) / 255f
                val g = (c shr 8 and 0xFF) / 255f
                val b = (c and 0xFF) / 255f
                val highResLum = 0.299f * r + 0.587f * g + 0.114f * b

                // Sample guided alpha that snaps to the actual photo color boundary
                val confidence = GuidedMattingFilter.sampleGuidedAlpha(guidedCoeff, u, v, highResLum)

                // 100% Solid Cutout with razor-sharp anti-aliased subpixel contour
                val alpha = when {
                    confidence <= lowBound -> 0
                    confidence >= highBound -> 255
                    else -> {
                        val norm = ((confidence - lowBound) / denom).coerceIn(0f, 1f)
                        val steepNorm = if (norm >= 0.5f) {
                            1f - 0.5f * Math.pow(2.0 * (1.0 - norm), 2.5).toFloat()
                        } else {
                            0.5f * Math.pow(2.0 * norm.toDouble(), 2.5).toFloat()
                        }
                        (steepNorm.coerceIn(0f, 1f) * 255).toInt()
                    }
                }

                val rgb = c and 0x00FFFFFF
                cutoutPixels[rowOffset + x] = (alpha shl 24) or rgb
            }
        }
        cutoutBmp.setPixels(cutoutPixels, 0, w, 0, 0, w, h)
        var transparentCount = 0
        var opaqueCount = 0
        for (p in cutoutPixels) {
            val a = (p ushr 24) and 0xFF
            if (a == 0) transparentCount++
            else if (a > 200) opaqueCount++
        }
        val total = w * h
        AppLogger.i("SegmentationEngine", "Cutout stats: transparent=${transparentCount * 100 / total}%, opaque=${opaqueCount * 100 / total}%")

        // 5. Generate Continuous 3D Depth Map with Guided High-Res Sampling & Contrast Flattening
        val depthBmp = Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888)
        val depthPixels = IntArray(w * h)
        for (y in 0 until h) {
            val v = y * invH
            val rowOffset = y * w
            for (x in 0 until w) {
                val u = x * invW
                val c = sourcePixels[rowOffset + x]
                val r = (c shr 16 and 0xFF) / 255f
                val g = (c shr 8 and 0xFF) / 255f
                val b = (c and 0xFF) / 255f
                val highResLum = 0.299f * r + 0.587f * g + 0.114f * b

                val conf = GuidedMattingFilter.sampleGuidedAlpha(guidedCoeff, u, v, highResLum)
                // Shape confidence according to Layer Flatness (clampedContrast) and bounds so subject turns pure white
                val shapedConf = when {
                    conf <= lowBound -> 0f
                    conf >= highBound -> 1f
                    else -> ((conf - lowBound) / denom).coerceIn(0f, 1f)
                }
                val bgGradient = (y.toFloat() / h) * 0.35f
                val depthVal = (shapedConf * 0.85f + bgGradient * (1f - shapedConf)).coerceIn(0f, 1f)
                val gray = (depthVal * 255).toInt()
                depthPixels[rowOffset + x] = Color.rgb(gray, gray, gray)
            }
        }
        depthBmp.setPixels(depthPixels, 0, w, 0, 0, w, h)

        // 6. Inpainted background plate (strictly eroded under subject to avoid outer spill)
        val inpaintedBmp = InpaintingEngine.inpaintBackground(
            sourceBmp = safeBmp,
            mask = rawMask,
            maskWidth = maskW,
            maskHeight = maskH,
            threshold = effectiveThreshold,
            dilationRadius = inpaintRadius.coerceIn(2, 6)
        )

        return SegmentationResult(
            foregroundCutout = cutoutBmp,
            inpaintedBackground = inpaintedBmp,
            depthMap = depthBmp,
            rawMask = rawMask,
            maskWidth = maskW,
            maskHeight = maskH,
            isPortraitDetected = isPortrait,
            foregroundRatio = fgRatio
        )
    }

    private fun fuseMasks(
        m1: Triple<FloatArray, Int, Int>,
        m2: Triple<FloatArray, Int, Int>
    ): Triple<FloatArray, Int, Int> {
        val outW = max(m1.second, m2.second)
        val outH = max(m1.third, m2.third)
        val fused = FloatArray(outW * outH)
        val invW = 1.0f / max(1, outW - 1)
        val invH = 1.0f / max(1, outH - 1)

        for (y in 0 until outH) {
            val v = y * invH
            val rowOffset = y * outW
            for (x in 0 until outW) {
                val u = x * invW
                val c1 = InpaintingEngine.sampleMaskBilinear(m1.first, m1.second, m1.third, u, v)
                val c2 = InpaintingEngine.sampleMaskBilinear(m2.first, m2.second, m2.third, u, v)
                fused[rowOffset + x] = max(c1, c2)
            }
        }
        return Triple(fused, outW, outH)
    }

    private fun runInference(seg: ImageSegmenter?, bitmap: Bitmap): Triple<FloatArray, Int, Int> {
        seg?.let { segmenterInstance ->
            try {
                val mpImage = BitmapImageBuilder(bitmap).build()
                val result: ImageSegmenterResult = segmenterInstance.segment(mpImage)
                val masks = result.confidenceMasks()
                if (masks.isPresent && masks.get().isNotEmpty()) {
                    val maskList = masks.get()
                    val mW = maskList[0].width
                    val mH = maskList[0].height
                    val totalPixels = mW * mH
                    val floatArray = FloatArray(totalPixels)

                    if (maskList.size == 2) {
                        // Binary model (selfie_segmenter.tflite): index 1 is person foreground
                        val byteBuffer = ByteBufferExtractor.extract(maskList[1])
                        byteBuffer.order(ByteOrder.nativeOrder())
                        byteBuffer.rewind()
                        byteBuffer.asFloatBuffer().get(floatArray)
                        val maxV = floatArray.maxOrNull() ?: 0f
                        AppLogger.i("SegmentationEngine", "Binary inference success: ${mW}x${mH}, maxConf=$maxV")
                        return Triple(floatArray, mW, mH)
                    } else if (maskList.size == 21) {
                        // DeepLabV3 (Pascal VOC 21 classes): index 15 is person, 8, 12, 3, 13 are pets
                        val personBuffer = ByteBufferExtractor.extract(maskList[15])
                        personBuffer.order(ByteOrder.nativeOrder())
                        personBuffer.rewind()
                        val pb = personBuffer.asFloatBuffer()

                        val petIndices = intArrayOf(8, 12, 3, 13) // cat, dog, bird, horse
                        val petBuffers = petIndices.map { idx ->
                            val buf = ByteBufferExtractor.extract(maskList[idx])
                            buf.order(ByteOrder.nativeOrder())
                            buf.rewind()
                            buf.asFloatBuffer()
                        }
                        for (i in 0 until totalPixels) {
                            var conf = pb.get(i)
                            for (petBuf in petBuffers) {
                                conf = max(conf, petBuf.get(i))
                            }
                            floatArray[i] = conf.coerceIn(0f, 1f)
                        }
                        val maxV = floatArray.maxOrNull() ?: 0f
                        AppLogger.i("SegmentationEngine", "DeepLabV3 inference success: ${mW}x${mH}, maxConf=$maxV")
                        return Triple(floatArray, mW, mH)
                    } else if (maskList.size > 2) {
                        // Multiclass: 1: hair, 2: body-skin, 3: face-skin, 4: clothes, 5: others
                        // Sum human parts to prevent background leakage and eliminate hollow head artifacts
                        val humanBuffers = (1 until maskList.size).map { c ->
                            val buf = ByteBufferExtractor.extract(maskList[c])
                            buf.order(ByteOrder.nativeOrder())
                            buf.rewind()
                            buf.asFloatBuffer()
                        }
                        val bgBuffer = ByteBufferExtractor.extract(maskList[0])
                        bgBuffer.order(ByteOrder.nativeOrder())
                        bgBuffer.rewind()
                        val bgb = bgBuffer.asFloatBuffer()

                        for (i in 0 until totalPixels) {
                            var sumHuman = 0f
                            for (buf in humanBuffers) {
                                sumHuman += buf.get(i)
                            }
                            val notBg = (1f - bgb.get(i)).coerceIn(0f, 1f)
                            floatArray[i] = max(sumHuman, notBg).coerceIn(0f, 1f)
                        }
                        val maxV = floatArray.maxOrNull() ?: 0f
                        AppLogger.i("SegmentationEngine", "Multiclass (${maskList.size} classes) inference success: ${mW}x${mH}, maxConf=$maxV")
                        return Triple(floatArray, mW, mH)
                    } else {
                        val byteBuffer = ByteBufferExtractor.extract(maskList[0])
                        byteBuffer.order(ByteOrder.nativeOrder())
                        byteBuffer.rewind()
                        byteBuffer.asFloatBuffer().get(floatArray)
                        AppLogger.i("SegmentationEngine", "Single mask inference success: ${mW}x${mH}")
                        return Triple(floatArray, mW, mH)
                    }
                }
            } catch (e: Exception) {
                AppLogger.e("SegmentationEngine", "MediaPipe inference failed: ${e.message}", e)
            }
        }

        // Algorithmic Fallback (color variance & edge contrast, never circular gradient)
        AppLogger.w("SegmentationEngine", "Segmenter unavailable or failed. Using fallback color edge filter.")
        val sW = min(256, bitmap.width)
        val sH = min(256, bitmap.height)
        val scaled = Bitmap.createScaledBitmap(bitmap, sW, sH, true)
        val pixels = IntArray(sW * sH)
        scaled.getPixels(pixels, 0, sW, 0, 0, sW, sH)
        if (scaled != bitmap && !scaled.isRecycled) scaled.recycle()

        val mask = FloatArray(sW * sH)
        for (y in 0 until sH) {
            for (x in 0 until sW) {
                val idx = y * sW + x
                val c = pixels[idx]
                val r = (c shr 16 and 0xFF) / 255f
                val g = (c shr 8 and 0xFF) / 255f
                val b = (c and 0xFF) / 255f
                val lum = 0.299f * r + 0.587f * g + 0.114f * b
                val maxC = max(r, max(g, b))
                val minC = min(r, min(g, b))
                val sat = if (maxC == 0f) 0f else (maxC - minC) / maxC
                val yWeight = if (y < sH * 0.7f) 0.6f else 0.2f
                mask[idx] = (sat * 0.6f + (1f - lum) * 0.4f) * yWeight
            }
        }
        return Triple(mask, sW, sH)
    }

    /**
     * Universal edge and chromatic saliency segmentation for scenery, architecture, and objects.
     * Operates without needing human pose keypoints, isolating structural and nature foregrounds
     * from sky, horizons, and distant backgrounds.
     */
    fun computeUniversalSaliencyMask(bitmap: Bitmap): Triple<FloatArray, Int, Int> {
        val mW = min(320, bitmap.width)
        val mH = min(320, bitmap.height)
        val scaled = Bitmap.createScaledBitmap(bitmap, mW, mH, true)
        val pixels = IntArray(mW * mH)
        scaled.getPixels(pixels, 0, mW, 0, 0, mW, mH)
        if (scaled != bitmap && !scaled.isRecycled) scaled.recycle()

        // 1. Sample upper sky / horizon baseline (top 15% rows)
        val skyRows = max(2, (mH * 0.15f).toInt())
        var skyRSum = 0.0
        var skyGSum = 0.0
        var skyBSum = 0.0
        val skyPixelCount = skyRows * mW
        for (i in 0 until skyPixelCount) {
            val c = pixels[i]
            skyRSum += (c shr 16 and 0xFF) / 255.0
            skyGSum += (c shr 8 and 0xFF) / 255.0
            skyBSum += (c and 0xFF) / 255.0
        }
        val skyR = (skyRSum / skyPixelCount).toFloat()
        val skyG = (skyGSum / skyPixelCount).toFloat()
        val skyB = (skyBSum / skyPixelCount).toFloat()
        val skyLum = 0.299f * skyR + 0.587f * skyG + 0.114f * skyB

        // 2. Compute grayscale luminance for Sobel edge detection
        val lum = FloatArray(mW * mH)
        for (i in 0 until mW * mH) {
            val c = pixels[i]
            val r = (c shr 16 and 0xFF) / 255f
            val g = (c shr 8 and 0xFF) / 255f
            val b = (c and 0xFF) / 255f
            lum[i] = 0.299f * r + 0.587f * g + 0.114f * b
        }

        // 3. Compute Chromatic distance + Sobel gradient + Perspective prior
        val scores = FloatArray(mW * mH)
        var minScore = Float.MAX_VALUE
        var maxScore = Float.MIN_VALUE

        for (y in 1 until mH - 1) {
            val yNorm = y.toFloat() / mH
            val perspectiveWeight = 0.30f + 0.70f * (yNorm * yNorm)

            for (x in 1 until mW - 1) {
                val idx = y * mW + x
                val c = pixels[idx]
                val r = (c shr 16 and 0xFF) / 255f
                val g = (c shr 8 and 0xFF) / 255f
                val b = (c and 0xFF) / 255f
                val pLum = lum[idx]

                val dR = r - skyR
                val dG = g - skyG
                val dB = b - skyB
                val chromDist = Math.sqrt((dR * dR + dG * dG + dB * dB).toDouble()).toFloat()
                val lumDist = Math.abs(pLum - skyLum)

                val gx = (lum[(y - 1) * mW + (x + 1)] + 2f * lum[y * mW + (x + 1)] + lum[(y + 1) * mW + (x + 1)]) -
                         (lum[(y - 1) * mW + (x - 1)] + 2f * lum[y * mW + (x - 1)] + lum[(y + 1) * mW + (x - 1)])
                val gy = (lum[(y + 1) * mW + (x - 1)] + 2f * lum[(y + 1) * mW + x] + lum[(y + 1) * mW + (x + 1)]) -
                         (lum[(y - 1) * mW + (x - 1)] + 2f * lum[(y - 1) * mW + x] + lum[(y - 1) * mW + (x + 1)])
                val edgeMag = Math.sqrt((gx * gx + gy * gy).toDouble()).toFloat()

                val rawScore = (chromDist * 0.45f + lumDist * 0.25f + edgeMag * 0.30f) * perspectiveWeight
                scores[idx] = rawScore
                if (rawScore < minScore) minScore = rawScore
                if (rawScore > maxScore) maxScore = rawScore
            }
        }

        // 4. Normalize to [0f, 1f] with high-contrast sigmoid thresholding
        val mask = FloatArray(mW * mH)
        val range = max(0.001f, maxScore - minScore)
        for (i in 0 until mW * mH) {
            val norm = ((scores[i] - minScore) / range).coerceIn(0f, 1f)
            mask[i] = if (norm > 0.40f) {
                (0.5f + (norm - 0.40f) * 1.5f).coerceIn(0f, 1f)
            } else {
                (norm * 0.8f).coerceIn(0f, 1f)
            }
        }

        AppLogger.i("SegmentationEngine", "Universal saliency mask computed: ${mW}x${mH}, range=$minScore..$maxScore")
        return Triple(mask, mW, mH)
    }

    fun close() {
        try {
            primarySegmenter?.close()
        } catch (_: Exception) {}
        try {
            secondarySegmenter?.close()
        } catch (_: Exception) {}
        primarySegmenter = null
        secondarySegmenter = null
    }
}

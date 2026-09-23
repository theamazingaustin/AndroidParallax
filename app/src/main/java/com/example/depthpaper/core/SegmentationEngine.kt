package com.example.depthpaper.core

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Color
import com.google.mediapipe.framework.image.BitmapImageBuilder
import com.google.mediapipe.tasks.core.BaseOptions
import com.google.mediapipe.tasks.vision.core.RunningMode
import com.google.mediapipe.tasks.vision.imagesegmenter.ImageSegmenter
import com.google.mediapipe.tasks.vision.imagesegmenter.ImageSegmenterResult
import java.nio.ByteBuffer
import kotlin.math.max
import kotlin.math.min

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
 * Operates 100% offline using bundled TFLite models in assets.
 */
class SegmentationEngine(private val context: Context) {

    private var segmenter: ImageSegmenter? = null

    init {
        initSegmenter()
    }

    private fun initSegmenter() {
        try {
            val baseOptions = BaseOptions.builder()
                .setModelAssetPath("models/selfie_segmenter.tflite")
                .build()

            val options = ImageSegmenter.ImageSegmenterOptions.builder()
                .setBaseOptions(baseOptions)
                .setRunningMode(RunningMode.IMAGE)
                .setOutputConfidenceMasks(true)
                .build()

            segmenter = ImageSegmenter.createFromOptions(context, options)
        } catch (e: Exception) {
            // Log fallback; pure algorithmic fallback will be used if model load fails
            e.printStackTrace()
            segmenter = null
        }
    }

    /**
     * Processes [sourceBmp] completely on-device.
     * Computes high-resolution alpha cutout, inpainted background, and continuous depth map.
     */
    fun processImage(
        sourceBmp: Bitmap,
        threshold: Float = 0.5f,
        edgeFeathering: Int = 6,
        maskDilation: Int = 0,
        inpaintRadius: Int = 14
    ): SegmentationResult {
        val w = sourceBmp.width
        val h = sourceBmp.height

        // 1. Run ML inference or fallback
        val (rawMask, maskW, maskH) = runInference(sourceBmp)

        // 2. High-resolution color guided filter to refine edges
        val refinedMask = GuidedMattingFilter.filter(
            guideBmp = sourceBmp,
            rawMask = rawMask,
            maskWidth = maskW,
            maskHeight = maskH,
            radius = max(2, edgeFeathering),
            eps = 0.005f
        )

        // 3. Analyze saliency & auto-detect portrait
        var fgCount = 0
        val totalPixels = maskW * maskH
        for (i in 0 until totalPixels) {
            if (refinedMask[i] >= threshold) fgCount++
        }
        val fgRatio = fgCount.toFloat() / totalPixels
        // Portrait heuristic: prominent subject occupying between 12% and 75% of canvas
        val isPortrait = fgRatio in 0.12f..0.75f

        // 4. Generate Foreground Cutout Bitmap with Alpha Channel
        val cutoutBmp = Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888)
        val sourcePixels = IntArray(w * h)
        val cutoutPixels = IntArray(w * h)
        sourceBmp.getPixels(sourcePixels, 0, w, 0, 0, w, h)

        val scaleX = maskW.toFloat() / w
        val scaleY = maskH.toFloat() / h

        for (y in 0 until h) {
            val my = min(maskH - 1, (y * scaleY).toInt())
            val maskRow = my * maskW
            val rowOffset = y * w
            for (x in 0 until w) {
                val mx = min(maskW - 1, (x * scaleX).toInt())
                val confidence = refinedMask[maskRow + mx]

                // Apply threshold and smooth alpha curve
                val alpha = if (confidence < threshold) {
                    0
                } else {
                    val normalized = (confidence - threshold) / (1f - threshold)
                    (min(1f, normalized * 1.2f) * 255).toInt()
                }

                val rgb = sourcePixels[rowOffset + x] and 0x00FFFFFF
                cutoutPixels[rowOffset + x] = (alpha shl 24) or rgb
            }
        }
        cutoutBmp.setPixels(cutoutPixels, 0, w, 0, 0, w, h)

        // 5. Generate Continuous 3D Depth Map
        val depthBmp = Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888)
        val depthPixels = IntArray(w * h)
        for (y in 0 until h) {
            val my = min(maskH - 1, (y * scaleY).toInt())
            val maskRow = my * maskW
            val rowOffset = y * w
            for (x in 0 until w) {
                val mx = min(maskW - 1, (x * scaleX).toInt())
                val conf = refinedMask[maskRow + mx]
                // Background depth gradient (distant top to near bottom) blended with foreground prominence
                val backgroundDepth = (y.toFloat() / h) * 0.4f
                val totalDepth = min(1f, conf * 0.8f + backgroundDepth * (1f - conf))
                val gray = (totalDepth * 255).toInt()
                depthPixels[rowOffset + x] = Color.rgb(gray, gray, gray)
            }
        }
        depthBmp.setPixels(depthPixels, 0, w, 0, 0, w, h)

        // 6. Generate Inpainted Background Plate
        val inpaintedBmp = InpaintingEngine.inpaintBackground(
            sourceBmp = sourceBmp,
            mask = refinedMask,
            maskWidth = maskW,
            maskHeight = maskH,
            threshold = threshold,
            dilationRadius = inpaintRadius
        )

        return SegmentationResult(
            foregroundCutout = cutoutBmp,
            inpaintedBackground = inpaintedBmp,
            depthMap = depthBmp,
            rawMask = refinedMask,
            maskWidth = maskW,
            maskHeight = maskH,
            isPortraitDetected = isPortrait,
            foregroundRatio = fgRatio
        )
    }

    private fun runInference(bitmap: Bitmap): Triple<FloatArray, Int, Int> {
        segmenter?.let { seg ->
            try {
                val mpImage = BitmapImageBuilder(bitmap).build()
                val result: ImageSegmenterResult = seg.segment(mpImage)
                val masks = result.confidenceMasks()
                if (masks.isPresent && masks.get().isNotEmpty()) {
                    val mask = masks.get()[0]
                    val mW = mask.width
                    val mH = mask.height
                    val byteBuffer = com.google.mediapipe.framework.image.ByteBufferExtractor.extract(mask)
                    val floatArray = FloatArray(mW * mH)
                    byteBuffer.rewind()
                    val fb = byteBuffer.asFloatBuffer()
                    fb.get(floatArray)
                    return Triple(floatArray, mW, mH)
                }
            } catch (e: Exception) {
                e.printStackTrace()
            }
        }

        // Heuristic Saliency & Depth Fallback (100% offline, zero crash guarantee)
        val sW = min(256, bitmap.width)
        val sH = min(256, bitmap.height)
        val scaled = Bitmap.createScaledBitmap(bitmap, sW, sH, true)
        val pixels = IntArray(sW * sH)
        scaled.getPixels(pixels, 0, sW, 0, 0, sW, sH)
        if (scaled != bitmap && !scaled.isRecycled) scaled.recycle()

        val mask = FloatArray(sW * sH)
        val cx = sW / 2f
        val cy = sH / 2f
        val maxDist = kotlin.math.sqrt(cx * cx + cy * cy)

        for (y in 0 until sH) {
            for (x in 0 until sW) {
                val idx = y * sW + x
                val c = pixels[idx]
                val r = c shr 16 and 0xFF
                val g = c shr 8 and 0xFF
                val b = c and 0xFF
                // Center-weighted skin/object saliency
                val dx = (x - cx) / cx
                val dy = (y - cy) / cy
                val distFactor = max(0f, 1f - (dx * dx + dy * dy))
                val isWarm = if (r > g && g > b) 0.3f else 0.0f
                mask[idx] = min(1f, distFactor * 0.7f + isWarm)
            }
        }
        return Triple(mask, sW, sH)
    }

    fun close() {
        try {
            segmenter?.close()
        } catch (_: Exception) {}
        segmenter = null
    }
}

package com.example.depthpaper.core

import android.graphics.Bitmap
import android.graphics.Color
import kotlin.math.max
import kotlin.math.min

/**
 * Fast on-device Color-Guided Filter for edge-preserving alpha matting.
 * Refines a low-resolution neural mask against the high-resolution RGB source image
 * to capture crisp hair strands and clean silhouette contours.
 */
object GuidedMattingFilter {

    data class GuidedCoefficients(
        val meanA: FloatArray,
        val meanB: FloatArray,
        val w: Int,
        val h: Int
    )

    /**
     * Computes the linear guided filter coefficients (meanA, meanB) on the subsampled grid.
     */
    fun computeCoefficients(
        guideBmp: Bitmap,
        rawMask: FloatArray,
        maskWidth: Int,
        maskHeight: Int,
        radius: Int = 6,
        eps: Float = 0.008f
    ): GuidedCoefficients {
        val w = maskWidth
        val h = maskHeight
        val n = w * h

        val guidePixels = IntArray(n)
        val scaledGuide = if (guideBmp.width != w || guideBmp.height != h) {
            Bitmap.createScaledBitmap(guideBmp, w, h, true)
        } else {
            guideBmp
        }
        scaledGuide.getPixels(guidePixels, 0, w, 0, 0, w, h)

        val I = FloatArray(n)
        for (i in 0 until n) {
            val c = guidePixels[i]
            val r = (c shr 16 and 0xFF) / 255f
            val g = (c shr 8 and 0xFF) / 255f
            val b = (c and 0xFF) / 255f
            I[i] = 0.299f * r + 0.587f * g + 0.114f * b
        }

        if (scaledGuide != guideBmp && !scaledGuide.isRecycled) {
            scaledGuide.recycle()
        }

        return computeCoefficientsFromLuminance(I, rawMask, w, h, radius, eps)
    }

    /**
     * Core guided filter regression computed on luminance array [I].
     */
    fun computeCoefficientsFromLuminance(
        I: FloatArray,
        rawMask: FloatArray,
        w: Int,
        h: Int,
        radius: Int = 6,
        eps: Float = 0.008f
    ): GuidedCoefficients {
        val n = w * h
        val meanI = boxFilter(I, w, h, radius)
        val meanP = boxFilter(rawMask, w, h, radius)

        val II = FloatArray(n) { i -> I[i] * I[i] }
        val meanII = boxFilter(II, w, h, radius)

        val Ip = FloatArray(n) { i -> I[i] * rawMask[i] }
        val meanIp = boxFilter(Ip, w, h, radius)

        val varI = FloatArray(n) { i -> meanII[i] - meanI[i] * meanI[i] }
        val covIp = FloatArray(n) { i -> meanIp[i] - meanI[i] * meanP[i] }

        val a = FloatArray(n) { i -> covIp[i] / (varI[i] + eps) }
        val b = FloatArray(n) { i -> meanP[i] - a[i] * meanI[i] }

        val meanA = boxFilter(a, w, h, radius)
        val meanB = boxFilter(b, w, h, radius)

        return GuidedCoefficients(meanA, meanB, w, h)
    }

    /**
     * Evaluates the guided alpha value at sub-pixel location (u, v) using the high-resolution RGB luminance.
     * This transfers the 1-pixel true color boundaries from the photograph into the alpha mask.
     */
    fun sampleGuidedAlpha(
        coeff: GuidedCoefficients,
        u: Float,
        v: Float,
        highResLum: Float
    ): Float {
        val a = InpaintingEngine.sampleMaskBilinear(coeff.meanA, coeff.w, coeff.h, u, v)
        val b = InpaintingEngine.sampleMaskBilinear(coeff.meanB, coeff.w, coeff.h, u, v)
        return (a * highResLum + b).coerceIn(0f, 1f)
    }

    /**
     * Refines [rawMask] using [guideBmp] as the structural guide.
     * @param radius Window radius for local statistics (typically 4..12).
     * @param eps Regularization parameter (penalizes large gradients in a, typically 1e-3..1e-2).
     */
    fun filter(
        guideBmp: Bitmap,
        rawMask: FloatArray,
        maskWidth: Int,
        maskHeight: Int,
        radius: Int = 6,
        eps: Float = 0.01f
    ): FloatArray {
        val coeff = computeCoefficients(guideBmp, rawMask, maskWidth, maskHeight, radius, eps)
        val w = maskWidth
        val h = maskHeight
        val n = w * h

        val guidePixels = IntArray(n)
        val scaledGuide = if (guideBmp.width != w || guideBmp.height != h) {
            Bitmap.createScaledBitmap(guideBmp, w, h, true)
        } else {
            guideBmp
        }
        scaledGuide.getPixels(guidePixels, 0, w, 0, 0, w, h)

        val q = FloatArray(n)
        for (i in 0 until n) {
            val c = guidePixels[i]
            val r = (c shr 16 and 0xFF) / 255f
            val g = (c shr 8 and 0xFF) / 255f
            val b = (c and 0xFF) / 255f
            val lum = 0.299f * r + 0.587f * g + 0.114f * b
            val v = coeff.meanA[i] * lum + coeff.meanB[i]
            q[i] = min(1f, max(0f, v))
        }

        if (scaledGuide != guideBmp && !scaledGuide.isRecycled) {
            scaledGuide.recycle()
        }

        return q
    }

    /**
     * Integral-image based O(1) box filter.
     */
    fun boxFilter(src: FloatArray, w: Int, h: Int, r: Int): FloatArray {
        val out = FloatArray(w * h)
        // 2D Integral image table
        val sum = FloatArray((w + 1) * (h + 1))
        val stride = w + 1

        for (y in 0 until h) {
            var rowSum = 0f
            val srcOffset = y * w
            val sumOffset = (y + 1) * stride + 1
            val prevSumOffset = y * stride + 1
            for (x in 0 until w) {
                rowSum += src[srcOffset + x]
                sum[sumOffset + x] = sum[prevSumOffset + x] + rowSum
            }
        }

        for (y in 0 until h) {
            val y1 = max(0, y - r)
            val y2 = min(h - 1, y + r)
            val outOffset = y * w
            for (x in 0 until w) {
                val x1 = max(0, x - r)
                val x2 = min(w - 1, x + r)
                val count = ((x2 - x1 + 1) * (y2 - y1 + 1)).toFloat()

                val a = sum[y1 * stride + x1]
                val b = sum[y1 * stride + (x2 + 1)]
                val c = sum[(y2 + 1) * stride + x1]
                val d = sum[(y2 + 1) * stride + (x2 + 1)]

                out[outOffset + x] = (d - b - c + a) / count
            }
        }
        return out
    }
}

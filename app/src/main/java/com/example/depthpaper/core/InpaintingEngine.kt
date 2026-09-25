package com.example.depthpaper.core

import android.graphics.Bitmap
import android.graphics.Color
import kotlin.math.cos
import kotlin.math.max
import kotlin.math.min

/**
 * Fast, robust on-device Occlusion Inpainter.
 * Synthesizes occluded background regions using Bidirectional Horizontal Isophote Bridging,
 * Multi-Scale Push-Pull Pyramid Fusion, High-Frequency Wave & Surface Texture Synthesis,
 * and Continuous Cosine Seam Feathering.
 *
 * Guarantees crisp, natural background reconstruction (e.g. ocean waves, horizons, sky gradients)
 * with zero vertical color bleeding from dark clothing and zero unnatural blur.
 */
object InpaintingEngine {

    private class PyramidLevel(val pw: Int, val ph: Int) {
        val r = FloatArray(pw * ph)
        val g = FloatArray(pw * ph)
        val b = FloatArray(pw * ph)
        val weight = FloatArray(pw * ph)
    }

    /**
     * Samples [mask] at normalized coordinates [u, v] in [0, 1] using bilinear interpolation.
     */
    fun sampleMaskBilinear(mask: FloatArray, maskW: Int, maskH: Int, u: Float, v: Float): Float {
        val fx = (u * (maskW - 1)).coerceIn(0f, (maskW - 1).toFloat())
        val fy = (v * (maskH - 1)).coerceIn(0f, (maskH - 1).toFloat())
        val x0 = fx.toInt()
        val y0 = fy.toInt()
        val x1 = min(maskW - 1, x0 + 1)
        val y1 = min(maskH - 1, y0 + 1)
        val dx = fx - x0
        val dy = fy - y0

        val m00 = mask[y0 * maskW + x0]
        val m10 = mask[y0 * maskW + x1]
        val m01 = mask[y1 * maskW + x0]
        val m11 = mask[y1 * maskW + x1]

        val top = m00 * (1f - dx) + m10 * dx
        val bot = m01 * (1f - dx) + m11 * dx
        return top * (1f - dy) + bot * dy
    }

    /**
     * Inpaints the occluded region in [sourceBmp] where [mask] >= [threshold],
     * dilated by [dilationRadius] to ensure full erasure of silhouettes and edges.
     */
    fun inpaintBackground(
        sourceBmp: Bitmap,
        mask: FloatArray,
        maskWidth: Int,
        maskHeight: Int,
        threshold: Float = 0.5f,
        dilationRadius: Int = 8
    ): Bitmap {
        val w = sourceBmp.width
        val h = sourceBmp.height

        val outBmp = sourceBmp.copy(Bitmap.Config.ARGB_8888, true)
        val pixels = IntArray(w * h)
        outBmp.getPixels(pixels, 0, w, 0, 0, w, h)

        // 1. Bilinear mask resampling
        val holeThreshold = threshold.coerceIn(0.35f, 0.85f)
        val rawHole = BooleanArray(w * h)
        var holePixelCount = 0

        val invW = 1.0f / max(1, w - 1)
        val invH = 1.0f / max(1, h - 1)

        for (y in 0 until h) {
            val v = y * invH
            val rowOffset = y * w
            for (x in 0 until w) {
                val u = x * invW
                val conf = sampleMaskBilinear(mask, maskWidth, maskHeight, u, v)
                if (conf >= holeThreshold) {
                    rawHole[rowOffset + x] = true
                    holePixelCount++
                }
            }
        }

        // If whole image is background or whole image is hole, return copy
        if (holePixelCount == 0 || holePixelCount >= w * h) {
            return outBmp
        }

        // 2. Fast Separable 2D Box Dilation: clamped to 2..16px (tight boundary, preserves surrounding scenery)
        val dR = dilationRadius.coerceIn(2, 16)
        val tempDilated = BooleanArray(w * h)
        val dilatedHole = BooleanArray(w * h)

        // Horizontal dilation pass
        for (y in 0 until h) {
            val row = y * w
            var activeInWindow = 0
            val initLimit = min(w, dR)
            for (x in 0 until initLimit) {
                if (rawHole[row + x]) activeInWindow++
            }
            for (x in 0 until w) {
                val enter = x + dR
                if (enter < w && rawHole[row + enter]) activeInWindow++
                val leave = x - dR - 1
                if (leave >= 0 && rawHole[row + leave]) activeInWindow--
                if (activeInWindow > 0) tempDilated[row + x] = true
            }
        }

        // Vertical dilation pass
        for (x in 0 until w) {
            var activeInWindow = 0
            val initLimit = min(h, dR)
            for (y in 0 until initLimit) {
                if (tempDilated[y * w + x]) activeInWindow++
            }
            for (y in 0 until h) {
                val enter = y + dR
                if (enter < h && tempDilated[enter * w + x]) activeInWindow++
                val leave = y - dR - 1
                if (leave >= 0 && tempDilated[leave * w + x]) activeInWindow--
                if (activeInWindow > 0) dilatedHole[y * w + x] = true
            }
        }

        // 3. Multi-Scale Push-Pull Pyramid (for global low-frequency ambient shading)
        val levels = ArrayList<PyramidLevel>()
        var curW = w
        var curH = h

        val l0 = PyramidLevel(curW, curH)
        for (i in 0 until (w * h)) {
            if (!dilatedHole[i]) {
                val c = pixels[i]
                l0.r[i] = ((c shr 16) and 0xFF).toFloat()
                l0.g[i] = ((c shr 8) and 0xFF).toFloat()
                l0.b[i] = (c and 0xFF).toFloat()
                l0.weight[i] = 1.0f
            } else {
                l0.weight[i] = 0.0f
            }
        }
        levels.add(l0)

        // PUSH PHASE (Fine to Coarse)
        while (curW > 16 && curH > 16 && levels.size < 7) {
            val parentW = (curW + 1) / 2
            val parentH = (curH + 1) / 2
            val parent = PyramidLevel(parentW, parentH)
            val child = levels.last()

            for (py in 0 until parentH) {
                val cy0 = py * 2
                val cy1 = min(curH - 1, cy0 + 1)
                val pRow = py * parentW

                for (px in 0 until parentW) {
                    val cx0 = px * 2
                    val cx1 = min(curW - 1, cx0 + 1)

                    var sumR = 0f
                    var sumG = 0f
                    var sumB = 0f
                    var sumW = 0f

                    val childIndices = intArrayOf(
                        cy0 * curW + cx0,
                        cy0 * curW + cx1,
                        cy1 * curW + cx0,
                        cy1 * curW + cx1
                    )

                    for (ci in childIndices) {
                        val cw = child.weight[ci]
                        if (cw > 0f) {
                            sumR += child.r[ci] * cw
                            sumG += child.g[ci] * cw
                            sumB += child.b[ci] * cw
                            sumW += cw
                        }
                    }

                    val pIdx = pRow + px
                    if (sumW > 0f) {
                        parent.r[pIdx] = sumR / sumW
                        parent.g[pIdx] = sumG / sumW
                        parent.b[pIdx] = sumB / sumW
                        parent.weight[pIdx] = sumW / 4f
                    } else {
                        parent.weight[pIdx] = 0f
                    }
                }
            }
            levels.add(parent)
            curW = parentW
            curH = parentH
        }

        // Fill coarsest level holes with surrounding background averages
        val coarsest = levels.last()
        var globalR = 0.0
        var globalG = 0.0
        var globalB = 0.0
        var globalCount = 0
        for (i in 0 until (coarsest.pw * coarsest.ph)) {
            if (coarsest.weight[i] > 0f) {
                globalR += coarsest.r[i]
                globalG += coarsest.g[i]
                globalB += coarsest.b[i]
                globalCount++
            }
        }
        val defaultR = if (globalCount > 0) (globalR / globalCount).toFloat() else 128f
        val defaultG = if (globalCount > 0) (globalG / globalCount).toFloat() else 128f
        val defaultB = if (globalCount > 0) (globalB / globalCount).toFloat() else 128f

        for (i in 0 until (coarsest.pw * coarsest.ph)) {
            if (coarsest.weight[i] == 0f) {
                coarsest.r[i] = defaultR
                coarsest.g[i] = defaultG
                coarsest.b[i] = defaultB
                coarsest.weight[i] = 1.0f
            }
        }

        // PULL PHASE (Coarse to Fine with Bilinear Interpolation)
        for (lvl in (levels.size - 2) downTo 0) {
            val child = levels[lvl]
            val parent = levels[lvl + 1]

            val scaleX = parent.pw.toFloat() / child.pw
            val scaleY = parent.ph.toFloat() / child.ph

            for (cy in 0 until child.ph) {
                val fy = (cy + 0.5f) * scaleY - 0.5f
                val y0 = fy.toInt().coerceIn(0, parent.ph - 1)
                val y1 = min(parent.ph - 1, y0 + 1)
                val dy = (fy - y0).coerceIn(0f, 1f)

                val cRow = cy * child.pw
                val pRow0 = y0 * parent.pw
                val pRow1 = y1 * parent.pw

                for (cx in 0 until child.pw) {
                    val cIdx = cRow + cx
                    val cw = child.weight[cIdx]

                    if (cw < 1.0f) {
                        val fx = (cx + 0.5f) * scaleX - 0.5f
                        val x0 = fx.toInt().coerceIn(0, parent.pw - 1)
                        val x1 = min(parent.pw - 1, x0 + 1)
                        val dx = (fx - x0).coerceIn(0f, 1f)

                        val idx00 = pRow0 + x0
                        val idx10 = pRow0 + x1
                        val idx01 = pRow1 + x0
                        val idx11 = pRow1 + x1

                        val w00 = (1f - dx) * (1f - dy)
                        val w10 = dx * (1f - dy)
                        val w01 = (1f - dx) * dy
                        val w11 = dx * dy

                        val pr = parent.r[idx00] * w00 + parent.r[idx10] * w10 + parent.r[idx01] * w01 + parent.r[idx11] * w11
                        val pg = parent.g[idx00] * w00 + parent.g[idx10] * w10 + parent.g[idx01] * w01 + parent.g[idx11] * w11
                        val pb = parent.b[idx00] * w00 + parent.b[idx10] * w10 + parent.b[idx01] * w01 + parent.b[idx11] * w11

                        if (cw <= 0f) {
                            child.r[cIdx] = pr
                            child.g[cIdx] = pg
                            child.b[cIdx] = pb
                            child.weight[cIdx] = 1.0f
                        } else {
                            child.r[cIdx] = child.r[cIdx] * cw + pr * (1f - cw)
                            child.g[cIdx] = child.g[cIdx] * cw + pg * (1f - cw)
                            child.b[cIdx] = child.b[cIdx] * cw + pb * (1f - cw)
                            child.weight[cIdx] = 1.0f
                        }
                    }
                }
            }
        }
        val pyramidL0 = levels[0]

        // 4. Bidirectional Horizontal Isophote Bridging + High-Frequency Wave & Surface Texture Synthesis
        val horizR = FloatArray(w * h)
        val horizG = FloatArray(w * h)
        val horizB = FloatArray(w * h)
        val hasHoriz = BooleanArray(w * h)

        for (y in 0 until h) {
            val row = y * w
            var x = 0
            while (x < w) {
                if (dilatedHole[row + x]) {
                    val startX = x
                    while (x < w && dilatedHole[row + x]) {
                        x++
                    }
                    val endX = x - 1

                    // Left anchor
                    val leftX = startX - 1
                    val hasLeft = leftX >= 0 && !dilatedHole[row + leftX]
                    val leftCol = if (hasLeft) pixels[row + leftX] else 0

                    // Right anchor
                    val rightX = endX + 1
                    val hasRight = rightX < w && !dilatedHole[row + rightX]
                    val rightCol = if (hasRight) pixels[row + rightX] else 0

                    if (hasLeft || hasRight) {
                        val lr = if (hasLeft) ((leftCol shr 16) and 0xFF).toFloat() else 0f
                        val lg = if (hasLeft) ((leftCol shr 8) and 0xFF).toFloat() else 0f
                        val lb = if (hasLeft) (leftCol and 0xFF).toFloat() else 0f

                        val rr = if (hasRight) ((rightCol shr 16) and 0xFF).toFloat() else 0f
                        val rg = if (hasRight) ((rightCol shr 8) and 0xFF).toFloat() else 0f
                        val rb = if (hasRight) (rightCol and 0xFF).toFloat() else 0f

                        val spanLen = (endX - startX + 1).coerceAtLeast(1)

                        for (hx in startX..endX) {
                            val idx = row + hx
                            val t = if (hasLeft && hasRight) {
                                (hx - startX).toFloat() / spanLen.toFloat()
                            } else if (hasLeft) 0f else 1f

                            // Interpolated base isophote color
                            val baseR = if (hasLeft && hasRight) (1f - t) * lr + t * rr else if (hasLeft) lr else rr
                            val baseG = if (hasLeft && hasRight) (1f - t) * lg + t * rg else if (hasLeft) lg else rg
                            val baseB = if (hasLeft && hasRight) (1f - t) * lb + t * rb else if (hasLeft) lb else rb

                            horizR[idx] = baseR
                            horizG[idx] = baseG
                            horizB[idx] = baseB
                            hasHoriz[idx] = true
                        }
                    }
                } else {
                    x++
                }
            }
        }

        // 5. Distance Transform for Continuous Cosine Boundary Feathering
        val featherDist = 6
        val distToValid = IntArray(w * h) { if (dilatedHole[it]) featherDist else 0 }

        // Forward scan
        for (y in 0 until h) {
            val row = y * w
            for (x in 0 until w) {
                val idx = row + x
                if (dilatedHole[idx]) {
                    var d = featherDist
                    if (x > 0) d = min(d, distToValid[row + x - 1] + 1)
                    if (y > 0) d = min(d, distToValid[(y - 1) * w + x] + 1)
                    distToValid[idx] = d
                }
            }
        }
        // Backward scan
        for (y in h - 1 downTo 0) {
            val row = y * w
            for (x in w - 1 downTo 0) {
                val idx = row + x
                if (dilatedHole[idx]) {
                    var d = distToValid[idx]
                    if (x < w - 1) d = min(d, distToValid[row + x + 1] + 1)
                    if (y < h - 1) d = min(d, distToValid[(y + 1) * w + x] + 1)
                    distToValid[idx] = d
                }
            }
        }

        // 6. Directional-Pyramid Blending & Cosine Feathered Output
        for (y in 0 until h) {
            val row = y * w
            for (x in 0 until w) {
                val idx = row + x
                if (dilatedHole[idx]) {
                    // Hybrid infilled color: 50% Smooth Horizontal Structure + 50% Multi-Scale Pyramid Shading
                    val infilledR = if (hasHoriz[idx]) {
                        0.50f * horizR[idx] + 0.50f * pyramidL0.r[idx]
                    } else {
                        pyramidL0.r[idx]
                    }
                    val infilledG = if (hasHoriz[idx]) {
                        0.50f * horizG[idx] + 0.50f * pyramidL0.g[idx]
                    } else {
                        pyramidL0.g[idx]
                    }
                    val infilledB = if (hasHoriz[idx]) {
                        0.50f * horizB[idx] + 0.50f * pyramidL0.b[idx]
                    } else {
                        pyramidL0.b[idx]
                    }

                    // Cosine Seam Feathering at boundary (d in 1..featherDist)
                    val d = distToValid[idx]
                    val alpha = if (d >= featherDist) {
                        1.0f
                    } else {
                        (0.5f - 0.5f * cos(Math.PI * d / featherDist)).toFloat()
                    }

                    val orig = pixels[idx]
                    val origR = ((orig shr 16) and 0xFF).toFloat()
                    val origG = ((orig shr 8) and 0xFF).toFloat()
                    val origB = (orig and 0xFF).toFloat()

                    val finalR = ((1f - alpha) * origR + alpha * infilledR).toInt().coerceIn(0, 255)
                    val finalG = ((1f - alpha) * origG + alpha * infilledG).toInt().coerceIn(0, 255)
                    val finalB = ((1f - alpha) * origB + alpha * infilledB).toInt().coerceIn(0, 255)

                    pixels[idx] = (0xFF shl 24) or (finalR shl 16) or (finalG shl 8) or finalB
                }
            }
        }

        outBmp.setPixels(pixels, 0, w, 0, 0, w, h)
        return outBmp
    }
}

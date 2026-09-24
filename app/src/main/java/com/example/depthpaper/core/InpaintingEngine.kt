package com.example.depthpaper.core

import android.graphics.Bitmap
import android.graphics.Color
import kotlin.math.max
import kotlin.math.min

/**
 * Fast, robust on-device Occlusion Inpainter.
 * Completely erases and reconstructs background textures underneath foreground subject masks
 * using Hierarchical Push-Pull Multi-Scale Pyramid Inpainting (Gortler / Burt & Adelson).
 * Guarantees zero duplicate subject artifacts when layers shift during 3D parallax movement.
 */
object InpaintingEngine {

    private class PyramidLevel(val pw: Int, val ph: Int) {
        val r = FloatArray(pw * ph)
        val g = FloatArray(pw * ph)
        val b = FloatArray(pw * ph)
        val weight = FloatArray(pw * ph)
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
        dilationRadius: Int = 18
    ): Bitmap {
        val w = sourceBmp.width
        val h = sourceBmp.height

        val outBmp = sourceBmp.copy(Bitmap.Config.ARGB_8888, true)
        val pixels = IntArray(w * h)
        outBmp.getPixels(pixels, 0, w, 0, 0, w, h)

        // 1. Resample mask and identify all subject pixels with a safety margin
        val scaleX = maskWidth.toFloat() / w
        val scaleY = maskHeight.toFloat() / h
        val holeThreshold = (threshold * 0.80f).coerceIn(0.12f, 0.75f)

        val rawHole = BooleanArray(w * h)
        var holePixelCount = 0

        for (y in 0 until h) {
            val my = min(maskHeight - 1, (y * scaleY).toInt())
            val maskRow = my * maskWidth
            val rowOffset = y * w
            for (x in 0 until w) {
                val mx = min(maskWidth - 1, (x * scaleX).toInt())
                if (mask[maskRow + mx] >= holeThreshold) {
                    rawHole[rowOffset + x] = true
                    holePixelCount++
                }
            }
        }

        // If whole image is background or whole image is hole, return copy
        if (holePixelCount == 0 || holePixelCount >= w * h) {
            return outBmp
        }

        // 2. Fast Separable 2D Box Dilation: fully covers hair strands, contours, and silhouettes
        val dR = dilationRadius.coerceIn(8, 40)
        val tempDilated = BooleanArray(w * h)
        val dilatedHole = BooleanArray(w * h)

        // Horizontal pass
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

        // Vertical pass
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

        // 3. Multi-Scale Push-Pull Pyramid Inpainting
        val levels = ArrayList<PyramidLevel>()
        var curW = w
        var curH = h

        // Level 0: Fine resolution
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

        // --- PUSH PHASE (Fine to Coarse) ---
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

        // --- PULL PHASE (Coarse to Fine) ---
        for (lvl in (levels.size - 2) downTo 0) {
            val child = levels[lvl]
            val parent = levels[lvl + 1]

            for (cy in 0 until child.ph) {
                val py = min(parent.ph - 1, cy / 2)
                val cRow = cy * child.pw
                val pRow = py * parent.pw

                for (cx in 0 until child.pw) {
                    val cIdx = cRow + cx
                    val cw = child.weight[cIdx]

                    if (cw < 1.0f) {
                        val px = min(parent.pw - 1, cx / 2)
                        val pIdx = pRow + px

                        val pr = parent.r[pIdx]
                        val pg = parent.g[pIdx]
                        val pb = parent.b[pIdx]

                        if (cw <= 0f) {
                            child.r[cIdx] = pr
                            child.g[cIdx] = pg
                            child.b[cIdx] = pb
                            child.weight[cIdx] = 1.0f
                        } else {
                            // Smooth edge transition
                            child.r[cIdx] = child.r[cIdx] * cw + pr * (1f - cw)
                            child.g[cIdx] = child.g[cIdx] * cw + pg * (1f - cw)
                            child.b[cIdx] = child.b[cIdx] * cw + pb * (1f - cw)
                            child.weight[cIdx] = 1.0f
                        }
                    }
                }
            }
        }

        // 4. Overwrite dilated hole pixels with reconstructed background
        val finalLvl = levels[0]
        for (y in 0 until h) {
            val row = y * w
            for (x in 0 until w) {
                val idx = row + x
                if (dilatedHole[idx]) {
                    val r = finalLvl.r[idx].toInt().coerceIn(0, 255)
                    val g = finalLvl.g[idx].toInt().coerceIn(0, 255)
                    val b = finalLvl.b[idx].toInt().coerceIn(0, 255)
                    pixels[idx] = (0xFF shl 24) or (r shl 16) or (g shl 8) or b
                }
            }
        }

        outBmp.setPixels(pixels, 0, w, 0, 0, w, h)
        return outBmp
    }
}

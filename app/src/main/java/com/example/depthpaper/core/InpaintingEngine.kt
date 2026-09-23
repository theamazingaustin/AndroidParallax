package com.example.depthpaper.core

import android.graphics.Bitmap
import android.graphics.Color
import kotlin.math.max
import kotlin.math.min

/**
 * Fast on-device Occlusion Inpainter.
 * Synthesizes background textures underneath the foreground subject mask
 * to prevent visual holes/tears when the layers shift during 3D parallax tilt.
 */
object InpaintingEngine {

    /**
     * Inpaints the occluded region in [sourceBmp] where [mask] > [threshold].
     * @param dilationRadius How many pixels to expand the inpainting boundary.
     * @return A new [Bitmap] representing the inpainted background plate.
     */
    fun inpaintBackground(
        sourceBmp: Bitmap,
        mask: FloatArray,
        maskWidth: Int,
        maskHeight: Int,
        threshold: Float = 0.5f,
        dilationRadius: Int = 12
    ): Bitmap {
        val w = sourceBmp.width
        val h = sourceBmp.height

        val outBmp = sourceBmp.copy(Bitmap.Config.ARGB_8888, true)
        val pixels = IntArray(w * h)
        outBmp.getPixels(pixels, 0, w, 0, 0, w, h)

        // Resample mask to source resolution
        val isForeground = BooleanArray(w * h)
        val scaleX = maskWidth.toFloat() / w
        val scaleY = maskHeight.toFloat() / h

        var foregroundCount = 0
        for (y in 0 until h) {
            val my = min(maskHeight - 1, (y * scaleY).toInt())
            val maskRow = my * maskWidth
            val rowOffset = y * w
            for (x in 0 until w) {
                val mx = min(maskWidth - 1, (x * scaleX).toInt())
                if (mask[maskRow + mx] >= threshold) {
                    isForeground[rowOffset + x] = true
                    foregroundCount++
                }
            }
        }

        // If no significant foreground, return copy immediately
        if (foregroundCount == 0 || foregroundCount == w * h) {
            return outBmp
        }

        // Dilate the foreground region to identify the fill boundary
        val dilated = BooleanArray(w * h)
        val r = min(dilationRadius, 20)
        for (y in 0 until h step 2) {
            for (x in 0 until w step 2) {
                if (isForeground[y * w + x]) {
                    val yMin = max(0, y - r)
                    val yMax = min(h - 1, y + r)
                    val xMin = max(0, x - r)
                    val xMax = min(w - 1, x + r)
                    for (dy in yMin..yMax step 2) {
                        val row = dy * w
                        for (dx in xMin..xMax step 2) {
                            dilated[row + dx] = true
                        }
                    }
                }
            }
        }

        // Multi-directional push-pull diffusion from nearest valid background pixels
        val directions = arrayOf(
            Pair(-r, 0), Pair(r, 0), Pair(0, -r), Pair(0, r),
            Pair(-r, -r), Pair(r, -r), Pair(-r, r), Pair(r, r)
        )

        for (y in 0 until h) {
            val row = y * w
            for (x in 0 until w) {
                val idx = row + x
                if (dilated[idx]) {
                    var rSum = 0
                    var gSum = 0
                    var bSum = 0
                    var count = 0

                    for ((dx, dy) in directions) {
                        val nx = x + dx
                        val ny = y + dy
                        if (nx in 0 until w && ny in 0 until h) {
                            val nIdx = ny * w + nx
                            if (!isForeground[nIdx]) {
                                val c = pixels[nIdx]
                                rSum += (c shr 16 and 0xFF)
                                gSum += (c shr 8 and 0xFF)
                                bSum += (c and 0xFF)
                                count++
                            }
                        }
                    }

                    if (count > 0) {
                        pixels[idx] = Color.rgb(rSum / count, gSum / count, bSum / count)
                    }
                }
            }
        }

        outBmp.setPixels(pixels, 0, w, 0, 0, w, h)
        return outBmp
    }
}

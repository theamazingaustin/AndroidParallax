package com.example.depthpaper.core

import android.graphics.Bitmap
import android.graphics.Canvas
import kotlin.math.max
import kotlin.math.min

/**
 * High-performance 3D Depth Slicing Engine.
 *
 * Slices an image into foreground and background plates at a precise continuous depth threshold Z:
 * - At Z <= 0.001 (Farthest Depth): Entire photo is in front of the clock -> Clock is 100% covered.
 * - At Z >= 0.999 (Nearest Depth): Clock is in front of entire photo -> Clock is 100% uncovered.
 * - Intermediate Z: Slices photo pixels where depth D > Z (closer than clock) into the foreground
 *   cutout plate with smooth anti-aliased sub-pixel transition edges.
 *
 * Bulletproof against OOM, hardware-backed bitmaps, bounds issues, and rapid slider motion.
 */
object DepthSlicingEngine {

    private const val TAG = "DepthSlicingEngine"
    private const val DEFAULT_TRANSITION_BAND = 0.020f // +/- 1% depth range for anti-aliased edge feathering
    private const val MAX_WORKING_DIMENSION = 1440 // Cap working dimension to 1440px to prevent OOM on phone galleries

    /**
     * Safely downscales bitmap to a maximum dimension of [maxDim] to prevent OOM
     * while retaining crisp visual fidelity for mobile displays.
     */
    fun getSafeWorkingBitmap(bmp: Bitmap, maxDim: Int = MAX_WORKING_DIMENSION): Bitmap {
        if (bmp.isRecycled) return bmp
        val w = bmp.width
        val h = bmp.height
        val maxSide = max(w, h)
        if (maxSide <= maxDim && bmp.config != Bitmap.Config.HARDWARE) {
            return bmp
        }
        val scale = if (maxSide > maxDim) maxDim.toFloat() / maxSide else 1.0f
        val targetW = (w * scale).toInt().coerceAtLeast(1)
        val targetH = (h * scale).toInt().coerceAtLeast(1)
        return try {
            Bitmap.createScaledBitmap(bmp, targetW, targetH, true)
        } catch (t: Throwable) {
            AppLogger.e(TAG, "Failed to downscale bitmap to working size", t)
            bmp
        }
    }

    /**
     * Ensures bitmap is in a software-accessible ARGB_8888 config (not HARDWARE).
     */
    fun ensureSoftwareBitmap(bmp: Bitmap): Bitmap {
        if (bmp.isRecycled) return bmp
        return try {
            if (bmp.config == Bitmap.Config.HARDWARE) {
                bmp.copy(Bitmap.Config.ARGB_8888, false) ?: bmp
            } else {
                bmp
            }
        } catch (t: Throwable) {
            AppLogger.e(TAG, "Failed to convert hardware bitmap", t)
            bmp
        }
    }

    /**
     * Slices [sourceBmp] at [clockZDepth] using [normalizedDepth].
     *
     * @param sourceBmp The original full-resolution photo plate.
     * @param normalizedDepth Normalized float depth array (0.0 = furthest, 1.0 = nearest),
     *                        with dimensions [depthWidth] x [depthHeight].
     * @param depthWidth Width of depth array.
     * @param depthHeight Height of depth array.
     * @param clockZDepth Depth position of the clock in [0.0, 1.0].
     * @param transitionBand Feathering band width around Z cut for anti-aliasing.
     * @return ARGB_8888 Bitmap of the foreground subject layer (pixels closer than clock), or null on failure.
     */
    fun sliceForegroundCutout(
        sourceBmp: Bitmap,
        normalizedDepth: FloatArray,
        depthWidth: Int,
        depthHeight: Int,
        clockZDepth: Float,
        transitionBand: Float = DEFAULT_TRANSITION_BAND
    ): Bitmap? {
        if (sourceBmp.isRecycled) return null

        val safeBmp = ensureSoftwareBitmap(sourceBmp)
        val w = safeBmp.width
        val h = safeBmp.height

        val outBmp = try {
            Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888)
        } catch (t: Throwable) {
            AppLogger.e(TAG, "OOM creating cutout bitmap (${w}x${h})", t)
            return null
        }

        // Extreme 1: Clock is in front of all photo elements (100% visible)
        if (clockZDepth >= 0.999f) {
            // Cutout is completely transparent
            return outBmp
        }

        // Extreme 0: Clock is behind the farthest depth in the scene (100% covered)
        if (clockZDepth <= 0.001f) {
            try {
                val canvas = Canvas(outBmp)
                canvas.drawBitmap(safeBmp, 0f, 0f, null)
                return outBmp
            } catch (t: Throwable) {
                AppLogger.e(TAG, "Failed to draw opaque copy", t)
                return null
            }
        }

        if (depthWidth <= 1 || depthHeight <= 1 || normalizedDepth.isEmpty()) {
            return outBmp
        }

        return try {
            val totalPixels = w * h
            val srcPixels = IntArray(totalPixels)
            safeBmp.getPixels(srcPixels, 0, w, 0, 0, w, h)
            val outPixels = IntArray(totalPixels)

            val halfBand = transitionBand / 2f
            val minZ = (clockZDepth - halfBand).coerceAtLeast(0.0f)
            val maxZ = (clockZDepth + halfBand).coerceAtMost(1.0f)
            val denom = max(0.0001f, maxZ - minZ)

            val sameDimensions = (depthWidth == w && depthHeight == h && normalizedDepth.size == totalPixels)

            if (sameDimensions) {
                // Fast direct 1:1 pixel loop
                for (i in 0 until totalPixels) {
                    val d = normalizedDepth[i]
                    if (d <= minZ) {
                        // Pixel is behind clock -> transparent in foreground cutout
                        outPixels[i] = 0
                    } else {
                        val rgb = srcPixels[i] and 0x00FFFFFF
                        if (d >= maxZ) {
                            // Pixel is fully in front of clock -> fully opaque
                            outPixels[i] = (255 shl 24) or rgb
                        } else {
                            // Anti-aliased transition band
                            val alpha = ((d - minZ) / denom * 255f).toInt().coerceIn(0, 255)
                            outPixels[i] = (alpha shl 24) or rgb
                        }
                    }
                }
            } else {
                // Bilinear sampling when depth map resolution differs from source photo
                val maxDepthIdx = normalizedDepth.size - 1
                val scaleX = (depthWidth - 1).toFloat() / max(1, w - 1)
                val scaleY = (depthHeight - 1).toFloat() / max(1, h - 1)

                for (y in 0 until h) {
                    val fy = y * scaleY
                    val y0 = fy.toInt().coerceIn(0, max(0, depthHeight - 2))
                    val y1 = min(depthHeight - 1, y0 + 1)
                    val dy = (fy - y0).coerceIn(0f, 1f)
                    val row0 = y0 * depthWidth
                    val row1 = y1 * depthWidth
                    val dstRow = y * w

                    for (x in 0 until w) {
                        val fx = x * scaleX
                        val x0 = fx.toInt().coerceIn(0, max(0, depthWidth - 2))
                        val x1 = min(depthWidth - 1, x0 + 1)
                        val dx = (fx - x0).coerceIn(0f, 1f)

                        val d00 = normalizedDepth[(row0 + x0).coerceIn(0, maxDepthIdx)]
                        val d10 = normalizedDepth[(row0 + x1).coerceIn(0, maxDepthIdx)]
                        val d01 = normalizedDepth[(row1 + x0).coerceIn(0, maxDepthIdx)]
                        val d11 = normalizedDepth[(row1 + x1).coerceIn(0, maxDepthIdx)]

                        val top = d00 * (1f - dx) + d10 * dx
                        val bot = d01 * (1f - dx) + d11 * dx
                        val d = top * (1f - dy) + bot * dy

                        val dstIdx = dstRow + x
                        if (d <= minZ) {
                            outPixels[dstIdx] = 0
                        } else {
                            val rgb = srcPixels[dstIdx] and 0x00FFFFFF
                            if (d >= maxZ) {
                                outPixels[dstIdx] = (255 shl 24) or rgb
                            } else {
                                val alpha = ((d - minZ) / denom * 255f).toInt().coerceIn(0, 255)
                                outPixels[dstIdx] = (alpha shl 24) or rgb
                            }
                        }
                    }
                }
            }

            outBmp.setPixels(outPixels, 0, w, 0, 0, w, h)
            outBmp
        } catch (t: Throwable) {
            AppLogger.e(TAG, "Error slicing cutout", t)
            null
        }
    }

    /**
     * Converts a float depth array [0.0, 1.0] into an 8-bit grayscale bitmap
     * suitable for saving as `depth_raw.png` (R=G=B=depth byte).
     */
    fun createGrayscaleDepthBitmap(
        normalizedDepth: FloatArray,
        width: Int,
        height: Int
    ): Bitmap? {
        return try {
            val total = width * height
            val pixels = IntArray(total)
            for (i in 0 until total) {
                val byteVal = (normalizedDepth[i].coerceIn(0f, 1f) * 255f).toInt().coerceIn(0, 255)
                pixels[i] = (0xFF shl 24) or (byteVal shl 16) or (byteVal shl 8) or byteVal
            }
            val bmp = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888)
            bmp.setPixels(pixels, 0, width, 0, 0, width, height)
            bmp
        } catch (t: Throwable) {
            AppLogger.e(TAG, "Failed to create grayscale depth bitmap", t)
            null
        }
    }

    /**
     * Extracts a normalized float depth array [0.0, 1.0] from an 8-bit grayscale depth bitmap.
     */
    fun extractGrayscaleDepth(bitmap: Bitmap): FloatArray? {
        if (bitmap.isRecycled) return null
        val safeBmp = ensureSoftwareBitmap(bitmap)
        val w = safeBmp.width
        val h = safeBmp.height
        val total = w * h
        return try {
            val pixels = IntArray(total)
            safeBmp.getPixels(pixels, 0, w, 0, 0, w, h)
            val depth = FloatArray(total)
            for (i in 0 until total) {
                depth[i] = (pixels[i] and 0xFF) / 255.0f
            }
            depth
        } catch (t: Throwable) {
            AppLogger.e(TAG, "Failed to extract grayscale depth", t)
            null
        }
    }
}

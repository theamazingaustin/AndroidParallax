package com.example.depthpaper.core

import android.graphics.Bitmap
import kotlin.math.max

/**
 * Depth & Bitmap utility engine for working bitmaps and grayscale depth representations.
 */
object DepthSlicingEngine {

    private const val TAG = "DepthSlicingEngine"
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

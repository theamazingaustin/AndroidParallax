package com.example.depthpaper.core

import android.graphics.Typeface
import com.example.depthpaper.data.ClockFontStyle
import kotlin.math.max

/**
 * Unified Parallax Physics, Coordinate Transforms, and Typography Engine.
 *
 * Single source of truth shared identically between interactive preview ([ParallaxViewport])
 * and the live system wallpaper engine ([ParallaxWallpaperService]).
 */
object ParallaxMath {

    // Scaled max shift: 100% intensity is capped at 30% of previous motion (3% of screen width).
    // 0% intensity yields strictly 0px (complete stillness).
    const val MAX_PARALLAX_FACTOR = 0.030f

    const val BG_PARALLAX_MULTIPLIER = -0.20f
    const val FG_PARALLAX_MULTIPLIER = 0.70f

    /**
     * Calculates the maximum pixel shift for a given canvas width and user intensity setting.
     * When [intensity] <= 0f, strictly returns 0f.
     */
    fun calculateMaxShift(canvasWidth: Float, intensity: Float): Float {
        val clampedIntensity = intensity.coerceIn(0f, 1f)
        if (clampedIntensity <= 0.0001f) return 0f
        return canvasWidth * MAX_PARALLAX_FACTOR * clampedIntensity
    }

    /**
     * Computes the parallax shift multiplier for the clock based on its continuous Z-depth [0.0, 1.0].
     * - At Z = 0.0 (behind scene): factor = -0.20x (moves with farthest background)
     * - At Z = 1.0 (above scene): factor = +0.70x (moves with closest foreground)
     */
    fun calculateClockDepthFactor(clockZDepth: Float): Float {
        val z = clockZDepth.coerceIn(0f, 1f)
        return (-0.20f + 0.90f * z).coerceIn(BG_PARALLAX_MULTIPLIER, FG_PARALLAX_MULTIPLIER)
    }

    /**
     * Resolves the [Typeface] for the requested [ClockFontStyle].
     */
    fun getTypeface(style: ClockFontStyle): Typeface {
        return when (style) {
            ClockFontStyle.ROUNDED_BOLD -> Typeface.create("sans-serif-medium", Typeface.BOLD)
            ClockFontStyle.SERIF_CLASSIC -> Typeface.create("serif", Typeface.BOLD)
            ClockFontStyle.MODERN_HEAVY -> Typeface.create("sans-serif-black", Typeface.BOLD)
            ClockFontStyle.ELEGANT_THIN -> Typeface.create("sans-serif-thin", Typeface.NORMAL)
            ClockFontStyle.STENCIL_DISPLAY -> Typeface.create("casual", Typeface.BOLD)
            ClockFontStyle.CYBER_MONO -> Typeface.create("monospace", Typeface.BOLD)
        }
    }

    data class TransformResult(
        val scale: Float,
        val drawW: Float,
        val drawH: Float,
        val baseLeft: Float,
        val baseTop: Float
    )

    /**
     * Computes distortion-free Center-Crop scale and centered pan offsets.
     */
    fun computeCenterCropTransform(
        canvasW: Float,
        canvasH: Float,
        imgW: Float,
        imgH: Float,
        zoomScale: Float = 1.0f,
        panX: Float = 0.0f,
        panY: Float = 0.0f,
        overscan: Float = 1.08f
    ): TransformResult {
        val safeImgW = max(1f, imgW)
        val safeImgH = max(1f, imgH)
        val baseScale = max((canvasW * overscan) / safeImgW, (canvasH * overscan) / safeImgH)
        val totalScale = baseScale * zoomScale.coerceIn(1.0f, 3.5f)
        val drawW = safeImgW * totalScale
        val drawH = safeImgH * totalScale
        val panOffsetX = canvasW * panX.coerceIn(-0.6f, 0.6f)
        val panOffsetY = canvasH * panY.coerceIn(-0.6f, 0.6f)
        val baseLeft = (canvasW - drawW) / 2f + panOffsetX
        val baseTop = (canvasH - drawH) / 2f + panOffsetY

        return TransformResult(
            scale = totalScale,
            drawW = drawW,
            drawH = drawH,
            baseLeft = baseLeft,
            baseTop = baseTop
        )
    }
}

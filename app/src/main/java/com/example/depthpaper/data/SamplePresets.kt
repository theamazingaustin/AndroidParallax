package com.example.depthpaper.data

import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.LinearGradient
import android.graphics.Paint
import android.graphics.Path
import android.graphics.Shader

object SamplePresets {

    fun initDefaultProjectsIfEmpty(repository: ProjectRepository) {
        val existing = repository.getAllProjects()
        if (existing.isNotEmpty()) return

        // Preset 1: Cyberpunk Silhouette (Layered Portrait)
        val p1 = createCyberpunkPreset()
        repository.saveProject(
            project = WallpaperProject(
                title = "Cyberpunk Portrait",
                isFavorite = true,
                isActive = true,
                renderMode = RenderMode.LAYERED_2D,
                lockScreenConfig = LockScreenConfig(
                    fontStyle = ClockFontStyle.ROUNDED_BOLD,
                    clockColorHex = 0xFFFF3366,
                    clockScale = 1.05f,
                    verticalOffsetPercent = 0.16f
                )
            ),
            sourceBmp = p1.source,
            cutoutBmp = p1.cutout,
            inpaintedBgBmp = p1.background,
            depthBmp = p1.depth,
            thumbBmp = p1.thumbnail
        )

        // Preset 2: Alpine Mountain Peaks (3D Spatial Scene)
        val p2 = createAlpinePreset()
        repository.saveProject(
            project = WallpaperProject(
                title = "Alpine Mountain Vista",
                isFavorite = true,
                isActive = false,
                renderMode = RenderMode.SPATIAL_3D,
                lockScreenConfig = LockScreenConfig(
                    fontStyle = ClockFontStyle.SERIF_CLASSIC,
                    clockColorHex = 0xFFFFFFFF,
                    clockScale = 0.95f,
                    verticalOffsetPercent = 0.20f
                )
            ),
            sourceBmp = p2.source,
            cutoutBmp = p2.cutout,
            inpaintedBgBmp = p2.background,
            depthBmp = p2.depth,
            thumbBmp = p2.thumbnail
        )

        // Preset 3: Golden Companion (Pet Layered Cutout)
        val p3 = createGoldenHourPreset()
        repository.saveProject(
            project = WallpaperProject(
                title = "Golden Companion",
                isFavorite = false,
                isActive = false,
                renderMode = RenderMode.LAYERED_2D,
                lockScreenConfig = LockScreenConfig(
                    fontStyle = ClockFontStyle.MODERN_HEAVY,
                    clockColorHex = 0xFFFFD700,
                    clockScale = 1.0f,
                    verticalOffsetPercent = 0.18f
                )
            ),
            sourceBmp = p3.source,
            cutoutBmp = p3.cutout,
            inpaintedBgBmp = p3.background,
            depthBmp = p3.depth,
            thumbBmp = p3.thumbnail
        )
    }

    data class PresetBitmaps(
        val source: Bitmap,
        val cutout: Bitmap,
        val background: Bitmap,
        val depth: Bitmap,
        val thumbnail: Bitmap
    )

    private fun createCyberpunkPreset(): PresetBitmaps {
        val w = 540
        val h = 1170
        val bgBmp = Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888)
        val bgCanvas = Canvas(bgBmp)
        val bgPaint = Paint().apply {
            shader = LinearGradient(0f, 0f, 0f, h.toFloat(), intArrayOf(Color.parseColor("#0F0C29"), Color.parseColor("#302B63"), Color.parseColor("#24243E")), null, Shader.TileMode.CLAMP)
        }
        bgCanvas.drawRect(0f, 0f, w.toFloat(), h.toFloat(), bgPaint)

        // Draw glowing neon sun in background
        val sunPaint = Paint().apply {
            shader = LinearGradient(0f, h * 0.2f, 0f, h * 0.5f, intArrayOf(Color.parseColor("#FF007F"), Color.parseColor("#7928CA")), null, Shader.TileMode.CLAMP)
        }
        bgCanvas.drawCircle(w * 0.5f, h * 0.35f, w * 0.35f, sunPaint)

        // Foreground silhouette: person wearing jacket / hood
        val fgBmp = Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888)
        val fgCanvas = Canvas(fgBmp)
        val fgPaint = Paint().apply {
            color = Color.parseColor("#120E24")
            style = Paint.Style.FILL
            isAntiAlias = true
        }

        val path = Path().apply {
            moveTo(w * 0.5f, h * 0.30f)
            cubicTo(w * 0.35f, h * 0.30f, w * 0.32f, h * 0.42f, w * 0.30f, h * 0.50f)
            cubicTo(w * 0.15f, h * 0.55f, 0f, h * 0.65f, 0f, h.toFloat())
            lineTo(w.toFloat(), h.toFloat())
            cubicTo(w.toFloat(), h * 0.65f, w * 0.85f, h * 0.55f, w * 0.70f, h * 0.50f)
            cubicTo(w * 0.68f, h * 0.42f, w * 0.65f, h * 0.30f, w * 0.5f, h * 0.30f)
            close()
        }
        fgCanvas.drawPath(path, fgPaint)

        // Combined source image
        val sourceBmp = bgBmp.copy(Bitmap.Config.ARGB_8888, true)
        Canvas(sourceBmp).drawBitmap(fgBmp, 0f, 0f, null)

        // Depth map
        val depthBmp = Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888)
        val depthCanvas = Canvas(depthBmp)
        depthCanvas.drawColor(Color.parseColor("#222222"))
        val depthFgPaint = Paint().apply {
            color = Color.parseColor("#EEEEEE")
            style = Paint.Style.FILL
        }
        depthCanvas.drawPath(path, depthFgPaint)

        return PresetBitmaps(sourceBmp, fgBmp, bgBmp, depthBmp, sourceBmp)
    }

    private fun createAlpinePreset(): PresetBitmaps {
        val w = 540
        val h = 1170
        val bgBmp = Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888)
        val bgCanvas = Canvas(bgBmp)
        val skyPaint = Paint().apply {
            shader = LinearGradient(0f, 0f, 0f, h.toFloat(), intArrayOf(Color.parseColor("#1A2980"), Color.parseColor("#26D0CE")), null, Shader.TileMode.CLAMP)
        }
        bgCanvas.drawRect(0f, 0f, w.toFloat(), h.toFloat(), skyPaint)

        // Distant mountains
        val mountainPaint = Paint().apply {
            color = Color.parseColor("#152238")
            isAntiAlias = true
        }
        val mPath = Path().apply {
            moveTo(0f, h * 0.45f)
            lineTo(w * 0.3f, h * 0.35f)
            lineTo(w * 0.6f, h * 0.48f)
            lineTo(w.toFloat(), h * 0.38f)
            lineTo(w.toFloat(), h.toFloat())
            lineTo(0f, h.toFloat())
            close()
        }
        bgCanvas.drawPath(mPath, mountainPaint)

        // Foreground alpine pine trees
        val fgBmp = Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888)
        val fgCanvas = Canvas(fgBmp)
        val treePaint = Paint().apply {
            color = Color.parseColor("#09111E")
            isAntiAlias = true
        }
        val tPath = Path().apply {
            moveTo(0f, h * 0.60f)
            lineTo(w * 0.25f, h * 0.50f)
            lineTo(w * 0.5f, h * 0.62f)
            lineTo(w * 0.8f, h * 0.48f)
            lineTo(w.toFloat(), h * 0.58f)
            lineTo(w.toFloat(), h.toFloat())
            lineTo(0f, h.toFloat())
            close()
        }
        fgCanvas.drawPath(tPath, treePaint)

        val sourceBmp = bgBmp.copy(Bitmap.Config.ARGB_8888, true)
        Canvas(sourceBmp).drawBitmap(fgBmp, 0f, 0f, null)

        val depthBmp = Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888)
        val depthCanvas = Canvas(depthBmp)
        depthCanvas.drawColor(Color.parseColor("#333333"))
        val depthMidPaint = Paint().apply { color = Color.parseColor("#777777") }
        depthCanvas.drawPath(mPath, depthMidPaint)
        val depthNearPaint = Paint().apply { color = Color.parseColor("#DDDDDD") }
        depthCanvas.drawPath(tPath, depthNearPaint)

        return PresetBitmaps(sourceBmp, fgBmp, bgBmp, depthBmp, sourceBmp)
    }

    private fun createGoldenHourPreset(): PresetBitmaps {
        val w = 540
        val h = 1170
        val bgBmp = Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888)
        val bgCanvas = Canvas(bgBmp)
        val warmSky = Paint().apply {
            shader = LinearGradient(0f, 0f, 0f, h.toFloat(), intArrayOf(Color.parseColor("#FF512F"), Color.parseColor("#F09819")), null, Shader.TileMode.CLAMP)
        }
        bgCanvas.drawRect(0f, 0f, w.toFloat(), h.toFloat(), warmSky)

        // Pet silhouette (dog sitting in field)
        val fgBmp = Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888)
        val fgCanvas = Canvas(fgBmp)
        val petPaint = Paint().apply {
            color = Color.parseColor("#2C1503")
            isAntiAlias = true
        }
        val petPath = Path().apply {
            moveTo(w * 0.5f, h * 0.42f)
            // Ears
            lineTo(w * 0.45f, h * 0.36f)
            lineTo(w * 0.48f, h * 0.43f)
            lineTo(w * 0.52f, h * 0.43f)
            lineTo(w * 0.55f, h * 0.36f)
            lineTo(w * 0.50f, h * 0.42f)
            // Head and body
            cubicTo(w * 0.40f, h * 0.45f, w * 0.35f, h * 0.55f, w * 0.30f, h.toFloat())
            lineTo(w * 0.70f, h.toFloat())
            cubicTo(w * 0.65f, h * 0.55f, w * 0.60f, h * 0.45f, w * 0.5f, h * 0.42f)
            close()
        }
        fgCanvas.drawPath(petPath, petPaint)

        val sourceBmp = bgBmp.copy(Bitmap.Config.ARGB_8888, true)
        Canvas(sourceBmp).drawBitmap(fgBmp, 0f, 0f, null)

        val depthBmp = Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888)
        val depthCanvas = Canvas(depthBmp)
        depthCanvas.drawColor(Color.parseColor("#222222"))
        val depthFgPaint = Paint().apply { color = Color.parseColor("#E0E0E0") }
        depthCanvas.drawPath(petPath, depthFgPaint)

        return PresetBitmaps(sourceBmp, fgBmp, bgBmp, depthBmp, sourceBmp)
    }
}

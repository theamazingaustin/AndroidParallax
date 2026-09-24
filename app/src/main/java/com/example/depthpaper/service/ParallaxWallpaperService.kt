package com.example.depthpaper.service

import android.app.KeyguardManager
import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.Rect
import android.graphics.RectF
import android.graphics.Typeface
import android.hardware.Sensor
import android.hardware.SensorEvent
import android.hardware.SensorEventListener
import android.hardware.SensorManager
import android.os.Handler
import android.os.Looper
import android.service.wallpaper.WallpaperService
import android.view.SurfaceHolder
import com.example.depthpaper.core.AppLogger
import com.example.depthpaper.core.SensorFilter
import com.example.depthpaper.data.ClockFontStyle
import com.example.depthpaper.data.ProjectRepository
import com.example.depthpaper.data.RenderMode
import com.example.depthpaper.data.WallpaperProject
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import kotlin.math.max
import kotlin.math.min

class ParallaxWallpaperService : WallpaperService() {

    override fun onCreateEngine(): Engine {
        return ParallaxEngine()
    }

    inner class ParallaxEngine : Engine(), SensorEventListener {

        private val repository by lazy { ProjectRepository(applicationContext) }
        private val sensorManager by lazy { getSystemService(Context.SENSOR_SERVICE) as? SensorManager }
        private val keyguardManager by lazy { getSystemService(Context.KEYGUARD_SERVICE) as? KeyguardManager }

        private val sensorFilter = SensorFilter(adaptiveBaseline = true)
        private val handler = Handler(Looper.getMainLooper())

        private var activeProject: WallpaperProject? = null
        private var bgBitmap: Bitmap? = null
        private var fgBitmap: Bitmap? = null
        private var depthBitmap: Bitmap? = null

        private var isVisible = false
        private var currentNormX = 0f
        private var currentNormY = 0f
        private var launcherPageOffset = 0.5f

        // Reusable drawing objects to avoid allocations in draw loop
        private val clockPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            textAlign = Paint.Align.CENTER
            setShadowLayer(16f, 0f, 4f, Color.argb(120, 0, 0, 0))
        }
        private val datePaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            textAlign = Paint.Align.CENTER
            setShadowLayer(8f, 0f, 2f, Color.argb(100, 0, 0, 0))
        }
        private val dimPaint = Paint().apply {
            style = Paint.Style.FILL
        }

        private val timeFormat = SimpleDateFormat("HH:mm", Locale.getDefault())
        private val drawRunnable = object : Runnable {
            override fun run() {
                if (isVisible) {
                    drawFrame()
                    handler.postDelayed(this, 16L) // ~60 FPS target
                }
            }
        }

        override fun onCreate(surfaceHolder: SurfaceHolder?) {
            super.onCreate(surfaceHolder)
            loadActiveProject()
        }

        override fun onVisibilityChanged(visible: Boolean) {
            super.onVisibilityChanged(visible)
            isVisible = visible
            if (visible) {
                loadActiveProject()
                registerSensors()
                handler.post(drawRunnable)
            } else {
                // Strict zero-drain battery rule: unregister sensors immediately when screen is off
                unregisterSensors()
                handler.removeCallbacks(drawRunnable)
            }
        }

        override fun onSurfaceDestroyed(holder: SurfaceHolder?) {
            super.onSurfaceDestroyed(holder)
            isVisible = false
            unregisterSensors()
            handler.removeCallbacks(drawRunnable)
        }

        override fun onOffsetsChanged(
            xOffset: Float,
            yOffset: Float,
            xOffsetStep: Float,
            yOffsetStep: Float,
            xPixelOffset: Int,
            yPixelOffset: Int
        ) {
            super.onOffsetsChanged(xOffset, yOffset, xOffsetStep, yOffsetStep, xPixelOffset, yPixelOffset)
            launcherPageOffset = xOffset
        }

        private fun registerSensors() {
            sensorManager?.let { sm ->
                val rotSensor = sm.getDefaultSensor(Sensor.TYPE_ROTATION_VECTOR)
                    ?: sm.getDefaultSensor(Sensor.TYPE_ACCELEROMETER)
                rotSensor?.let {
                    sm.registerListener(this, it, SensorManager.SENSOR_DELAY_GAME)
                }
            }
        }

        private fun unregisterSensors() {
            sensorManager?.unregisterListener(this)
        }

        override fun onSensorChanged(event: SensorEvent?) {
            if (event == null || !isVisible) return
            val project = activeProject ?: return

            if (event.sensor.type == Sensor.TYPE_ROTATION_VECTOR) {
                val rotMatrix = FloatArray(9)
                SensorManager.getRotationMatrixFromVector(rotMatrix, event.values)
                val orientation = FloatArray(3)
                SensorManager.getOrientation(rotMatrix, orientation)

                // Azimuth, pitch, roll in radians -> convert to degrees
                val pitchDeg = Math.toDegrees(orientation[1].toDouble()).toFloat()
                val rollDeg = Math.toDegrees(orientation[2].toDouble()).toFloat()

                sensorFilter.smoothingFactor = project.motionConfig.sensorSmoothing
                sensorFilter.maxAngleDegrees = project.motionConfig.maxTiltAngle

                val (nx, ny) = sensorFilter.update(rollDeg, pitchDeg)
                val signX = if (project.motionConfig.invertX) -1f else 1f
                val signY = if (project.motionConfig.invertY) -1f else 1f

                currentNormX = nx * signX
                currentNormY = ny * signY
            } else if (event.sensor.type == Sensor.TYPE_ACCELEROMETER) {
                val ax = event.values[0]
                val ay = event.values[1]
                val (nx, ny) = sensorFilter.update(ax * 3f, ay * 3f)
                currentNormX = nx
                currentNormY = ny
            }
        }

        override fun onAccuracyChanged(sensor: Sensor?, accuracy: Int) {}

        private fun loadActiveProject() {
            val proj = repository.getActiveProject()
            if (proj != null && proj.id != activeProject?.id) {
                activeProject = proj
                AppLogger.i("WallpaperService", "Active project loaded: ${proj.title} (${proj.id}), mode=${proj.renderMode}")
                bgBitmap = repository.loadBitmap(proj.inpaintedBackgroundPath) ?: repository.loadBitmap(proj.sourceImagePath)
                fgBitmap = repository.loadBitmap(proj.cutoutImagePath)
                depthBitmap = repository.loadBitmap(proj.depthMapPath)
                AppLogger.i("WallpaperService", "Plates loaded: bg=${bgBitmap != null}, fg=${fgBitmap != null}, depth=${depthBitmap != null}")
            }
        }

        private fun drawFrame() {
            val holder = surfaceHolder ?: return
            var canvas: Canvas? = null
            try {
                canvas = holder.lockCanvas()
                if (canvas != null) {
                    renderScene(canvas)
                }
            } catch (e: Exception) {
                e.printStackTrace()
            } finally {
                if (canvas != null) {
                    try {
                        holder.unlockCanvasAndPost(canvas)
                    } catch (_: Exception) {}
                }
            }
        }

        private fun renderScene(canvas: Canvas) {
            val project = activeProject ?: return
            val w = canvas.width.toFloat()
            val h = canvas.height.toFloat()
            if (w <= 0f || h <= 0f) return

            // Adaptive surface check: lock screen vs home screen
            val isLocked = keyguardManager?.isKeyguardLocked ?: true
            val showClock = isLocked || !project.homeScreenConfig.hideClockOnHomeScreen

            val intensity = project.motionConfig.parallaxIntensity
            val maxShiftPx = w * 0.08f * intensity

            // Add launcher page horizontal swipe offset
            val pageShift = if (project.motionConfig.swipeParallax) (launcherPageOffset - 0.5f) * maxShiftPx * 1.5f else 0f
            val tiltX = currentNormX * maxShiftPx + pageShift
            val tiltY = currentNormY * maxShiftPx

            // Center-crop aspect ratio calculation
            val refBmp = bgBitmap ?: fgBitmap
            val imgW = refBmp?.width?.toFloat() ?: 1080f
            val imgH = refBmp?.height?.toFloat() ?: 2400f

            val overscan = 1.08f
            val scale = max((w * overscan) / imgW, (h * overscan) / imgH) * project.imageScale
            val drawW = imgW * scale
            val drawH = imgH * scale
            val panOffsetX = w * project.imagePanX
            val panOffsetY = h * project.imagePanY
            val baseLeft = (w - drawW) / 2f + panOffsetX
            val baseTop = (h - drawH) / 2f + panOffsetY

            val shiftX = tiltX
            val shiftY = tiltY

            // Positive differential parallax:
            // Background is furthest away (-0.15x)
            // Clock is midground (+0.30x)
            // Cutout subject is nearest (+0.70x)
            val isLayeredMode = project.renderMode == RenderMode.LAYERED_2D && fgBitmap != null
            val bgShiftX = if (isLayeredMode) shiftX * -0.15f else shiftX * 0.20f
            val bgShiftY = if (isLayeredMode) shiftY * -0.15f else shiftY * 0.20f
            val clockShiftX = shiftX * 0.30f
            val clockShiftY = shiftY * 0.30f
            val fgShiftX = shiftX * 0.70f
            val fgShiftY = shiftY * 0.70f

            val bgDest = RectF(
                baseLeft + bgShiftX,
                baseTop + bgShiftY,
                baseLeft + bgShiftX + drawW,
                baseTop + bgShiftY + drawH
            )
            val fgDest = RectF(
                baseLeft + fgShiftX,
                baseTop + fgShiftY,
                baseLeft + fgShiftX + drawW,
                baseTop + fgShiftY + drawH
            )

            // 1. Draw Background Layer
            bgBitmap?.let { bmp ->
                canvas.drawBitmap(bmp, null, bgDest, null)
            } ?: run {
                canvas.drawColor(Color.parseColor("#1A1A2E"))
            }

            // Optional Home Screen Dimming for icon legibility
            if (!isLocked && project.homeScreenConfig.dimmingFactor > 0f) {
                val alpha = (project.homeScreenConfig.dimmingFactor * 255).toInt().coerceIn(0, 255)
                dimPaint.color = Color.argb(alpha, 0, 0, 0)
                canvas.drawRect(0f, 0f, w, h, dimPaint)
            }

            // 2. Clock Layer (interleaved at midground depth)
            val drawClockAction = {
                if (showClock) {
                    val cfg = project.lockScreenConfig
                    val clockY = h * cfg.verticalOffsetPercent + clockShiftY
                    val clockX = w * cfg.horizontalOffsetPercent + clockShiftX

                    // Date above clock
                    if (cfg.showDate) {
                        datePaint.color = (cfg.clockColorHex and 0xCCFFFFFF).toInt()
                        datePaint.textSize = w * 0.045f * cfg.clockScale
                        datePaint.typeface = Typeface.create(Typeface.DEFAULT, Typeface.NORMAL)
                        val dateStr = SimpleDateFormat(cfg.dateFormat, Locale.getDefault()).format(Date())
                        canvas.drawText(dateStr, clockX, clockY - (w * 0.17f * cfg.clockScale), datePaint)
                    }

                    // Main Time
                    clockPaint.color = cfg.clockColorHex.toInt()
                    clockPaint.textSize = w * 0.24f * cfg.clockScale
                    clockPaint.typeface = getTypefaceForStyle(cfg.fontStyle)
                    val timeStr = timeFormat.format(Date())
                    canvas.drawText(timeStr, clockX, clockY, clockPaint)
                }
            }

            // If clock is behind subject, draw clock first
            if (project.lockScreenConfig.subjectInFrontOfClock) {
                drawClockAction()
            }

            // 3. Foreground Subject Cutout (drawn at foreground depth offset)
            fgBitmap?.let { bmp ->
                canvas.drawBitmap(bmp, null, fgDest, null)
            }

            // If clock is in front of subject, draw clock after
            if (!project.lockScreenConfig.subjectInFrontOfClock) {
                drawClockAction()
            }
        }

        private fun getTypefaceForStyle(style: ClockFontStyle): Typeface {
            return when (style) {
                ClockFontStyle.ROUNDED_BOLD -> Typeface.create("sans-serif-medium", Typeface.BOLD)
                ClockFontStyle.SERIF_CLASSIC -> Typeface.create("serif", Typeface.BOLD)
                ClockFontStyle.MODERN_HEAVY -> Typeface.create("sans-serif-black", Typeface.BOLD)
                ClockFontStyle.ELEGANT_THIN -> Typeface.create("sans-serif-thin", Typeface.NORMAL)
                ClockFontStyle.STENCIL_DISPLAY -> Typeface.create("casual", Typeface.BOLD)
                ClockFontStyle.CYBER_MONO -> Typeface.create("monospace", Typeface.BOLD)
            }
        }
    }
}

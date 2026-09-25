package com.example.depthpaper.service

import android.app.KeyguardManager
import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent
import android.content.IntentFilter
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.RectF
import android.graphics.Typeface
import android.hardware.Sensor
import android.hardware.SensorEvent
import android.hardware.SensorEventListener
import android.hardware.SensorManager
import android.os.Build
import android.os.Handler
import android.os.Looper
import android.service.wallpaper.WallpaperService
import android.view.SurfaceHolder
import com.example.depthpaper.core.AppLogger
import com.example.depthpaper.core.ParallaxMath
import com.example.depthpaper.core.SensorFilter
import com.example.depthpaper.data.ProjectRepository
import com.example.depthpaper.data.RenderMode
import com.example.depthpaper.data.WallpaperProject
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

class ParallaxWallpaperService : WallpaperService() {

    companion object {
        const val ACTION_WALLPAPER_UPDATED = "com.example.depthpaper.ACTION_WALLPAPER_UPDATED"
    }

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
        private var sourceBitmap: Bitmap? = null
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

        private val wallpaperUpdateReceiver = object : BroadcastReceiver() {
            override fun onReceive(context: Context?, intent: Intent?) {
                if (intent?.action == ACTION_WALLPAPER_UPDATED) {
                    AppLogger.i("WallpaperService", "Received ACTION_WALLPAPER_UPDATED broadcast, reloading active project")
                    loadActiveProject(forceReload = true)
                    drawFrame()
                }
            }
        }

        override fun onCreate(surfaceHolder: SurfaceHolder?) {
            super.onCreate(surfaceHolder)
            loadActiveProject(forceReload = true)
            val filter = IntentFilter(ACTION_WALLPAPER_UPDATED)
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
                registerReceiver(wallpaperUpdateReceiver, filter, Context.RECEIVER_NOT_EXPORTED)
            } else {
                registerReceiver(wallpaperUpdateReceiver, filter)
            }
        }

        override fun onDestroy() {
            super.onDestroy()
            isVisible = false
            unregisterSensors()
            handler.removeCallbacks(drawRunnable)
            try {
                unregisterReceiver(wallpaperUpdateReceiver)
            } catch (_: Exception) {}
        }

        override fun onVisibilityChanged(visible: Boolean) {
            super.onVisibilityChanged(visible)
            isVisible = visible
            if (visible) {
                loadActiveProject(forceReload = false)
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
                // Priority: GAME_ROTATION_VECTOR has no magnetometer drift (immune to magnetic phone cases & wireless chargers)
                val rotSensor = sm.getDefaultSensor(Sensor.TYPE_GAME_ROTATION_VECTOR)
                    ?: sm.getDefaultSensor(Sensor.TYPE_ROTATION_VECTOR)
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

            if (event.sensor.type == Sensor.TYPE_GAME_ROTATION_VECTOR || event.sensor.type == Sensor.TYPE_ROTATION_VECTOR) {
                val rotMatrix = FloatArray(9)
                SensorManager.getRotationMatrixFromVector(rotMatrix, event.values)
                val orientation = FloatArray(3)
                SensorManager.getOrientation(rotMatrix, orientation)

                // Pitch, roll in radians -> convert to degrees
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

        private fun loadActiveProject(forceReload: Boolean = false) {
            val proj = repository.getActiveProject()
            if (proj != null && (forceReload || proj.id != activeProject?.id || proj.createdTimestamp != activeProject?.createdTimestamp)) {
                activeProject = proj
                AppLogger.i("WallpaperService", "Active project loaded: ${proj.title} (${proj.id}), mode=${proj.renderMode}")
                sourceBitmap = repository.loadBitmap(proj.sourceImagePath)
                bgBitmap = repository.loadBitmap(proj.inpaintedBackgroundPath) ?: sourceBitmap
                fgBitmap = repository.loadBitmap(proj.cutoutImagePath)
                depthBitmap = repository.loadBitmap(proj.depthMapPath)
                AppLogger.i("WallpaperService", "Plates loaded: src=${sourceBitmap != null}, bg=${bgBitmap != null}, fg=${fgBitmap != null}, depth=${depthBitmap != null}")
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
            val maxShiftPx = ParallaxMath.calculateMaxShift(w, intensity)

            // Add launcher page horizontal swipe offset
            val pageShift = if (project.motionConfig.swipeParallax) (launcherPageOffset - 0.5f) * maxShiftPx * 1.5f else 0f
            val tiltX = currentNormX * maxShiftPx + pageShift
            val tiltY = currentNormY * maxShiftPx

            // Center-crop aspect ratio calculation
            val refBmp = sourceBitmap ?: bgBitmap ?: fgBitmap
            val imgW = refBmp?.width?.toFloat() ?: 1080f
            val imgH = refBmp?.height?.toFloat() ?: 2400f

            val transform = ParallaxMath.computeCenterCropTransform(
                canvasW = w,
                canvasH = h,
                imgW = imgW,
                imgH = imgH,
                zoomScale = project.imageScale,
                panX = project.imagePanX,
                panY = project.imagePanY
            )

            val isLayeredMode = project.renderMode == RenderMode.LAYERED_2D && fgBitmap != null
            val isInFrontOfEverything = project.clockZDepth >= 0.98f || !project.lockScreenConfig.subjectInFrontOfClock
            val isBehindEverything = project.clockZDepth <= 0.02f
            val isZeroParallax = project.motionConfig.parallaxIntensity <= 0.001f

            val bgShiftX = if (isLayeredMode) tiltX * ParallaxMath.BG_PARALLAX_MULTIPLIER else tiltX * 0.20f
            val bgShiftY = if (isLayeredMode) tiltY * ParallaxMath.BG_PARALLAX_MULTIPLIER else tiltY * 0.20f
            val clockDepthFactor = ParallaxMath.calculateClockDepthFactor(project.clockZDepth)
            val clockShiftX = tiltX * clockDepthFactor
            val clockShiftY = tiltY * clockDepthFactor
            val fgShiftX = tiltX * ParallaxMath.FG_PARALLAX_MULTIPLIER
            val fgShiftY = tiltY * ParallaxMath.FG_PARALLAX_MULTIPLIER

            val bgDest = RectF(
                transform.baseLeft + bgShiftX,
                transform.baseTop + bgShiftY,
                transform.baseLeft + bgShiftX + transform.drawW,
                transform.baseTop + bgShiftY + transform.drawH
            )
            val fgDest = RectF(
                transform.baseLeft + fgShiftX,
                transform.baseTop + fgShiftY,
                transform.baseLeft + fgShiftX + transform.drawW,
                transform.baseTop + fgShiftY + transform.drawH
            )

            // Clock Layer definition
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
                    clockPaint.typeface = ParallaxMath.getTypeface(cfg.fontStyle)
                    val timeStr = timeFormat.format(Date())
                    canvas.drawText(timeStr, clockX, clockY, clockPaint)
                }
            }

            // 1. Draw Background Layer
            // When Clock Z is in front of everything or parallax is disabled, draw the pristine source photo (no inpainting smudge!)
            val baseBmp = if (isInFrontOfEverything || isZeroParallax) {
                sourceBitmap ?: bgBitmap
            } else {
                if (isLayeredMode) (bgBitmap ?: sourceBitmap) else (sourceBitmap ?: bgBitmap)
            }

            // If clock is behind everything, draw clock behind the base photo plate so it's fully covered
            if (isBehindEverything && isLayeredMode) {
                drawClockAction()
            }

            baseBmp?.let { bmp ->
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

            // Midground clock (drawn behind cutout subject only when in midground)
            if (!isInFrontOfEverything && !isBehindEverything && isLayeredMode) {
                drawClockAction()
            }

            // 3. Foreground Subject Cutout (drawn ONLY in Layered 2D mode when clock is in midground)
            if (!isInFrontOfEverything && !isBehindEverything && isLayeredMode) {
                fgBitmap?.let { bmp ->
                    canvas.drawBitmap(bmp, null, fgDest, null)
                }
            }

            // In 3D Perspective mode or when clock is in front of everything, draw clock on top
            if (isInFrontOfEverything || !isLayeredMode) {
                drawClockAction()
            }
        }
    }
}

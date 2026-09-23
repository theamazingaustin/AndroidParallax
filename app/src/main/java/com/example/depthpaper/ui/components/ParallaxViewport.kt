package com.example.depthpaper.ui.components

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Typeface
import android.hardware.Sensor
import android.hardware.SensorEvent
import android.hardware.SensorEventListener
import android.hardware.SensorManager
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
import androidx.compose.foundation.gestures.detectDragGestures
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableFloatStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Rect
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.graphics.drawscope.drawIntoCanvas
import androidx.compose.ui.graphics.nativeCanvas
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.unit.IntOffset
import androidx.compose.ui.unit.IntSize
import com.example.depthpaper.core.SensorFilter
import com.example.depthpaper.data.ClockFontStyle
import com.example.depthpaper.data.RenderMode
import com.example.depthpaper.data.WallpaperProject
import com.example.depthpaper.ui.PreviewSurface
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import kotlin.math.roundToInt

@Composable
fun ParallaxViewport(
    project: WallpaperProject,
    previewSurface: PreviewSurface,
    sourceBmp: Bitmap?,
    cutoutBmp: Bitmap?,
    backgroundBmp: Bitmap?,
    depthBmp: Bitmap?,
    simulatedTiltX: Float,
    simulatedTiltY: Float,
    onTiltChanged: (Float, Float) -> Unit,
    modifier: Modifier = Modifier
) {
    val context = LocalContext.current
    var sensorTiltX by remember { mutableFloatStateOf(0f) }
    var sensorTiltY by remember { mutableFloatStateOf(0f) }
    val sensorFilter = remember { SensorFilter() }

    // Listen to real hardware gyroscope if present
    DisposableEffect(project) {
        val sm = context.getSystemService(Context.SENSOR_SERVICE) as? SensorManager
        val rotSensor = sm?.getDefaultSensor(Sensor.TYPE_ROTATION_VECTOR)
            ?: sm?.getDefaultSensor(Sensor.TYPE_ACCELEROMETER)

        val listener = object : SensorEventListener {
            override fun onSensorChanged(event: SensorEvent?) {
                if (event == null) return
                if (event.sensor.type == Sensor.TYPE_ROTATION_VECTOR) {
                    val rotMatrix = FloatArray(9)
                    SensorManager.getRotationMatrixFromVector(rotMatrix, event.values)
                    val orientation = FloatArray(3)
                    SensorManager.getOrientation(rotMatrix, orientation)

                    val pitchDeg = Math.toDegrees(orientation[1].toDouble()).toFloat()
                    val rollDeg = Math.toDegrees(orientation[2].toDouble()).toFloat()

                    sensorFilter.smoothingFactor = project.motionConfig.sensorSmoothing
                    sensorFilter.maxAngleDegrees = project.motionConfig.maxTiltAngle

                    val (nx, ny) = sensorFilter.update(rollDeg, pitchDeg)
                    val signX = if (project.motionConfig.invertX) -1f else 1f
                    val signY = if (project.motionConfig.invertY) -1f else 1f

                    sensorTiltX = nx * signX
                    sensorTiltY = ny * signY
                }
            }
            override fun onAccuracyChanged(sensor: Sensor?, accuracy: Int) {}
        }

        rotSensor?.let { sm?.registerListener(listener, it, SensorManager.SENSOR_DELAY_GAME) }

        onDispose {
            sm?.unregisterListener(listener)
        }
    }

    // Combine drag tilt and sensor tilt
    val totalTiltX = (simulatedTiltX + sensorTiltX).coerceIn(-1f, 1f)
    val totalTiltY = (simulatedTiltY + sensorTiltY).coerceIn(-1f, 1f)

    Box(
        modifier = modifier
            .fillMaxSize()
            .background(Color.Black)
            .pointerInput(project) {
                detectDragGestures(
                    onDragEnd = { onTiltChanged(0f, 0f) },
                    onDragCancel = { onTiltChanged(0f, 0f) }
                ) { change, dragAmount ->
                    change.consume()
                    val dx = (dragAmount.x / size.width) * 3f
                    val dy = (dragAmount.y / size.height) * 3f
                    onTiltChanged(
                        (simulatedTiltX + dx).coerceIn(-1f, 1f),
                        (simulatedTiltY + dy).coerceIn(-1f, 1f)
                    )
                }
            }
    ) {
        Canvas(modifier = Modifier.fillMaxSize()) {
            val canvasW = size.width
            val canvasH = size.height

            if (previewSurface == PreviewSurface.AOD) {
                // AOD mode: OLED true black with monochrome contour & dim clock
                drawRect(Color.Black, size = size)
                cutoutBmp?.let { bmp ->
                    val imageBitmap = bmp.asImageBitmap()
                    drawImage(
                        image = imageBitmap,
                        dstSize = IntSize(canvasW.roundToInt(), canvasH.roundToInt()),
                        alpha = 0.35f
                    )
                }

                // Minimal dim AOD clock
                drawIntoCanvas { nativeCanvas ->
                    val paint = android.graphics.Paint(android.graphics.Paint.ANTI_ALIAS_FLAG).apply {
                        color = android.graphics.Color.argb(200, 220, 220, 220)
                        textSize = canvasW * 0.18f
                        textAlign = android.graphics.Paint.Align.CENTER
                        typeface = Typeface.create("sans-serif-thin", Typeface.NORMAL)
                    }
                    val timeStr = SimpleDateFormat("HH:mm", Locale.getDefault()).format(Date())
                    nativeCanvas.nativeCanvas.drawText(timeStr, canvasW / 2f, canvasH * 0.22f, paint)
                }
                return@Canvas
            }

            // Normal or 3D Parallax rendering
            val intensity = project.motionConfig.parallaxIntensity
            val maxShift = canvasW * 0.08f * intensity
            val shiftX = totalTiltX * maxShift
            val shiftY = totalTiltY * maxShift

            val overscan = 1.08f
            val osW = canvasW * overscan
            val osH = canvasH * overscan
            val osLeft = -canvasW * (overscan - 1f) / 2f
            val osTop = -canvasH * (overscan - 1f) / 2f

            // 1. Draw Background
            val bgBmp = backgroundBmp ?: sourceBmp
            bgBmp?.let { bmp ->
                val bgX = (osLeft - shiftX * 0.4f).roundToInt()
                val bgY = (osTop - shiftY * 0.4f).roundToInt()
                drawImage(
                    image = bmp.asImageBitmap(),
                    dstOffset = IntOffset(bgX, bgY),
                    dstSize = IntSize(osW.roundToInt(), osH.roundToInt())
                )
            }

            // Home screen dimming
            if (previewSurface == PreviewSurface.HOME_SCREEN && project.homeScreenConfig.dimmingFactor > 0f) {
                drawRect(
                    color = Color.Black.copy(alpha = project.homeScreenConfig.dimmingFactor),
                    size = size
                )
            }

            // 2. Draw Clock
            val showClock = previewSurface == PreviewSurface.LOCK_SCREEN || !project.homeScreenConfig.hideClockOnHomeScreen
            val drawClockLambda = {
                if (showClock) {
                    val cfg = project.lockScreenConfig
                    val clockX = canvasW / 2f - shiftX * 0.15f
                    val clockY = canvasH * cfg.verticalOffsetPercent - shiftY * 0.15f

                    drawIntoCanvas { nativeCanvas ->
                        val datePaint = android.graphics.Paint(android.graphics.Paint.ANTI_ALIAS_FLAG).apply {
                            color = (cfg.clockColorHex and 0xCCFFFFFF).toInt()
                            textSize = canvasW * 0.045f * cfg.clockScale
                            textAlign = android.graphics.Paint.Align.CENTER
                            setShadowLayer(8f, 0f, 2f, android.graphics.Color.argb(120, 0, 0, 0))
                        }
                        val clockPaint = android.graphics.Paint(android.graphics.Paint.ANTI_ALIAS_FLAG).apply {
                            color = cfg.clockColorHex.toInt()
                            textSize = canvasW * 0.24f * cfg.clockScale
                            textAlign = android.graphics.Paint.Align.CENTER
                            typeface = when (cfg.fontStyle) {
                                ClockFontStyle.ROUNDED_BOLD -> Typeface.create("sans-serif-medium", Typeface.BOLD)
                                ClockFontStyle.SERIF_CLASSIC -> Typeface.create("serif", Typeface.BOLD)
                                ClockFontStyle.MODERN_HEAVY -> Typeface.create("sans-serif-black", Typeface.BOLD)
                                ClockFontStyle.ELEGANT_THIN -> Typeface.create("sans-serif-thin", Typeface.NORMAL)
                                ClockFontStyle.STENCIL_DISPLAY -> Typeface.create("casual", Typeface.BOLD)
                                ClockFontStyle.CYBER_MONO -> Typeface.create("monospace", Typeface.BOLD)
                            }
                            setShadowLayer(16f, 0f, 4f, android.graphics.Color.argb(140, 0, 0, 0))
                        }

                        if (cfg.showDate) {
                            val dateStr = SimpleDateFormat(cfg.dateFormat, Locale.getDefault()).format(Date())
                            nativeCanvas.nativeCanvas.drawText(dateStr, clockX, clockY - (canvasW * 0.18f * cfg.clockScale), datePaint)
                        }

                        val timeStr = SimpleDateFormat("HH:mm", Locale.getDefault()).format(Date())
                        nativeCanvas.nativeCanvas.drawText(timeStr, clockX, clockY, clockPaint)
                    }
                }
            }

            if (project.lockScreenConfig.subjectInFrontOfClock) {
                drawClockLambda()
            }

            // 3. Foreground Cutout Plate
            cutoutBmp?.let { bmp ->
                val fgX = (osLeft + shiftX * 0.8f).roundToInt()
                val fgY = (osTop + shiftY * 0.8f).roundToInt()
                drawImage(
                    image = bmp.asImageBitmap(),
                    dstOffset = IntOffset(fgX, fgY),
                    dstSize = IntSize(osW.roundToInt(), osH.roundToInt())
                )
            }

            if (!project.lockScreenConfig.subjectInFrontOfClock) {
                drawClockLambda()
            }

            // 4. Simulated Home Screen app icons overlay if in Home Screen preview
            if (previewSurface == PreviewSurface.HOME_SCREEN) {
                val iconCols = 4
                val iconRows = 5
                val startY = canvasH * 0.28f
                val colSpacing = canvasW / (iconCols + 1)
                val rowSpacing = (canvasH * 0.55f) / iconRows
                val iconRadius = colSpacing * 0.35f

                for (r in 0 until iconRows) {
                    for (c in 0 until iconCols) {
                        val cx = colSpacing * (c + 1)
                        val cy = startY + r * rowSpacing
                        drawCircle(
                            color = Color.White.copy(alpha = 0.25f),
                            radius = iconRadius,
                            center = Offset(cx, cy)
                        )
                    }
                }
            }
        }
    }
}

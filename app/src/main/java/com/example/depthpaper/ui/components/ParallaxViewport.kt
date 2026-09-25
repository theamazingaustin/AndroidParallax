package com.example.depthpaper.ui.components

import android.content.Context
import android.graphics.Bitmap
import android.graphics.DashPathEffect
import android.graphics.Typeface
import android.hardware.Sensor
import android.hardware.SensorEvent
import android.hardware.SensorEventListener
import android.hardware.SensorManager
import android.os.Handler
import android.os.Looper
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
import androidx.compose.foundation.gestures.awaitEachGesture
import androidx.compose.foundation.gestures.awaitFirstDown
import androidx.compose.foundation.gestures.detectDragGestures
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableFloatStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberUpdatedState
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
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
import kotlin.math.max
import kotlin.math.min
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
    onClockPositionChanged: (Float, Float) -> Unit,
    onImageTransformChanged: ((Float, Float, Float) -> Unit)? = null,
    onTapDepthPoint: ((Float, Float) -> Unit)? = null,
    modifier: Modifier = Modifier
) {
    val context = LocalContext.current
    var sensorTiltX by remember { mutableFloatStateOf(0f) }
    var sensorTiltY by remember { mutableFloatStateOf(0f) }
    val sensorFilter = remember { SensorFilter(adaptiveBaseline = true) }

    var isDraggingClock by remember { mutableStateOf(false) }
    var currentClockX by remember(project.id) { mutableFloatStateOf(project.lockScreenConfig.horizontalOffsetPercent) }
    var currentClockY by remember(project.id) { mutableFloatStateOf(project.lockScreenConfig.verticalOffsetPercent) }

    var currentScale by remember(project.id, project.imageScale) { mutableFloatStateOf(project.imageScale) }
    var currentPanX by remember(project.id, project.imagePanX) { mutableFloatStateOf(project.imagePanX) }
    var currentPanY by remember(project.id, project.imagePanY) { mutableFloatStateOf(project.imagePanY) }

    LaunchedEffect(project.lockScreenConfig.horizontalOffsetPercent, project.lockScreenConfig.verticalOffsetPercent) {
        if (!isDraggingClock) {
            currentClockX = project.lockScreenConfig.horizontalOffsetPercent
            currentClockY = project.lockScreenConfig.verticalOffsetPercent
        }
    }

    val currentMotionConfig by rememberUpdatedState(project.motionConfig)

    // Gyroscope tracking (runs continuously without disposing on project metadata updates)
    DisposableEffect(Unit) {
        val mainHandler = Handler(Looper.getMainLooper())
        val sm = context.getSystemService(Context.SENSOR_SERVICE) as? SensorManager
        val rotSensor = sm?.getDefaultSensor(Sensor.TYPE_ROTATION_VECTOR)
            ?: sm?.getDefaultSensor(Sensor.TYPE_GAME_ROTATION_VECTOR)
            ?: sm?.getDefaultSensor(Sensor.TYPE_GRAVITY)
            ?: sm?.getDefaultSensor(Sensor.TYPE_ACCELEROMETER)

        val listener = object : SensorEventListener {
            override fun onSensorChanged(event: SensorEvent?) {
                if (event == null) return
                val cfg = currentMotionConfig
                sensorFilter.smoothingFactor = cfg.sensorSmoothing
                sensorFilter.maxAngleDegrees = cfg.maxTiltAngle

                val (rawX, rawY) = when (event.sensor.type) {
                    Sensor.TYPE_ROTATION_VECTOR, Sensor.TYPE_GAME_ROTATION_VECTOR -> {
                        val rotMatrix = FloatArray(9)
                        SensorManager.getRotationMatrixFromVector(rotMatrix, event.values)
                        val orientation = FloatArray(3)
                        SensorManager.getOrientation(rotMatrix, orientation)

                        val pitchDeg = Math.toDegrees(orientation[1].toDouble()).toFloat()
                        val rollDeg = Math.toDegrees(orientation[2].toDouble()).toFloat()
                        sensorFilter.update(rollDeg, pitchDeg)
                    }
                    Sensor.TYPE_GRAVITY, Sensor.TYPE_ACCELEROMETER -> {
                        val ax = event.values[0]
                        val ay = event.values[1]
                        sensorFilter.update(ax * 3.5f, ay * 3.5f)
                    }
                    else -> Pair(0f, 0f)
                }

                val signX = if (cfg.invertX) -1f else 1f
                val signY = if (cfg.invertY) -1f else 1f

                mainHandler.post {
                    sensorTiltX = (rawX * signX).coerceIn(-1f, 1f)
                    sensorTiltY = (rawY * signY).coerceIn(-1f, 1f)
                }
            }
            override fun onAccuracyChanged(sensor: Sensor?, accuracy: Int) {}
        }

        rotSensor?.let { sm?.registerListener(listener, it, SensorManager.SENSOR_DELAY_GAME) }

        onDispose {
            sm?.unregisterListener(listener)
        }
    }

    val totalTiltX = (simulatedTiltX + sensorTiltX).coerceIn(-1f, 1f)
    val totalTiltY = (simulatedTiltY + sensorTiltY).coerceIn(-1f, 1f)

    Box(
        modifier = modifier
            .fillMaxSize()
            .background(Color.Black)
            .pointerInput(project.id) {
                awaitEachGesture {
                    val firstDown = awaitFirstDown(requireUnconsumed = false)
                    val downPos = firstDown.position
                    val cfg = project.lockScreenConfig
                    val clockTargetX = size.width * currentClockX
                    val clockTargetY = size.height * currentClockY
                    val hitRadiusX = size.width * 0.45f * cfg.clockScale
                    // Clock text baseline is at clockTargetY, digits & date extend above
                    val boxTop = clockTargetY - (size.width * 0.28f * cfg.clockScale)
                    val boxBottom = clockTargetY + 40f

                    val isClockTouch = firstDown.position.x in (clockTargetX - hitRadiusX)..(clockTargetX + hitRadiusX) &&
                                       firstDown.position.y in boxTop..boxBottom
                    var isTwoFinger = false
                    var totalDragDistance = 0f

                    do {
                        val event = awaitPointerEvent()
                        val activePointers = event.changes.filter { it.pressed }

                        if (activePointers.size >= 2) {
                            // Two-finger pinch to zoom & pan the image
                            isTwoFinger = true
                            isDraggingClock = false
                            val p0 = activePointers[0]
                            val p1 = activePointers[1]
                            val prevDist = (p0.previousPosition - p1.previousPosition).getDistance()
                            val currDist = (p0.position - p1.position).getDistance()
                            if (prevDist > 0f) {
                                val zoomFactor = currDist / prevDist
                                currentScale = (currentScale * zoomFactor).coerceIn(1.0f, 3.5f)
                            }

                            val prevCenter = (p0.previousPosition + p1.previousPosition) / 2f
                            val currCenter = (p0.position + p1.position) / 2f
                            val panDelta = currCenter - prevCenter

                            currentPanX = (currentPanX + panDelta.x / size.width).coerceIn(-0.6f, 0.6f)
                            currentPanY = (currentPanY + panDelta.y / size.height).coerceIn(-0.6f, 0.6f)

                            event.changes.forEach { it.consume() }
                        } else if (activePointers.size == 1 && !isTwoFinger) {
                            val change = activePointers[0]
                            val dragAmount = change.position - change.previousPosition
                            totalDragDistance += dragAmount.getDistance()

                            if (isClockTouch) {
                                isDraggingClock = true
                                currentClockX = (currentClockX + dragAmount.x / size.width).coerceIn(0.10f, 0.90f)
                                currentClockY = (currentClockY + dragAmount.y / size.height).coerceIn(0.06f, 0.85f)
                                change.consume()
                            } else {
                                // Dragging scene tilts the 3D parallax
                                val dx = (dragAmount.x / size.width) * 3f
                                val dy = (dragAmount.y / size.height) * 3f
                                onTiltChanged(
                                    (simulatedTiltX + dx).coerceIn(-1f, 1f),
                                    (simulatedTiltY + dy).coerceIn(-1f, 1f)
                                )
                                change.consume()
                            }
                        }
                    } while (activePointers.isNotEmpty())

                    if (isDraggingClock) {
                        onClockPositionChanged(currentClockX, currentClockY)
                    } else if (!isTwoFinger && totalDragDistance < 15f && onTapDepthPoint != null) {
                        // User tapped on preview to pick 3D depth layer!
                        // Map touch coordinate on canvas back to original image normalized coordinate
                        val canvasW = size.width.toFloat()
                        val canvasH = size.height.toFloat()
                        val refBmp = sourceBmp ?: backgroundBmp ?: cutoutBmp ?: depthBmp
                        val imgW = refBmp?.width?.toFloat() ?: canvasW
                        val imgH = refBmp?.height?.toFloat() ?: canvasH
                        val overscan = 1.08f
                        val scale = max((canvasW * overscan) / imgW, (canvasH * overscan) / imgH) * currentScale
                        val drawW = imgW * scale
                        val drawH = imgH * scale
                        val panOffsetX = canvasW * currentPanX
                        val panOffsetY = canvasH * currentPanY
                        val baseLeft = (canvasW - drawW) / 2f + panOffsetX
                        val baseTop = (canvasH - drawH) / 2f + panOffsetY

                        val normX = ((downPos.x - baseLeft) / drawW).coerceIn(0f, 1f)
                        val normY = ((downPos.y - baseTop) / drawH).coerceIn(0f, 1f)
                        onTapDepthPoint.invoke(normX, normY)
                    }

                    if (isTwoFinger) {
                        onImageTransformChanged?.invoke(currentScale, currentPanX, currentPanY)
                    }
                    isDraggingClock = false
                    onTiltChanged(0f, 0f)
                }
            }
    ) {
        Canvas(modifier = Modifier.fillMaxSize()) {
            val canvasW = size.width
            val canvasH = size.height

            // --- AOD Mode ---
            if (previewSurface == PreviewSurface.AOD) {
                drawRect(Color.Black, size = size)
                cutoutBmp?.let { bmp ->
                    val scale = max(canvasW / bmp.width, canvasH / bmp.height) * currentScale
                    val dW = (bmp.width * scale).roundToInt()
                    val dH = (bmp.height * scale).roundToInt()
                    val panOffsetX = (canvasW * currentPanX).roundToInt()
                    val panOffsetY = (canvasH * currentPanY).roundToInt()
                    val left = ((canvasW - dW) / 2f).roundToInt() + panOffsetX
                    val top = ((canvasH - dH) / 2f).roundToInt() + panOffsetY
                    drawImage(
                        image = bmp.asImageBitmap(),
                        dstOffset = IntOffset(left, top),
                        dstSize = IntSize(dW, dH),
                        alpha = 0.30f
                    )
                }

                // Minimal dim AOD clock
                drawIntoCanvas { nativeCanvas ->
                    val paint = android.graphics.Paint(android.graphics.Paint.ANTI_ALIAS_FLAG).apply {
                        color = android.graphics.Color.argb(180, 240, 240, 240)
                        textSize = canvasW * 0.18f
                        textAlign = android.graphics.Paint.Align.CENTER
                        typeface = Typeface.create("sans-serif-thin", Typeface.NORMAL)
                    }
                    val timeStr = SimpleDateFormat("HH:mm", Locale.getDefault()).format(Date())
                    nativeCanvas.nativeCanvas.drawText(
                        timeStr,
                        canvasW * currentClockX,
                        canvasH * currentClockY,
                        paint
                    )
                }
                return@Canvas
            }

            // --- 3D Depth Map Inspection Mode ---
            if (previewSurface == PreviewSurface.DEPTH_MAP) {
                drawRect(Color.Black, size = size)
                val targetBmp = depthBmp ?: sourceBmp
                targetBmp?.let { bmp ->
                    val overscan = 1.08f
                    val scale = max((canvasW * overscan) / bmp.width, (canvasH * overscan) / bmp.height) * currentScale
                    val drawW = (bmp.width * scale).roundToInt()
                    val drawH = (bmp.height * scale).roundToInt()
                    val panOffsetX = (canvasW * currentPanX).roundToInt()
                    val panOffsetY = (canvasH * currentPanY).roundToInt()
                    val baseLeft = ((canvasW - drawW) / 2f).roundToInt() + panOffsetX
                    val baseTop = ((canvasH - drawH) / 2f).roundToInt() + panOffsetY
                    val shiftX = totalTiltX * canvasW * 0.04f * project.motionConfig.parallaxIntensity
                    val shiftY = totalTiltY * canvasW * 0.04f * project.motionConfig.parallaxIntensity

                    drawImage(
                        image = bmp.asImageBitmap(),
                        dstOffset = IntOffset(baseLeft + shiftX.roundToInt(), baseTop + shiftY.roundToInt()),
                        dstSize = IntSize(drawW, drawH)
                    )
                }

                drawIntoCanvas { nativeCanvas ->
                    val paint = android.graphics.Paint(android.graphics.Paint.ANTI_ALIAS_FLAG).apply {
                        color = android.graphics.Color.WHITE
                        textSize = canvasW * 0.038f
                        textAlign = android.graphics.Paint.Align.CENTER
                        setShadowLayer(10f, 0f, 2f, android.graphics.Color.BLACK)
                    }
                    nativeCanvas.nativeCanvas.drawText("3D DEPTH MAP (WHITE = FOREGROUND)", canvasW * 0.5f, canvasH * 0.94f, paint)
                }
                return@Canvas
            }

            // --- Normal & Parallax Rendering ---
            val intensity = project.motionConfig.parallaxIntensity.coerceAtLeast(0.3f)
            val maxShift = canvasW * 0.12f * intensity
            val shiftX = totalTiltX * maxShift
            val shiftY = totalTiltY * maxShift

            // 1. Compute Center-Crop Aspect Ratio Scale (NEVER distort!)
            val refBmp = sourceBmp ?: backgroundBmp ?: cutoutBmp
            val imgW = refBmp?.width?.toFloat() ?: 1000f
            val imgH = refBmp?.height?.toFloat() ?: 1000f

            val overscan = 1.08f
            val scale = max((canvasW * overscan) / imgW, (canvasH * overscan) / imgH) * currentScale
            val drawW = (imgW * scale).roundToInt()
            val drawH = (imgH * scale).roundToInt()
            val panOffsetX = (canvasW * currentPanX).roundToInt()
            val panOffsetY = (canvasH * currentPanY).roundToInt()
            val baseLeft = ((canvasW - drawW) / 2f).roundToInt() + panOffsetX
            val baseTop = ((canvasH - drawH) / 2f).roundToInt() + panOffsetY
            val drawSize = IntSize(drawW, drawH)

            // Positive differential parallax:
            // Background is furthest away (-0.25x)
            // Clock is midground (+0.35x)
            // Cutout subject is nearest (+0.75x)
            val isLayeredMode = project.renderMode == RenderMode.LAYERED_2D && cutoutBmp != null
            val bgShiftX = if (isLayeredMode) shiftX * -0.25f else shiftX * 0.20f
            val bgShiftY = if (isLayeredMode) shiftY * -0.25f else shiftY * 0.20f
            val clockParallaxShiftX = shiftX * 0.35f
            val clockParallaxShiftY = shiftY * 0.35f
            val fgShiftX = shiftX * 0.75f
            val fgShiftY = shiftY * 0.75f

            val bgLeft = baseLeft + bgShiftX.roundToInt()
            val bgTop = baseTop + bgShiftY.roundToInt()
            val fgLeft = baseLeft + fgShiftX.roundToInt()
            val fgTop = baseTop + fgShiftY.roundToInt()

            // 1. Draw Background Photo Plate
            val bgBmp = if (isLayeredMode) (backgroundBmp ?: sourceBmp) else (sourceBmp ?: backgroundBmp)
            bgBmp?.let { bmp ->
                drawImage(
                    image = bmp.asImageBitmap(),
                    dstOffset = IntOffset(bgLeft, bgTop),
                    dstSize = drawSize
                )
            }

            // Home screen icon contrast dimming
            if (previewSurface == PreviewSurface.HOME_SCREEN && project.homeScreenConfig.dimmingFactor > 0f) {
                drawRect(
                    color = Color.Black.copy(alpha = project.homeScreenConfig.dimmingFactor),
                    size = size
                )
            }

            // 2. Draw Lock Screen Clock
            val showClock = previewSurface == PreviewSurface.LOCK_SCREEN || !project.homeScreenConfig.hideClockOnHomeScreen
            val drawClock = {
                if (showClock) {
                    val cfg = project.lockScreenConfig
                    val clockX = canvasW * currentClockX + clockParallaxShiftX
                    val clockY = canvasH * currentClockY + clockParallaxShiftY

                    drawIntoCanvas { nativeCanvas ->
                        val datePaint = android.graphics.Paint(android.graphics.Paint.ANTI_ALIAS_FLAG).apply {
                            color = (cfg.clockColorHex and 0xCCFFFFFF).toInt()
                            textSize = canvasW * 0.045f * cfg.clockScale
                            textAlign = android.graphics.Paint.Align.CENTER
                            setShadowLayer(10f, 0f, 2f, android.graphics.Color.argb(140, 0, 0, 0))
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
                            setShadowLayer(20f, 0f, 4f, android.graphics.Color.argb(160, 0, 0, 0))
                        }

                        if (cfg.showDate) {
                            val dateStr = SimpleDateFormat(cfg.dateFormat, Locale.getDefault()).format(Date())
                            nativeCanvas.nativeCanvas.drawText(
                                dateStr,
                                clockX,
                                clockY - (canvasW * 0.17f * cfg.clockScale),
                                datePaint
                            )
                        }

                        val timeStr = SimpleDateFormat("HH:mm", Locale.getDefault()).format(Date())
                        nativeCanvas.nativeCanvas.drawText(timeStr, clockX, clockY, clockPaint)

                        // If user is actively dragging the clock, draw a sleek boundary guide
                        if (isDraggingClock) {
                            val guidePaint = android.graphics.Paint(android.graphics.Paint.ANTI_ALIAS_FLAG).apply {
                                color = android.graphics.Color.parseColor("#00E5FF")
                                style = android.graphics.Paint.Style.STROKE
                                strokeWidth = 3f
                                pathEffect = DashPathEffect(floatArrayOf(12f, 12f), 0f)
                            }
                            val boxHalfW = canvasW * 0.38f * cfg.clockScale
                            val boxTop = clockY - (canvasW * 0.24f * cfg.clockScale)
                            val boxBottom = clockY + 24f
                            nativeCanvas.nativeCanvas.drawRoundRect(
                                clockX - boxHalfW, boxTop, clockX + boxHalfW, boxBottom,
                                20f, 20f, guidePaint
                            )
                        }
                    }
                }
            }

            // Depth order: In Layered 2D, Clock is drawn behind subject cutout
            if (project.lockScreenConfig.subjectInFrontOfClock && isLayeredMode) {
                drawClock()
            }

            // 3. Draw Foreground Cutout Plate (drawn ONLY in Layered 2D mode)
            if (isLayeredMode) {
                cutoutBmp.let { bmp ->
                    drawImage(
                        image = bmp.asImageBitmap(),
                        dstOffset = IntOffset(fgLeft, fgTop),
                        dstSize = drawSize
                    )
                }
            }

            // In 3D Perspective mode or when Subject in Front is disabled, clock is drawn on top
            if (!project.lockScreenConfig.subjectInFrontOfClock || !isLayeredMode) {
                drawClock()
            }
        }
    }
}

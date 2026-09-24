package com.example.depthpaper.ui

import android.graphics.Bitmap
import android.graphics.ImageDecoder
import android.net.Uri
import android.os.Build
import android.provider.MediaStore
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.aspectRatio
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.navigationBarsPadding
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.statusBarsPadding
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.lazy.LazyRow
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.ArrowBack
import androidx.compose.material.icons.filled.AddPhotoAlternate
import androidx.compose.material.icons.filled.BugReport
import androidx.compose.material.icons.filled.Info
import androidx.compose.material.icons.filled.Layers
import androidx.compose.material.icons.filled.Refresh
import androidx.compose.material.icons.filled.Schedule
import androidx.compose.material.icons.filled.TouchApp
import androidx.compose.material.icons.filled.Tune
import androidx.compose.material.icons.filled.ViewInAr
import androidx.compose.material.icons.filled.Wallpaper
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.FilterChip
import androidx.compose.material3.FilterChipDefaults
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.Slider
import androidx.compose.material3.Surface
import androidx.compose.material3.Switch
import androidx.compose.material3.SwitchDefaults
import androidx.compose.material3.Tab
import androidx.compose.material3.TabRow
import androidx.compose.material3.TabRowDefaults
import androidx.compose.material3.TabRowDefaults.tabIndicatorOffset
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableFloatStateOf
import androidx.compose.runtime.mutableIntStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.example.depthpaper.core.AppLogger
import com.example.depthpaper.core.SegmentationModelType
import com.example.depthpaper.data.ClockFontStyle
import com.example.depthpaper.data.RenderMode
import com.example.depthpaper.ui.components.ParallaxViewport
import java.util.Locale

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun StudioScreen(
    viewModel: StudioViewModel,
    onNavigateToGallery: () -> Unit,
    modifier: Modifier = Modifier
) {
    val state by viewModel.uiState.collectAsState()
    val context = LocalContext.current

    val photoPickerLauncher = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.GetContent()
    ) { uri: Uri? ->
        uri?.let {
            val bitmap = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.P) {
                ImageDecoder.decodeBitmap(ImageDecoder.createSource(context.contentResolver, it)) { decoder, _, _ ->
                    decoder.isMutableRequired = true
                    decoder.allocator = ImageDecoder.ALLOCATOR_SOFTWARE
                }
            } else {
                @Suppress("DEPRECATION")
                MediaStore.Images.Media.getBitmap(context.contentResolver, it)
            }
            viewModel.importNewImage(bitmap, "My Photo")
        }
    }

    Box(
        modifier = modifier
            .fillMaxSize()
            .background(Color(0xFF0A0A12))
            .statusBarsPadding()
            .navigationBarsPadding()
    ) {
        Column(modifier = Modifier.fillMaxSize()) {

            // Top Action Bar
            Row(
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(horizontal = 12.dp, vertical = 6.dp),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically
            ) {
                IconButton(onClick = onNavigateToGallery) {
                    Icon(
                        Icons.AutoMirrored.Filled.ArrowBack,
                        contentDescription = "Gallery",
                        tint = Color.White
                    )
                }

                // Render Mode chip
                FilterChip(
                    selected = true,
                    onClick = { viewModel.toggleRenderMode() },
                    label = {
                        Text(
                            text = if (state.currentProject.renderMode == RenderMode.LAYERED_2D) "2.5D Layered" else "3D Perspective",
                            fontSize = 12.sp,
                            fontWeight = FontWeight.Bold
                        )
                    },
                    leadingIcon = {
                        Icon(
                            if (state.currentProject.renderMode == RenderMode.LAYERED_2D) Icons.Default.Layers else Icons.Default.ViewInAr,
                            contentDescription = null,
                            modifier = Modifier.size(16.dp)
                        )
                    },
                    colors = FilterChipDefaults.filterChipColors(
                        selectedContainerColor = if (state.currentProject.renderMode == RenderMode.LAYERED_2D) Color(0xFF6C5CE7) else Color(0xFF00B894),
                        selectedLabelColor = Color.White,
                        selectedLeadingIconColor = Color.White
                    ),
                    shape = RoundedCornerShape(12.dp)
                )

                Row(verticalAlignment = Alignment.CenterVertically) {
                    IconButton(onClick = { AppLogger.copyLogsToClipboard(context) }) {
                        Icon(
                            Icons.Default.BugReport,
                            contentDescription = "Copy Diagnostic Logs",
                            tint = Color(0xFF00E5FF)
                        )
                    }

                    Spacer(modifier = Modifier.width(4.dp))

                    Button(
                        onClick = { viewModel.setActiveWallpaper(state.currentProject.id, context) },
                        colors = ButtonDefaults.buttonColors(containerColor = Color(0xFF00E5FF)),
                        shape = RoundedCornerShape(12.dp)
                    ) {
                        Icon(Icons.Default.Wallpaper, contentDescription = null, tint = Color.Black, modifier = Modifier.size(16.dp))
                        Spacer(modifier = Modifier.width(6.dp))
                        Text("Apply", color = Color.Black, fontWeight = FontWeight.Bold, fontSize = 13.sp)
                    }
                }
            }

            // Viewport Interactive Preview (3D Parallax & Drag)
            Box(
                modifier = Modifier
                    .fillMaxWidth()
                    .weight(1f)
                    .padding(horizontal = 10.dp)
                    .clip(RoundedCornerShape(20.dp))
                    .background(Color.Black)
            ) {
                ParallaxViewport(
                    project = state.currentProject,
                    previewSurface = state.previewSurface,
                    sourceBmp = state.sourceBitmap,
                    cutoutBmp = state.cutoutBitmap,
                    backgroundBmp = state.backgroundBitmap,
                    depthBmp = state.depthBitmap,
                    simulatedTiltX = state.simulatedTiltX,
                    simulatedTiltY = state.simulatedTiltY,
                    onTiltChanged = { x, y -> viewModel.updateTilt(x, y) },
                    onClockPositionChanged = { x, y -> viewModel.updateClockPosition(x, y) },
                    modifier = Modifier.fillMaxSize()
                )

                // Processing overlay
                if (state.isProcessing) {
                    Box(
                        modifier = Modifier
                            .fillMaxSize()
                            .background(Color.Black.copy(alpha = 0.65f)),
                        contentAlignment = Alignment.Center
                    ) {
                        Column(horizontalAlignment = Alignment.CenterHorizontally) {
                            CircularProgressIndicator(color = Color(0xFF00E5FF), strokeWidth = 3.dp)
                            Spacer(modifier = Modifier.height(12.dp))
                            Text(
                                text = state.statusMessage ?: "AI processing on-device...",
                                color = Color.White,
                                fontSize = 13.sp,
                                fontWeight = FontWeight.Medium
                            )
                        }
                    }
                }

                // Drag hint pill at bottom of viewport
                Surface(
                    color = Color.Black.copy(alpha = 0.55f),
                    shape = RoundedCornerShape(12.dp),
                    modifier = Modifier
                        .align(Alignment.BottomCenter)
                        .padding(bottom = 12.dp)
                ) {
                    Text(
                        text = if (state.previewSurface == PreviewSurface.DEPTH_MAP) "3D Depth Map Active • Tilt to Inspect" else "Touch & drag clock to reposition",
                        color = Color.White.copy(alpha = 0.85f),
                        fontSize = 11.sp,
                        modifier = Modifier.padding(horizontal = 12.dp, vertical = 4.dp)
                    )
                }
            }

            Spacer(modifier = Modifier.height(6.dp))

            // Combined Bottom Control Panel
            StudioBottomControlPanel(
                viewModel = viewModel,
                state = state,
                onPickPhoto = { photoPickerLauncher.launch("image/*") }
            )
        }
    }
}

/**
 * Clean 2-Tab Material 3 Control Panel:
 * Tab 0: "Clock & Wallpaper" (Photo selector, live font previews, colors, size, depth ordering)
 * Tab 1: "Layers & 3D Motion" (Preview surface modes with 3D map inspection, AI models, granular inpaint/sensitivity sliders)
 */
@Composable
fun StudioBottomControlPanel(
    viewModel: StudioViewModel,
    state: StudioUiState,
    onPickPhoto: () -> Unit
) {
    var selectedTab by remember { mutableIntStateOf(0) }
    val tabs = listOf("Clock & Wallpaper", "Layers & 3D Motion")

    Surface(
        modifier = Modifier.fillMaxWidth(),
        color = Color(0xFF141422),
        shape = RoundedCornerShape(topStart = 20.dp, topEnd = 20.dp)
    ) {
        Column(modifier = Modifier.padding(14.dp)) {
            TabRow(
                selectedTabIndex = selectedTab,
                containerColor = Color.Transparent,
                contentColor = Color(0xFF00E5FF),
                indicator = { tabPositions ->
                    TabRowDefaults.SecondaryIndicator(
                        Modifier.tabIndicatorOffset(tabPositions[selectedTab]),
                        color = Color(0xFF00E5FF)
                    )
                }
            ) {
                tabs.forEachIndexed { index, title ->
                    Tab(
                        selected = selectedTab == index,
                        onClick = { selectedTab = index },
                        text = {
                            Row(verticalAlignment = Alignment.CenterVertically) {
                                Icon(
                                    imageVector = if (index == 0) Icons.Default.Schedule else Icons.Default.Layers,
                                    contentDescription = null,
                                    modifier = Modifier.size(16.dp),
                                    tint = if (selectedTab == index) Color(0xFF00E5FF) else Color.Gray
                                )
                                Spacer(modifier = Modifier.width(6.dp))
                                Text(
                                    text = title,
                                    fontSize = 12.sp,
                                    fontWeight = if (selectedTab == index) FontWeight.Bold else FontWeight.Normal
                                )
                            }
                        }
                    )
                }
            }

            Spacer(modifier = Modifier.height(12.dp))

            when (selectedTab) {
                0 -> ClockAndWallpaperTab(viewModel, state, onPickPhoto)
                1 -> LayersAndMotionTab(viewModel, state)
            }
        }
    }
}

/**
 * Tab 0: Clock typography with live font preview cards, color palette, scale, and photo switching.
 */
@Composable
fun ClockAndWallpaperTab(
    viewModel: StudioViewModel,
    state: StudioUiState,
    onPickPhoto: () -> Unit
) {
    val cfg = state.currentProject.lockScreenConfig

    Column(
        modifier = Modifier
            .fillMaxWidth()
            .verticalScroll(rememberScrollState()),
        verticalArrangement = Arrangement.spacedBy(12.dp)
    ) {
        // 1. Wallpaper Photo Row
        Card(
            colors = CardDefaults.cardColors(containerColor = Color(0xFF1B1B2C)),
            shape = RoundedCornerShape(12.dp),
            modifier = Modifier.fillMaxWidth()
        ) {
            Row(
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(10.dp),
                verticalAlignment = Alignment.CenterVertically,
                horizontalArrangement = Arrangement.SpaceBetween
            ) {
                Row(verticalAlignment = Alignment.CenterVertically) {
                    ProjectThumbnail(
                        path = state.currentProject.thumbnailPath.ifBlank { state.currentProject.sourceImagePath },
                        modifier = Modifier
                            .size(36.dp, 54.dp)
                            .clip(RoundedCornerShape(6.dp))
                    )
                    Spacer(modifier = Modifier.width(10.dp))
                    Column {
                        Text(
                            text = state.currentProject.title,
                            fontSize = 13.sp,
                            fontWeight = FontWeight.Bold,
                            color = Color.White
                        )
                        Text(
                            text = "Photo Wallpaper",
                            fontSize = 11.sp,
                            color = Color.Gray
                        )
                    }
                }

                Button(
                    onClick = onPickPhoto,
                    colors = ButtonDefaults.buttonColors(containerColor = Color(0xFF28283E)),
                    shape = RoundedCornerShape(8.dp),
                    contentPadding = PaddingValues(horizontal = 12.dp, vertical = 6.dp)
                ) {
                    Icon(Icons.Default.AddPhotoAlternate, contentDescription = null, tint = Color(0xFF00E5FF), modifier = Modifier.size(14.dp))
                    Spacer(modifier = Modifier.width(6.dp))
                    Text("Change", color = Color(0xFF00E5FF), fontSize = 11.sp, fontWeight = FontWeight.Bold)
                }
            }
        }

        // 2. Typography Styles (Visual Live Previews)
        Column(verticalArrangement = Arrangement.spacedBy(6.dp)) {
            Text("Clock Typography Style", fontSize = 12.sp, color = Color.Gray)
            LazyRow(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                items(ClockFontStyle.values()) { style ->
                    val (fontFam, fontWt) = getFontFamilyAndWeight(style)
                    val isSelected = cfg.fontStyle == style
                    Surface(
                        modifier = Modifier
                            .width(115.dp)
                            .clip(RoundedCornerShape(10.dp))
                            .border(
                                width = if (isSelected) 2.dp else 1.dp,
                                color = if (isSelected) Color(0xFF00E5FF) else Color(0xFF28283E),
                                shape = RoundedCornerShape(10.dp)
                            )
                            .clickable { viewModel.updateClockStyle(style) },
                        color = if (isSelected) Color(0xFF1E2838) else Color(0xFF181828)
                    ) {
                        Column(
                            modifier = Modifier.padding(horizontal = 8.dp, vertical = 8.dp),
                            horizontalAlignment = Alignment.CenterHorizontally
                        ) {
                            Text(
                                text = "12:34",
                                fontSize = 18.sp,
                                fontFamily = fontFam,
                                fontWeight = fontWt,
                                color = if (isSelected) Color(0xFF00E5FF) else Color.White
                            )
                            Spacer(modifier = Modifier.height(2.dp))
                            Text(
                                text = style.name.replace('_', ' ').lowercase().replaceFirstChar { it.titlecase(Locale.getDefault()) },
                                fontSize = 9.sp,
                                color = Color.LightGray,
                                maxLines = 1
                            )
                        }
                    }
                }
            }
        }

        // 3. Color Palette
        Column(verticalArrangement = Arrangement.spacedBy(6.dp)) {
            Text("Clock Color", fontSize = 12.sp, color = Color.Gray)
            val colors = listOf(
                0xFFFFFFFF to "White",
                0xFFFF3366 to "Rose",
                0xFFFFD700 to "Gold",
                0xFF00E5FF to "Cyan",
                0xFFA29BFE to "Lavender",
                0xFF55EFC4 to "Mint",
                0xFFFF7675 to "Coral",
                0xFF00CEC9 to "Teal"
            )
            Row(horizontalArrangement = Arrangement.spacedBy(10.dp)) {
                colors.forEach { (colorHex, _) ->
                    Box(
                        modifier = Modifier
                            .size(28.dp)
                            .clip(CircleShape)
                            .background(Color(colorHex))
                            .border(
                                width = if (cfg.clockColorHex == colorHex) 3.dp else 1.dp,
                                color = if (cfg.clockColorHex == colorHex) Color.White else Color.Transparent,
                                shape = CircleShape
                            )
                            .clickable { viewModel.updateClockColor(colorHex) }
                    )
                }
            }
        }

        // 4. Clock Scale Slider
        Row(verticalAlignment = Alignment.CenterVertically) {
            Text("Clock Size", fontSize = 12.sp, color = Color.Gray, modifier = Modifier.width(90.dp))
            Slider(
                value = cfg.clockScale,
                onValueChange = { viewModel.updateClockScale(it) },
                valueRange = 0.7f..1.4f,
                modifier = Modifier.weight(1f)
            )
            Text("${(cfg.clockScale * 100).toInt()}%", fontSize = 11.sp, color = Color.LightGray, modifier = Modifier.width(36.dp))
        }

        // 5. Subject in Front Toggle (M3 Switch)
        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.SpaceBetween,
            verticalAlignment = Alignment.CenterVertically
        ) {
            Column {
                Text("Subject in Front of Clock", fontSize = 13.sp, color = Color.White, fontWeight = FontWeight.SemiBold)
                Text("Layer clock behind foreground subject", fontSize = 11.sp, color = Color.Gray)
            }
            Switch(
                checked = cfg.subjectInFrontOfClock,
                onCheckedChange = { viewModel.toggleSubjectInFrontOfClock() },
                colors = SwitchDefaults.colors(checkedThumbColor = Color(0xFF00E5FF))
            )
        }
    }
}

/**
 * Tab 1: Combines Surface preview modes (including 3D depth map inspection),
 * AI segmentation model tuning (with detailed info cards), and 3D parallax motion.
 */
@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun LayersAndMotionTab(viewModel: StudioViewModel, state: StudioUiState) {
    val project = state.currentProject
    val m = project.motionConfig
    var threshold by remember(project.id, project.threshold) { mutableFloatStateOf(project.threshold) }
    var feathering by remember(project.id, project.edgeFeathering) { mutableIntStateOf(project.edgeFeathering) }
    var inpaintRadius by remember(project.id, project.inpaintRadius) { mutableIntStateOf(project.inpaintRadius) }
    var selectedModel by remember { mutableStateOf(viewModel.getCurrentModelType()) }

    Column(
        modifier = Modifier
            .fillMaxWidth()
            .verticalScroll(rememberScrollState()),
        verticalArrangement = Arrangement.spacedBy(14.dp)
    ) {
        // 1. Preview Mode Surface Selector (combines Surface tab & 3D Depth Map inspection)
        Column(verticalArrangement = Arrangement.spacedBy(4.dp)) {
            Text("Preview Screen Mode", fontSize = 12.sp, color = Color.Gray)
            Row(horizontalArrangement = Arrangement.spacedBy(6.dp)) {
                FilterChip(
                    selected = state.previewSurface == PreviewSurface.LOCK_SCREEN,
                    onClick = { viewModel.setPreviewSurface(PreviewSurface.LOCK_SCREEN) },
                    label = { Text("Lock Screen", fontSize = 11.sp) }
                )
                FilterChip(
                    selected = state.previewSurface == PreviewSurface.HOME_SCREEN,
                    onClick = { viewModel.setPreviewSurface(PreviewSurface.HOME_SCREEN) },
                    label = { Text("Home Screen", fontSize = 11.sp) }
                )
                FilterChip(
                    selected = state.previewSurface == PreviewSurface.AOD,
                    onClick = { viewModel.setPreviewSurface(PreviewSurface.AOD) },
                    label = { Text("AOD (OLED)", fontSize = 11.sp) }
                )
                FilterChip(
                    selected = state.previewSurface == PreviewSurface.DEPTH_MAP,
                    onClick = { viewModel.setPreviewSurface(PreviewSurface.DEPTH_MAP) },
                    label = { Text("3D Depth Map", fontSize = 11.sp) },
                    leadingIcon = { Icon(Icons.Default.ViewInAr, contentDescription = null, modifier = Modifier.size(14.dp)) }
                )
            }
            Text(
                text = when (state.previewSurface) {
                    PreviewSurface.LOCK_SCREEN -> "Showing full lock screen depth wallpaper with floating clock."
                    PreviewSurface.HOME_SCREEN -> "Simulating launcher with icons. Clock is auto-hidden to prevent clutter."
                    PreviewSurface.AOD -> "Power-saving pure black OLED display."
                    PreviewSurface.DEPTH_MAP -> "Visualizing AI continuous depth map (white = foreground, dark = background). Tilt phone to inspect depth planes."
                },
                fontSize = 11.sp,
                color = Color.LightGray
            )

            // Home screen specific controls
            if (state.previewSurface == PreviewSurface.HOME_SCREEN) {
                Spacer(modifier = Modifier.height(4.dp))
                Row(verticalAlignment = Alignment.CenterVertically) {
                    Text("Icon Dimming", fontSize = 11.sp, color = Color.Gray, modifier = Modifier.width(90.dp))
                    Slider(
                        value = project.homeScreenConfig.dimmingFactor,
                        onValueChange = { viewModel.updateHomeScreenDimming(it) },
                        valueRange = 0f..0.4f,
                        modifier = Modifier.weight(1f)
                    )
                }
            }
        }

        // 2. AI Segmentation Engine with Explanatory Card
        Column(verticalArrangement = Arrangement.spacedBy(6.dp)) {
            Text("AI Segmentation Engine", fontSize = 12.sp, color = Color.Gray)
            Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                FilterChip(
                    selected = selectedModel == SegmentationModelType.GROUP_MULTICLASS,
                    onClick = { selectedModel = SegmentationModelType.GROUP_MULTICLASS },
                    label = { Text("Group / Multi-Subject", fontSize = 11.sp) }
                )
                FilterChip(
                    selected = selectedModel == SegmentationModelType.SELFIE_FAST,
                    onClick = { selectedModel = SegmentationModelType.SELFIE_FAST },
                    label = { Text("Selfie Portrait", fontSize = 11.sp) }
                )
            }

            // Detailed Model Clarification Card
            Card(
                colors = CardDefaults.cardColors(containerColor = Color(0xFF1B1B2C)),
                shape = RoundedCornerShape(10.dp),
                modifier = Modifier.fillMaxWidth()
            ) {
                Row(
                    modifier = Modifier.padding(10.dp),
                    verticalAlignment = Alignment.CenterVertically
                ) {
                    Icon(Icons.Default.Info, contentDescription = null, tint = Color(0xFF00E5FF), modifier = Modifier.size(18.dp))
                    Spacer(modifier = Modifier.width(8.dp))
                    Text(
                        text = if (selectedModel == SegmentationModelType.GROUP_MULTICLASS) {
                            "Group & Multi-Subject Engine: Detects multiple people, full-body poses, and objects. Best for family and group photos."
                        } else {
                            "Selfie Portrait Engine: Ultra-fast neural model tuned for single or close-up portraits, with hair-strand precision."
                        },
                        fontSize = 11.sp,
                        color = Color.LightGray
                    )
                }
            }
        }

        // 3. Granular AI Tuning Controls (with detailed descriptions)
        Column(verticalArrangement = Arrangement.spacedBy(10.dp)) {
            Text("Granular AI Tuning", fontSize = 12.sp, color = Color.Gray)

            // Sensitivity / Threshold
            Column {
                Row(verticalAlignment = Alignment.CenterVertically) {
                    Text("Sensitivity", fontSize = 12.sp, color = Color.White, modifier = Modifier.width(90.dp))
                    Slider(
                        value = threshold,
                        onValueChange = { threshold = it },
                        valueRange = 0.15f..0.85f,
                        modifier = Modifier.weight(1f)
                    )
                    Text("${(threshold * 100).toInt()}%", fontSize = 11.sp, color = Color.LightGray, modifier = Modifier.width(36.dp))
                }
                Text(
                    text = "Adjusts detection threshold. Lower values capture hair & clothing outlines; higher isolates core subjects tightly. (Subjects remain 100% solid & opaque).",
                    fontSize = 10.sp,
                    color = Color.Gray
                )
            }

            // Edge Softness / Feathering
            Column {
                Row(verticalAlignment = Alignment.CenterVertically) {
                    Text("Edge Softness", fontSize = 12.sp, color = Color.White, modifier = Modifier.width(90.dp))
                    Slider(
                        value = feathering.toFloat(),
                        onValueChange = { feathering = it.toInt() },
                        valueRange = 1f..16f,
                        steps = 15,
                        modifier = Modifier.weight(1f)
                    )
                    Text("${feathering}px", fontSize = 11.sp, color = Color.LightGray, modifier = Modifier.width(36.dp))
                }
                Text(
                    text = "Guided matting feather radius to anti-alias and soften subject silhouette edges.",
                    fontSize = 10.sp,
                    color = Color.Gray
                )
            }

            // Inpaint Fill / Background Erasure Radius
            Column {
                Row(verticalAlignment = Alignment.CenterVertically) {
                    Text("Inpaint Fill", fontSize = 12.sp, color = Color.White, modifier = Modifier.width(90.dp))
                    Slider(
                        value = inpaintRadius.toFloat(),
                        onValueChange = { inpaintRadius = it.toInt() },
                        valueRange = 8f..40f,
                        steps = 32,
                        modifier = Modifier.weight(1f)
                    )
                    Text("${inpaintRadius}px", fontSize = 11.sp, color = Color.LightGray, modifier = Modifier.width(36.dp))
                }
                Text(
                    text = "Expansion radius to completely erase subjects from the background plate using multi-scale pyramid synthesis, eliminating duplicate reflections.",
                    fontSize = 10.sp,
                    color = Color.Gray
                )
            }

            // 3D Parallax Intensity
            Column {
                Row(verticalAlignment = Alignment.CenterVertically) {
                    Text("3D Parallax", fontSize = 12.sp, color = Color.White, modifier = Modifier.width(90.dp))
                    Slider(
                        value = m.parallaxIntensity,
                        onValueChange = { viewModel.updateParallaxIntensity(it) },
                        valueRange = 0f..1f,
                        modifier = Modifier.weight(1f)
                    )
                    Text("${(m.parallaxIntensity * 100).toInt()}%", fontSize = 11.sp, color = Color.LightGray, modifier = Modifier.width(36.dp))
                }
                Text(
                    text = "Controls how much layers move with gyroscope phone tilt and touch dragging.",
                    fontSize = 10.sp,
                    color = Color.Gray
                )
            }
        }

        // 4. Action Button: Re-Segment & Update Layers
        Button(
            onClick = {
                viewModel.reprocessWithTuning(
                    threshold = threshold,
                    feathering = feathering,
                    inpaintRadius = inpaintRadius,
                    modelType = selectedModel
                )
            },
            enabled = !state.isProcessing && state.sourceBitmap != null,
            colors = ButtonDefaults.buttonColors(containerColor = Color(0xFF00E5FF)),
            shape = RoundedCornerShape(12.dp),
            modifier = Modifier.fillMaxWidth()
        ) {
            if (state.isProcessing) {
                CircularProgressIndicator(
                    modifier = Modifier.size(16.dp),
                    color = Color.Black,
                    strokeWidth = 2.dp
                )
                Spacer(modifier = Modifier.width(8.dp))
                Text("Segmenting & Inpainting...", color = Color.Black, fontWeight = FontWeight.Bold)
            } else {
                Icon(Icons.Default.Refresh, contentDescription = null, tint = Color.Black, modifier = Modifier.size(18.dp))
                Spacer(modifier = Modifier.width(8.dp))
                Text("Re-Segment & Update Layers", color = Color.Black, fontWeight = FontWeight.Bold)
            }
        }
    }
}

/**
 * Returns the matching Compose FontFamily and FontWeight for a ClockFontStyle.
 */
private fun getFontFamilyAndWeight(style: ClockFontStyle): Pair<FontFamily, FontWeight> {
    return when (style) {
        ClockFontStyle.ROUNDED_BOLD -> FontFamily.Default to FontWeight.Bold
        ClockFontStyle.SERIF_CLASSIC -> FontFamily.Serif to FontWeight.Bold
        ClockFontStyle.MODERN_HEAVY -> FontFamily.SansSerif to FontWeight.Black
        ClockFontStyle.ELEGANT_THIN -> FontFamily.SansSerif to FontWeight.Light
        ClockFontStyle.STENCIL_DISPLAY -> FontFamily.Cursive to FontWeight.Bold
        ClockFontStyle.CYBER_MONO -> FontFamily.Monospace to FontWeight.Bold
    }
}

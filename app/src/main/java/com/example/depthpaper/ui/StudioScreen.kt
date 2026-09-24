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
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
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
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.ArrowBack
import androidx.compose.material.icons.filled.AddPhotoAlternate
import androidx.compose.material.icons.filled.BugReport
import androidx.compose.material.icons.filled.Layers
import androidx.compose.material.icons.filled.Refresh
import androidx.compose.material.icons.filled.ViewInAr
import androidx.compose.material.icons.filled.Wallpaper
import com.example.depthpaper.core.AppLogger
import com.example.depthpaper.core.SegmentationModelType
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.FilterChip
import androidx.compose.material3.FilterChipDefaults
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.ScrollableTabRow
import androidx.compose.material3.Slider
import androidx.compose.material3.Surface
import androidx.compose.material3.Switch
import androidx.compose.material3.SwitchDefaults
import androidx.compose.material3.Tab
import androidx.compose.material3.TabRow
import androidx.compose.material3.TabRowDefaults
import androidx.compose.material3.TabRowDefaults.tabIndicatorOffset
import androidx.compose.material3.Text
import androidx.compose.foundation.layout.aspectRatio
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
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.example.depthpaper.data.ClockFontStyle
import com.example.depthpaper.data.RenderMode
import com.example.depthpaper.ui.components.ParallaxViewport

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

            // Clean Top Action Bar (no overlap with status bar)
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

                // Mode toggle
                FilterChip(
                    selected = true,
                    onClick = { viewModel.toggleRenderMode() },
                    label = {
                        Text(
                            text = if (state.currentProject.renderMode == RenderMode.LAYERED_2D) "Layered Depth" else "3D Perspective",
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

            // Interactive Viewport (Drag clock directly on screen!)
            Box(
                modifier = Modifier
                    .fillMaxWidth()
                    .weight(1f)
                    .clip(RoundedCornerShape(20.dp))
                    .padding(horizontal = 8.dp)
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

                // Subtle hint at bottom of viewport
                Text(
                    text = "Touch & drag clock to reposition",
                    color = Color.White.copy(alpha = 0.5f),
                    fontSize = 11.sp,
                    modifier = Modifier
                        .align(Alignment.BottomCenter)
                        .padding(bottom = 10.dp)
                        .background(Color.Black.copy(alpha = 0.4f), RoundedCornerShape(12.dp))
                        .padding(horizontal = 10.dp, vertical = 4.dp)
                )

                // Processing indicator
                if (state.isProcessing) {
                    Box(
                        modifier = Modifier
                            .fillMaxSize()
                            .background(Color.Black.copy(alpha = 0.75f)),
                        contentAlignment = Alignment.Center
                    ) {
                        Column(horizontalAlignment = Alignment.CenterHorizontally) {
                            CircularProgressIndicator(color = Color(0xFF00E5FF))
                            Spacer(modifier = Modifier.height(12.dp))
                            Text(
                                text = state.statusMessage ?: "Processing on-device...",
                                color = Color.White,
                                fontSize = 14.sp
                            )
                        }
                    }
                }
            }

            Spacer(modifier = Modifier.height(8.dp))

            // Bottom Customization Sheet
            StudioBottomControlPanel(
                viewModel = viewModel,
                state = state,
                onPickPhoto = { photoPickerLauncher.launch("image/*") }
            )
        }
    }
}

@Composable
fun StudioBottomControlPanel(
    viewModel: StudioViewModel,
    state: StudioUiState,
    onPickPhoto: () -> Unit
) {
    var selectedTab by remember { mutableIntStateOf(0) }
    val tabs = listOf("Clock", "Layers & AI", "Motion", "Surface", "Photos")

    Surface(
        modifier = Modifier.fillMaxWidth(),
        color = Color(0xFF141422),
        shape = RoundedCornerShape(topStart = 20.dp, topEnd = 20.dp)
    ) {
        Column(modifier = Modifier.padding(14.dp)) {
            ScrollableTabRow(
                selectedTabIndex = selectedTab,
                containerColor = Color.Transparent,
                contentColor = Color(0xFF00E5FF),
                edgePadding = 0.dp,
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
                            Text(
                                text = title,
                                fontSize = 12.sp,
                                fontWeight = if (selectedTab == index) FontWeight.Bold else FontWeight.Normal
                            )
                        }
                    )
                }
            }

            Spacer(modifier = Modifier.height(12.dp))

            when (selectedTab) {
                0 -> ClockControlTab(viewModel, state)
                1 -> LayersAndAiControlTab(viewModel, state)
                2 -> MotionControlTab(viewModel, state)
                3 -> SurfaceControlTab(viewModel, state)
                4 -> PhotosTab(viewModel, state, onPickPhoto)
            }
        }
    }
}

@Composable
fun ClockControlTab(viewModel: StudioViewModel, state: StudioUiState) {
    val cfg = state.currentProject.lockScreenConfig

    Column(verticalArrangement = Arrangement.spacedBy(10.dp)) {
        // Typography Styles
        Text("Typography Style", fontSize = 11.sp, color = Color.Gray)
        LazyRow(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            items(ClockFontStyle.values()) { style ->
                FilterChip(
                    selected = cfg.fontStyle == style,
                    onClick = { viewModel.updateClockStyle(style) },
                    label = { Text(style.name.replace('_', ' '), fontSize = 11.sp) }
                )
            }
        }

        // Color Palette
        Text("Color", fontSize = 11.sp, color = Color.Gray)
        val colors = listOf(
            0xFFFFFFFF to "White",
            0xFFFF3366 to "Rose",
            0xFFFFD700 to "Gold",
            0xFF00E5FF to "Cyan",
            0xFFA29BFE to "Lavender",
            0xFF55EFC4 to "Mint"
        )
        Row(horizontalArrangement = Arrangement.spacedBy(12.dp)) {
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

        // Clock Scale Slider
        Row(verticalAlignment = Alignment.CenterVertically) {
            Text("Clock Size", fontSize = 12.sp, color = Color.Gray, modifier = Modifier.width(90.dp))
            Slider(
                value = cfg.clockScale,
                onValueChange = { viewModel.updateClockScale(it) },
                valueRange = 0.7f..1.4f,
                modifier = Modifier.weight(1f)
            )
        }

        // Subject in front toggle
        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.SpaceBetween,
            verticalAlignment = Alignment.CenterVertically
        ) {
            Column {
                Text("Subject in Front of Clock", fontSize = 13.sp, color = Color.White, fontWeight = FontWeight.SemiBold)
                Text("Depth effect behind subject", fontSize = 11.sp, color = Color.Gray)
            }
            Switch(
                checked = cfg.subjectInFrontOfClock,
                onCheckedChange = { viewModel.toggleSubjectInFrontOfClock() },
                colors = SwitchDefaults.colors(checkedThumbColor = Color(0xFF00E5FF))
            )
        }
    }
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun LayersAndAiControlTab(viewModel: StudioViewModel, state: StudioUiState) {
    val project = state.currentProject
    var threshold by remember(project.id, project.threshold) { mutableFloatStateOf(project.threshold) }
    var feathering by remember(project.id, project.edgeFeathering) { mutableIntStateOf(project.edgeFeathering) }
    var inpaintRadius by remember(project.id, project.inpaintRadius) { mutableIntStateOf(project.inpaintRadius) }
    var selectedModel by remember { mutableStateOf(viewModel.getCurrentModelType()) }

    Column(verticalArrangement = Arrangement.spacedBy(10.dp)) {
        // 1. Render Mode Switcher
        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.SpaceBetween,
            verticalAlignment = Alignment.CenterVertically
        ) {
            Column {
                Text("Layer Render Mode", fontSize = 13.sp, color = Color.White, fontWeight = FontWeight.SemiBold)
                Text(
                    text = if (project.renderMode == RenderMode.LAYERED_2D) "2.5D Layered Cutout" else "3D Perspective Depth",
                    fontSize = 11.sp,
                    color = Color(0xFF00E5FF)
                )
            }
            FilterChip(
                selected = project.renderMode == RenderMode.LAYERED_2D,
                onClick = { viewModel.toggleRenderMode() },
                label = { Text(if (project.renderMode == RenderMode.LAYERED_2D) "2.5D Cutout" else "3D Spatial") }
            )
        }

        // 2. AI Model Engine Selector
        Text("AI Segmentation Engine", fontSize = 11.sp, color = Color.Gray)
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

        // 3. Threshold Slider
        Row(verticalAlignment = Alignment.CenterVertically) {
            Text("Sensitivity", fontSize = 12.sp, color = Color.Gray, modifier = Modifier.width(90.dp))
            Slider(
                value = threshold,
                onValueChange = { threshold = it },
                valueRange = 0.15f..0.85f,
                modifier = Modifier.weight(1f)
            )
            Text("${(threshold * 100).toInt()}%", fontSize = 11.sp, color = Color.LightGray, modifier = Modifier.width(36.dp))
        }

        // 4. Edge Feathering
        Row(verticalAlignment = Alignment.CenterVertically) {
            Text("Edge Softness", fontSize = 12.sp, color = Color.Gray, modifier = Modifier.width(90.dp))
            Slider(
                value = feathering.toFloat(),
                onValueChange = { feathering = it.toInt() },
                valueRange = 1f..16f,
                steps = 15,
                modifier = Modifier.weight(1f)
            )
            Text("${feathering}px", fontSize = 11.sp, color = Color.LightGray, modifier = Modifier.width(36.dp))
        }

        // 5. Inpaint Occlusion Hole Fill Radius
        Row(verticalAlignment = Alignment.CenterVertically) {
            Text("Inpaint Fill", fontSize = 12.sp, color = Color.Gray, modifier = Modifier.width(90.dp))
            Slider(
                value = inpaintRadius.toFloat(),
                onValueChange = { inpaintRadius = it.toInt() },
                valueRange = 4f..30f,
                steps = 26,
                modifier = Modifier.weight(1f)
            )
            Text("${inpaintRadius}px", fontSize = 11.sp, color = Color.LightGray, modifier = Modifier.width(36.dp))
        }

        // 6. Action Button
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
            shape = RoundedCornerShape(10.dp),
            modifier = Modifier.fillMaxWidth()
        ) {
            if (state.isProcessing) {
                CircularProgressIndicator(
                    modifier = Modifier.size(16.dp),
                    color = Color.Black,
                    strokeWidth = 2.dp
                )
                Spacer(modifier = Modifier.width(8.dp))
                Text("Segmenting On-Device...", color = Color.Black, fontWeight = FontWeight.Bold)
            } else {
                Icon(Icons.Default.Refresh, contentDescription = null, tint = Color.Black, modifier = Modifier.size(18.dp))
                Spacer(modifier = Modifier.width(8.dp))
                Text("Re-Segment & Update Layers", color = Color.Black, fontWeight = FontWeight.Bold)
            }
        }
    }
}

@Composable
fun MotionControlTab(viewModel: StudioViewModel, state: StudioUiState) {
    val m = state.currentProject.motionConfig

    Column(verticalArrangement = Arrangement.spacedBy(10.dp)) {
        Row(verticalAlignment = Alignment.CenterVertically) {
            Text("3D Parallax", fontSize = 12.sp, color = Color.Gray, modifier = Modifier.width(90.dp))
            Slider(
                value = m.parallaxIntensity,
                onValueChange = { viewModel.updateParallaxIntensity(it) },
                valueRange = 0f..1f,
                modifier = Modifier.weight(1f)
            )
        }

        Row(verticalAlignment = Alignment.CenterVertically) {
            Text("Icon Dimming", fontSize = 12.sp, color = Color.Gray, modifier = Modifier.width(90.dp))
            Slider(
                value = state.currentProject.homeScreenConfig.dimmingFactor,
                onValueChange = { viewModel.updateHomeScreenDimming(it) },
                valueRange = 0f..0.4f,
                modifier = Modifier.weight(1f)
            )
        }

        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.SpaceBetween,
            verticalAlignment = Alignment.CenterVertically
        ) {
            Column {
                Text("Hide Clock on Home Screen", fontSize = 13.sp, color = Color.White, fontWeight = FontWeight.SemiBold)
                Text("Auto-hide clock when unlocked", fontSize = 11.sp, color = Color.Gray)
            }
            Switch(
                checked = state.currentProject.homeScreenConfig.hideClockOnHomeScreen,
                onCheckedChange = { viewModel.toggleHideClockOnHomeScreen() },
                colors = SwitchDefaults.colors(checkedThumbColor = Color(0xFF00E5FF))
            )
        }
    }
}

@Composable
fun SurfaceControlTab(viewModel: StudioViewModel, state: StudioUiState) {
    Column(verticalArrangement = Arrangement.spacedBy(12.dp)) {
        Text("Preview Screen Surface", fontSize = 12.sp, color = Color.Gray)
        Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            FilterChip(
                selected = state.previewSurface == PreviewSurface.LOCK_SCREEN,
                onClick = { viewModel.setPreviewSurface(PreviewSurface.LOCK_SCREEN) },
                label = { Text("Lock Screen") }
            )
            FilterChip(
                selected = state.previewSurface == PreviewSurface.HOME_SCREEN,
                onClick = { viewModel.setPreviewSurface(PreviewSurface.HOME_SCREEN) },
                label = { Text("Home Screen") }
            )
            FilterChip(
                selected = state.previewSurface == PreviewSurface.AOD,
                onClick = { viewModel.setPreviewSurface(PreviewSurface.AOD) },
                label = { Text("AOD (OLED Black)") }
            )
        }

        Text(
            text = when (state.previewSurface) {
                PreviewSurface.LOCK_SCREEN -> "Showing full lock screen with layered depth clock."
                PreviewSurface.HOME_SCREEN -> "Clock is hidden so it doesn't clash with launcher icons."
                PreviewSurface.AOD -> "Pure black OLED power saving screen with zero sensor motion."
            },
            fontSize = 11.sp,
            color = Color.LightGray
        )
    }
}

@Composable
fun PhotosTab(viewModel: StudioViewModel, state: StudioUiState, onPickPhoto: () -> Unit) {
    Column(verticalArrangement = Arrangement.spacedBy(12.dp)) {
        Button(
            onClick = onPickPhoto,
            modifier = Modifier.fillMaxWidth(),
            colors = ButtonDefaults.buttonColors(containerColor = Color(0xFF00E5FF)),
            shape = RoundedCornerShape(12.dp)
        ) {
            Icon(Icons.Default.AddPhotoAlternate, contentDescription = null, tint = Color.Black)
            Spacer(modifier = Modifier.width(8.dp))
            Text("Choose from Device Gallery", color = Color.Black, fontWeight = FontWeight.Bold)
        }

        Text("Switch Wallpaper Project", fontSize = 11.sp, color = Color.Gray)
        LazyRow(horizontalArrangement = Arrangement.spacedBy(10.dp)) {
            items(state.allProjects, key = { it.id }) { proj ->
                Box(
                    modifier = Modifier
                        .width(90.dp)
                        .aspectRatio(9f / 16f)
                        .clip(RoundedCornerShape(10.dp))
                        .border(
                            width = if (proj.id == state.currentProject.id) 2.dp else 1.dp,
                            color = if (proj.id == state.currentProject.id) Color(0xFF00E5FF) else Color(0xFF333348),
                            shape = RoundedCornerShape(10.dp)
                        )
                        .clickable { viewModel.selectProject(proj) }
                ) {
                    ProjectThumbnail(
                        path = proj.thumbnailPath.ifBlank { proj.sourceImagePath },
                        modifier = Modifier.fillMaxSize()
                    )
                    Text(
                        text = proj.title,
                        fontSize = 10.sp,
                        color = Color.White,
                        maxLines = 1,
                        modifier = Modifier
                            .align(Alignment.BottomCenter)
                            .background(Color(0xAA000000))
                            .fillMaxWidth()
                            .padding(4.dp)
                    )
                }
            }
        }
    }
}

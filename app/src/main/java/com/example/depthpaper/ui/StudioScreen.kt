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
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.lazy.LazyRow
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.ArrowBack
import androidx.compose.material.icons.filled.AddPhotoAlternate
import androidx.compose.material.icons.filled.Check
import androidx.compose.material.icons.filled.CropRotate
import androidx.compose.material.icons.filled.Layers
import androidx.compose.material.icons.filled.Schedule
import androidx.compose.material.icons.filled.Tune
import androidx.compose.material.icons.filled.ViewInAr
import androidx.compose.material.icons.filled.Wallpaper
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.FilterChip
import androidx.compose.material3.FilterChipDefaults
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Slider
import androidx.compose.material3.SliderDefaults
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

    // Photo picker launcher
    val photoPickerLauncher = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.GetContent()
    ) { uri: Uri? ->
        uri?.let {
            val bitmap = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.P) {
                ImageDecoder.decodeBitmap(ImageDecoder.createSource(context.contentResolver, it)) { decoder, _, _ ->
                    decoder.isMutableRequired = true
                }
            } else {
                @Suppress("DEPRECATION")
                MediaStore.Images.Media.getBitmap(context.contentResolver, it)
            }
            viewModel.importNewImage(bitmap, "Custom Wallpaper")
        }
    }

    Box(modifier = modifier.fillMaxSize().background(Color(0xFF0F0F1A))) {
        Column(modifier = Modifier.fillMaxSize()) {

            // Top Action Bar
            Row(
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(horizontal = 12.dp, vertical = 8.dp),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically
            ) {
                IconButton(onClick = onNavigateToGallery) {
                    Icon(Icons.AutoMirrored.Filled.ArrowBack, contentDescription = "Gallery", tint = Color.White)
                }

                // Render Mode Toggle Badge
                FilterChip(
                    selected = true,
                    onClick = { viewModel.toggleRenderMode() },
                    label = {
                        Text(
                            text = if (state.currentProject.renderMode == RenderMode.LAYERED_2D) "Layered 2.5D" else "3D Spatial Scene",
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

                // "Set Wallpaper" action
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

            // Viewport + Surface Pill Switcher
            Box(
                modifier = Modifier
                    .fillMaxWidth()
                    .weight(1f)
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
                    modifier = Modifier.fillMaxSize()
                )

                // Surface selector pills overlay (Lock Screen | Home Screen | AOD)
                Row(
                    modifier = Modifier
                        .align(Alignment.TopCenter)
                        .padding(top = 12.dp)
                        .clip(RoundedCornerShape(20.dp))
                        .background(Color.Black.copy(alpha = 0.65f))
                        .padding(4.dp),
                    horizontalArrangement = Arrangement.spacedBy(4.dp)
                ) {
                    SurfacePill(
                        text = "Lock Screen",
                        selected = state.previewSurface == PreviewSurface.LOCK_SCREEN,
                        onClick = { viewModel.setPreviewSurface(PreviewSurface.LOCK_SCREEN) }
                    )
                    SurfacePill(
                        text = "Home Screen",
                        selected = state.previewSurface == PreviewSurface.HOME_SCREEN,
                        onClick = { viewModel.setPreviewSurface(PreviewSurface.HOME_SCREEN) }
                    )
                    SurfacePill(
                        text = "AOD",
                        selected = state.previewSurface == PreviewSurface.AOD,
                        onClick = { viewModel.setPreviewSurface(PreviewSurface.AOD) }
                    )
                }

                // Loading overlay
                if (state.isProcessing) {
                    Box(
                        modifier = Modifier
                            .fillMaxSize()
                            .background(Color.Black.copy(alpha = 0.7f)),
                        contentAlignment = Alignment.Center
                    ) {
                        Column(horizontalAlignment = Alignment.CenterHorizontally) {
                            CircularProgressIndicator(color = Color(0xFF00E5FF))
                            Spacer(modifier = Modifier.height(12.dp))
                            Text(
                                text = state.statusMessage ?: "Processing image on-device...",
                                color = Color.White,
                                fontSize = 14.sp
                            )
                        }
                    }
                }
            }

            // Bottom Customization Control Panel
            StudioBottomControlPanel(
                viewModel = viewModel,
                state = state,
                onPickPhoto = { photoPickerLauncher.launch("image/*") }
            )
        }
    }
}

@Composable
fun SurfacePill(text: String, selected: Boolean, onClick: () -> Unit) {
    Box(
        modifier = Modifier
            .clip(RoundedCornerShape(16.dp))
            .background(if (selected) Color(0xFF00E5FF) else Color.Transparent)
            .clickable { onClick() }
            .padding(horizontal = 12.dp, vertical = 6.dp)
    ) {
        Text(
            text = text,
            color = if (selected) Color.Black else Color.LightGray,
            fontSize = 12.sp,
            fontWeight = if (selected) FontWeight.Bold else FontWeight.Normal
        )
    }
}

@Composable
fun StudioBottomControlPanel(
    viewModel: StudioViewModel,
    state: StudioUiState,
    onPickPhoto: () -> Unit
) {
    var selectedTab by remember { mutableIntStateOf(0) }
    val tabs = listOf("Clock", "Motion", "AI Tune", "Photos")

    Surface(
        modifier = Modifier.fillMaxWidth(),
        color = Color(0xFF161626),
        shape = RoundedCornerShape(topStart = 24.dp, topEnd = 24.dp)
    ) {
        Column(modifier = Modifier.padding(16.dp)) {
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
                            Text(
                                text = title,
                                fontSize = 13.sp,
                                fontWeight = if (selectedTab == index) FontWeight.Bold else FontWeight.Normal
                            )
                        }
                    )
                }
            }

            Spacer(modifier = Modifier.height(14.dp))

            when (selectedTab) {
                0 -> ClockControlTab(viewModel, state)
                1 -> MotionControlTab(viewModel, state)
                2 -> AiTuneControlTab(viewModel, state)
                3 -> PhotosTab(viewModel, state, onPickPhoto)
            }
        }
    }
}

@Composable
fun ClockControlTab(viewModel: StudioViewModel, state: StudioUiState) {
    val cfg = state.currentProject.lockScreenConfig

    Column(verticalArrangement = Arrangement.spacedBy(10.dp)) {
        // Typography Styles
        Text("Typography Style", fontSize = 12.sp, color = Color.Gray)
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
        Text("Color & Shadow", fontSize = 12.sp, color = Color.Gray)
        val colors = listOf(
            0xFFFFFFFF to "White",
            0xFFFF3366 to "Neon Rose",
            0xFFFFD700 to "Gold",
            0xFF00E5FF to "Cyan",
            0xFFA29BFE to "Lavender",
            0xFF55EFC4 to "Mint"
        )
        Row(horizontalArrangement = Arrangement.spacedBy(12.dp)) {
            colors.forEach { (colorHex, _) ->
                Box(
                    modifier = Modifier
                        .size(32.dp)
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

        // Vertical Placement Slider
        Row(verticalAlignment = Alignment.CenterVertically) {
            Text("Vertical Position", fontSize = 12.sp, color = Color.Gray, modifier = Modifier.width(110.dp))
            Slider(
                value = cfg.verticalOffsetPercent,
                onValueChange = { viewModel.updateClockVerticalOffset(it) },
                valueRange = 0.10f..0.45f,
                modifier = Modifier.weight(1f)
            )
        }

        // Clock Scale Slider
        Row(verticalAlignment = Alignment.CenterVertically) {
            Text("Clock Size", fontSize = 12.sp, color = Color.Gray, modifier = Modifier.width(110.dp))
            Slider(
                value = cfg.clockScale,
                onValueChange = { viewModel.updateClockScale(it) },
                valueRange = 0.7f..1.4f,
                modifier = Modifier.weight(1f)
            )
        }

        // Full creative control: Subject in front toggle
        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.SpaceBetween,
            verticalAlignment = Alignment.CenterVertically
        ) {
            Column {
                Text("Subject In Front of Clock", fontSize = 13.sp, color = Color.White, fontWeight = FontWeight.SemiBold)
                Text("Place clock behind subject depth layer", fontSize = 11.sp, color = Color.Gray)
            }
            Switch(
                checked = cfg.subjectInFrontOfClock,
                onCheckedChange = { viewModel.toggleSubjectInFrontOfClock() },
                colors = SwitchDefaults.colors(checkedThumbColor = Color(0xFF00E5FF))
            )
        }
    }
}

@Composable
fun MotionControlTab(viewModel: StudioViewModel, state: StudioUiState) {
    val m = state.currentProject.motionConfig

    Column(verticalArrangement = Arrangement.spacedBy(10.dp)) {
        Row(verticalAlignment = Alignment.CenterVertically) {
            Text("3D Tilt Intensity", fontSize = 12.sp, color = Color.Gray, modifier = Modifier.width(110.dp))
            Slider(
                value = m.parallaxIntensity,
                onValueChange = { viewModel.updateParallaxIntensity(it) },
                valueRange = 0f..1f,
                modifier = Modifier.weight(1f)
            )
        }

        // Home screen dimming
        Row(verticalAlignment = Alignment.CenterVertically) {
            Text("Icon Dimming", fontSize = 12.sp, color = Color.Gray, modifier = Modifier.width(110.dp))
            Slider(
                value = state.currentProject.homeScreenConfig.dimmingFactor,
                onValueChange = { viewModel.updateHomeScreenDimming(it) },
                valueRange = 0f..0.4f,
                modifier = Modifier.weight(1f)
            )
        }

        // Auto-hide clock on Home Screen toggle
        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.SpaceBetween,
            verticalAlignment = Alignment.CenterVertically
        ) {
            Column {
                Text("Hide Clock on Home Screen", fontSize = 13.sp, color = Color.White, fontWeight = FontWeight.SemiBold)
                Text("Only show depth clock on Lock Screen", fontSize = 11.sp, color = Color.Gray)
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
fun AiTuneControlTab(viewModel: StudioViewModel, state: StudioUiState) {
    var threshold by remember(state.currentProject) { mutableFloatStateOf(state.currentProject.threshold) }
    var feathering by remember(state.currentProject) { mutableIntStateOf(state.currentProject.edgeFeathering) }
    var inpaintRadius by remember(state.currentProject) { mutableIntStateOf(state.currentProject.inpaintRadius) }

    Column(verticalArrangement = Arrangement.spacedBy(10.dp)) {
        Row(verticalAlignment = Alignment.CenterVertically) {
            Text("Cutout Threshold", fontSize = 12.sp, color = Color.Gray, modifier = Modifier.width(110.dp))
            Slider(
                value = threshold,
                onValueChange = { threshold = it },
                valueRange = 0.2f..0.85f,
                modifier = Modifier.weight(1f)
            )
        }

        Row(verticalAlignment = Alignment.CenterVertically) {
            Text("Edge Feathering", fontSize = 12.sp, color = Color.Gray, modifier = Modifier.width(110.dp))
            Slider(
                value = feathering.toFloat(),
                onValueChange = { feathering = it.toInt() },
                valueRange = 2f..14f,
                modifier = Modifier.weight(1f)
            )
        }

        Row(verticalAlignment = Alignment.CenterVertically) {
            Text("Inpaint Fill", fontSize = 12.sp, color = Color.Gray, modifier = Modifier.width(110.dp))
            Slider(
                value = inpaintRadius.toFloat(),
                onValueChange = { inpaintRadius = it.toInt() },
                valueRange = 6f..24f,
                modifier = Modifier.weight(1f)
            )
        }

        Button(
            onClick = { viewModel.reprocessWithTuning(threshold, feathering, inpaintRadius) },
            modifier = Modifier.fillMaxWidth(),
            colors = ButtonDefaults.buttonColors(containerColor = Color(0xFF6C5CE7)),
            shape = RoundedCornerShape(12.dp)
        ) {
            Icon(Icons.Default.Tune, contentDescription = null, modifier = Modifier.size(16.dp))
            Spacer(modifier = Modifier.width(6.dp))
            Text("Apply AI Edge Refinement")
        }
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

        Text("Or Switch Wallpaper Project", fontSize = 12.sp, color = Color.Gray)
        LazyRow(horizontalArrangement = Arrangement.spacedBy(10.dp)) {
            items(state.allProjects, key = { it.id }) { proj ->
                ProjectScreenshotCard(
                    project = proj,
                    isCompact = true,
                    onCardClick = { viewModel.selectProject(proj) },
                    onSetActiveClick = { /* Handled in studio */ },
                    onFavoriteClick = { viewModel.toggleFavorite(proj.id) }
                )
            }
        }
    }
}

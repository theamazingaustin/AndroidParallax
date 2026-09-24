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
import androidx.compose.foundation.interaction.MutableInteractionSource
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.aspectRatio
import androidx.compose.foundation.layout.fillMaxHeight
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.heightIn
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
import androidx.compose.material3.BottomSheetScaffold
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
import androidx.compose.material3.SheetValue
import androidx.compose.material3.Slider
import androidx.compose.material3.Surface
import androidx.compose.material3.Switch
import androidx.compose.material3.SwitchDefaults
import androidx.compose.material3.Tab
import androidx.compose.material3.TabRow
import androidx.compose.material3.TabRowDefaults
import androidx.compose.material3.TabRowDefaults.tabIndicatorOffset
import androidx.compose.material3.Text
import androidx.compose.material3.rememberBottomSheetScaffoldState
import androidx.compose.runtime.Composable
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableFloatStateOf
import androidx.compose.runtime.mutableIntStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalConfiguration
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import kotlinx.coroutines.launch
import com.example.depthpaper.core.AppLogger
import com.example.depthpaper.core.AiModelChoice
import com.example.depthpaper.core.AiPipelineChoice
import com.example.depthpaper.core.ProcessingMode
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
            viewModel.changeProjectImage(bitmap)
        }
    }

    val scaffoldState = rememberBottomSheetScaffoldState()
    val coroutineScope = rememberCoroutineScope()

    BottomSheetScaffold(
        scaffoldState = scaffoldState,
        sheetPeekHeight = 88.dp,
        sheetContainerColor = Color(0xFF141422),
        sheetContentColor = Color.White,
        sheetShape = RoundedCornerShape(topStart = 24.dp, topEnd = 24.dp),
        sheetShadowElevation = 16.dp,
        sheetTonalElevation = 8.dp,
        sheetDragHandle = {
            Box(
                modifier = Modifier
                    .padding(top = 10.dp, bottom = 6.dp)
                    .width(40.dp)
                    .height(4.dp)
                    .background(Color(0xFF4A4A65), RoundedCornerShape(2.dp))
            )
        },
        sheetContent = {
            StudioBottomControlPanel(
                viewModel = viewModel,
                state = state,
                onPickPhoto = { photoPickerLauncher.launch("image/*") },
                onTabSelected = {
                    coroutineScope.launch {
                        scaffoldState.bottomSheetState.expand()
                    }
                }
            )
        },
        containerColor = Color(0xFF0A0A12),
        topBar = {
            // Top Action Bar
            Row(
                modifier = Modifier
                    .fillMaxWidth()
                    .statusBarsPadding()
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
        },
        modifier = modifier.fillMaxSize()
    ) { innerPadding ->
        // Phone-proportioned Interactive Preview Canvas (Full-screen sized)
        val configuration = LocalConfiguration.current
        val phoneAspectRatio = (configuration.screenWidthDp.toFloat() / configuration.screenHeightDp.toFloat()).coerceIn(0.42f, 0.65f)

        Box(
            modifier = Modifier
                .fillMaxSize()
                .padding(innerPadding)
                .clickable(
                    interactionSource = remember { MutableInteractionSource() },
                    indication = null
                ) {
                    // Tap on preview canvas collapses expanded bottom sheet
                    if (scaffoldState.bottomSheetState.currentValue == SheetValue.Expanded) {
                        coroutineScope.launch {
                            scaffoldState.bottomSheetState.partialExpand()
                        }
                    }
                },
            contentAlignment = Alignment.Center
        ) {
            Box(
                modifier = Modifier
                    .fillMaxHeight(0.96f)
                    .aspectRatio(phoneAspectRatio, matchHeightConstraintsFirst = true)
                    .clip(RoundedCornerShape(26.dp))
                    .border(2.dp, Color(0xFF28283E), RoundedCornerShape(26.dp))
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
                    onImageTransformChanged = { scale, panX, panY ->
                        viewModel.updateImageTransform(scale, panX, panY)
                    },
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
    onPickPhoto: () -> Unit,
    onTabSelected: () -> Unit = {}
) {
    var selectedTab by remember { mutableIntStateOf(0) }
    val tabs = listOf("Clock & Wallpaper", "Layers & 3D Motion")

    Column(
        modifier = Modifier
            .fillMaxWidth()
            .heightIn(min = 400.dp, max = 560.dp)
            .padding(horizontal = 14.dp, vertical = 4.dp)
            .navigationBarsPadding()
    ) {
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
                    onClick = {
                        selectedTab = index
                        onTabSelected()
                    },
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
    var maskExpansion by remember(project.id, project.maskExpansion) { mutableIntStateOf(project.maskExpansion) }
    var inpaintRadius by remember(project.id, project.inpaintRadius) { mutableIntStateOf(project.inpaintRadius) }
    var cutoutContrast by remember(project.id, project.cutoutContrast) { mutableFloatStateOf(project.cutoutContrast) }
    var processingMode by remember(project.id, project.processingMode) { mutableStateOf(project.processingMode) }
    var selectedModel by remember(project.id, project.selectedModel) { mutableStateOf(project.selectedModel) }
    var selectedPipeline by remember(project.id, project.selectedPipeline) { mutableStateOf(project.selectedPipeline) }
    var enablePreprocessing by remember(project.id, project.enablePreprocessing) { mutableStateOf(project.enablePreprocessing) }

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

        // 2. Pre-Processing Enhancement Option (Works with ANY model or pipeline)
        Card(
            colors = CardDefaults.cardColors(containerColor = Color(0xFF1E1F38)),
            shape = RoundedCornerShape(12.dp),
            modifier = Modifier.fillMaxWidth()
        ) {
            Column(modifier = Modifier.padding(12.dp), verticalArrangement = Arrangement.spacedBy(6.dp)) {
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    verticalAlignment = Alignment.CenterVertically,
                    horizontalArrangement = Arrangement.SpaceBetween
                ) {
                    Row(verticalAlignment = Alignment.CenterVertically) {
                        Icon(Icons.Default.Tune, contentDescription = null, tint = Color(0xFF00E5FF), modifier = Modifier.size(18.dp))
                        Spacer(modifier = Modifier.width(8.dp))
                        Text("Pre-Process Image (CLAHE & Bilateral)", fontSize = 12.sp, fontWeight = FontWeight.SemiBold, color = Color.White)
                    }
                    Switch(
                        checked = enablePreprocessing,
                        onCheckedChange = { enablePreprocessing = it }
                    )
                }
                Text(
                    text = "Applies Contrast-Limited Adaptive Equalization & Bilateral Denoising prior to inference. Anchors subject contrast gradients and removes sensor grain (crucial for grey hoodies on rocks, dark forest paths, or low lighting).",
                    fontSize = 11.sp,
                    color = Color.LightGray
                )
            }
        }

        // 3. AI Processing Engine (Standalone Models & Multi-Model Pipelines)
        Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
            Text("AI Processing Engine", fontSize = 12.sp, color = Color.Gray)

            // Section A: Standalone AI Models
            Text("Standalone AI Models", fontSize = 11.sp, fontWeight = FontWeight.Medium, color = Color(0xFFB0B0C0))
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.spacedBy(6.dp)
            ) {
                AiModelChoice.entries.forEach { model ->
                    FilterChip(
                        selected = processingMode == ProcessingMode.SINGLE_MODEL && selectedModel == model,
                        onClick = {
                            selectedModel = model
                            processingMode = ProcessingMode.SINGLE_MODEL
                        },
                        label = { Text(model.shortLabel, fontSize = 11.sp) }
                    )
                }
            }

            // Section B: Multi-Model High-Precision Pipelines (Visually Separated)
            Spacer(modifier = Modifier.height(2.dp))
            Row(verticalAlignment = Alignment.CenterVertically) {
                Box(modifier = Modifier.weight(1f).height(1.dp).background(Color(0xFF2E2E4A)))
                Text(
                    "  MULTI-MODEL PIPELINES  ",
                    fontSize = 9.sp,
                    fontWeight = FontWeight.Bold,
                    color = Color(0xFF00E5FF)
                )
                Box(modifier = Modifier.weight(1f).height(1.dp).background(Color(0xFF2E2E4A)))
            }
            Spacer(modifier = Modifier.height(2.dp))

            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.spacedBy(6.dp)
            ) {
                AiPipelineChoice.entries.forEach { pipeline ->
                    FilterChip(
                        selected = processingMode == ProcessingMode.PIPELINE && selectedPipeline == pipeline,
                        onClick = {
                            selectedPipeline = pipeline
                            processingMode = ProcessingMode.PIPELINE
                        },
                        label = { Text(pipeline.shortLabel, fontSize = 11.sp) }
                    )
                }
            }

            // Detailed Model / Pipeline Clarification Card
            Card(
                colors = CardDefaults.cardColors(containerColor = Color(0xFF1B1B2C)),
                shape = RoundedCornerShape(10.dp),
                modifier = Modifier.fillMaxWidth()
            ) {
                Column(
                    modifier = Modifier.padding(12.dp),
                    verticalArrangement = Arrangement.spacedBy(6.dp)
                ) {
                    Row(verticalAlignment = Alignment.CenterVertically) {
                        Surface(
                            color = if (processingMode == ProcessingMode.PIPELINE) Color(0xFF00E5FF).copy(alpha = 0.15f) else Color(0xFFFF9100).copy(alpha = 0.15f),
                            shape = RoundedCornerShape(4.dp)
                        ) {
                            Text(
                                text = if (processingMode == ProcessingMode.PIPELINE) "PIPELINE CASCADE" else "STANDALONE MODEL",
                                color = if (processingMode == ProcessingMode.PIPELINE) Color(0xFF00E5FF) else Color(0xFFFF9100),
                                fontSize = 9.sp,
                                fontWeight = FontWeight.Bold,
                                modifier = Modifier.padding(horizontal = 6.dp, vertical = 2.dp)
                            )
                        }
                        Spacer(modifier = Modifier.width(8.dp))
                        Text(
                            text = if (processingMode == ProcessingMode.PIPELINE) selectedPipeline.pipelineName else selectedModel.modelName,
                            fontSize = 12.sp,
                            fontWeight = FontWeight.Bold,
                            color = Color.White
                        )
                    }

                    Text(
                        text = if (processingMode == ProcessingMode.PIPELINE) selectedPipeline.bestAt else selectedModel.bestAt,
                        fontSize = 11.sp,
                        color = Color.LightGray
                    )

                    Row(verticalAlignment = Alignment.CenterVertically) {
                        Icon(Icons.Default.Info, contentDescription = null, tint = Color(0xFF4CAF50), modifier = Modifier.size(13.dp))
                        Spacer(modifier = Modifier.width(4.dp))
                        Text(
                            text = if (processingMode == ProcessingMode.PIPELINE) selectedPipeline.license else selectedModel.license,
                            fontSize = 10.sp,
                            color = Color(0xFF81C784)
                        )
                    }
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
                        value = threshold.coerceIn(0.35f, 0.75f),
                        onValueChange = { threshold = it },
                        valueRange = 0.35f..0.75f,
                        modifier = Modifier.weight(1f)
                    )
                    Text("${(threshold.coerceIn(0.35f, 0.75f) * 100).toInt()}%", fontSize = 11.sp, color = Color.LightGray, modifier = Modifier.width(36.dp))
                }
                Text(
                    text = "Adjusts detection threshold (35%–75%). Lower values capture fine hair & clothing contours; higher isolates core subjects tightly. Background sky noise is filtered out for pristine clock visibility.",
                    fontSize = 10.sp,
                    color = Color.Gray
                )
            }

            // Mask Expansion / Contraction (Choke)
            Column {
                Row(verticalAlignment = Alignment.CenterVertically) {
                    Text("Mask Margin", fontSize = 12.sp, color = Color.White, modifier = Modifier.width(90.dp))
                    Slider(
                        value = maskExpansion.toFloat(),
                        onValueChange = { maskExpansion = it.toInt() },
                        valueRange = -10f..10f,
                        steps = 20,
                        modifier = Modifier.weight(1f)
                    )
                    val label = when {
                        maskExpansion > 0 -> "+${maskExpansion}px"
                        maskExpansion < 0 -> "${maskExpansion}px"
                        else -> "0px"
                    }
                    Text(label, fontSize = 11.sp, color = Color.LightGray, modifier = Modifier.width(42.dp))
                }
                Text(
                    text = "Expands (+) or contracts/chokes (-) subject cutout borders to eliminate background halos or capture extra hair.",
                    fontSize = 10.sp,
                    color = Color.Gray
                )
            }

            // Layer Flatness / Contrast (eliminates translucency and ghost fuzziness)
            Column {
                Row(verticalAlignment = Alignment.CenterVertically) {
                    Text("Layer Flatness", fontSize = 12.sp, color = Color.White, modifier = Modifier.width(90.dp))
                    Slider(
                        value = cutoutContrast,
                        onValueChange = { cutoutContrast = it },
                        valueRange = 0.50f..1.0f,
                        modifier = Modifier.weight(1f)
                    )
                    Text("${(cutoutContrast * 100).toInt()}%", fontSize = 11.sp, color = Color.LightGray, modifier = Modifier.width(36.dp))
                }
                Text(
                    text = "Increases layer opacity contrast to 100% solid. Flattens subjects to completely eliminate semi-transparent fuzziness and ghost text bleed.",
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
                        valueRange = 2f..16f,
                        steps = 14,
                        modifier = Modifier.weight(1f)
                    )
                    Text("${inpaintRadius}px", fontSize = 11.sp, color = Color.LightGray, modifier = Modifier.width(36.dp))
                }
                Text(
                    text = "Dilation margin to erase subjects underneath and reconstruct background using multi-scale pyramid synthesis with bilinear upsampling.",
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
                    maskExpansion = maskExpansion,
                    inpaintRadius = inpaintRadius,
                    modelType = selectedModel,
                    cutoutContrast = cutoutContrast,
                    processingMode = processingMode,
                    pipelineChoice = selectedPipeline,
                    enablePreprocessing = enablePreprocessing
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

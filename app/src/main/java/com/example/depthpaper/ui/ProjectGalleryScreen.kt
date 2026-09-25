package com.example.depthpaper.ui

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import androidx.activity.compose.BackHandler
import androidx.compose.foundation.BorderStroke
import androidx.compose.foundation.ExperimentalFoundationApi
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.combinedClickable
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
import androidx.compose.foundation.lazy.grid.GridCells
import androidx.compose.foundation.lazy.grid.GridItemSpan
import androidx.compose.foundation.lazy.grid.LazyVerticalGrid
import androidx.compose.foundation.lazy.grid.items
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Add
import androidx.compose.material.icons.filled.AddPhotoAlternate
import androidx.compose.material.icons.filled.Check
import androidx.compose.material.icons.filled.Close
import androidx.compose.material.icons.filled.ContentCopy
import androidx.compose.material.icons.filled.Delete
import androidx.compose.material.icons.filled.Favorite
import androidx.compose.material.icons.filled.FavoriteBorder
import androidx.compose.material.icons.filled.Info
import androidx.compose.material.icons.filled.MoreVert
import androidx.compose.material.icons.filled.OpenInNew
import androidx.compose.material.icons.filled.SystemUpdate
import androidx.compose.material.icons.filled.Wallpaper
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.Checkbox
import androidx.compose.material3.CheckboxDefaults
import androidx.compose.material3.DropdownMenu
import androidx.compose.material3.DropdownMenuItem
import androidx.compose.material3.FloatingActionButton
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableFloatStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalUriHandler
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.text.style.TextDecoration
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.example.depthpaper.BuildConfig
import com.example.depthpaper.core.AppUpdater
import com.example.depthpaper.core.UpdateCheckResult
import com.example.depthpaper.core.UpdateInfo
import com.example.depthpaper.data.RenderMode
import com.example.depthpaper.data.WallpaperProject
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.File
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import kotlin.math.max
import kotlin.math.min

@Composable
fun ProjectGalleryScreen(
    projects: List<WallpaperProject>,
    onSelectProject: (WallpaperProject) -> Unit,
    onSetActive: (String) -> Unit,
    onToggleFavorite: (String) -> Unit,
    onDuplicate: (String) -> Unit,
    onDelete: (String) -> Unit,
    onDeleteMultiple: (Set<String>) -> Unit = {},
    onNewProjectClick: () -> Unit,
    modifier: Modifier = Modifier
) {
    val context = LocalContext.current
    val coroutineScope = rememberCoroutineScope()
    val uriHandler = LocalUriHandler.current

    // Multi-Select State
    var isSelectMode by remember { mutableStateOf(false) }
    var selectedIds by remember { mutableStateOf(setOf<String>()) }
    var showDeleteConfirmDialog by remember { mutableStateOf(false) }
    var showAttributionsDialog by remember { mutableStateOf(false) }
    var galleryMenuExpanded by remember { mutableStateOf(false) }

    // Intercept Android Back button to exit multi-select mode
    BackHandler(enabled = isSelectMode) {
        selectedIds = emptySet()
        isSelectMode = false
    }

    // Updater State
    val prefs = remember { context.getSharedPreferences("depthpaper_prefs", android.content.Context.MODE_PRIVATE) }
    var updateInfo by remember { mutableStateOf<UpdateInfo?>(null) }
    var isDownloading by remember { mutableStateOf(false) }
    var downloadProgress by remember { mutableFloatStateOf(0f) }
    var updateError by remember { mutableStateOf<String?>(null) }
    var isCheckingUpdate by remember { mutableStateOf(false) }

    val dismissUpdate = {
        updateInfo?.let { info ->
            prefs.edit().putString("dismissed_update_tag", info.versionTag).apply()
        }
        updateInfo = null
    }

    // Automatic update check upon launch
    LaunchedEffect(Unit) {
        val dismissedTag = prefs.getString("dismissed_update_tag", "")
        when (val result = AppUpdater.checkForUpdate()) {
            is UpdateCheckResult.UpdateAvailable -> {
                if (result.updateInfo.versionTag != dismissedTag) {
                    updateInfo = result.updateInfo
                }
            }
            else -> Unit
        }
    }

    // Update Dialog
    updateInfo?.let { info ->
        AlertDialog(
            onDismissRequest = { if (!isDownloading) dismissUpdate() },
            containerColor = Color(0xFF1E1E2E),
            titleContentColor = Color.White,
            textContentColor = Color.LightGray,
            title = {
                Row(verticalAlignment = Alignment.CenterVertically) {
                    Icon(
                        Icons.Default.SystemUpdate,
                        contentDescription = null,
                        tint = Color(0xFF00E5FF),
                        modifier = Modifier.size(24.dp)
                    )
                    Spacer(modifier = Modifier.width(8.dp))
                    Text(
                        text = "Update Available: ${info.versionTag}",
                        fontWeight = FontWeight.Bold,
                        fontSize = 18.sp
                    )
                }
            },
            text = {
                Column(verticalArrangement = Arrangement.spacedBy(10.dp)) {
                    val sizeMb = info.apkSizeBytes / (1024f * 1024f)
                    Text(
                        text = "A new version of DepthPaper is available on GitHub (~${"%.1f".format(sizeMb)} MB).",
                        fontSize = 14.sp
                    )
                    if (info.releaseNotes.isNotBlank()) {
                        Text(
                            text = info.releaseNotes.take(180) + if (info.releaseNotes.length > 180) "..." else "",
                            fontSize = 12.sp,
                            color = Color.Gray
                        )
                    }

                    if (isDownloading) {
                        Spacer(modifier = Modifier.height(4.dp))
                        LinearProgressIndicator(
                            progress = { downloadProgress },
                            modifier = Modifier.fillMaxWidth(),
                            color = Color(0xFF00E5FF),
                            trackColor = Color(0xFF2A2A40)
                        )
                        Text(
                            text = "Downloading ${(downloadProgress * 100).toInt()}%...",
                            fontSize = 12.sp,
                            color = Color(0xFF00E5FF)
                        )
                    }

                    updateError?.let { err ->
                        Text(text = err, color = Color(0xFFFF5252), fontSize = 12.sp)
                    }
                }
            },
            confirmButton = {
                Button(
                    onClick = {
                        if (!isDownloading) {
                            isDownloading = true
                            updateError = null
                            coroutineScope.launch {
                                AppUpdater.downloadAndInstallApk(
                                    context = context,
                                    update = info,
                                    onProgress = { p -> downloadProgress = p },
                                    onError = { err ->
                                        updateError = err
                                        isDownloading = false
                                    }
                                )
                                isDownloading = false
                            }
                        }
                    },
                    enabled = !isDownloading,
                    colors = ButtonDefaults.buttonColors(containerColor = Color(0xFF00E5FF)),
                    shape = RoundedCornerShape(10.dp)
                ) {
                    Text(
                        text = if (isDownloading) "Downloading..." else "Install Update",
                        color = Color.Black,
                        fontWeight = FontWeight.Bold
                    )
                }
            },
            dismissButton = {
                if (!isDownloading) {
                    TextButton(onClick = { dismissUpdate() }) {
                        Text("Later", color = Color.Gray)
                    }
                }
            }
        )
    }

    // Permanent Deletion Confirmation Dialog
    if (showDeleteConfirmDialog) {
        val count = selectedIds.size
        AlertDialog(
            onDismissRequest = { showDeleteConfirmDialog = false },
            containerColor = Color(0xFF1E1E2E),
            titleContentColor = Color.White,
            textContentColor = Color.LightGray,
            title = {
                Row(verticalAlignment = Alignment.CenterVertically) {
                    Icon(
                        Icons.Default.Delete,
                        contentDescription = null,
                        tint = Color(0xFFFF5252),
                        modifier = Modifier.size(24.dp)
                    )
                    Spacer(modifier = Modifier.width(8.dp))
                    Text(
                        text = "Delete $count Wallpaper${if (count > 1) "s" else ""}?",
                        fontWeight = FontWeight.Bold,
                        fontSize = 18.sp
                    )
                }
            },
            text = {
                Text(
                    text = "Are you sure you want to permanently delete $count selected wallpaper${if (count > 1) "s" else ""}? This action cannot be undone.",
                    fontSize = 14.sp
                )
            },
            confirmButton = {
                Button(
                    onClick = {
                        onDeleteMultiple(selectedIds)
                        selectedIds = emptySet()
                        isSelectMode = false
                        showDeleteConfirmDialog = false
                    },
                    colors = ButtonDefaults.buttonColors(containerColor = Color(0xFFFF5252)),
                    shape = RoundedCornerShape(10.dp)
                ) {
                    Text("Delete Permanently", color = Color.White, fontWeight = FontWeight.Bold)
                }
            },
            dismissButton = {
                TextButton(onClick = { showDeleteConfirmDialog = false }) {
                    Text("Cancel", color = Color.LightGray)
                }
            }
        )
    }

    // Legal & Open Source Credits & Attributions Dialog
    if (showAttributionsDialog) {
        AttributionsDialog(onDismiss = { showAttributionsDialog = false })
    }

    Box(
        modifier = modifier
            .fillMaxSize()
            .background(Color(0xFF0F0F1A))
            .statusBarsPadding()
            .navigationBarsPadding()
    ) {
        // 3-Images-Wide Lock Screen Preview Gallery Grid scrolling downward
        LazyVerticalGrid(
            columns = GridCells.Fixed(3),
            modifier = Modifier.fillMaxSize(),
            contentPadding = PaddingValues(start = 10.dp, end = 10.dp, top = 12.dp, bottom = 88.dp),
            horizontalArrangement = Arrangement.spacedBy(8.dp),
            verticalArrangement = Arrangement.spacedBy(10.dp)
        ) {
            // Header spans all 3 columns
            item(span = { GridItemSpan(3) }) {
                Row(
                    modifier = Modifier
                        .fillMaxWidth()
                        .padding(bottom = 6.dp),
                    horizontalArrangement = Arrangement.SpaceBetween,
                    verticalAlignment = Alignment.CenterVertically
                ) {
                    if (isSelectMode) {
                        // Multi-select header mode
                        Row(verticalAlignment = Alignment.CenterVertically) {
                            IconButton(onClick = {
                                selectedIds = emptySet()
                                isSelectMode = false
                            }) {
                                Icon(
                                    Icons.Default.Close,
                                    contentDescription = "Cancel Selection",
                                    tint = Color.White
                                )
                            }
                            Spacer(modifier = Modifier.width(4.dp))
                            Text(
                                text = "${selectedIds.size} Selected",
                                fontSize = 18.sp,
                                fontWeight = FontWeight.Bold,
                                color = Color.White
                            )
                        }

                        // Trash Can action
                        if (selectedIds.isNotEmpty()) {
                            IconButton(onClick = { showDeleteConfirmDialog = true }) {
                                Icon(
                                    Icons.Default.Delete,
                                    contentDescription = "Delete Selected",
                                    tint = Color(0xFFFF5252),
                                    modifier = Modifier.size(26.dp)
                                )
                            }
                        }
                    } else {
                        // Normal header mode
                        Column {
                            Row(verticalAlignment = Alignment.CenterVertically) {
                                Text(
                                    text = "Depth Studio",
                                    fontSize = 24.sp,
                                    fontWeight = FontWeight.Bold,
                                    color = Color.White
                                )
                                Spacer(modifier = Modifier.width(8.dp))
                                Surface(
                                    color = Color(0xFF222238),
                                    shape = RoundedCornerShape(8.dp),
                                    modifier = Modifier.clickable {
                                        if (!isCheckingUpdate) {
                                            isCheckingUpdate = true
                                            coroutineScope.launch {
                                                val result = AppUpdater.checkForUpdate()
                                                isCheckingUpdate = false
                                                when (result) {
                                                    is UpdateCheckResult.UpdateAvailable -> {
                                                        updateInfo = result.updateInfo
                                                    }
                                                    is UpdateCheckResult.UpToDate -> {
                                                        android.widget.Toast.makeText(
                                                            context,
                                                            "DepthPaper is up to date (v${BuildConfig.VERSION_NAME})",
                                                            android.widget.Toast.LENGTH_SHORT
                                                        ).show()
                                                    }
                                                    is UpdateCheckResult.Error -> {
                                                        android.widget.Toast.makeText(
                                                            context,
                                                            "Check failed: ${result.message}",
                                                            android.widget.Toast.LENGTH_SHORT
                                                        ).show()
                                                    }
                                                }
                                            }
                                        }
                                    }
                                ) {
                                    Text(
                                        text = if (isCheckingUpdate) "Checking..." else "v${BuildConfig.VERSION_NAME}",
                                        fontSize = 11.sp,
                                        color = Color(0xFF00E5FF),
                                        fontWeight = FontWeight.Bold,
                                        modifier = Modifier.padding(horizontal = 8.dp, vertical = 3.dp)
                                    )
                                }
                            }
                            Text(
                                text = "${projects.size} Wallpapers • 3D Parallax",
                                fontSize = 12.sp,
                                color = Color.Gray
                            )
                        }

                        // 3-Dots Overflow Menu
                        Box {
                            IconButton(onClick = { galleryMenuExpanded = true }) {
                                Icon(
                                    Icons.Default.MoreVert,
                                    contentDescription = "Options",
                                    tint = Color.White
                                )
                            }

                            DropdownMenu(
                                expanded = galleryMenuExpanded,
                                onDismissRequest = { galleryMenuExpanded = false },
                                containerColor = Color(0xFF1E1E30)
                            ) {
                                DropdownMenuItem(
                                    text = { Text("Open Source & Legal Credits", color = Color.White, fontSize = 13.sp) },
                                    onClick = {
                                        galleryMenuExpanded = false
                                        showAttributionsDialog = true
                                    },
                                    leadingIcon = {
                                        Icon(
                                            Icons.Default.Info,
                                            contentDescription = null,
                                            tint = Color(0xFF00E5FF),
                                            modifier = Modifier.size(18.dp)
                                        )
                                    }
                                )
                            }
                        }
                    }
                }
            }

            if (projects.isEmpty()) {
                item(span = { GridItemSpan(3) }) {
                    Box(
                        modifier = Modifier
                            .fillMaxWidth()
                            .padding(top = 100.dp),
                        contentAlignment = Alignment.Center
                    ) {
                        Column(
                            horizontalAlignment = Alignment.CenterHorizontally,
                            verticalArrangement = Arrangement.spacedBy(12.dp)
                        ) {
                            Icon(
                                Icons.Default.AddPhotoAlternate,
                                contentDescription = null,
                                tint = Color(0xFF00E5FF),
                                modifier = Modifier.size(64.dp)
                            )
                            Text(
                                text = "No Wallpapers Yet",
                                fontSize = 18.sp,
                                fontWeight = FontWeight.Bold,
                                color = Color.White
                            )
                            Text(
                                text = "Tap the + button below to import your photos\nand generate 3D layered depth wallpapers.",
                                fontSize = 13.sp,
                                color = Color.Gray,
                                textAlign = TextAlign.Center
                            )
                        }
                    }
                }
            } else {
                items(projects, key = { it.id }) { proj ->
                    val isSelected = proj.id in selectedIds
                    LockScreenPreviewCard(
                        project = proj,
                        isSelectMode = isSelectMode,
                        isSelected = isSelected,
                        onClick = {
                            if (isSelectMode) {
                                val next = if (isSelected) selectedIds - proj.id else selectedIds + proj.id
                                selectedIds = next
                                if (next.isEmpty()) isSelectMode = false
                            } else {
                                onSelectProject(proj)
                            }
                        },
                        onLongClick = {
                            if (!isSelectMode) {
                                isSelectMode = true
                                selectedIds = setOf(proj.id)
                            }
                        },
                        onToggleSelect = {
                            val next = if (isSelected) selectedIds - proj.id else selectedIds + proj.id
                            selectedIds = next
                            if (next.isEmpty()) isSelectMode = false
                        },
                        onSetActive = { onSetActive(proj.id) },
                        onToggleFavorite = { onToggleFavorite(proj.id) },
                        onDuplicate = { onDuplicate(proj.id) },
                        onDelete = { onDelete(proj.id) }
                    )
                }
            }
        }

        // Material 3 floating add (plus) button bottom right
        if (!isSelectMode) {
            FloatingActionButton(
                onClick = onNewProjectClick,
                modifier = Modifier
                    .align(Alignment.BottomEnd)
                    .padding(24.dp),
                containerColor = Color(0xFF00E5FF),
                contentColor = Color.Black,
                shape = RoundedCornerShape(18.dp)
            ) {
                Icon(
                    imageVector = Icons.Default.Add,
                    contentDescription = "Create New Wallpaper",
                    modifier = Modifier.size(30.dp)
                )
            }
        }
    }
}

/**
 * Vertical 9:16 lock screen preview card showing photo thumbnail,
 * mini live clock overlay, active badge, and action menu.
 * Supports multi-select long-press and selection checkboxes.
 */
@OptIn(ExperimentalFoundationApi::class)
@Composable
fun LockScreenPreviewCard(
    project: WallpaperProject,
    isSelectMode: Boolean,
    isSelected: Boolean,
    onClick: () -> Unit,
    onLongClick: () -> Unit,
    onToggleSelect: () -> Unit,
    onSetActive: () -> Unit,
    onToggleFavorite: () -> Unit,
    onDuplicate: () -> Unit,
    onDelete: () -> Unit
) {
    var menuExpanded by remember { mutableStateOf(false) }

    Card(
        modifier = Modifier
            .fillMaxWidth()
            .aspectRatio(9f / 16f)
            .clip(RoundedCornerShape(12.dp))
            .border(
                width = if (isSelected) 2.5.dp else if (project.isActive) 2.dp else 1.dp,
                color = if (isSelected) Color(0xFF00E5FF) else if (project.isActive) Color(0xFF00E5FF).copy(alpha = 0.7f) else Color(0xFF252538),
                shape = RoundedCornerShape(12.dp)
            )
            .combinedClickable(
                onClick = onClick,
                onLongClick = onLongClick
            ),
        shape = RoundedCornerShape(12.dp),
        colors = CardDefaults.cardColors(containerColor = Color(0xFF141424))
    ) {
        Box(modifier = Modifier.fillMaxSize()) {
            // 1. Photo Thumbnail
            ProjectThumbnail(
                path = project.thumbnailPath.ifBlank { project.sourceImagePath },
                modifier = Modifier.fillMaxSize()
            )

            // 2. Lock Screen Mini Clock Overlay
            val cfg = project.lockScreenConfig
            val clockColor = Color(cfg.clockColorHex)
            Column(
                modifier = Modifier
                    .align(Alignment.TopCenter)
                    .padding(top = 14.dp),
                horizontalAlignment = Alignment.CenterHorizontally
            ) {
                if (cfg.showDate) {
                    Text(
                        text = SimpleDateFormat("MMM d", Locale.getDefault()).format(Date()),
                        fontSize = 7.sp,
                        fontWeight = FontWeight.Medium,
                        color = clockColor.copy(alpha = 0.85f),
                        maxLines = 1
                    )
                }
                Text(
                    text = SimpleDateFormat("HH:mm", Locale.getDefault()).format(Date()),
                    fontSize = 16.sp,
                    fontWeight = FontWeight.Bold,
                    color = clockColor,
                    maxLines = 1
                )
            }

            // 3. Top Badges: Active & Favorite / Checkbox
            if (project.isActive && !isSelectMode) {
                Surface(
                    color = Color(0xFF00E5FF),
                    shape = RoundedCornerShape(4.dp),
                    modifier = Modifier
                        .align(Alignment.TopStart)
                        .padding(5.dp)
                ) {
                    Text(
                        text = "ACTIVE",
                        fontSize = 7.sp,
                        fontWeight = FontWeight.Black,
                        color = Color.Black,
                        modifier = Modifier.padding(horizontal = 4.dp, vertical = 1.dp)
                    )
                }
            }

            if (isSelectMode) {
                Surface(
                    color = Color.Black.copy(alpha = 0.5f),
                    shape = RoundedCornerShape(6.dp),
                    modifier = Modifier
                        .align(Alignment.TopEnd)
                        .padding(4.dp)
                ) {
                    Checkbox(
                        checked = isSelected,
                        onCheckedChange = { onToggleSelect() },
                        modifier = Modifier.size(24.dp),
                        colors = CheckboxDefaults.colors(
                            checkedColor = Color(0xFF00E5FF),
                            uncheckedColor = Color.White.copy(alpha = 0.8f),
                            checkmarkColor = Color.Black
                        )
                    )
                }
            } else {
                IconButton(
                    onClick = onToggleFavorite,
                    modifier = Modifier
                        .align(Alignment.TopEnd)
                        .size(26.dp)
                        .padding(2.dp)
                ) {
                    Icon(
                        imageVector = if (project.isFavorite) Icons.Default.Favorite else Icons.Default.FavoriteBorder,
                        contentDescription = "Favorite",
                        tint = if (project.isFavorite) Color(0xFFFF4081) else Color.White.copy(alpha = 0.7f),
                        modifier = Modifier.size(15.dp)
                    )
                }
            }

            // Selected subtle tint overlay
            if (isSelected) {
                Box(
                    modifier = Modifier
                        .fillMaxSize()
                        .background(Color(0xFF00E5FF).copy(alpha = 0.20f))
                )
            }

            // 4. Bottom Scrim with Title and 3-dot Menu (Only when not in select mode)
            if (!isSelectMode) {
                Box(
                    modifier = Modifier
                        .align(Alignment.BottomCenter)
                        .fillMaxWidth()
                        .background(
                            Brush.verticalGradient(
                                colors = listOf(Color.Transparent, Color(0xCC000000), Color(0xEE000000))
                            )
                        )
                        .padding(horizontal = 6.dp, vertical = 4.dp)
                ) {
                    Row(
                        modifier = Modifier.fillMaxWidth(),
                        verticalAlignment = Alignment.CenterVertically,
                        horizontalArrangement = Arrangement.SpaceBetween
                    ) {
                        Column(modifier = Modifier.weight(1f)) {
                            Text(
                                text = project.title,
                                fontSize = 10.sp,
                                fontWeight = FontWeight.SemiBold,
                                color = Color.White,
                                maxLines = 1,
                                overflow = TextOverflow.Ellipsis
                            )
                            Text(
                                text = if (project.renderMode == RenderMode.LAYERED_2D) "2.5D Layer" else "3D Depth",
                                fontSize = 7.sp,
                                color = Color(0xFF00E5FF)
                            )
                        }

                        Box {
                            IconButton(
                                onClick = { menuExpanded = true },
                                modifier = Modifier.size(20.dp)
                            ) {
                                Icon(
                                    Icons.Default.MoreVert,
                                    contentDescription = "Options",
                                    tint = Color.LightGray,
                                    modifier = Modifier.size(14.dp)
                                )
                            }

                            DropdownMenu(
                                expanded = menuExpanded,
                                onDismissRequest = { menuExpanded = false },
                                containerColor = Color(0xFF1E1E30)
                            ) {
                                if (!project.isActive) {
                                    DropdownMenuItem(
                                        text = { Text("Set as Active", fontSize = 12.sp, color = Color(0xFF00E5FF)) },
                                        onClick = {
                                            menuExpanded = false
                                            onSetActive()
                                        },
                                        leadingIcon = {
                                            Icon(Icons.Default.Check, contentDescription = null, tint = Color(0xFF00E5FF), modifier = Modifier.size(16.dp))
                                        }
                                    )
                                }
                                DropdownMenuItem(
                                    text = { Text("Duplicate", fontSize = 12.sp, color = Color.White) },
                                    onClick = {
                                        menuExpanded = false
                                        onDuplicate()
                                    },
                                    leadingIcon = {
                                        Icon(Icons.Default.ContentCopy, contentDescription = null, tint = Color.White, modifier = Modifier.size(16.dp))
                                    }
                                )
                                DropdownMenuItem(
                                    text = { Text("Delete", fontSize = 12.sp, color = Color(0xFFFF5252)) },
                                    onClick = {
                                        menuExpanded = false
                                        onDelete()
                                    },
                                    leadingIcon = {
                                        Icon(Icons.Default.Delete, contentDescription = null, tint = Color(0xFFFF5252), modifier = Modifier.size(16.dp))
                                    }
                                )
                            }
                        }
                    }
                }
            }
        }
    }
}

/**
 * Asynchronous, memory-efficient downsampled thumbnail loader.
 */
@Composable
fun ProjectThumbnail(path: String, modifier: Modifier = Modifier) {
    var bitmap by remember(path) { mutableStateOf<Bitmap?>(null) }

    LaunchedEffect(path) {
        if (path.isNotBlank()) {
            withContext(Dispatchers.IO) {
                val file = File(path)
                if (file.exists()) {
                    val boundsOpts = BitmapFactory.Options().apply { inJustDecodeBounds = true }
                    BitmapFactory.decodeFile(file.absolutePath, boundsOpts)
                    val sample = max(1, min(boundsOpts.outWidth / 360, boundsOpts.outHeight / 640))
                    val opts = BitmapFactory.Options().apply {
                        inSampleSize = sample
                        inPreferredConfig = Bitmap.Config.RGB_565
                    }
                    val loaded = BitmapFactory.decodeFile(file.absolutePath, opts)
                    withContext(Dispatchers.Main) {
                        bitmap = loaded
                    }
                }
            }
        }
    }

    if (bitmap != null) {
        Image(
            bitmap = bitmap!!.asImageBitmap(),
            contentDescription = "Thumbnail",
            contentScale = ContentScale.Crop,
            modifier = modifier
        )
    } else {
        Box(
            modifier = modifier.background(Color(0xFF222238)),
            contentAlignment = Alignment.Center
        ) {
            Icon(Icons.Default.Wallpaper, contentDescription = null, tint = Color.Gray, modifier = Modifier.size(24.dp))
        }
    }
}

/**
 * Comprehensive Legal & Open Source Credits & Attributions Dialog.
 * Complies with Apache 2.0 / MIT licensing notices and attribution recommendations.
 */
@Composable
private fun AttributionsDialog(onDismiss: () -> Unit) {
    val uriHandler = LocalUriHandler.current

    AlertDialog(
        onDismissRequest = onDismiss,
        containerColor = Color(0xFF1E1E2E),
        titleContentColor = Color.White,
        textContentColor = Color.LightGray,
        title = {
            Row(verticalAlignment = Alignment.CenterVertically) {
                Icon(
                    Icons.Default.Info,
                    contentDescription = null,
                    tint = Color(0xFF00E5FF),
                    modifier = Modifier.size(24.dp)
                )
                Spacer(modifier = Modifier.width(8.dp))
                Text("Open Source & Legal Credits", fontWeight = FontWeight.Bold, fontSize = 18.sp)
            }
        },
        text = {
            Column(
                modifier = Modifier
                    .fillMaxWidth()
                    .verticalScroll(rememberScrollState()),
                verticalArrangement = Arrangement.spacedBy(14.dp)
            ) {
                Text(
                    text = "DepthPaper operates 100% on-device and is powered by state-of-the-art open-source research and technologies. We gratefully acknowledge the creators and contributors:",
                    fontSize = 13.sp,
                    color = Color.LightGray
                )

                // 1. Depth Anything V2
                AttributionSection(
                    title = "Depth Anything V2 (ViT-Small)",
                    authors = "Lihe Yang, Bingyi Kang, Zilong Huang, Xiaogang Xu, Jiashi Feng, Hengshuang Zhao (The University of Hong Kong & ByteDance Seed)",
                    license = "Apache License 2.0",
                    url = "https://github.com/DepthAnything/Depth-Anything-V2",
                    uriHandler = uriHandler
                )

                // 2. MediaPipe Selfie Multiclass & Fast Selfie
                AttributionSection(
                    title = "MediaPipe Selfie Segmenters",
                    authors = "Google LLC / MediaPipe Solutions Team",
                    license = "Apache License 2.0",
                    url = "https://developers.google.com/mediapipe/solutions/vision/image_segmenter",
                    uriHandler = uriHandler
                )

                // 3. DeepLab V3 MobileNet
                AttributionSection(
                    title = "DeepLab v3 MobileNet-v2",
                    authors = "Liang-Chieh Chen, Yukun Zhu, George Papandreou, Florian Schroff, Hartwig Adam (Google Research)",
                    license = "Apache License 2.0",
                    url = "https://github.com/tensorflow/models/tree/master/research/deeplab",
                    uriHandler = uriHandler
                )

                // 4. Guided Filtering & Inpainting
                AttributionSection(
                    title = "Guided Image Filter & Multi-Scale Push-Pull",
                    authors = "Kaiming He et al. (Guided Filtering) & Steven J. Gortler et al. (Push-Pull Lumigraph)",
                    license = "Apache License 2.0 Implementation",
                    url = "https://kaiminghe.github.io/publications/eccv10guidedfilter.pdf",
                    uriHandler = uriHandler
                )

                // 5. TensorFlow Lite & Delegates
                AttributionSection(
                    title = "TensorFlow Lite & GPU/NNAPI Acceleration",
                    authors = "The TensorFlow Authors / Google LLC",
                    license = "Apache License 2.0",
                    url = "https://www.tensorflow.org/lite",
                    uriHandler = uriHandler
                )

                // 6. Android Jetpack & Kotlin
                AttributionSection(
                    title = "Android Jetpack Compose & Kotlin",
                    authors = "Google LLC & JetBrains s.r.o.",
                    license = "Apache License 2.0",
                    url = "https://developer.android.com/jetpack/compose",
                    uriHandler = uriHandler
                )
            }
        },
        confirmButton = {
            Button(
                onClick = onDismiss,
                colors = ButtonDefaults.buttonColors(containerColor = Color(0xFF00E5FF)),
                shape = RoundedCornerShape(10.dp)
            ) {
                Text("Close", color = Color.Black, fontWeight = FontWeight.Bold)
            }
        }
    )
}

@Composable
private fun AttributionSection(
    title: String,
    authors: String,
    license: String,
    url: String,
    uriHandler: androidx.compose.ui.platform.UriHandler
) {
    Surface(
        color = Color(0xFF26263A),
        shape = RoundedCornerShape(8.dp),
        modifier = Modifier.fillMaxWidth()
    ) {
        Column(modifier = Modifier.padding(10.dp)) {
            Text(title, fontWeight = FontWeight.Bold, color = Color.White, fontSize = 13.sp)
            Spacer(modifier = Modifier.height(2.dp))
            Text(authors, fontSize = 11.sp, color = Color.LightGray)
            Spacer(modifier = Modifier.height(2.dp))
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically
            ) {
                Text(license, fontSize = 10.sp, color = Color(0xFF00E5FF), fontWeight = FontWeight.SemiBold)
                Row(
                    modifier = Modifier.clickable { uriHandler.openUri(url) },
                    verticalAlignment = Alignment.CenterVertically
                ) {
                    Text(
                        text = "Source",
                        fontSize = 10.sp,
                        color = Color(0xFF00E5FF),
                        textDecoration = TextDecoration.Underline
                    )
                    Spacer(modifier = Modifier.width(2.dp))
                    Icon(
                        Icons.Default.OpenInNew,
                        contentDescription = null,
                        tint = Color(0xFF00E5FF),
                        modifier = Modifier.size(10.dp)
                    )
                }
            }
        }
    }
}

package com.example.depthpaper.ui

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import androidx.compose.foundation.BorderStroke
import androidx.compose.foundation.Image
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
import androidx.compose.foundation.lazy.grid.GridCells
import androidx.compose.foundation.lazy.grid.GridItemSpan
import androidx.compose.foundation.lazy.grid.LazyVerticalGrid
import androidx.compose.foundation.lazy.grid.items
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Add
import androidx.compose.material.icons.filled.AddPhotoAlternate
import androidx.compose.material.icons.filled.BugReport
import androidx.compose.material.icons.filled.Check
import androidx.compose.material.icons.filled.ContentCopy
import androidx.compose.material.icons.filled.Delete
import androidx.compose.material.icons.filled.Favorite
import androidx.compose.material.icons.filled.FavoriteBorder
import androidx.compose.material.icons.filled.MoreVert
import androidx.compose.material.icons.filled.SystemUpdate
import androidx.compose.material.icons.filled.Wallpaper
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
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
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.example.depthpaper.BuildConfig
import com.example.depthpaper.core.AppLogger
import com.example.depthpaper.data.RenderMode
import com.example.depthpaper.data.WallpaperProject
import com.example.depthpaper.core.AppUpdater
import com.example.depthpaper.core.UpdateInfo
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
    onNewProjectClick: () -> Unit,
    modifier: Modifier = Modifier
) {
    val context = LocalContext.current
    val coroutineScope = rememberCoroutineScope()

    var updateInfo by remember { mutableStateOf<UpdateInfo?>(null) }
    var isDownloading by remember { mutableStateOf(false) }
    var downloadProgress by remember { mutableFloatStateOf(0f) }
    var updateError by remember { mutableStateOf<String?>(null) }
    var isCheckingUpdate by remember { mutableStateOf(false) }

    // Automatic update check upon launch
    LaunchedEffect(Unit) {
        val latest = AppUpdater.checkForUpdate()
        if (latest != null) {
            updateInfo = latest
        }
    }

    // Update Dialog
    updateInfo?.let { info ->
        AlertDialog(
            onDismissRequest = { if (!isDownloading) updateInfo = null },
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
                        isDownloading = true
                        updateError = null
                        coroutineScope.launch {
                            AppUpdater.downloadAndInstallApk(
                                context = context,
                                update = info,
                                onProgress = { downloadProgress = it },
                                onError = {
                                    updateError = it
                                    isDownloading = false
                                }
                            )
                        }
                    },
                    enabled = !isDownloading,
                    colors = ButtonDefaults.buttonColors(containerColor = Color(0xFF00E5FF)),
                    shape = RoundedCornerShape(10.dp)
                ) {
                    Text(
                        text = if (isDownloading) "Downloading..." else "Download & Install",
                        color = Color.Black,
                        fontWeight = FontWeight.Bold
                    )
                }
            },
            dismissButton = {
                if (!isDownloading) {
                    TextButton(onClick = { updateInfo = null }) {
                        Text("Later", color = Color.Gray)
                    }
                }
            }
        )
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
                                            val latest = AppUpdater.checkForUpdate()
                                            isCheckingUpdate = false
                                            if (latest != null) {
                                                updateInfo = latest
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

                    // Diagnostic Logs button
                    OutlinedButton(
                        onClick = { AppLogger.copyLogsToClipboard(context) },
                        colors = ButtonDefaults.outlinedButtonColors(contentColor = Color(0xFF00E5FF)),
                        border = BorderStroke(1.dp, Color(0xFF00E5FF).copy(alpha = 0.5f)),
                        shape = RoundedCornerShape(10.dp),
                        contentPadding = PaddingValues(horizontal = 10.dp, vertical = 4.dp)
                    ) {
                        Icon(
                            Icons.Default.BugReport,
                            contentDescription = "Copy Diagnostic Logs",
                            modifier = Modifier.size(16.dp)
                        )
                        Spacer(modifier = Modifier.width(4.dp))
                        Text(text = "Logs", fontSize = 11.sp, fontWeight = FontWeight.Bold)
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
                    LockScreenPreviewCard(
                        project = proj,
                        onClick = { onSelectProject(proj) },
                        onSetActive = { onSetActive(proj.id) },
                        onToggleFavorite = { onToggleFavorite(proj.id) },
                        onDuplicate = { onDuplicate(proj.id) },
                        onDelete = { onDelete(proj.id) }
                    )
                }
            }
        }

        // Material 3 floating add (plus) button bottom right
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

/**
 * Vertical 9:16 lock screen preview card showing photo thumbnail,
 * mini live clock overlay, active badge, and action menu.
 */
@Composable
fun LockScreenPreviewCard(
    project: WallpaperProject,
    onClick: () -> Unit,
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
                width = if (project.isActive) 2.dp else 1.dp,
                color = if (project.isActive) Color(0xFF00E5FF) else Color(0xFF252538),
                shape = RoundedCornerShape(12.dp)
            )
            .clickable { onClick() },
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

            // 3. Top Badges: Active & Favorite
            if (project.isActive) {
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

            // 4. Bottom Scrim with Title and 3-dot Menu
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

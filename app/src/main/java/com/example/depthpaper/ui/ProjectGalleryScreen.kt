package com.example.depthpaper.ui

import android.graphics.BitmapFactory
import androidx.compose.foundation.BorderStroke
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
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
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.LazyRow
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Add
import androidx.compose.material.icons.filled.CheckCircle
import androidx.compose.material.icons.filled.ContentCopy
import androidx.compose.material.icons.filled.Delete
import androidx.compose.material.icons.filled.Edit
import androidx.compose.material.icons.filled.Favorite
import androidx.compose.material.icons.filled.FavoriteBorder
import androidx.compose.material.icons.filled.SystemUpdate
import androidx.compose.material.icons.filled.Wallpaper
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.FloatingActionButton
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
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
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.example.depthpaper.BuildConfig
import com.example.depthpaper.core.AppUpdater
import com.example.depthpaper.core.UpdateInfo
import com.example.depthpaper.data.WallpaperProject
import kotlinx.coroutines.launch
import java.io.File
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

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
    val activeProject = projects.firstOrNull { it.isActive }
    val favoriteProjects = projects.filter { it.isFavorite && !it.isActive }
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
        LazyColumn(
            modifier = Modifier.fillMaxSize(),
            contentPadding = PaddingValues(16.dp),
            verticalArrangement = Arrangement.spacedBy(20.dp)
        ) {
            // Header
            item {
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.SpaceBetween,
                    verticalAlignment = Alignment.CenterVertically
                ) {
                    Column {
                        Row(verticalAlignment = Alignment.CenterVertically) {
                            Text(
                                text = "Depth Studio",
                                fontSize = 28.sp,
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
                                    modifier = Modifier.padding(horizontal = 8.dp, vertical = 4.dp)
                                )
                            }
                        }
                        Text(
                            text = "Layered & 3D Parallax Wallpapers",
                            fontSize = 14.sp,
                            color = Color.Gray
                        )
                    }
                }
            }

            // 1. ACTIVE WALLPAPER (Pinned at Top)
            activeProject?.let { active ->
                item {
                    Text(
                        text = "ACTIVE WALLPAPER",
                        fontSize = 12.sp,
                        fontWeight = FontWeight.Bold,
                        color = Color(0xFF00E5FF),
                        letterSpacing = 1.sp
                    )
                    Spacer(modifier = Modifier.height(8.dp))
                    ActiveProjectCard(
                        project = active,
                        onEditClick = { onSelectProject(active) },
                        onFavoriteClick = { onToggleFavorite(active.id) }
                    )
                }
            }

            // 2. FAVORITES ROW (Pinned below Active)
            if (favoriteProjects.isNotEmpty()) {
                item {
                    Text(
                        text = "FAVORITES",
                        fontSize = 12.sp,
                        fontWeight = FontWeight.Bold,
                        color = Color(0xFFFF4081),
                        letterSpacing = 1.sp
                    )
                    Spacer(modifier = Modifier.height(8.dp))
                    LazyRow(
                        horizontalArrangement = Arrangement.spacedBy(14.dp)
                    ) {
                        items(favoriteProjects, key = { it.id }) { fav ->
                            ProjectScreenshotCard(
                                project = fav,
                                isCompact = true,
                                onCardClick = { onSelectProject(fav) },
                                onSetActiveClick = { onSetActive(fav.id) },
                                onFavoriteClick = { onToggleFavorite(fav.id) }
                            )
                        }
                    }
                }
            }

            // 3. ALL PROJECTS
            item {
                Text(
                    text = "ALL CREATIONS (${projects.size})",
                    fontSize = 12.sp,
                    fontWeight = FontWeight.Bold,
                    color = Color.LightGray,
                    letterSpacing = 1.sp
                )
            }

            items(projects, key = { it.id }) { proj ->
                ProjectListItemCard(
                    project = proj,
                    onEditClick = { onSelectProject(proj) },
                    onSetActiveClick = { onSetActive(proj.id) },
                    onFavoriteClick = { onToggleFavorite(proj.id) },
                    onDuplicateClick = { onDuplicate(proj.id) },
                    onDeleteClick = { onDelete(proj.id) }
                )
            }

            item {
                Spacer(modifier = Modifier.height(60.dp))
            }
        }

        // Floating button to import new photo
        FloatingActionButton(
            onClick = onNewProjectClick,
            modifier = Modifier
                .align(Alignment.BottomEnd)
                .padding(24.dp),
            containerColor = Color(0xFF00E5FF),
            contentColor = Color.Black
        ) {
            Row(
                modifier = Modifier.padding(horizontal = 16.dp),
                verticalAlignment = Alignment.CenterVertically
            ) {
                Icon(Icons.Default.Add, contentDescription = "New Wallpaper")
                Spacer(modifier = Modifier.width(6.dp))
                Text("New Wallpaper", fontWeight = FontWeight.SemiBold)
            }
        }
    }
}

@Composable
fun ActiveProjectCard(
    project: WallpaperProject,
    onEditClick: () -> Unit,
    onFavoriteClick: () -> Unit
) {
    Card(
        modifier = Modifier
            .fillMaxWidth()
            .clickable { onEditClick() },
        shape = RoundedCornerShape(20.dp),
        colors = CardDefaults.cardColors(containerColor = Color(0xFF1E1E30)),
        border = BorderStroke(2.dp, Color(0xFF00E5FF))
    ) {
        Row(
            modifier = Modifier.padding(16.dp),
            verticalAlignment = Alignment.CenterVertically
        ) {
            ProjectThumbnail(
                path = project.thumbnailPath.ifBlank { project.sourceImagePath },
                modifier = Modifier
                    .size(width = 75.dp, height = 120.dp)
                    .clip(RoundedCornerShape(14.dp))
            )
            Spacer(modifier = Modifier.width(16.dp))
            Column(modifier = Modifier.weight(1f)) {
                Row(verticalAlignment = Alignment.CenterVertically) {
                    Icon(
                        Icons.Default.CheckCircle,
                        contentDescription = "Active",
                        tint = Color(0xFF00E5FF),
                        modifier = Modifier.size(16.dp)
                    )
                    Spacer(modifier = Modifier.width(6.dp))
                    Text(
                        text = "CURRENT SYSTEM WALLPAPER",
                        fontSize = 11.sp,
                        fontWeight = FontWeight.Bold,
                        color = Color(0xFF00E5FF)
                    )
                }
                Spacer(modifier = Modifier.height(4.dp))
                Text(
                    text = project.title,
                    fontSize = 18.sp,
                    fontWeight = FontWeight.Bold,
                    color = Color.White
                )
                Text(
                    text = "Mode: ${project.renderMode.name.replace('_', ' ')}",
                    fontSize = 12.sp,
                    color = Color.Gray
                )
                Spacer(modifier = Modifier.height(10.dp))
                Row {
                    Button(
                        onClick = onEditClick,
                        colors = ButtonDefaults.buttonColors(containerColor = Color(0xFF00E5FF)),
                        shape = RoundedCornerShape(10.dp),
                        contentPadding = PaddingValues(horizontal = 14.dp, vertical = 6.dp)
                    ) {
                        Icon(Icons.Default.Edit, contentDescription = null, tint = Color.Black, modifier = Modifier.size(14.dp))
                        Spacer(modifier = Modifier.width(6.dp))
                        Text("Customize", color = Color.Black, fontSize = 12.sp, fontWeight = FontWeight.Bold)
                    }
                }
            }
            IconButton(onClick = onFavoriteClick) {
                Icon(
                    if (project.isFavorite) Icons.Default.Favorite else Icons.Default.FavoriteBorder,
                    contentDescription = "Favorite",
                    tint = if (project.isFavorite) Color(0xFFFF4081) else Color.Gray
                )
            }
        }
    }
}

@Composable
fun ProjectScreenshotCard(
    project: WallpaperProject,
    isCompact: Boolean = false,
    onCardClick: () -> Unit,
    onSetActiveClick: () -> Unit,
    onFavoriteClick: () -> Unit
) {
    Card(
        modifier = Modifier
            .width(if (isCompact) 140.dp else 170.dp)
            .clickable { onCardClick() },
        shape = RoundedCornerShape(16.dp),
        colors = CardDefaults.cardColors(containerColor = Color(0xFF1E1E30))
    ) {
        Column(modifier = Modifier.padding(10.dp)) {
            Box(
                modifier = Modifier
                    .fillMaxWidth()
                    .aspectRatio(9f / 16f)
                    .clip(RoundedCornerShape(12.dp))
            ) {
                ProjectThumbnail(
                    path = project.thumbnailPath.ifBlank { project.sourceImagePath },
                    modifier = Modifier.fillMaxSize()
                )
                IconButton(
                    onClick = onFavoriteClick,
                    modifier = Modifier.align(Alignment.TopEnd)
                ) {
                    Icon(
                        if (project.isFavorite) Icons.Default.Favorite else Icons.Default.FavoriteBorder,
                        contentDescription = "Favorite",
                        tint = if (project.isFavorite) Color(0xFFFF4081) else Color.White
                    )
                }
            }
            Spacer(modifier = Modifier.height(8.dp))
            Text(
                text = project.title,
                fontSize = 14.sp,
                fontWeight = FontWeight.SemiBold,
                color = Color.White,
                maxLines = 1
            )
            Spacer(modifier = Modifier.height(6.dp))
            Button(
                onClick = onSetActiveClick,
                colors = ButtonDefaults.buttonColors(containerColor = Color(0xFF2A2A44)),
                shape = RoundedCornerShape(8.dp),
                modifier = Modifier.fillMaxWidth(),
                contentPadding = PaddingValues(vertical = 4.dp)
            ) {
                Text(
                    text = if (project.isActive) "Active" else "Set Active",
                    fontSize = 11.sp,
                    color = if (project.isActive) Color(0xFF00E5FF) else Color.White
                )
            }
        }
    }
}

@Composable
fun ProjectListItemCard(
    project: WallpaperProject,
    onEditClick: () -> Unit,
    onSetActiveClick: () -> Unit,
    onFavoriteClick: () -> Unit,
    onDuplicateClick: () -> Unit,
    onDeleteClick: () -> Unit
) {
    Card(
        modifier = Modifier
            .fillMaxWidth()
            .clickable { onEditClick() },
        shape = RoundedCornerShape(16.dp),
        colors = CardDefaults.cardColors(containerColor = Color(0xFF1E1E30)),
        border = if (project.isActive) BorderStroke(1.5.dp, Color(0xFF00E5FF)) else null
    ) {
        Row(
            modifier = Modifier.padding(12.dp),
            verticalAlignment = Alignment.CenterVertically
        ) {
            ProjectThumbnail(
                path = project.thumbnailPath.ifBlank { project.sourceImagePath },
                modifier = Modifier
                    .size(width = 60.dp, height = 96.dp)
                    .clip(RoundedCornerShape(10.dp))
            )
            Spacer(modifier = Modifier.width(14.dp))
            Column(modifier = Modifier.weight(1f)) {
                Text(
                    text = project.title,
                    fontSize = 16.sp,
                    fontWeight = FontWeight.Bold,
                    color = Color.White
                )
                Text(
                    text = "${project.renderMode.name.replace('_', ' ')} • ${SimpleDateFormat("MMM d, yyyy", Locale.getDefault()).format(Date(project.createdTimestamp))}",
                    fontSize = 12.sp,
                    color = Color.Gray
                )
                Spacer(modifier = Modifier.height(8.dp))
                Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                    Button(
                        onClick = onSetActiveClick,
                        colors = ButtonDefaults.buttonColors(containerColor = if (project.isActive) Color(0xFF00E5FF) else Color(0xFF2E2E48)),
                        shape = RoundedCornerShape(8.dp),
                        contentPadding = PaddingValues(horizontal = 10.dp, vertical = 4.dp)
                    ) {
                        Text(
                            text = if (project.isActive) "Active" else "Set Active",
                            fontSize = 11.sp,
                            fontWeight = FontWeight.Bold,
                            color = if (project.isActive) Color.Black else Color.White
                        )
                    }
                    OutlinedButton(
                        onClick = onEditClick,
                        shape = RoundedCornerShape(8.dp),
                        contentPadding = PaddingValues(horizontal = 10.dp, vertical = 4.dp)
                    ) {
                        Text("Edit", fontSize = 11.sp, color = Color.White)
                    }
                }
            }
            Column(horizontalAlignment = Alignment.CenterHorizontally) {
                IconButton(onClick = onFavoriteClick) {
                    Icon(
                        if (project.isFavorite) Icons.Default.Favorite else Icons.Default.FavoriteBorder,
                        contentDescription = "Favorite",
                        tint = if (project.isFavorite) Color(0xFFFF4081) else Color.Gray,
                        modifier = Modifier.size(20.dp)
                    )
                }
                IconButton(onClick = onDuplicateClick) {
                    Icon(
                        Icons.Default.ContentCopy,
                        contentDescription = "Duplicate",
                        tint = Color.Gray,
                        modifier = Modifier.size(18.dp)
                    )
                }
                IconButton(onClick = onDeleteClick) {
                    Icon(
                        Icons.Default.Delete,
                        contentDescription = "Delete",
                        tint = Color(0xFFFF5252),
                        modifier = Modifier.size(18.dp)
                    )
                }
            }
        }
    }
}

@Composable
fun ProjectThumbnail(path: String, modifier: Modifier = Modifier) {
    val bitmap = remember(path) {
        if (path.isNotBlank()) {
            val file = File(path)
            if (file.exists()) BitmapFactory.decodeFile(file.absolutePath) else null
        } else null
    }

    if (bitmap != null) {
        Image(
            bitmap = bitmap.asImageBitmap(),
            contentDescription = "Thumbnail",
            contentScale = ContentScale.Crop,
            modifier = modifier
        )
    } else {
        Box(
            modifier = modifier.background(Color(0xFF2A2A40)),
            contentAlignment = Alignment.Center
        ) {
            Icon(Icons.Default.Wallpaper, contentDescription = null, tint = Color.Gray)
        }
    }
}

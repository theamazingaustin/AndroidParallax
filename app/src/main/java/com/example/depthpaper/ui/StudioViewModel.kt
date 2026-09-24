package com.example.depthpaper.ui

import android.app.WallpaperManager
import android.content.ComponentName
import android.content.Context
import android.content.Intent
import android.graphics.Bitmap
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.example.depthpaper.core.SegmentationEngine
import com.example.depthpaper.core.SegmentationModelType
import com.example.depthpaper.data.ClockFontStyle
import com.example.depthpaper.data.ProjectRepository
import com.example.depthpaper.data.RenderMode
import com.example.depthpaper.data.WallpaperProject
import com.example.depthpaper.service.ParallaxWallpaperService
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

enum class PreviewSurface {
    LOCK_SCREEN,
    HOME_SCREEN,
    AOD
}

enum class StudioTab {
    LOCK_SCREEN,
    HOME_SCREEN,
    AOD,
    FINE_TUNING,
    PROJECTS_GALLERY
}

data class StudioUiState(
    val currentProject: WallpaperProject = WallpaperProject(),
    val allProjects: List<WallpaperProject> = emptyList(),
    val sourceBitmap: Bitmap? = null,
    val cutoutBitmap: Bitmap? = null,
    val backgroundBitmap: Bitmap? = null,
    val depthBitmap: Bitmap? = null,
    val isProcessing: Boolean = false,
    val statusMessage: String? = null,
    val previewSurface: PreviewSurface = PreviewSurface.LOCK_SCREEN,
    val activeTab: StudioTab = StudioTab.LOCK_SCREEN,
    val simulatedTiltX: Float = 0f,
    val simulatedTiltY: Float = 0f
)

class StudioViewModel(
    private val repository: ProjectRepository,
    private val segmentationEngine: SegmentationEngine
) : ViewModel() {

    private val _uiState = MutableStateFlow(StudioUiState())
    val uiState: StateFlow<StudioUiState> = _uiState.asStateFlow()

    init {
        viewModelScope.launch(Dispatchers.IO) {
            // Clean up any legacy vector preset projects
            val legacyTitles = setOf("Cyberpunk Portrait", "Alpine Mountain Vista", "Golden Companion")
            repository.getAllProjects().filter { it.title in legacyTitles }.forEach {
                repository.deleteProject(it.id)
            }
            refreshProjectsList()
            val active = repository.getActiveProject() ?: repository.getAllProjects().firstOrNull()
            active?.let { loadProjectBitmaps(it) }
        }
    }

    fun refreshProjectsList() {
        val list = repository.getAllProjects()
        _uiState.value = _uiState.value.copy(allProjects = list)
    }

    fun selectProject(project: WallpaperProject) {
        viewModelScope.launch(Dispatchers.IO) {
            loadProjectBitmaps(project)
        }
    }

    private suspend fun loadProjectBitmaps(project: WallpaperProject) = withContext(Dispatchers.IO) {
        val src = repository.loadBitmap(project.sourceImagePath)
        val cut = repository.loadBitmap(project.cutoutImagePath)
        val bg = repository.loadBitmap(project.inpaintedBackgroundPath) ?: src
        val depth = repository.loadBitmap(project.depthMapPath)

        _uiState.value = _uiState.value.copy(
            currentProject = project,
            sourceBitmap = src,
            cutoutBitmap = cut,
            backgroundBitmap = bg,
            depthBitmap = depth
        )
    }

    fun importNewImage(bitmap: Bitmap, title: String = "My Wallpaper") {
        _uiState.value = _uiState.value.copy(isProcessing = true, statusMessage = "AI segmenting photo on-device...")
        viewModelScope.launch(Dispatchers.Default) {
            val result = segmentationEngine.processImage(
                sourceBmp = bitmap,
                threshold = 0.5f,
                edgeFeathering = 6,
                inpaintRadius = 14
            )

            // Auto-detect mode: Portrait -> LAYERED_2D, Scenic/Other -> SPATIAL_3D
            val detectedMode = if (result.isPortraitDetected) RenderMode.LAYERED_2D else RenderMode.SPATIAL_3D

            val newProject = WallpaperProject(
                title = title,
                renderMode = detectedMode,
                isActive = true
            )

            val thumbScale = 480f / maxOf(bitmap.width, bitmap.height)
            val thumbW = if (thumbScale < 1f) (bitmap.width * thumbScale).toInt() else bitmap.width
            val thumbH = if (thumbScale < 1f) (bitmap.height * thumbScale).toInt() else bitmap.height
            val thumbBmp = if (thumbScale < 1f) Bitmap.createScaledBitmap(bitmap, thumbW, thumbH, true) else bitmap

            val saved = repository.saveProject(
                project = newProject,
                sourceBmp = bitmap,
                cutoutBmp = result.foregroundCutout,
                inpaintedBgBmp = result.inpaintedBackground,
                depthBmp = result.depthMap,
                thumbBmp = thumbBmp
            )

            repository.setActiveProject(saved.id)

            withContext(Dispatchers.Main) {
                _uiState.value = _uiState.value.copy(
                    currentProject = saved,
                    sourceBitmap = bitmap,
                    cutoutBitmap = result.foregroundCutout,
                    backgroundBitmap = result.inpaintedBackground,
                    depthBitmap = result.depthMap,
                    isProcessing = false,
                    statusMessage = if (result.isPortraitDetected) "Detected Portrait → Layered Cutout" else "Detected Scene → 3D Spatial Depth"
                )
                refreshProjectsList()
            }
        }
    }

    fun toggleRenderMode() {
        val cur = _uiState.value.currentProject
        val newMode = if (cur.renderMode == RenderMode.LAYERED_2D) RenderMode.SPATIAL_3D else RenderMode.LAYERED_2D
        val updated = cur.copy(renderMode = newMode)
        repository.saveProjectMetaOnly(updated)
        _uiState.value = _uiState.value.copy(currentProject = updated)
        refreshProjectsList()
    }

    fun setPreviewSurface(surface: PreviewSurface) {
        _uiState.value = _uiState.value.copy(previewSurface = surface)
    }

    fun setActiveTab(tab: StudioTab) {
        _uiState.value = _uiState.value.copy(activeTab = tab)
    }

    fun updateTilt(x: Float, y: Float) {
        _uiState.value = _uiState.value.copy(simulatedTiltX = x.coerceIn(-1f, 1f), simulatedTiltY = y.coerceIn(-1f, 1f))
    }

    fun updateClockStyle(style: ClockFontStyle) {
        val cur = _uiState.value.currentProject
        val updated = cur.copy(lockScreenConfig = cur.lockScreenConfig.copy(fontStyle = style))
        repository.saveProjectMetaOnly(updated)
        _uiState.value = _uiState.value.copy(currentProject = updated)
    }

    fun updateClockColor(colorHex: Long) {
        val cur = _uiState.value.currentProject
        val updated = cur.copy(lockScreenConfig = cur.lockScreenConfig.copy(clockColorHex = colorHex))
        repository.saveProjectMetaOnly(updated)
        _uiState.value = _uiState.value.copy(currentProject = updated)
    }

    fun updateClockPosition(xPercent: Float, yPercent: Float) {
        val cur = _uiState.value.currentProject
        val updated = cur.copy(
            lockScreenConfig = cur.lockScreenConfig.copy(
                horizontalOffsetPercent = xPercent.coerceIn(0.15f, 0.85f),
                verticalOffsetPercent = yPercent.coerceIn(0.08f, 0.70f)
            )
        )
        repository.saveProjectMetaOnly(updated)
        _uiState.value = _uiState.value.copy(currentProject = updated)
    }

    fun updateClockVerticalOffset(offsetPercent: Float) {
        val cur = _uiState.value.currentProject
        val updated = cur.copy(lockScreenConfig = cur.lockScreenConfig.copy(verticalOffsetPercent = offsetPercent))
        repository.saveProjectMetaOnly(updated)
        _uiState.value = _uiState.value.copy(currentProject = updated)
    }

    fun updateClockScale(scale: Float) {
        val cur = _uiState.value.currentProject
        val updated = cur.copy(lockScreenConfig = cur.lockScreenConfig.copy(clockScale = scale))
        repository.saveProjectMetaOnly(updated)
        _uiState.value = _uiState.value.copy(currentProject = updated)
    }

    fun toggleSubjectInFrontOfClock() {
        val cur = _uiState.value.currentProject
        val updated = cur.copy(lockScreenConfig = cur.lockScreenConfig.copy(subjectInFrontOfClock = !cur.lockScreenConfig.subjectInFrontOfClock))
        repository.saveProjectMetaOnly(updated)
        _uiState.value = _uiState.value.copy(currentProject = updated)
    }

    fun toggleHideClockOnHomeScreen() {
        val cur = _uiState.value.currentProject
        val updated = cur.copy(homeScreenConfig = cur.homeScreenConfig.copy(hideClockOnHomeScreen = !cur.homeScreenConfig.hideClockOnHomeScreen))
        repository.saveProjectMetaOnly(updated)
        _uiState.value = _uiState.value.copy(currentProject = updated)
    }

    fun updateHomeScreenDimming(dim: Float) {
        val cur = _uiState.value.currentProject
        val updated = cur.copy(homeScreenConfig = cur.homeScreenConfig.copy(dimmingFactor = dim))
        repository.saveProjectMetaOnly(updated)
        _uiState.value = _uiState.value.copy(currentProject = updated)
    }

    fun updateParallaxIntensity(intensity: Float) {
        val cur = _uiState.value.currentProject
        val updated = cur.copy(motionConfig = cur.motionConfig.copy(parallaxIntensity = intensity))
        repository.saveProjectMetaOnly(updated)
        _uiState.value = _uiState.value.copy(currentProject = updated)
    }

    fun reprocessWithTuning(
        threshold: Float = _uiState.value.currentProject.threshold,
        feathering: Int = _uiState.value.currentProject.edgeFeathering,
        inpaintRadius: Int = _uiState.value.currentProject.inpaintRadius,
        modelType: SegmentationModelType? = null
    ) {
        val src = _uiState.value.sourceBitmap ?: return
        _uiState.value = _uiState.value.copy(isProcessing = true, statusMessage = "Refining segmentation & layers...")

        viewModelScope.launch(Dispatchers.Default) {
            modelType?.let { segmentationEngine.setModelType(it) }
            val result = segmentationEngine.processImage(
                sourceBmp = src,
                threshold = threshold,
                edgeFeathering = feathering,
                inpaintRadius = inpaintRadius
            )

            val cur = _uiState.value.currentProject.copy(
                threshold = threshold,
                edgeFeathering = feathering,
                inpaintRadius = inpaintRadius
            )

            val saved = repository.saveProject(
                project = cur,
                cutoutBmp = result.foregroundCutout,
                inpaintedBgBmp = result.inpaintedBackground,
                depthBmp = result.depthMap
            )

            withContext(Dispatchers.Main) {
                _uiState.value = _uiState.value.copy(
                    currentProject = saved,
                    cutoutBitmap = result.foregroundCutout,
                    backgroundBitmap = result.inpaintedBackground,
                    depthBitmap = result.depthMap,
                    isProcessing = false,
                    statusMessage = null
                )
            }
        }
    }

    fun getCurrentModelType(): SegmentationModelType = segmentationEngine.currentModelType

    fun toggleFavorite(projectId: String) {
        repository.toggleFavorite(projectId)
        refreshProjectsList()
        if (_uiState.value.currentProject.id == projectId) {
            _uiState.value = _uiState.value.copy(
                currentProject = _uiState.value.currentProject.copy(isFavorite = !_uiState.value.currentProject.isFavorite)
            )
        }
    }

    fun setActiveWallpaper(projectId: String, context: Context) {
        repository.setActiveProject(projectId)
        refreshProjectsList()
        val p = repository.getProjectById(projectId)
        if (p != null) {
            selectProject(p)
        }

        // Launch system wallpaper picker pointing to ParallaxWallpaperService
        try {
            val intent = Intent(WallpaperManager.ACTION_CHANGE_LIVE_WALLPAPER).apply {
                putExtra(
                    WallpaperManager.EXTRA_LIVE_WALLPAPER_COMPONENT,
                    ComponentName(context, ParallaxWallpaperService::class.java)
                )
                addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)
            }
            context.startActivity(intent)
        } catch (_: Exception) {
            // Fallback for devices without direct component selector
            val fallback = Intent(WallpaperManager.ACTION_LIVE_WALLPAPER_CHOOSER).apply {
                addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)
            }
            context.startActivity(fallback)
        }
    }

    fun duplicateProject(projectId: String) {
        repository.duplicateProject(projectId)
        refreshProjectsList()
    }

    fun deleteProject(projectId: String) {
        repository.deleteProject(projectId)
        refreshProjectsList()
        val remaining = repository.getActiveProject() ?: repository.getAllProjects().firstOrNull()
        if (remaining != null) {
            selectProject(remaining)
        }
    }
}

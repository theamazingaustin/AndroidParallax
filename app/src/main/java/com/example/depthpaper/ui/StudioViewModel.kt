package com.example.depthpaper.ui

import android.app.WallpaperManager
import android.content.ComponentName
import android.content.Context
import android.content.Intent
import android.graphics.Bitmap
import android.graphics.Color
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.example.depthpaper.core.AiModelChoice
import com.example.depthpaper.core.AiPipelineChoice
import com.example.depthpaper.core.AppLogger
import com.example.depthpaper.core.DepthSlicingEngine
import com.example.depthpaper.core.ProcessingMode
import com.example.depthpaper.core.SegmentationEngine
import com.example.depthpaper.core.SegmentationModelType
import com.example.depthpaper.data.ClockFontStyle
import com.example.depthpaper.data.ProjectRepository
import com.example.depthpaper.data.RenderMode
import com.example.depthpaper.data.WallpaperProject
import com.example.depthpaper.service.ParallaxWallpaperService
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.delay
import kotlin.math.max
import kotlin.math.min
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

enum class PreviewSurface {
    LOCK_SCREEN,
    HOME_SCREEN,
    AOD,
    DEPTH_MAP
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

    private var saveJob: Job? = null
    private var sliceJob: Job? = null
    private var cachedNormalizedDepth: FloatArray? = null
    private var cachedDepthW: Int = 0
    private var cachedDepthH: Int = 0

    private fun ensureCachedDepth(): Boolean {
        if (cachedNormalizedDepth != null && cachedDepthW > 0 && cachedDepthH > 0) return true
        val projId = _uiState.value.currentProject.id
        if (projId.isNotBlank()) {
            val rawBmp = repository.loadRawDepthBitmap(projId)
            if (rawBmp != null) {
                val d = DepthSlicingEngine.extractGrayscaleDepth(rawBmp)
                if (d != null) {
                    cachedNormalizedDepth = d
                    cachedDepthW = rawBmp.width
                    cachedDepthH = rawBmp.height
                    return true
                }
            }
        }
        val depthBmp = _uiState.value.depthBitmap ?: return false
        val safeDepthBmp = DepthSlicingEngine.ensureSoftwareBitmap(depthBmp)
        val w = safeDepthBmp.width
        val h = safeDepthBmp.height
        if (w <= 0 || h <= 0) return false
        return try {
            val pixels = IntArray(w * h)
            safeDepthBmp.getPixels(pixels, 0, w, 0, 0, w, h)
            val depthArr = FloatArray(w * h)
            for (i in 0 until (w * h)) {
                depthArr[i] = (Color.red(pixels[i]) / 255.0f).coerceIn(0f, 1f)
            }
            cachedNormalizedDepth = depthArr
            cachedDepthW = w
            cachedDepthH = h
            true
        } catch (t: Throwable) {
            AppLogger.e("StudioViewModel", "ensureCachedDepth failed", t)
            false
        }
    }

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
        val rawSrc = repository.loadBitmap(project.sourceImagePath)
        val src = rawSrc?.let { DepthSlicingEngine.getSafeWorkingBitmap(it) }
        var cut = repository.loadBitmap(project.cutoutImagePath)
        val bg = repository.loadBitmap(project.inpaintedBackgroundPath) ?: src
        val depth = repository.loadBitmap(project.depthMapPath)
        val rawDepthBmp = repository.loadRawDepthBitmap(project.id)

        if (rawDepthBmp != null) {
            val d = DepthSlicingEngine.extractGrayscaleDepth(rawDepthBmp)
            if (d != null) {
                cachedNormalizedDepth = d
                cachedDepthW = rawDepthBmp.width
                cachedDepthH = rawDepthBmp.height
            }
        }

        // If cutout is null and we have depth and source, generate it immediately at clockZDepth!
        if (cut == null && src != null && ensureCachedDepth()) {
            val d = cachedNormalizedDepth
            val dw = cachedDepthW
            val dh = cachedDepthH
            if (d != null && dw > 0 && dh > 0) {
                cut = DepthSlicingEngine.sliceForegroundCutout(
                    sourceBmp = src,
                    normalizedDepth = d,
                    depthWidth = dw,
                    depthHeight = dh,
                    clockZDepth = project.clockZDepth
                )
            }
        }

        _uiState.value = _uiState.value.copy(
            currentProject = project,
            sourceBitmap = src,
            cutoutBitmap = cut,
            backgroundBitmap = bg,
            depthBitmap = depth
        )
    }

    fun importNewImage(bitmap: Bitmap, title: String = "My Wallpaper") {
        val safeWorkingBmp = DepthSlicingEngine.getSafeWorkingBitmap(bitmap)
        _uiState.value = _uiState.value.copy(isProcessing = true, statusMessage = "AI estimating 3D depth on-device...")
        viewModelScope.launch(Dispatchers.Default) {
            val result = segmentationEngine.processImage(
                sourceBmp = safeWorkingBmp,
                threshold = 0.50f,
                edgeFeathering = 6,
                maskExpansion = 0,
                inpaintRadius = 6,
                processingMode = ProcessingMode.SINGLE_MODEL,
                modelChoice = AiModelChoice.DEPTH_ANYTHING_V2,
                clockZDepth = 0.50f
            )

            val newProject = WallpaperProject(
                title = title,
                renderMode = RenderMode.LAYERED_2D,
                processingMode = ProcessingMode.SINGLE_MODEL,
                selectedModel = AiModelChoice.DEPTH_ANYTHING_V2,
                clockZDepth = 0.50f,
                isActive = true
            )

            val thumbScale = 480f / maxOf(safeWorkingBmp.width, safeWorkingBmp.height)
            val thumbW = if (thumbScale < 1f) (safeWorkingBmp.width * thumbScale).toInt() else safeWorkingBmp.width
            val thumbH = if (thumbScale < 1f) (safeWorkingBmp.height * thumbScale).toInt() else safeWorkingBmp.height
            val thumbBmp = if (thumbScale < 1f) Bitmap.createScaledBitmap(safeWorkingBmp, thumbW, thumbH, true) else safeWorkingBmp

            val rawDepthBmp = if (result.normalizedDepth != null) {
                cachedNormalizedDepth = result.normalizedDepth
                cachedDepthW = result.depthWidth
                cachedDepthH = result.depthHeight
                DepthSlicingEngine.createGrayscaleDepthBitmap(result.normalizedDepth, result.depthWidth, result.depthHeight)
            } else null

            // Initial slice at clockZDepth = 0.50f
            val initialCutout = if (result.normalizedDepth != null) {
                DepthSlicingEngine.sliceForegroundCutout(
                    sourceBmp = safeWorkingBmp,
                    normalizedDepth = result.normalizedDepth,
                    depthWidth = result.depthWidth,
                    depthHeight = result.depthHeight,
                    clockZDepth = 0.50f
                )
            } else {
                result.foregroundCutout
            }

            val saved = repository.saveProject(
                project = newProject,
                sourceBmp = safeWorkingBmp,
                cutoutBmp = initialCutout,
                inpaintedBgBmp = result.inpaintedBackground,
                depthBmp = result.depthMap,
                rawDepthBmp = rawDepthBmp,
                thumbBmp = thumbBmp
            )

            repository.setActiveProject(saved.id)

            withContext(Dispatchers.Main) {
                _uiState.value = _uiState.value.copy(
                    currentProject = saved,
                    sourceBitmap = safeWorkingBmp,
                    cutoutBitmap = initialCutout,
                    backgroundBitmap = result.inpaintedBackground,
                    depthBitmap = result.depthMap,
                    isProcessing = false,
                    statusMessage = "Continuous 3D Depth Map ready"
                )
                refreshProjectsList()
            }
        }
    }

    /**
     * Changes photo in existing project without creating a new project ID or resetting customized clock/motion configs.
     */
    fun changeProjectImage(bitmap: Bitmap) {
        val cur = _uiState.value.currentProject
        _uiState.value = _uiState.value.copy(isProcessing = true, statusMessage = "AI updating wallpaper photo...")
        viewModelScope.launch(Dispatchers.Default) {
            val result = segmentationEngine.processImage(
                sourceBmp = bitmap,
                threshold = cur.threshold,
                edgeFeathering = cur.edgeFeathering,
                maskExpansion = cur.maskExpansion,
                inpaintRadius = cur.inpaintRadius,
                cutoutContrast = cur.cutoutContrast,
                processingMode = cur.processingMode,
                modelChoice = cur.selectedModel,
                pipelineChoice = cur.selectedPipeline,
                clockZDepth = cur.clockZDepth,
                depthPlaneOffset = cur.depthPlaneOffset,
                fusionBalance = cur.fusionBalance,
                enableHoleFilling = cur.enableHoleFilling,
                holeFillingRadius = cur.holeFillingRadius
            )

            val detectedMode = if (result.isPortraitDetected) RenderMode.LAYERED_2D else RenderMode.SPATIAL_3D

            val thumbScale = 480f / maxOf(bitmap.width, bitmap.height)
            val thumbW = if (thumbScale < 1f) (bitmap.width * thumbScale).toInt() else bitmap.width
            val thumbH = if (thumbScale < 1f) (bitmap.height * thumbScale).toInt() else bitmap.height
            val thumbBmp = if (thumbScale < 1f) Bitmap.createScaledBitmap(bitmap, thumbW, thumbH, true) else bitmap

            val updated = cur.copy(
                renderMode = detectedMode
            )

            val rawDepthBmp = if (result.normalizedDepth != null) {
                cachedNormalizedDepth = result.normalizedDepth
                cachedDepthW = result.depthWidth
                cachedDepthH = result.depthHeight
                DepthSlicingEngine.createGrayscaleDepthBitmap(result.normalizedDepth, result.depthWidth, result.depthHeight)
            } else null

            val initialCutout = if (result.normalizedDepth != null) {
                DepthSlicingEngine.sliceForegroundCutout(
                    sourceBmp = bitmap,
                    normalizedDepth = result.normalizedDepth,
                    depthWidth = result.depthWidth,
                    depthHeight = result.depthHeight,
                    clockZDepth = cur.clockZDepth
                )
            } else {
                result.foregroundCutout
            }

            val saved = repository.saveProject(
                project = updated,
                sourceBmp = bitmap,
                cutoutBmp = initialCutout,
                inpaintedBgBmp = result.inpaintedBackground,
                depthBmp = result.depthMap,
                rawDepthBmp = rawDepthBmp,
                thumbBmp = thumbBmp
            )

            withContext(Dispatchers.Main) {
                _uiState.value = _uiState.value.copy(
                    currentProject = saved,
                    sourceBitmap = bitmap,
                    cutoutBitmap = initialCutout,
                    backgroundBitmap = result.inpaintedBackground,
                    depthBitmap = result.depthMap,
                    isProcessing = false,
                    statusMessage = null
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

    fun updateImageTransform(scale: Float, panX: Float, panY: Float) {
        val cur = _uiState.value.currentProject
        val updated = cur.copy(
            imageScale = scale.coerceIn(1.0f, 3.5f),
            imagePanX = panX.coerceIn(-0.6f, 0.6f),
            imagePanY = panY.coerceIn(-0.6f, 0.6f)
        )
        repository.saveProjectMetaOnly(updated)
        _uiState.value = _uiState.value.copy(currentProject = updated)
    }

    private var reprocessJob: Job? = null

    /**
     * Reactively reprocesses the image with new tuning parameters.
     * Supports debouncing for smooth slider dragging (cancels in-flight tasks),
     * and immediate execution (0ms) for model/pipeline/toggle clicks.
     */
    fun onTuningChanged(
        threshold: Float = _uiState.value.currentProject.threshold,
        feathering: Int = _uiState.value.currentProject.edgeFeathering,
        maskExpansion: Int = _uiState.value.currentProject.maskExpansion,
        inpaintRadius: Int = _uiState.value.currentProject.inpaintRadius,
        modelType: AiModelChoice? = null,
        cutoutContrast: Float = _uiState.value.currentProject.cutoutContrast,
        processingMode: ProcessingMode = _uiState.value.currentProject.processingMode,
        pipelineChoice: AiPipelineChoice = _uiState.value.currentProject.selectedPipeline,
        clockZDepth: Float = _uiState.value.currentProject.clockZDepth,
        depthPlaneOffset: Float = _uiState.value.currentProject.depthPlaneOffset,
        fusionBalance: Float = _uiState.value.currentProject.fusionBalance,
        enableHoleFilling: Boolean = _uiState.value.currentProject.enableHoleFilling,
        holeFillingRadius: Int = _uiState.value.currentProject.holeFillingRadius,
        depthLayerCount: Int = _uiState.value.currentProject.depthLayerCount,
        debounceMs: Long = 250L
    ) {
        val src = _uiState.value.sourceBitmap ?: return
        val activeModel = modelType ?: _uiState.value.currentProject.selectedModel

        val updatedMeta = _uiState.value.currentProject.copy(
            threshold = threshold,
            edgeFeathering = feathering,
            maskExpansion = maskExpansion,
            inpaintRadius = inpaintRadius,
            cutoutContrast = cutoutContrast,
            depthPlaneOffset = depthPlaneOffset,
            clockZDepth = clockZDepth,
            depthLayerCount = depthLayerCount,
            fusionBalance = fusionBalance,
            enableHoleFilling = enableHoleFilling,
            holeFillingRadius = holeFillingRadius,
            processingMode = processingMode,
            selectedModel = activeModel,
            selectedPipeline = pipelineChoice
        )

        // Instantly update project state for smooth UI reactivity
        _uiState.value = _uiState.value.copy(
            currentProject = updatedMeta,
            isProcessing = true,
            statusMessage = "AI Reprocessing (${if (processingMode == ProcessingMode.PIPELINE) pipelineChoice.shortLabel else activeModel.shortLabel})..."
        )

        reprocessJob?.cancel()
        reprocessJob = viewModelScope.launch(Dispatchers.Default) {
            if (debounceMs > 0) {
                delay(debounceMs)
            }

            segmentationEngine.setProcessingMode(processingMode)
            if (processingMode == ProcessingMode.SINGLE_MODEL) {
                segmentationEngine.setModelChoice(activeModel)
            } else {
                segmentationEngine.setPipelineChoice(pipelineChoice)
            }

            val result = segmentationEngine.processImage(
                sourceBmp = src,
                threshold = threshold,
                edgeFeathering = feathering,
                maskExpansion = maskExpansion,
                inpaintRadius = inpaintRadius,
                cutoutContrast = cutoutContrast,
                processingMode = processingMode,
                modelChoice = activeModel,
                pipelineChoice = pipelineChoice,
                clockZDepth = clockZDepth,
                depthPlaneOffset = depthPlaneOffset,
                fusionBalance = fusionBalance,
                enableHoleFilling = enableHoleFilling,
                holeFillingRadius = holeFillingRadius,
                depthLayerCount = depthLayerCount
            )

            val rawDepthBmp = if (result.normalizedDepth != null) {
                cachedNormalizedDepth = result.normalizedDepth
                cachedDepthW = result.depthWidth
                cachedDepthH = result.depthHeight
                DepthSlicingEngine.createGrayscaleDepthBitmap(result.normalizedDepth, result.depthWidth, result.depthHeight)
            } else null

            val initialCutout = if (result.normalizedDepth != null && _uiState.value.sourceBitmap != null) {
                DepthSlicingEngine.sliceForegroundCutout(
                    sourceBmp = _uiState.value.sourceBitmap!!,
                    normalizedDepth = result.normalizedDepth,
                    depthWidth = result.depthWidth,
                    depthHeight = result.depthHeight,
                    clockZDepth = clockZDepth
                )
            } else {
                result.foregroundCutout
            }

            val saved = repository.saveProject(
                project = updatedMeta,
                cutoutBmp = initialCutout,
                inpaintedBgBmp = result.inpaintedBackground,
                depthBmp = result.depthMap,
                rawDepthBmp = rawDepthBmp
            )

            withContext(Dispatchers.Main) {
                _uiState.value = _uiState.value.copy(
                    currentProject = saved,
                    cutoutBitmap = initialCutout,
                    backgroundBitmap = result.inpaintedBackground,
                    depthBitmap = result.depthMap,
                    isProcessing = false,
                    statusMessage = null
                )
            }
        }
    }

    fun reprocessWithTuning(
        threshold: Float = _uiState.value.currentProject.threshold,
        feathering: Int = _uiState.value.currentProject.edgeFeathering,
        maskExpansion: Int = _uiState.value.currentProject.maskExpansion,
        inpaintRadius: Int = _uiState.value.currentProject.inpaintRadius,
        modelType: AiModelChoice? = null,
        cutoutContrast: Float = _uiState.value.currentProject.cutoutContrast,
        processingMode: ProcessingMode = _uiState.value.currentProject.processingMode,
        pipelineChoice: AiPipelineChoice = _uiState.value.currentProject.selectedPipeline,
        clockZDepth: Float = _uiState.value.currentProject.clockZDepth,
        depthPlaneOffset: Float = _uiState.value.currentProject.depthPlaneOffset,
        fusionBalance: Float = _uiState.value.currentProject.fusionBalance,
        enableHoleFilling: Boolean = _uiState.value.currentProject.enableHoleFilling,
        holeFillingRadius: Int = _uiState.value.currentProject.holeFillingRadius,
        depthLayerCount: Int = _uiState.value.currentProject.depthLayerCount
    ) {
        onTuningChanged(
            threshold = threshold,
            feathering = feathering,
            maskExpansion = maskExpansion,
            inpaintRadius = inpaintRadius,
            modelType = modelType,
            cutoutContrast = cutoutContrast,
            processingMode = processingMode,
            pipelineChoice = pipelineChoice,
            clockZDepth = clockZDepth,
            depthPlaneOffset = depthPlaneOffset,
            fusionBalance = fusionBalance,
            enableHoleFilling = enableHoleFilling,
            holeFillingRadius = holeFillingRadius,
            depthLayerCount = depthLayerCount,
            debounceMs = 0L
        )
    }

    /**
     * Dynamically slices foreground cutout from continuous 3D depth map
     * at 60 FPS in ~5ms as the user drags the Clock Z-Position slider.
     */
    fun onClockZDepthChanged(zDepth: Float) {
        val cur = _uiState.value.currentProject
        val newZ = zDepth.coerceIn(0.0f, 1.0f)
        val updatedMeta = cur.copy(clockZDepth = newZ)

        _uiState.value = _uiState.value.copy(currentProject = updatedMeta)

        val src = _uiState.value.sourceBitmap
        if (src != null && ensureCachedDepth()) {
            val depth = cachedNormalizedDepth
            val dW = cachedDepthW
            val dH = cachedDepthH
            if (depth != null && dW > 0 && dH > 0) {
                sliceJob?.cancel()
                sliceJob = viewModelScope.launch(Dispatchers.Default) {
                    try {
                        val sliced = DepthSlicingEngine.sliceForegroundCutout(
                            sourceBmp = src,
                            normalizedDepth = depth,
                            depthWidth = dW,
                            depthHeight = dH,
                            clockZDepth = newZ
                        )
                        if (sliced != null) {
                            withContext(Dispatchers.Main) {
                                _uiState.value = _uiState.value.copy(
                                    cutoutBitmap = sliced,
                                    currentProject = updatedMeta
                                )
                            }
                        }
                    } catch (t: Throwable) {
                        AppLogger.e("StudioViewModel", "sliceJob error", t)
                    }
                }
            }
        }

        saveJob?.cancel()
        saveJob = viewModelScope.launch(Dispatchers.IO) {
            try {
                delay(350)
                repository.saveProjectMetaOnly(updatedMeta)
                val currentCut = _uiState.value.cutoutBitmap
                if (currentCut != null) {
                    repository.saveCutoutOnly(updatedMeta.id, currentCut)
                }
            } catch (t: Throwable) {
                AppLogger.e("StudioViewModel", "saveJob error", t)
            }
        }
    }

    fun onHoleFillingChanged(enabled: Boolean, radius: Int = _uiState.value.currentProject.holeFillingRadius) {
        onTuningChanged(enableHoleFilling = enabled, holeFillingRadius = radius, debounceMs = 0L)
    }

    /**
     * Tapping on the preview samples continuous 3D depth,
     * and sets Clock Z-depth instantaneously (< 5ms).
     */
    fun onTapPreviewCoordinate(normX: Float, normY: Float) {
        if (!ensureCachedDepth()) return
        val depth = cachedNormalizedDepth ?: return
        val px = (normX * (cachedDepthW - 1)).toInt().coerceIn(0, cachedDepthW - 1)
        val py = (normY * (cachedDepthH - 1)).toInt().coerceIn(0, cachedDepthH - 1)
        val continuousZ = depth[py * cachedDepthW + px]

        onClockZDepthChanged(continuousZ.coerceIn(0.0f, 1.0f))
    }

    fun getCurrentModelType(): AiModelChoice = segmentationEngine.currentModelChoice
    fun getCurrentProcessingMode(): ProcessingMode = segmentationEngine.currentProcessingMode
    fun getCurrentPipelineChoice(): AiPipelineChoice = segmentationEngine.currentPipelineChoice

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

    fun deleteProjects(projectIds: Set<String>) {
        projectIds.forEach { repository.deleteProject(it) }
        refreshProjectsList()
        val currentId = _uiState.value.currentProject.id
        if (currentId in projectIds) {
            val remaining = repository.getActiveProject() ?: repository.getAllProjects().firstOrNull()
            if (remaining != null) {
                selectProject(remaining)
            } else {
                _uiState.value = _uiState.value.copy(
                    currentProject = WallpaperProject(),
                    sourceBitmap = null,
                    cutoutBitmap = null,
                    backgroundBitmap = null,
                    depthBitmap = null
                )
            }
        }
    }
}

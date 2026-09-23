package com.example.depthpaper

import android.graphics.ImageDecoder
import android.net.Uri
import android.os.Build
import android.os.Bundle
import android.provider.MediaStore
import androidx.activity.ComponentActivity
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.lifecycle.ViewModel
import androidx.lifecycle.ViewModelProvider
import com.example.depthpaper.core.SegmentationEngine
import com.example.depthpaper.data.ProjectRepository
import com.example.depthpaper.theme.DepthPaperTheme
import com.example.depthpaper.ui.ProjectGalleryScreen
import com.example.depthpaper.ui.StudioScreen
import com.example.depthpaper.ui.StudioViewModel

enum class AppScreen {
    STUDIO,
    GALLERY
}

class MainActivity : ComponentActivity() {

    private lateinit var repository: ProjectRepository
    private lateinit var segmentationEngine: SegmentationEngine
    private lateinit var viewModel: StudioViewModel

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()

        repository = ProjectRepository(applicationContext)
        segmentationEngine = SegmentationEngine(applicationContext)
        viewModel = ViewModelProvider(this, object : ViewModelProvider.Factory {
            @Suppress("UNCHECKED_CAST")
            override fun <T : ViewModel> create(modelClass: Class<T>): T {
                return StudioViewModel(repository, segmentationEngine) as T
            }
        })[StudioViewModel::class.java]

        setContent {
            DepthPaperTheme {
                Surface(
                    modifier = Modifier.fillMaxSize(),
                    color = MaterialTheme.colorScheme.background
                ) {
                    val context = LocalContext.current
                    val state by viewModel.uiState.collectAsState()
                    var currentScreen by remember { mutableStateOf(AppScreen.STUDIO) }

                    val galleryPhotoLauncher = rememberLauncherForActivityResult(
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
                            viewModel.importNewImage(bitmap, "New Wallpaper")
                            currentScreen = AppScreen.STUDIO
                        }
                    }

                    when (currentScreen) {
                        AppScreen.STUDIO -> {
                            StudioScreen(
                                viewModel = viewModel,
                                onNavigateToGallery = { currentScreen = AppScreen.GALLERY }
                            )
                        }
                        AppScreen.GALLERY -> {
                            ProjectGalleryScreen(
                                projects = state.allProjects,
                                onSelectProject = { proj ->
                                    viewModel.selectProject(proj)
                                    currentScreen = AppScreen.STUDIO
                                },
                                onSetActive = { id -> viewModel.setActiveWallpaper(id, context) },
                                onToggleFavorite = { id -> viewModel.toggleFavorite(id) },
                                onDuplicate = { id -> viewModel.duplicateProject(id) },
                                onDelete = { id -> viewModel.deleteProject(id) },
                                onNewProjectClick = { galleryPhotoLauncher.launch("image/*") }
                            )
                        }
                    }
                }
            }
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        segmentationEngine.close()
    }
}

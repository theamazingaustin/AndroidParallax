package com.example.depthpaper.core

import android.content.Context
import android.graphics.Bitmap
import com.google.android.gms.common.ConnectionResult
import com.google.android.gms.common.GoogleApiAvailability
import com.google.android.gms.common.moduleinstall.ModuleInstall
import com.google.android.gms.common.moduleinstall.ModuleInstallRequest
import com.google.android.gms.tasks.Tasks
import com.google.mlkit.vision.common.InputImage
import com.google.mlkit.vision.segmentation.subject.SubjectSegmentation
import com.google.mlkit.vision.segmentation.subject.SubjectSegmenter
import com.google.mlkit.vision.segmentation.subject.SubjectSegmenterOptions
import java.nio.FloatBuffer
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicBoolean

data class SubjectSegmentationResult(
    val mask: FloatArray,
    val width: Int,
    val height: Int
)

/**
 * On-device Subject Segmentation powered by Google Play Services ML Kit.
 * Extracts high-precision continuous confidence masks for arbitrary foreground subjects
 * (people, pets, plants, objects) across diverse scene geometries.
 */
class MlKitSubjectSegmenter(private val context: Context) {

    private val tag = "MlKitSubjectSegmenter"
    private var segmenter: SubjectSegmenter? = null
    private val isInstalling = AtomicBoolean(false)

    init {
        ensureModuleAvailable()
    }

    /**
     * Checks if Google Play Services is available and requests dynamic module
     * provisioning if not yet downloaded to the device.
     */
    fun ensureModuleAvailable() {
        try {
            val availability = GoogleApiAvailability.getInstance().isGooglePlayServicesAvailable(context)
            if (availability != ConnectionResult.SUCCESS) {
                AppLogger.w(tag, "Google Play Services not available (code $availability). ML Kit Subject will fallback to bundled models.")
                return
            }

            val client = getOrCreateSegmenter() ?: return
            val moduleInstallClient = ModuleInstall.getClient(context)

            moduleInstallClient.areModulesAvailable(client)
                .addOnSuccessListener { response ->
                    if (!response.areModulesAvailable() && isInstalling.compareAndSet(false, true)) {
                        AppLogger.i(tag, "Requesting on-device installation of ML Kit Subject Segmentation module...")
                        val request = ModuleInstallRequest.newBuilder()
                            .addApi(client)
                            .build()
                        moduleInstallClient.installModules(request)
                            .addOnSuccessListener {
                                AppLogger.i(tag, "ML Kit Subject Segmentation module installed successfully.")
                                isInstalling.set(false)
                            }
                            .addOnFailureListener { e ->
                                AppLogger.w(tag, "ML Kit module install failed: ${e.message}")
                                isInstalling.set(false)
                            }
                    } else {
                        AppLogger.d(tag, "ML Kit Subject Segmentation module is available.")
                    }
                }
                .addOnFailureListener { e ->
                    AppLogger.w(tag, "Could not check ML Kit module availability: ${e.message}")
                }
        } catch (e: Throwable) {
            AppLogger.w(tag, "Failed checking ML Kit module: ${e.message}")
        }
    }

    @Synchronized
    private fun getOrCreateSegmenter(): SubjectSegmenter? {
        if (segmenter != null) return segmenter
        return try {
            val options = SubjectSegmenterOptions.Builder()
                .enableForegroundConfidenceMask()
                .build()
            SubjectSegmentation.getClient(options).also { segmenter = it }
        } catch (e: Throwable) {
            AppLogger.w(tag, "Failed to create ML Kit SubjectSegmenter: ${e.message}")
            null
        }
    }

    /**
     * Runs subject segmentation on [bitmap] and returns a continuous float confidence mask [0.0..1.0].
     * Must be called from a background thread (e.g. Dispatchers.Default or Dispatchers.IO).
     * Returns null if Google Play Services or the module is unavailable, allowing seamless local fallback.
     */
    fun segment(bitmap: Bitmap): SubjectSegmentationResult? {
        val client = getOrCreateSegmenter() ?: return null
        return try {
            val inputImage = InputImage.fromBitmap(bitmap, 0)
            val task = client.process(inputImage)
            // Wait up to 5 seconds for on-device inference
            val result = Tasks.await(task, 5000, TimeUnit.MILLISECONDS)
            val floatBuffer: FloatBuffer = result.foregroundConfidenceMask ?: return null

            val maskW = bitmap.width
            val maskH = bitmap.height

            val total = floatBuffer.capacity()
            val array = FloatArray(total)
            floatBuffer.rewind()
            floatBuffer.get(array)

            AppLogger.i(tag, "ML Kit Subject segmented successfully: ${maskW}x${maskH}, $total pixels")
            SubjectSegmentationResult(
                mask = array,
                width = maskW,
                height = maskH
            )
        } catch (e: Throwable) {
            AppLogger.w(tag, "ML Kit Subject segmentation failed or not ready: ${e.message}. Falling back.")
            null
        }
    }

    @Synchronized
    fun close() {
        try {
            segmenter?.close()
        } catch (_: Exception) {}
        segmenter = null
    }
}

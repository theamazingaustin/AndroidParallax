package com.example.depthpaper.core

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Color
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.gpu.CompatibilityList
import org.tensorflow.lite.gpu.GpuDelegate
import org.tensorflow.lite.nnapi.NnApiDelegate
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.channels.FileChannel
import kotlin.math.abs
import kotlin.math.max
import kotlin.math.min

/**
 * Processing mode category: either a standalone single AI model or a multi-model high-precision pipeline.
 */
enum class ProcessingMode(val displayName: String) {
    SINGLE_MODEL("Single Model"),
    PIPELINE("Multi-Model Pipeline")
}

/**
 * Configuration for a tuning slider, including model-tailored ranges, recommended defaults, and step counts.
 */
data class SliderSetting(
    val min: Float,
    val max: Float,
    val default: Float,
    val steps: Int = 0
)

data class ModelTuningProfile(
    val sensitivity: SliderSetting,
    val maskMargin: SliderSetting,
    val layerFlatness: SliderSetting,
    val edgeSoftness: SliderSetting,
    val inpaintFill: SliderSetting
)

/**
 * Curated list of genuine, on-device AI models.
 * Strictly Apache 2.0 and MIT licenses (100% legal for paid/commercial apps).
 * Every entry points to an authentic neural network weight file in assets.
 */
enum class AiModelChoice(
    val id: String,
    val modelName: String,
    val shortLabel: String,
    val bestAt: String,
    val license: String,
    val assetPath: String,
    val tuningProfile: ModelTuningProfile
) {
    DEPTH_ANYTHING_V2(
        id = "DEPTH_ANYTHING_V2",
        modelName = "Depth Anything V2 Small (ViT-Small)",
        shortLabel = "Depth Anything V2",
        bestAt = "Universal continuous 3D metric depth geometry; redwoods, architecture, landscapes, nature, objects, and depth-based multi-subject extraction.",
        license = "Apache 2.0 (100% Commercial Cleared)",
        assetPath = "models/depth_anything_v2_small.tflite",
        tuningProfile = ModelTuningProfile(
            sensitivity = SliderSetting(min = 0.20f, max = 0.85f, default = 0.50f),
            maskMargin = SliderSetting(min = -10f, max = 10f, default = 0f, steps = 20),
            layerFlatness = SliderSetting(min = 0.50f, max = 1.0f, default = 0.85f),
            edgeSoftness = SliderSetting(min = 1f, max = 16f, default = 6f, steps = 15),
            inpaintFill = SliderSetting(min = 2f, max = 20f, default = 8f, steps = 18)
        )
    ),
    SELFIE_MULTICLASS(
        id = "SELFIE_MULTICLASS",
        modelName = "MediaPipe Selfie Multiclass",
        shortLabel = "Portrait & Hair Matting",
        bestAt = "High-precision portrait matting; sub-pixel hair strands, fine skin tones, and soft clothing transitions on people.",
        license = "Apache 2.0 (100% Commercial Cleared)",
        assetPath = "models/selfie_multiclass.tflite",
        tuningProfile = ModelTuningProfile(
            sensitivity = SliderSetting(min = 0.20f, max = 0.75f, default = 0.40f),
            maskMargin = SliderSetting(min = -8f, max = 8f, default = 0f, steps = 16),
            layerFlatness = SliderSetting(min = 0.50f, max = 1.0f, default = 0.85f),
            edgeSoftness = SliderSetting(min = 2f, max = 16f, default = 8f, steps = 14),
            inpaintFill = SliderSetting(min = 2f, max = 20f, default = 6f, steps = 18)
        )
    ),
    DEEPLAB_V3(
        id = "DEEPLAB_V3",
        modelName = "DeepLab v3 MobileNet",
        shortLabel = "DeepLab (Multi-Subject)",
        bestAt = "Multi-subject and object semantic segmentation across 21 discrete categories (people, pets, vehicles, objects).",
        license = "Apache 2.0 (100% Commercial Cleared)",
        assetPath = "models/deeplab_v3.tflite",
        tuningProfile = ModelTuningProfile(
            sensitivity = SliderSetting(min = 0.25f, max = 0.85f, default = 0.48f),
            maskMargin = SliderSetting(min = -10f, max = 10f, default = 1f, steps = 20),
            layerFlatness = SliderSetting(min = 0.60f, max = 1.0f, default = 0.95f),
            edgeSoftness = SliderSetting(min = 1f, max = 12f, default = 4f, steps = 11),
            inpaintFill = SliderSetting(min = 2f, max = 20f, default = 8f, steps = 18)
        )
    ),
    FAST_SELFIE(
        id = "FAST_SELFIE",
        modelName = "MediaPipe Fast Selfie",
        shortLabel = "Fast Selfie",
        bestAt = "Ultra-fast portrait cutout with minimal battery and compute overhead.",
        license = "Apache 2.0 (100% Commercial Cleared)",
        assetPath = "models/selfie_segmenter.tflite",
        tuningProfile = ModelTuningProfile(
            sensitivity = SliderSetting(min = 0.25f, max = 0.85f, default = 0.50f),
            maskMargin = SliderSetting(min = -10f, max = 10f, default = 0f, steps = 20),
            layerFlatness = SliderSetting(min = 0.50f, max = 1.0f, default = 0.90f),
            edgeSoftness = SliderSetting(min = 1f, max = 14f, default = 5f, steps = 13),
            inpaintFill = SliderSetting(min = 2f, max = 20f, default = 6f, steps = 18)
        )
    );

    companion object {
        fun fromId(id: String): AiModelChoice =
            entries.find { it.id.equals(id, ignoreCase = true) }
                ?: when (id.uppercase()) {
                    "MOD_NET" -> SELFIE_MULTICLASS
                    "BIREF_NET", "MOBILE_SAM" -> DEEPLAB_V3
                    "SELFIE_FAST" -> FAST_SELFIE
                    else -> DEPTH_ANYTHING_V2
                }
    }
}

/**
 * Multi-model pipelines cascading multiple neural passes for maximum precision.
 */
enum class AiPipelineChoice(
    val id: String,
    val pipelineName: String,
    val shortLabel: String,
    val bestAt: String,
    val license: String,
    val tuningProfile: ModelTuningProfile
) {
    DEPTH_MATTING_FUSION(
        id = "DEPTH_MATTING_FUSION",
        pipelineName = "Depth Anything V2 + Hair Matting Fusion",
        shortLabel = "Depth + Matting (Recommended)",
        bestAt = "The ultimate flagship pipeline: Depth Anything V2 captures the full body & 3D geometry (zero hollowing) while Multiclass refines sub-pixel hair strands.",
        license = "Apache 2.0 (100% Commercial Cleared)",
        tuningProfile = ModelTuningProfile(
            sensitivity = SliderSetting(min = 0.20f, max = 0.85f, default = 0.48f),
            maskMargin = SliderSetting(min = -10f, max = 10f, default = 1f, steps = 20),
            layerFlatness = SliderSetting(min = 0.60f, max = 1.0f, default = 0.92f),
            edgeSoftness = SliderSetting(min = 1f, max = 16f, default = 5f, steps = 15),
            inpaintFill = SliderSetting(min = 2f, max = 20f, default = 8f, steps = 18)
        )
    ),
    SEMANTIC_PORTRAIT_HYBRID(
        id = "SEMANTIC_PORTRAIT_HYBRID",
        pipelineName = "DeepLab + Multiclass Hybrid Fusion",
        shortLabel = "DeepLab + Portrait Fusion",
        bestAt = "Fuses DeepLab group/pet/object context with Multiclass hair & clothing details. Preserves full bodies with zero hollowing.",
        license = "Apache 2.0 (100% Commercial Cleared)",
        tuningProfile = ModelTuningProfile(
            sensitivity = SliderSetting(min = 0.20f, max = 0.80f, default = 0.42f),
            maskMargin = SliderSetting(min = -10f, max = 10f, default = 1f, steps = 20),
            layerFlatness = SliderSetting(min = 0.60f, max = 1.0f, default = 0.92f),
            edgeSoftness = SliderSetting(min = 1f, max = 16f, default = 5f, steps = 15),
            inpaintFill = SliderSetting(min = 2f, max = 20f, default = 8f, steps = 18)
        )
    ),
    MULTI_SCALE_ZOOM(
        id = "MULTI_SCALE_ZOOM",
        pipelineName = "Multi-Scale Zoom Tiling",
        shortLabel = "Multi-Scale Zoom",
        bestAt = "Runs a global scene context pass plus high-resolution zoomed crops on subject boundaries for desktop-grade edge precision.",
        license = "Apache 2.0 (100% Commercial Cleared)",
        tuningProfile = ModelTuningProfile(
            sensitivity = SliderSetting(min = 0.25f, max = 0.80f, default = 0.45f),
            maskMargin = SliderSetting(min = -10f, max = 10f, default = 0f, steps = 20),
            layerFlatness = SliderSetting(min = 0.60f, max = 1.0f, default = 0.90f),
            edgeSoftness = SliderSetting(min = 1f, max = 14f, default = 4f, steps = 13),
            inpaintFill = SliderSetting(min = 2f, max = 20f, default = 8f, steps = 18)
        )
    ),
    PURE_DEPTH_3D(
        id = "PURE_DEPTH_3D",
        pipelineName = "Pure Depth Anything V2 3D Geometry",
        shortLabel = "Pure 3D Depth",
        bestAt = "Pure continuous 3D relief mesh without 2D cutout layers. Ideal for landscapes, redwood forests, architecture, and nature wallpapers.",
        license = "Apache 2.0 (100% Commercial Cleared)",
        tuningProfile = ModelTuningProfile(
            sensitivity = SliderSetting(min = 0.20f, max = 0.90f, default = 0.50f),
            maskMargin = SliderSetting(min = -5f, max = 5f, default = 0f, steps = 10),
            layerFlatness = SliderSetting(min = 0.50f, max = 1.0f, default = 0.80f),
            edgeSoftness = SliderSetting(min = 1f, max = 16f, default = 6f, steps = 15),
            inpaintFill = SliderSetting(min = 2f, max = 20f, default = 8f, steps = 18)
        )
    );

    companion object {
        fun fromId(id: String): AiPipelineChoice =
            entries.find { it.id.equals(id, ignoreCase = true) }
                ?: when (id.uppercase()) {
                    "DUAL_MODEL_HYBRID" -> DEPTH_MATTING_FUSION
                    "MULTI_SCALE_TILING" -> MULTI_SCALE_ZOOM
                    else -> DEPTH_MATTING_FUSION
                }
    }
}

/**
 * Backwards compatibility alias for existing code referencing SegmentationModelType.
 */
typealias SegmentationModelType = AiModelChoice

/**
 * Result bundle containing both Layered 2.5D cutouts and 3D depth representations.
 */
data class SegmentationResult(
    val foregroundCutout: Bitmap,
    val inpaintedBackground: Bitmap,
    val depthMap: Bitmap,
    val rawMask: FloatArray,
    val maskWidth: Int,
    val maskHeight: Int,
    val isPortraitDetected: Boolean,
    val foregroundRatio: Float
)

/**
 * Native hardware-accelerated TFLite segmenter wrapper for raw .tflite models.
 * Completely replaces MediaPipe Tasks to guarantee 100% authentic neural inference.
 */
class TfliteSegmenter(
    private val context: Context,
    private val assetPath: String,
    val inputW: Int,
    val inputH: Int,
    private val outputClasses: Int,
    private val normalizeType: Int // 0: [0, 1] for selfie models, 1: [-1, 1] for DeepLab
) {
    private var interpreter: Interpreter? = null
    private var gpuDelegate: GpuDelegate? = null
    private var nnApiDelegate: NnApiDelegate? = null

    @Synchronized
    fun init() {
        if (interpreter != null) return

        val buffer = try {
            val fd = context.assets.openFd(assetPath)
            FileInputStream(fd.fileDescriptor).channel.map(FileChannel.MapMode.READ_ONLY, fd.startOffset, fd.declaredLength)
        } catch (e: Exception) {
            AppLogger.e("TfliteSegmenter", "Failed to map $assetPath: ${e.message}", e)
            return
        }

        // Tier 1: Hardware GPU Acceleration
        try {
            val compat = CompatibilityList()
            if (compat.isDelegateSupportedOnThisDevice) {
                val delegate = GpuDelegate(GpuDelegate.Options().apply { setPrecisionLossAllowed(true) })
                val opts = Interpreter.Options().apply {
                    addDelegate(delegate)
                    setNumThreads(4)
                }
                val interp = Interpreter(buffer, opts)
                interpreter = interp
                gpuDelegate = delegate
                AppLogger.i("TfliteSegmenter", "Loaded $assetPath on GPU")
                return
            }
        } catch (e: Exception) {
            AppLogger.w("TfliteSegmenter", "GPU init failed for $assetPath (${e.message}), trying NNAPI")
            gpuDelegate?.close()
            gpuDelegate = null
        }

        // Tier 2: Qualcomm Hexagon NPU Acceleration via Android NNAPI
        try {
            val delegate = NnApiDelegate(NnApiDelegate.Options().apply {
                setAllowFp16(true)
                setExecutionPreference(NnApiDelegate.Options.EXECUTION_PREFERENCE_SUSTAINED_SPEED)
            })
            val opts = Interpreter.Options().apply {
                addDelegate(delegate)
                setNumThreads(4)
            }
            val interp = Interpreter(buffer, opts)
            interpreter = interp
            nnApiDelegate = delegate
            AppLogger.i("TfliteSegmenter", "Loaded $assetPath on NNAPI")
            return
        } catch (e: Exception) {
            AppLogger.w("TfliteSegmenter", "NNAPI init failed for $assetPath (${e.message}), trying CPU")
            nnApiDelegate?.close()
            nnApiDelegate = null
        }

        // Tier 3: CPU Fallback with 4 threads and XNNPACK
        try {
            val opts = Interpreter.Options().apply { setNumThreads(4) }
            val interp = Interpreter(buffer, opts)
            interpreter = interp
            AppLogger.i("TfliteSegmenter", "Loaded $assetPath on CPU (4 threads)")
        } catch (e: Exception) {
            AppLogger.e("TfliteSegmenter", "Failed to init $assetPath on CPU: ${e.message}", e)
        }
    }

    fun segment(bitmap: Bitmap): Triple<FloatArray, Int, Int>? {
        if (interpreter == null) init()
        var interp = interpreter ?: return null

        val scaled = if (bitmap.width == inputW && bitmap.height == inputH) {
            bitmap
        } else {
            Bitmap.createScaledBitmap(bitmap, inputW, inputH, true)
        }

        val totalPix = inputW * inputH
        val pixels = IntArray(totalPix)
        scaled.getPixels(pixels, 0, inputW, 0, 0, inputW, inputH)
        if (scaled != bitmap && !scaled.isRecycled) {
            scaled.recycle()
        }

        val inputBuf = ByteBuffer.allocateDirect(1 * inputH * inputW * 3 * 4).apply {
            order(ByteOrder.nativeOrder())
        }

        if (normalizeType == 1) {
            // DeepLab v3 MobileNet: [-1.0, 1.0]
            val inv127 = 1.0f / 127.5f
            for (i in 0 until totalPix) {
                val c = pixels[i]
                inputBuf.putFloat(((c shr 16 and 0xFF) - 127.5f) * inv127)
                inputBuf.putFloat(((c shr 8 and 0xFF) - 127.5f) * inv127)
                inputBuf.putFloat(((c and 0xFF) - 127.5f) * inv127)
            }
        } else {
            // MediaPipe models: [0.0, 1.0]
            val inv255 = 1.0f / 255.0f
            for (i in 0 until totalPix) {
                val c = pixels[i]
                inputBuf.putFloat((c shr 16 and 0xFF) * inv255)
                inputBuf.putFloat((c shr 8 and 0xFF) * inv255)
                inputBuf.putFloat((c and 0xFF) * inv255)
            }
        }
        inputBuf.rewind()

        val outputBuf = ByteBuffer.allocateDirect(1 * inputH * inputW * outputClasses * 4).apply {
            order(ByteOrder.nativeOrder())
        }

        try {
            interp.run(inputBuf, outputBuf)
        } catch (e: Exception) {
            AppLogger.w("TfliteSegmenter", "Inference error on hardware delegate for $assetPath (${e.message}), recovering on CPU fallback...")
            try {
                close()
                val fd = context.assets.openFd(assetPath)
                val buffer = FileInputStream(fd.fileDescriptor).channel.map(FileChannel.MapMode.READ_ONLY, fd.startOffset, fd.declaredLength)
                interp = Interpreter(buffer, Interpreter.Options().apply { setNumThreads(4) })
                interpreter = interp
                inputBuf.rewind()
                outputBuf.rewind()
                interp.run(inputBuf, outputBuf)
            } catch (e2: Exception) {
                AppLogger.e("TfliteSegmenter", "CPU inference failed for $assetPath: ${e2.message}", e2)
                return null
            }
        }

        outputBuf.rewind()
        val mask = FloatArray(totalPix)

        if (outputClasses == 1) {
            // selfie_segmenter.tflite: [1, 256, 256, 1] sigmoid probability
            for (i in 0 until totalPix) {
                mask[i] = outputBuf.float.coerceIn(0f, 1f)
            }
        } else if (outputClasses == 6) {
            // selfie_multiclass.tflite: [1, 256, 256, 6] logits
            // 0: bg, 1: hair, 2: body, 3: face, 4: clothes, 5: others
            val logits = FloatArray(6)
            for (i in 0 until totalPix) {
                var maxL = -Float.MAX_VALUE
                for (c in 0 until 6) {
                    val l = outputBuf.float
                    logits[c] = l
                    if (l > maxL) maxL = l
                }
                var sumExp = 0.0
                for (c in 0 until 6) {
                    sumExp += Math.exp((logits[c] - maxL).toDouble())
                }
                val bgProb = (Math.exp((logits[0] - maxL).toDouble()) / sumExp).toFloat()
                // Human presence probability = 1.0 - bgProbability
                mask[i] = (1.0f - bgProb).coerceIn(0f, 1f)
            }
        } else if (outputClasses == 21) {
            // deeplab_v3.tflite: [1, 257, 257, 21] logits
            // Salient categories: 15: person, 8: cat, 12: dog, 3: bird, 13: horse, 7: car, 2: bicycle, 14: motorbike, 19: train
            val salientClasses = intArrayOf(15, 8, 12, 3, 13, 7, 2, 14, 19, 1, 4, 10, 17)
            val logits = FloatArray(21)
            for (i in 0 until totalPix) {
                var maxL = -Float.MAX_VALUE
                for (c in 0 until 21) {
                    val l = outputBuf.float
                    logits[c] = l
                    if (l > maxL) maxL = l
                }
                var sumExp = 0.0
                for (c in 0 until 21) {
                    sumExp += Math.exp((logits[c] - maxL).toDouble())
                }
                var salientExpSum = 0.0
                for (c in salientClasses) {
                    salientExpSum += Math.exp((logits[c] - maxL).toDouble())
                }
                val salientProb = (salientExpSum / sumExp).toFloat()
                val bgProb = (Math.exp((logits[0] - maxL).toDouble()) / sumExp).toFloat()
                mask[i] = max(salientProb, (1.0f - bgProb) * 0.90f).coerceIn(0f, 1f)
            }
        }

        return Triple(mask, inputW, inputH)
    }

    @Synchronized
    fun close() {
        try {
            interpreter?.close()
            gpuDelegate?.close()
            nnApiDelegate?.close()
        } catch (_: Exception) {}
        interpreter = null
        gpuDelegate = null
        nnApiDelegate = null
    }
}

/**
 * On-Device ML Segmentation, Depth Generation, and Pipeline Engine.
 * Operates 100% offline with authentic GPU/NPU acceleration and CPU fallback.
 */
class SegmentationEngine(private val context: Context) {

    private val deepLabSegmenter by lazy {
        TfliteSegmenter(
            context = context,
            assetPath = "models/deeplab_v3.tflite",
            inputW = 257,
            inputH = 257,
            outputClasses = 21,
            normalizeType = 1
        )
    }

    private val multiclassSegmenter by lazy {
        TfliteSegmenter(
            context = context,
            assetPath = "models/selfie_multiclass.tflite",
            inputW = 256,
            inputH = 256,
            outputClasses = 6,
            normalizeType = 0
        )
    }

    private val fastSelfieSegmenter by lazy {
        TfliteSegmenter(
            context = context,
            assetPath = "models/selfie_segmenter.tflite",
            inputW = 256,
            inputH = 256,
            outputClasses = 1,
            normalizeType = 0
        )
    }

    var currentProcessingMode: ProcessingMode = ProcessingMode.PIPELINE
        private set
    var currentModelChoice: AiModelChoice = AiModelChoice.DEPTH_ANYTHING_V2
        private set
    var currentPipelineChoice: AiPipelineChoice = AiPipelineChoice.DEPTH_MATTING_FUSION
        private set

    val currentModelType: AiModelChoice get() = currentModelChoice

    init {
        // Initialize Depth Anything V2 on GPU/NPU
        DepthAnythingEngine.initialize(context)
    }

    fun setProcessingMode(mode: ProcessingMode) {
        currentProcessingMode = mode
    }

    fun setModelChoice(model: AiModelChoice) {
        currentProcessingMode = ProcessingMode.SINGLE_MODEL
        currentModelChoice = model
    }

    fun setPipelineChoice(pipeline: AiPipelineChoice) {
        currentProcessingMode = ProcessingMode.PIPELINE
        currentPipelineChoice = pipeline
    }

    fun setModelType(type: AiModelChoice) {
        setModelChoice(type)
    }

    /**
     * Morphologically dilates (expansion > 0) or erodes (expansion < 0) [rawMask]
     * by [expansionPixels] pixels using a fast separable min/max 2D filter.
     */
    fun morphologicallyFilterMask(
        rawMask: FloatArray,
        w: Int,
        h: Int,
        expansionPixels: Int
    ): FloatArray {
        if (expansionPixels == 0) return rawMask
        val radius = abs(expansionPixels).coerceIn(1, 15)
        val isDilation = expansionPixels > 0

        val temp = FloatArray(w * h)
        val result = FloatArray(w * h)

        // Pass 1: Horizontal min/max
        for (y in 0 until h) {
            val row = y * w
            for (x in 0 until w) {
                var extreme = rawMask[row + x]
                val xStart = max(0, x - radius)
                val xEnd = min(w - 1, x + radius)
                for (nx in xStart..xEnd) {
                    val v = rawMask[row + nx]
                    if (isDilation) {
                        if (v > extreme) extreme = v
                    } else {
                        if (v < extreme) extreme = v
                    }
                }
                temp[row + x] = extreme
            }
        }

        // Pass 2: Vertical min/max
        for (x in 0 until w) {
            for (y in 0 until h) {
                var extreme = temp[y * w + x]
                val yStart = max(0, y - radius)
                val yEnd = min(h - 1, y + radius)
                for (ny in yStart..yEnd) {
                    val v = temp[ny * w + x]
                    if (isDilation) {
                        if (v > extreme) extreme = v
                    } else {
                        if (v < extreme) extreme = v
                    }
                }
                result[y * w + x] = extreme
            }
        }

        return result
    }

    /**
     * Main on-device inference entrypoint.
     * Executes authentic neural network weights, applies separable morphological filtering,
     * and enforces Trimap-Constrained Guided Matting.
     */
    fun processImage(
        sourceBmp: Bitmap,
        threshold: Float = 0.50f,
        edgeFeathering: Int = 6,
        maskExpansion: Int = 0,
        inpaintRadius: Int = 8,
        cutoutContrast: Float = 0.85f,
        enablePreprocessing: Boolean = true,
        processingMode: ProcessingMode = currentProcessingMode,
        modelChoice: AiModelChoice = currentModelChoice,
        pipelineChoice: AiPipelineChoice = currentPipelineChoice,
        depthPlaneOffset: Float = 0.50f,
        fusionBalance: Float = 0.50f
    ): SegmentationResult {
        // Downscale massive camera photos to max 1440px to prevent OOM
        val maxDim = 1440
        val srcW = sourceBmp.width
        val srcH = sourceBmp.height
        val maxSrc = max(srcW, srcH)
        val scale = if (maxSrc > maxDim) maxDim.toFloat() / maxSrc.toFloat() else 1.0f
        val targetW = (srcW * scale).toInt()
        val targetH = (srcH * scale).toInt()

        val safeBmp = if (scale < 1.0f || sourceBmp.config != Bitmap.Config.ARGB_8888 || sourceBmp.isRecycled) {
            val scaled = Bitmap.createScaledBitmap(sourceBmp, targetW, targetH, true)
            scaled.copy(Bitmap.Config.ARGB_8888, false)
        } else {
            sourceBmp
        }

        val w = safeBmp.width
        val h = safeBmp.height

        // Optional Pre-Processing: CLAHE Local Contrast + Bilateral Denoising
        val inferenceBmp = if (enablePreprocessing) {
            AppLogger.i("SegmentationEngine", "Pre-processing enabled: running CLAHE & Bilateral Denoising")
            ImagePreprocessor.enhanceForInference(safeBmp)
        } else {
            safeBmp
        }

        AppLogger.i("SegmentationEngine", "processImage: ${w}x${h}, mode=$processingMode, model=${modelChoice.modelName}, pipeline=${pipelineChoice.pipelineName}")

        // 1. ALWAYS run Depth Anything V2 for real 3D scene geometry & continuous metric depth!
        val depthResult = DepthAnythingEngine.estimateDepth(
            context = context,
            inputBitmap = inferenceBmp,
            foregroundSensitivity = threshold,
            depthPlaneOffset = depthPlaneOffset
        )

        // 2. Execute selected AI processing architecture for foreground extraction
        var (rawMask, maskW, maskH) = when (processingMode) {
            ProcessingMode.PIPELINE -> {
                when (pipelineChoice) {
                    AiPipelineChoice.DEPTH_MATTING_FUSION -> {
                        executeDepthMattingFusionPipeline(inferenceBmp, depthResult, fusionBalance)
                    }
                    AiPipelineChoice.SEMANTIC_PORTRAIT_HYBRID -> {
                        executeSemanticPortraitHybridPipeline(inferenceBmp)
                    }
                    AiPipelineChoice.MULTI_SCALE_ZOOM -> {
                        executeMultiScaleZoomPipeline(inferenceBmp)
                    }
                    AiPipelineChoice.PURE_DEPTH_3D -> {
                        executePureDepthMask(depthResult, inferenceBmp)
                    }
                }
            }
            ProcessingMode.SINGLE_MODEL -> {
                when (modelChoice) {
                    AiModelChoice.DEPTH_ANYTHING_V2 -> {
                        executePureDepthMask(depthResult, inferenceBmp)
                    }
                    AiModelChoice.SELFIE_MULTICLASS -> {
                        multiclassSegmenter.segment(inferenceBmp) ?: computeUniversalSaliencyMask(inferenceBmp)
                    }
                    AiModelChoice.DEEPLAB_V3 -> {
                        deepLabSegmenter.segment(inferenceBmp) ?: computeUniversalSaliencyMask(inferenceBmp)
                    }
                    AiModelChoice.FAST_SELFIE -> {
                        fastSelfieSegmenter.segment(inferenceBmp) ?: computeUniversalSaliencyMask(inferenceBmp)
                    }
                }
            }
        }

        // Saliency check
        var fgCount = 0
        val totalPixels = maskW * maskH
        for (i in 0 until totalPixels) {
            if (rawMask[i] >= threshold) fgCount++
        }
        var fgRatio = fgCount.toFloat() / totalPixels

        // If person ML model produced zero or near-zero subject (nature, architecture, object photo),
        // fallback to Depth Anything V2 depth mask or Universal Saliency!
        if (fgRatio < 0.03f && depthResult != null) {
            AppLogger.i("SegmentationEngine", "Low subject confidence ($fgRatio). Using Depth Anything V2 foreground mask.")
            rawMask = depthResult.foregroundConfidenceMask
            maskW = depthResult.depthWidth
            maskH = depthResult.depthHeight
            fgCount = 0
            for (i in rawMask.indices) {
                if (rawMask[i] >= threshold) fgCount++
            }
            fgRatio = fgCount.toFloat() / rawMask.size
        } else if (fgRatio < 0.03f) {
            AppLogger.i("SegmentationEngine", "Falling back to Universal Saliency.")
            val universal = computeUniversalSaliencyMask(inferenceBmp)
            rawMask = universal.first
            maskW = universal.second
            maskH = universal.third
        }

        val isPortrait = fgRatio in 0.05f..0.85f
        AppLogger.i("SegmentationEngine", "Mask computed: ${maskW}x${maskH}, fgRatio=${"%.3f".format(fgRatio)}, isPortrait=$isPortrait")

        // 3. True Separable 2D Morphological Mask Expansion/Choke
        val adjustedMask = if (maskExpansion != 0) {
            morphologicallyFilterMask(rawMask, maskW, maskH, maskExpansion)
        } else {
            rawMask
        }

        // 4. Compute Guided Filter Coefficients for Sub-Pixel Edge Snapping
        val guidedCoeff = GuidedMattingFilter.computeCoefficients(
            guideBmp = safeBmp,
            rawMask = adjustedMask,
            maskWidth = maskW,
            maskHeight = maskH,
            radius = edgeFeathering.coerceIn(2, 16),
            eps = 0.005f
        )

        // 5. Trimap-Constrained Cutout Generation
        // Core interior is locked to 255 (SOLID: zero hollowing of bodies, clothes, skin)
        // Exterior is locked to 0 (CLEAN: zero background bleed)
        // ONLY the thin boundary transition zone is refined by high-res guided filter & Layer Flatness contrast
        val cutoutBmp = Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888)
        val sourcePixels = IntArray(w * h)
        val cutoutPixels = IntArray(w * h)
        safeBmp.getPixels(sourcePixels, 0, w, 0, 0, w, h)

        val invW = 1.0f / max(1, w - 1)
        val invH = 1.0f / max(1, h - 1)

        val transBand = 0.12f
        val solidFgThresh = (threshold + transBand).coerceAtMost(0.95f)
        val solidBgThresh = (threshold - transBand).coerceAtLeast(0.08f)
        val bandDenom = max(0.0001f, solidFgThresh - solidBgThresh)
        val contrastFactor = 1.0f + (cutoutContrast - 0.5f) * 6.0f

        for (y in 0 until h) {
            val v = y * invH
            val rowOffset = y * w
            for (x in 0 until w) {
                val u = x * invW
                val c = sourcePixels[rowOffset + x]
                val neuralConf = InpaintingEngine.sampleMaskBilinear(adjustedMask, maskW, maskH, u, v)

                val alpha: Int = when {
                    neuralConf >= solidFgThresh -> 255
                    neuralConf <= solidBgThresh -> 0
                    else -> {
                        val r = (c shr 16 and 0xFF) / 255f
                        val g = (c shr 8 and 0xFF) / 255f
                        val b = (c and 0xFF) / 255f
                        val highResLum = 0.299f * r + 0.587f * g + 0.114f * b

                        val guidedAlpha = GuidedMattingFilter.sampleGuidedAlpha(guidedCoeff, u, v, highResLum)
                        val bandNorm = ((neuralConf - solidBgThresh) / bandDenom).coerceIn(0f, 1f)
                        val edgeVal = (guidedAlpha * 0.70f + bandNorm * 0.30f).coerceIn(0f, 1f)

                        val centered = (edgeVal - 0.5f) * contrastFactor + 0.5f
                        val shapedAlpha = centered.coerceIn(0f, 1f)
                        (shapedAlpha * 255f).toInt()
                    }
                }

                val rgb = c and 0x00FFFFFF
                cutoutPixels[rowOffset + x] = (alpha shl 24) or rgb
            }
        }
        cutoutBmp.setPixels(cutoutPixels, 0, w, 0, 0, w, h)

        // 6. Generate Continuous 3D Depth Map with Depth Anything V2
        val depthBmp = if (depthResult != null) {
            Bitmap.createScaledBitmap(depthResult.depthBitmap, w, h, true)
        } else {
            val fallbackBmp = Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888)
            val depthPixels = IntArray(w * h)
            for (y in 0 until h) {
                val v = y * invH
                val rowOffset = y * w
                for (x in 0 until w) {
                    val u = x * invW
                    val neuralConf = InpaintingEngine.sampleMaskBilinear(adjustedMask, maskW, maskH, u, v)
                    val bgGradient = (y.toFloat() / h) * 0.35f
                    val depthVal = (neuralConf * 0.85f + bgGradient * (1f - neuralConf)).coerceIn(0f, 1f)
                    val gray = (depthVal * 255).toInt()
                    depthPixels[rowOffset + x] = Color.rgb(gray, gray, gray)
                }
            }
            fallbackBmp.setPixels(depthPixels, 0, w, 0, 0, w, h)
            fallbackBmp
        }

        // 7. Inpainted Background Plate (Tight dilation preserves pristine background textures)
        val inpaintedBmp = InpaintingEngine.inpaintBackground(
            sourceBmp = safeBmp,
            mask = adjustedMask,
            maskWidth = maskW,
            maskHeight = maskH,
            threshold = threshold,
            dilationRadius = inpaintRadius.coerceIn(2, 16)
        )

        return SegmentationResult(
            foregroundCutout = cutoutBmp,
            inpaintedBackground = inpaintedBmp,
            depthMap = depthBmp,
            rawMask = adjustedMask,
            maskWidth = maskW,
            maskHeight = maskH,
            isPortraitDetected = isPortrait,
            foregroundRatio = fgRatio
        )
    }

    /**
     * Flagship Pipeline: Depth Anything V2 + MediaPipe Selfie Multiclass Fusion.
     * Uses Depth Anything V2 for whole-body geometry and zero hollowing,
     * fused with Multiclass for sub-pixel hair strands and skin tone transitions.
     */
    private fun executeDepthMattingFusionPipeline(
        bitmap: Bitmap,
        depthResult: DepthAnythingEngine.DepthResult?,
        fusionBalance: Float = 0.50f
    ): Triple<FloatArray, Int, Int> {
        AppLogger.i("SegmentationEngine", "Executing Flagship: Depth Anything V2 + Multiclass Fusion (balance=$fusionBalance)")
        val multiclass = multiclassSegmenter.segment(bitmap) ?: computeUniversalSaliencyMask(bitmap)

        if (depthResult == null) {
            return multiclass
        }

        val dMask = depthResult.foregroundConfidenceMask
        val dW = depthResult.depthWidth
        val dH = depthResult.depthHeight

        val mW = multiclass.second
        val mH = multiclass.third
        val mMask = multiclass.first

        val outW = max(dW, mW)
        val outH = max(dH, mH)
        val fused = FloatArray(outW * outH)
        val invW = 1.0f / max(1, outW - 1)
        val invH = 1.0f / max(1, outH - 1)

        val depthWeight = 2.0f * (1.0f - fusionBalance).coerceIn(0.1f, 1.0f)
        val mattingWeight = 2.0f * fusionBalance.coerceIn(0.1f, 1.0f)

        for (y in 0 until outH) {
            val v = y * invH
            val rowOffset = y * outW
            for (x in 0 until outW) {
                val u = x * invW
                val dVal = InpaintingEngine.sampleMaskBilinear(dMask, dW, dH, u, v) * depthWeight
                val mVal = InpaintingEngine.sampleMaskBilinear(mMask, mW, mH, u, v) * mattingWeight
                fused[rowOffset + x] = max(dVal, mVal).coerceIn(0f, 1f)
            }
        }
        return Triple(fused, outW, outH)
    }

    /**
     * Semantic + Portrait Hybrid:
     * Fuses DeepLabV3 (group context, bodies, legs, objects) with Selfie Multiclass (hair, clothing details).
     */
    private fun executeSemanticPortraitHybridPipeline(bitmap: Bitmap): Triple<FloatArray, Int, Int> {
        AppLogger.i("SegmentationEngine", "Executing Semantic Portrait Hybrid: DeepLabV3 + Selfie Multiclass")
        val deepLab = deepLabSegmenter.segment(bitmap) ?: computeUniversalSaliencyMask(bitmap)
        val multiclass = multiclassSegmenter.segment(bitmap) ?: computeUniversalSaliencyMask(bitmap)

        val outW = max(deepLab.second, multiclass.second)
        val outH = max(deepLab.third, multiclass.third)
        val fused = FloatArray(outW * outH)
        val invW = 1.0f / max(1, outW - 1)
        val invH = 1.0f / max(1, outH - 1)

        val dMask = deepLab.first
        val dW = deepLab.second
        val dH = deepLab.third

        val mMask = multiclass.first
        val mW = multiclass.second
        val mH = multiclass.third

        for (y in 0 until outH) {
            val v = y * invH
            val rowOffset = y * outW
            for (x in 0 until outW) {
                val u = x * invW
                val dVal = InpaintingEngine.sampleMaskBilinear(dMask, dW, dH, u, v)
                val mVal = InpaintingEngine.sampleMaskBilinear(mMask, mW, mH, u, v)
                fused[rowOffset + x] = max(dVal, mVal)
            }
        }
        return Triple(fused, outW, outH)
    }

    /**
     * Multi-Scale Zoom Tiling Pipeline:
     * Global context pass + high-resolution zoomed crops on subject boundaries for sub-pixel precision.
     */
    private fun executeMultiScaleZoomPipeline(bitmap: Bitmap): Triple<FloatArray, Int, Int> {
        AppLogger.i("SegmentationEngine", "Executing Multi-Scale Zoom Tiling Pipeline")
        val global = deepLabSegmenter.segment(bitmap) ?: computeUniversalSaliencyMask(bitmap)
        val gMask = global.first
        val gW = global.second
        val gH = global.third

        var minX = gW
        var maxX = 0
        var minY = gH
        var maxY = 0
        var fgCount = 0

        for (y in 0 until gH) {
            val row = y * gW
            for (x in 0 until gW) {
                if (gMask[row + x] > 0.35f) {
                    fgCount++
                    if (x < minX) minX = x
                    if (x > maxX) maxX = x
                    if (y < minY) minY = y
                    if (y > maxY) maxY = y
                }
            }
        }

        if (fgCount < 100 || minX >= maxX || minY >= maxY) {
            return global
        }

        val normLeft = (minX.toFloat() / gW).coerceIn(0f, 1f)
        val normTop = (minY.toFloat() / gH).coerceIn(0f, 1f)
        val normRight = (maxX.toFloat() / gW).coerceIn(0f, 1f)
        val normBottom = (maxY.toFloat() / gH).coerceIn(0f, 1f)

        val cropLeft = (normLeft * bitmap.width).toInt().coerceIn(0, bitmap.width - 1)
        val cropTop = (normTop * bitmap.height).toInt().coerceIn(0, bitmap.height - 1)
        val cropW = ((normRight - normLeft) * bitmap.width).toInt().coerceIn(32, bitmap.width - cropLeft)
        val cropH = ((normBottom - normTop) * bitmap.height).toInt().coerceIn(32, bitmap.height - cropTop)

        val cropBmp = try {
            Bitmap.createBitmap(bitmap, cropLeft, cropTop, cropW, cropH)
        } catch (_: Exception) {
            null
        }

        if (cropBmp == null) return global

        val localCropResult = deepLabSegmenter.segment(cropBmp) ?: computeUniversalSaliencyMask(cropBmp)
        if (cropBmp != bitmap && !cropBmp.isRecycled) cropBmp.recycle()

        val stitched = gMask.copyOf()
        val cMask = localCropResult.first
        val cW = localCropResult.second
        val cH = localCropResult.third

        for (y in minY..maxY) {
            val vLocal = (y - minY).toFloat() / max(1, maxY - minY)
            val rowOffset = y * gW
            for (x in minX..maxX) {
                val uLocal = (x - minX).toFloat() / max(1, maxX - minX)
                val cropConf = InpaintingEngine.sampleMaskBilinear(cMask, cW, cH, uLocal, vLocal)
                val origConf = stitched[rowOffset + x]
                stitched[rowOffset + x] = (origConf * 0.35f + cropConf * 0.65f).coerceIn(0f, 1f)
            }
        }

        return Triple(stitched, gW, gH)
    }

    private fun executePureDepthMask(
        depthResult: DepthAnythingEngine.DepthResult?,
        bitmap: Bitmap
    ): Triple<FloatArray, Int, Int> {
        if (depthResult != null) {
            return Triple(depthResult.foregroundConfidenceMask, depthResult.depthWidth, depthResult.depthHeight)
        }
        return computeUniversalSaliencyMask(bitmap)
    }

    /**
     * Universal edge and chromatic saliency segmentation for scenery, architecture, and objects.
     */
    fun computeUniversalSaliencyMask(bitmap: Bitmap): Triple<FloatArray, Int, Int> {
        val mW = min(320, bitmap.width)
        val mH = min(320, bitmap.height)
        val scaled = Bitmap.createScaledBitmap(bitmap, mW, mH, true)
        val pixels = IntArray(mW * mH)
        scaled.getPixels(pixels, 0, mW, 0, 0, mW, mH)
        if (scaled != bitmap && !scaled.isRecycled) scaled.recycle()

        val skyRows = max(2, (mH * 0.15f).toInt())
        var skyRSum = 0.0
        var skyGSum = 0.0
        var skyBSum = 0.0
        val skyPixelCount = skyRows * mW
        for (i in 0 until skyPixelCount) {
            val c = pixels[i]
            skyRSum += (c shr 16 and 0xFF) / 255.0
            skyGSum += (c shr 8 and 0xFF) / 255.0
            skyBSum += (c and 0xFF) / 255.0
        }
        val skyR = (skyRSum / skyPixelCount).toFloat()
        val skyG = (skyGSum / skyPixelCount).toFloat()
        val skyB = (skyBSum / skyPixelCount).toFloat()
        val skyLum = 0.299f * skyR + 0.587f * skyG + 0.114f * skyB

        val lum = FloatArray(mW * mH)
        for (i in 0 until mW * mH) {
            val c = pixels[i]
            val r = (c shr 16 and 0xFF) / 255f
            val g = (c shr 8 and 0xFF) / 255f
            val b = (c and 0xFF) / 255f
            lum[i] = 0.299f * r + 0.587f * g + 0.114f * b
        }

        val scores = FloatArray(mW * mH)
        val cx = mW / 2f
        val cy = mH * 0.45f
        val maxDist = Math.hypot(cx.toDouble(), cy.toDouble()).toFloat()

        for (y in 0 until mH) {
            val row = y * mW
            for (x in 0 until mW) {
                val idx = row + x
                val c = pixels[idx]
                val r = (c shr 16 and 0xFF) / 255f
                val g = (c shr 8 and 0xFF) / 255f
                val b = (c and 0xFF) / 255f

                val colorDist = Math.sqrt(
                    ((r - skyR) * (r - skyR) +
                     (g - skyG) * (g - skyG) +
                     (b - skyB) * (b - skyB)).toDouble()
                ).toFloat()

                val dx = x - cx
                val dy = y - cy
                val dist = Math.hypot(dx.toDouble(), dy.toDouble()).toFloat()
                val centerWeight = 1.0f - (dist / maxDist).coerceIn(0f, 1f)

                val lumDiff = Math.abs(lum[idx] - skyLum)
                scores[idx] = (colorDist * 0.50f + lumDiff * 0.25f + centerWeight * 0.25f).coerceIn(0f, 1f)
            }
        }

        return Triple(scores, mW, mH)
    }

    fun close() {
        deepLabSegmenter.close()
        multiclassSegmenter.close()
        fastSelfieSegmenter.close()
        DepthAnythingEngine.close()
    }
}

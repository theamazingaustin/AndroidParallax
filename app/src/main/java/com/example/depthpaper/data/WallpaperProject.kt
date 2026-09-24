package com.example.depthpaper.data

import org.json.JSONObject
import java.util.UUID

enum class RenderMode {
    LAYERED_2D,
    SPATIAL_3D
}

enum class ClockFontStyle {
    ROUNDED_BOLD,
    SERIF_CLASSIC,
    MODERN_HEAVY,
    ELEGANT_THIN,
    STENCIL_DISPLAY,
    CYBER_MONO
}

data class LockScreenConfig(
    val fontStyle: ClockFontStyle = ClockFontStyle.ROUNDED_BOLD,
    val clockColorHex: Long = 0xFFFFFFFF,
    val clockScale: Float = 1.0f,
    val verticalOffsetPercent: Float = 0.18f,
    val horizontalOffsetPercent: Float = 0.5f,
    val subjectInFrontOfClock: Boolean = true,
    val showDate: Boolean = true,
    val dateFormat: String = "EEEE, MMMM d",
    val showWidgets: Boolean = true
) {
    fun toJson(): JSONObject = JSONObject().apply {
        put("fontStyle", fontStyle.name)
        put("clockColorHex", clockColorHex)
        put("clockScale", clockScale.toDouble())
        put("verticalOffsetPercent", verticalOffsetPercent.toDouble())
        put("horizontalOffsetPercent", horizontalOffsetPercent.toDouble())
        put("subjectInFrontOfClock", subjectInFrontOfClock)
        put("showDate", showDate)
        put("dateFormat", dateFormat)
        put("showWidgets", showWidgets)
    }

    companion object {
        fun fromJson(json: JSONObject): LockScreenConfig = LockScreenConfig(
            fontStyle = runCatching { ClockFontStyle.valueOf(json.optString("fontStyle", "ROUNDED_BOLD")) }.getOrDefault(ClockFontStyle.ROUNDED_BOLD),
            clockColorHex = json.optLong("clockColorHex", 0xFFFFFFFF),
            clockScale = json.optDouble("clockScale", 1.0).toFloat(),
            verticalOffsetPercent = json.optDouble("verticalOffsetPercent", 0.18).toFloat(),
            horizontalOffsetPercent = json.optDouble("horizontalOffsetPercent", 0.5).toFloat(),
            subjectInFrontOfClock = json.optBoolean("subjectInFrontOfClock", true),
            showDate = json.optBoolean("showDate", true),
            dateFormat = json.optString("dateFormat", "EEEE, MMMM d"),
            showWidgets = json.optBoolean("showWidgets", true)
        )
    }
}

data class HomeScreenConfig(
    val hideClockOnHomeScreen: Boolean = true,
    val dimmingFactor: Float = 0.05f
) {
    fun toJson(): JSONObject = JSONObject().apply {
        put("hideClockOnHomeScreen", hideClockOnHomeScreen)
        put("dimmingFactor", dimmingFactor.toDouble())
    }

    companion object {
        fun fromJson(json: JSONObject): HomeScreenConfig = HomeScreenConfig(
            hideClockOnHomeScreen = json.optBoolean("hideClockOnHomeScreen", true),
            dimmingFactor = json.optDouble("dimmingFactor", 0.05).toFloat()
        )
    }
}

data class AodConfig(
    val enabled: Boolean = true,
    val monochromeOutline: Boolean = true,
    val oledBlackLevel: Float = 1.0f
) {
    fun toJson(): JSONObject = JSONObject().apply {
        put("enabled", enabled)
        put("monochromeOutline", monochromeOutline)
        put("oledBlackLevel", oledBlackLevel.toDouble())
    }

    companion object {
        fun fromJson(json: JSONObject): AodConfig = AodConfig(
            enabled = json.optBoolean("enabled", true),
            monochromeOutline = json.optBoolean("monochromeOutline", true),
            oledBlackLevel = json.optDouble("oledBlackLevel", 1.0).toFloat()
        )
    }
}

data class MotionConfig(
    val parallaxIntensity: Float = 0.6f,
    val invertX: Boolean = false,
    val invertY: Boolean = false,
    val sensorSmoothing: Float = 0.2f,
    val maxTiltAngle: Float = 15f,
    val swipeParallax: Boolean = true
) {
    fun toJson(): JSONObject = JSONObject().apply {
        put("parallaxIntensity", parallaxIntensity.toDouble())
        put("invertX", invertX)
        put("invertY", invertY)
        put("sensorSmoothing", sensorSmoothing.toDouble())
        put("maxTiltAngle", maxTiltAngle.toDouble())
        put("swipeParallax", swipeParallax)
    }

    companion object {
        fun fromJson(json: JSONObject): MotionConfig = MotionConfig(
            parallaxIntensity = json.optDouble("parallaxIntensity", 0.6).toFloat(),
            invertX = json.optBoolean("invertX", false),
            invertY = json.optBoolean("invertY", false),
            sensorSmoothing = json.optDouble("sensorSmoothing", 0.2).toFloat(),
            maxTiltAngle = json.optDouble("maxTiltAngle", 15.0).toFloat(),
            swipeParallax = json.optBoolean("swipeParallax", true)
        )
    }
}

data class WallpaperProject(
    val id: String = UUID.randomUUID().toString(),
    val title: String = "Untitled Wallpaper",
    val createdTimestamp: Long = System.currentTimeMillis(),
    val isFavorite: Boolean = false,
    val isActive: Boolean = false,
    val renderMode: RenderMode = RenderMode.LAYERED_2D,
    val sourceImagePath: String = "",
    val cutoutImagePath: String = "",
    val inpaintedBackgroundPath: String = "",
    val depthMapPath: String = "",
    val thumbnailPath: String = "",
    val threshold: Float = 0.5f,
    val edgeFeathering: Int = 6,
    val maskExpansion: Int = 0,
    val inpaintRadius: Int = 8,
    val motionConfig: MotionConfig = MotionConfig(),
    val lockScreenConfig: LockScreenConfig = LockScreenConfig(),
    val homeScreenConfig: HomeScreenConfig = HomeScreenConfig(),
    val aodConfig: AodConfig = AodConfig()
) {
    fun toJson(): JSONObject = JSONObject().apply {
        put("id", id)
        put("title", title)
        put("createdTimestamp", createdTimestamp)
        put("isFavorite", isFavorite)
        put("isActive", isActive)
        put("renderMode", renderMode.name)
        put("sourceImagePath", sourceImagePath)
        put("cutoutImagePath", cutoutImagePath)
        put("inpaintedBackgroundPath", inpaintedBackgroundPath)
        put("depthMapPath", depthMapPath)
        put("thumbnailPath", thumbnailPath)
        put("threshold", threshold.toDouble())
        put("edgeFeathering", edgeFeathering)
        put("maskExpansion", maskExpansion)
        put("inpaintRadius", inpaintRadius)
        put("motionConfig", motionConfig.toJson())
        put("lockScreenConfig", lockScreenConfig.toJson())
        put("homeScreenConfig", homeScreenConfig.toJson())
        put("aodConfig", aodConfig.toJson())
    }

    companion object {
        fun fromJson(json: JSONObject): WallpaperProject = WallpaperProject(
            id = json.optString("id", UUID.randomUUID().toString()),
            title = json.optString("title", "Wallpaper"),
            createdTimestamp = json.optLong("createdTimestamp", System.currentTimeMillis()),
            isFavorite = json.optBoolean("isFavorite", false),
            isActive = json.optBoolean("isActive", false),
            renderMode = runCatching { RenderMode.valueOf(json.optString("renderMode", "LAYERED_2D")) }.getOrDefault(RenderMode.LAYERED_2D),
            sourceImagePath = json.optString("sourceImagePath", ""),
            cutoutImagePath = json.optString("cutoutImagePath", ""),
            inpaintedBackgroundPath = json.optString("inpaintedBackgroundPath", ""),
            depthMapPath = json.optString("depthMapPath", ""),
            thumbnailPath = json.optString("thumbnailPath", ""),
            threshold = json.optDouble("threshold", 0.5).toFloat(),
            edgeFeathering = json.optInt("edgeFeathering", 6),
            maskExpansion = json.optInt("maskExpansion", 0),
            inpaintRadius = json.optInt("inpaintRadius", 8),
            motionConfig = json.optJSONObject("motionConfig")?.let { MotionConfig.fromJson(it) } ?: MotionConfig(),
            lockScreenConfig = json.optJSONObject("lockScreenConfig")?.let { LockScreenConfig.fromJson(it) } ?: LockScreenConfig(),
            homeScreenConfig = json.optJSONObject("homeScreenConfig")?.let { HomeScreenConfig.fromJson(it) } ?: HomeScreenConfig(),
            aodConfig = json.optJSONObject("aodConfig")?.let { AodConfig.fromJson(it) } ?: AodConfig()
        )
    }
}

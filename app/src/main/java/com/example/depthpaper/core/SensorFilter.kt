package com.example.depthpaper.core

import kotlin.math.max
import kotlin.math.min

/**
 * Sensor low-pass filter and tilt angle processor.
 * Fuses rotation vector inputs and applies exponential moving average (EMA)
 * smoothing with pitch/roll clamping to eliminate hand tremor.
 */
class SensorFilter(
    var smoothingFactor: Float = 0.2f, // 0.05 = heavy spring lag, 0.4 = snappy
    var maxAngleDegrees: Float = 15f,
    var adaptiveBaseline: Boolean = false
) {
    var filteredPitch: Float = 0f
        private set
    var filteredRoll: Float = 0f
        private set

    private var baselinePitch: Float? = null
    private var baselineRoll: Float? = null

    /**
     * Updates the filter with raw pitch and roll angles in degrees.
     * Clamps within [-maxAngleDegrees, maxAngleDegrees] and computes normalized [-1f, 1f] output.
     * When [adaptiveBaseline] is true, smoothly centers around the device's resting holding angle.
     * @return Pair(normalizedRoll, normalizedPitch) both clamped to [-1f, 1f].
     */
    fun update(rawRoll: Float, rawPitch: Float): Pair<Float, Float> {
        val inputRoll = if (adaptiveBaseline) {
            if (baselineRoll == null) baselineRoll = rawRoll
            baselineRoll = (baselineRoll ?: rawRoll) * 0.995f + rawRoll * 0.005f
            rawRoll - (baselineRoll ?: 0f)
        } else {
            rawRoll
        }

        val inputPitch = if (adaptiveBaseline) {
            if (baselinePitch == null) baselinePitch = rawPitch
            baselinePitch = (baselinePitch ?: rawPitch) * 0.995f + rawPitch * 0.005f
            rawPitch - (baselinePitch ?: 0f)
        } else {
            rawPitch
        }

        val clampedRoll = max(-maxAngleDegrees, min(maxAngleDegrees, inputRoll))
        val clampedPitch = max(-maxAngleDegrees, min(maxAngleDegrees, inputPitch))

        filteredRoll += (clampedRoll - filteredRoll) * smoothingFactor
        filteredPitch += (clampedPitch - filteredPitch) * smoothingFactor

        val normX = (filteredRoll / maxAngleDegrees).coerceIn(-1f, 1f)
        val normY = (filteredPitch / maxAngleDegrees).coerceIn(-1f, 1f)

        return Pair(normX, normY)
    }

    /**
     * Resets the filter state and recalibrates baseline back to center.
     */
    fun reset() {
        filteredPitch = 0f
        filteredRoll = 0f
        baselinePitch = null
        baselineRoll = null
    }
}

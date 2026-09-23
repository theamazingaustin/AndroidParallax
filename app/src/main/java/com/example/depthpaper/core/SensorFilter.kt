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
    var maxAngleDegrees: Float = 15f
) {
    var filteredPitch: Float = 0f
        private set
    var filteredRoll: Float = 0f
        private set

    /**
     * Updates the filter with raw pitch and roll angles in degrees.
     * Clamps within [-maxAngleDegrees, maxAngleDegrees] and computes normalized [-1f, 1f] output.
     * @return Pair(normalizedRoll, normalizedPitch) both clamped to [-1f, 1f].
     */
    fun update(rawRoll: Float, rawPitch: Float): Pair<Float, Float> {
        val clampedRoll = max(-maxAngleDegrees, min(maxAngleDegrees, rawRoll))
        val clampedPitch = max(-maxAngleDegrees, min(maxAngleDegrees, rawPitch))

        filteredRoll += (clampedRoll - filteredRoll) * smoothingFactor
        filteredPitch += (clampedPitch - filteredPitch) * smoothingFactor

        val normX = filteredRoll / maxAngleDegrees
        val normY = filteredPitch / maxAngleDegrees

        return Pair(normX, normY)
    }

    /**
     * Resets the filter state back to center.
     */
    fun reset() {
        filteredPitch = 0f
        filteredRoll = 0f
    }
}

package com.example.depthpaper.core

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class CoreEnginesTest {

    @Test
    fun testBoxFilterComputesCorrectAverages() {
        val w = 4
        val h = 4
        // Flat array of 1.0f
        val src = FloatArray(w * h) { 1.0f }
        val filtered = GuidedMattingFilter.boxFilter(src, w, h, 1)

        for (v in filtered) {
            assertEquals(1.0f, v, 0.001f)
        }
    }

    @Test
    fun testSensorFilterClampingAndSmoothing() {
        val filter = SensorFilter(smoothingFactor = 0.5f, maxAngleDegrees = 15f)

        // Raw input exceeds max angle 15 -> should clamp to 15
        val (normX, normY) = filter.update(rawRoll = 30f, rawPitch = -45f)

        // With smoothing 0.5: first step towards 15 is 7.5 -> 7.5 / 15 = 0.5f
        assertEquals(0.5f, normX, 0.01f)
        assertEquals(-0.5f, normY, 0.01f)

        // Second step towards 15: 7.5 + (15 - 7.5)*0.5 = 11.25 -> 11.25 / 15 = 0.75f
        val (step2X, step2Y) = filter.update(rawRoll = 30f, rawPitch = -45f)
        assertEquals(0.75f, step2X, 0.01f)
        assertEquals(-0.75f, step2Y, 0.01f)
    }
}

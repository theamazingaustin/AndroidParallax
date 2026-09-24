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

    @Test
    fun testAdaptiveBaselineCentersAtInitialHoldingAngle() {
        val filter = SensorFilter(smoothingFactor = 1.0f, maxAngleDegrees = 15f, adaptiveBaseline = true)
        // First step sets the baseline to (10f, -45f) -> delta is 0
        val (initX, initY) = filter.update(rawRoll = 10f, rawPitch = -45f)
        assertEquals(0f, initX, 0.01f)
        assertEquals(0f, initY, 0.01f)

        // Tilting 15 degrees right (from 10f to 25f)
        val (tiltX, _) = filter.update(rawRoll = 25f, rawPitch = -45f)
        assertTrue(tiltX > 0.8f)
    }

    @Test
    fun testGuidedFilterCoefficientsAndSampling() {
        val w = 8
        val h = 8
        val lum = FloatArray(w * h) { (it % 8) / 8.0f }
        val mask = FloatArray(w * h) { if (it < 32) 1.0f else 0.0f }

        val coeff = GuidedMattingFilter.computeCoefficientsFromLuminance(lum, mask, w, h, radius = 2)
        assertEquals(w, coeff.w)
        assertEquals(h, coeff.h)

        val alpha = GuidedMattingFilter.sampleGuidedAlpha(coeff, 0.5f, 0.1f, 0.5f)
        assertTrue(alpha in 0f..1f)
    }
}

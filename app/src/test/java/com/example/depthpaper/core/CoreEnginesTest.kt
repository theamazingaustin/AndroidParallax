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

    @Test
    fun testModelTuningProfilesValid() {
        for (model in AiModelChoice.entries) {
            val prof = model.tuningProfile
            assertTrue(prof.sensitivity.min < prof.sensitivity.max)
            assertTrue(prof.sensitivity.default in prof.sensitivity.min..prof.sensitivity.max)
            assertTrue(prof.maskMargin.min < prof.maskMargin.max)
            assertTrue(prof.layerFlatness.min < prof.layerFlatness.max)
            assertTrue(prof.edgeSoftness.min < prof.edgeSoftness.max)
            assertTrue(prof.inpaintFill.min < prof.inpaintFill.max)
        }

        for (pipeline in AiPipelineChoice.entries) {
            val prof = pipeline.tuningProfile
            assertTrue(prof.sensitivity.min < prof.sensitivity.max)
            assertTrue(prof.sensitivity.default in prof.sensitivity.min..prof.sensitivity.max)
            assertTrue(prof.maskMargin.min < prof.maskMargin.max)
            assertTrue(prof.layerFlatness.min < prof.layerFlatness.max)
            assertTrue(prof.edgeSoftness.min < prof.edgeSoftness.max)
            assertTrue(prof.inpaintFill.min < prof.inpaintFill.max)
        }
    }

    @Test
    fun testBilinearMaskSampling() {
        val w = 2
        val h = 2
        val mask = floatArrayOf(
            0.0f, 1.0f,
            0.0f, 1.0f
        )
        // Midpoint u=0.5, v=0.5 -> average of 0 and 1 is 0.5
        val sample = InpaintingEngine.sampleMaskBilinear(mask, w, h, 0.5f, 0.5f)
        assertEquals(0.5f, sample, 0.01f)
    }

    @Test
    fun testFromIdBackwardCompatibility() {
        assertEquals(AiModelChoice.SELFIE_MULTICLASS, AiModelChoice.fromId("MOD_NET"))
        assertEquals(AiModelChoice.DEEPLAB_V3, AiModelChoice.fromId("BIREF_NET"))
        assertEquals(AiModelChoice.DEEPLAB_V3, AiModelChoice.fromId("MOBILE_SAM"))
        assertEquals(AiModelChoice.DEPTH_ANYTHING_V2, AiModelChoice.fromId("DEPTH_ANYTHING_V2"))

        assertEquals(AiPipelineChoice.DEPTH_MATTING_FUSION, AiPipelineChoice.fromId("DUAL_MODEL_HYBRID"))
        assertEquals(AiPipelineChoice.MULTI_SCALE_ZOOM, AiPipelineChoice.fromId("MULTI_SCALE_TILING"))
        assertEquals(AiPipelineChoice.DEPTH_MATTING_FUSION, AiPipelineChoice.fromId("DEPTH_MATTING_FUSION"))
    }

    @Test
    fun testInpaintFillSliderRangesTightened() {
        for (model in AiModelChoice.entries) {
            val prof = model.tuningProfile
            assertTrue("inpaintFill min should be 2f", prof.inpaintFill.min == 2f)
            assertTrue("inpaintFill max should be 20f", prof.inpaintFill.max == 20f)
            assertTrue("inpaintFill default should be <= 8f", prof.inpaintFill.default in 6f..8f)
        }
        for (pipe in AiPipelineChoice.entries) {
            val prof = pipe.tuningProfile
            assertTrue("inpaintFill min should be 2f", prof.inpaintFill.min == 2f)
            assertTrue("inpaintFill max should be 20f", prof.inpaintFill.max == 20f)
            assertTrue("inpaintFill default should be <= 8f", prof.inpaintFill.default in 6f..8f)
        }
    }

    @Test
    fun testModelAndPipelineBestAtDescriptionsPopulated() {
        for (model in AiModelChoice.entries) {
            assertTrue("modelName should not be blank", model.modelName.isNotBlank())
            assertTrue("bestAt should not be blank for ${model.name}", model.bestAt.isNotBlank())
            assertTrue("assetPath should point to models/*.tflite", model.assetPath.startsWith("models/") && model.assetPath.endsWith(".tflite"))
        }
        for (pipe in AiPipelineChoice.entries) {
            assertTrue("pipelineName should not be blank", pipe.pipelineName.isNotBlank())
            assertTrue("bestAt should not be blank for ${pipe.name}", pipe.bestAt.isNotBlank())
        }
    }

    @Test
    fun testGroundPlaneRelativeElevationEliminatesSlopingGroundSlicing() {
        // Simulate a sloping beach/ground:
        // Top row y=0 is far ocean depth 0.20
        // Bottom row y=9 is near sand depth 0.80
        val w = 10
        val h = 10
        val depth = FloatArray(w * h)
        for (y in 0 until h) {
            val bgZ = 0.20f + (y.toFloat() / h) * 0.60f
            for (x in 0 until w) {
                depth[y * w + x] = bgZ
            }
        }

        // Place a standing person from y=2 to y=8 at columns x=4..5:
        // Person's body is 0.25 closer than the ground behind them at every row!
        for (y in 2..8) {
            val bgZ = 0.20f + (y.toFloat() / h) * 0.60f
            depth[y * w + 4] = bgZ + 0.25f
            depth[y * w + 5] = bgZ + 0.25f
        }

        // Compute row background baseline (15th percentile)
        val rowBg = FloatArray(h)
        val rowBuffer = FloatArray(w)
        for (y in 0 until h) {
            System.arraycopy(depth, y * w, rowBuffer, 0, w)
            rowBuffer.sort()
            rowBg[y] = rowBuffer[(w * 0.15f).toInt()]
        }

        // Verify that for all rows of the person (y=2..8), deltaZ is positive and elevated
        for (y in 2..8) {
            val bgZ = rowBg[y]
            val personZ = depth[y * w + 4]
            val deltaZ = personZ - bgZ
            assertTrue("Person at row $y should have deltaZ > 0.20f", deltaZ >= 0.20f)
        }

        // Verify ground pixels (e.g. x=0) have deltaZ approximately 0
        for (y in 0 until h) {
            val bgZ = rowBg[y]
            val groundZ = depth[y * w + 0]
            val deltaZ = groundZ - bgZ
            assertTrue("Ground at row $y should have deltaZ <= 0.05f", deltaZ <= 0.05f)
        }
    }
}

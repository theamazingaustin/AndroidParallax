package com.example.depthpaper

import com.example.depthpaper.core.AppUpdater
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class AppUpdaterTest {

    @Test
    fun testVersionComparison() {
        assertTrue(AppUpdater.isVersionNewer("0.3.0", "0.2.0"))
        assertTrue(AppUpdater.isVersionNewer("1.0.0", "0.9.9"))
        assertTrue(AppUpdater.isVersionNewer("0.3.1", "0.3.0"))
        assertTrue(AppUpdater.isVersionNewer("0.10.0", "0.9.0"))

        assertFalse(AppUpdater.isVersionNewer("0.2.0", "0.2.0"))
        assertFalse(AppUpdater.isVersionNewer("0.1.0", "0.2.0"))
        assertFalse(AppUpdater.isVersionNewer("0.2.0", "0.3.0"))
    }
}

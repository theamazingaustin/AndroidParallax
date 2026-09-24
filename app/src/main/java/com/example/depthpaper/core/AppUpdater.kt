package com.example.depthpaper.core

import android.content.Context
import android.content.Intent
import android.net.Uri
import android.os.Build
import android.provider.Settings
import androidx.core.content.FileProvider
import com.example.depthpaper.BuildConfig
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import org.json.JSONArray
import org.json.JSONObject
import java.io.File
import java.io.FileOutputStream
import java.net.HttpURLConnection
import java.net.URL

data class UpdateInfo(
    val versionName: String,
    val versionTag: String,
    val downloadUrl: String,
    val apkName: String,
    val apkSizeBytes: Long,
    val releaseNotes: String
)

sealed class UpdateCheckResult {
    data class UpdateAvailable(val updateInfo: UpdateInfo) : UpdateCheckResult()
    data class UpToDate(val currentVersion: String) : UpdateCheckResult()
    data class Error(val message: String) : UpdateCheckResult()
}

object AppUpdater {

    private const val GITHUB_REPO = "theamazingaustin/AndroidParallax"
    private const val RELEASES_URL = "https://api.github.com/repos/$GITHUB_REPO/releases"

    /**
     * Checks GitHub API for newer release than current BuildConfig.VERSION_NAME
     */
    suspend fun checkForUpdate(currentVersion: String = BuildConfig.VERSION_NAME): UpdateCheckResult = withContext(Dispatchers.IO) {
        try {
            AppLogger.i("AppUpdater", "Checking GitHub API for updates. Current version: $currentVersion")
            val url = URL(RELEASES_URL)
            val conn = (url.openConnection() as HttpURLConnection).apply {
                connectTimeout = 8000
                readTimeout = 8000
                setRequestProperty("User-Agent", "DepthPaper-App")
                setRequestProperty("Accept", "application/vnd.github+json")
            }

            if (conn.responseCode != 200) {
                AppLogger.w("AppUpdater", "GitHub API returned HTTP ${conn.responseCode}")
                return@withContext UpdateCheckResult.Error("HTTP ${conn.responseCode}")
            }

            val jsonStr = conn.inputStream.bufferedReader().use { it.readText() }
            val releasesArray = JSONArray(jsonStr)

            if (releasesArray.length() == 0) {
                return@withContext UpdateCheckResult.UpToDate(currentVersion)
            }

            var latestValidRelease: JSONObject? = null
            var apkUrl: String? = null
            var apkName: String? = null
            var apkSize: Long = 0L

            for (i in 0 until releasesArray.length()) {
                val rel = releasesArray.getJSONObject(i)
                if (rel.optBoolean("draft", false)) continue
                val tag = rel.optString("tag_name", "").trim()
                if (!tag.startsWith("v")) continue

                val assets = rel.optJSONArray("assets") ?: continue
                for (j in 0 until assets.length()) {
                    val asset = assets.getJSONObject(j)
                    val name = asset.optString("name", "")
                    if (name.endsWith(".apk", ignoreCase = true)) {
                        apkUrl = asset.optString("browser_download_url", "")
                        apkName = name
                        apkSize = asset.optLong("size", 0L)
                        latestValidRelease = rel
                        break
                    }
                }
                if (latestValidRelease != null) break
            }

            if (latestValidRelease == null || apkUrl.isNullOrEmpty() || apkName.isNullOrEmpty()) {
                return@withContext UpdateCheckResult.UpToDate(currentVersion)
            }

            val tagName = latestValidRelease.optString("tag_name", "").trim()
            val releaseNotes = latestValidRelease.optString("body", "")
            val cleanRemoteVersion = tagName.removePrefix("v").trim()
            val cleanCurrentVersion = currentVersion.removePrefix("v").trim()

            val newer = isVersionNewer(cleanRemoteVersion, cleanCurrentVersion)
            AppLogger.i("AppUpdater", "Remote tag: $tagName ($cleanRemoteVersion), current: $cleanCurrentVersion, isNewer: $newer")

            if (newer) {
                UpdateCheckResult.UpdateAvailable(
                    UpdateInfo(
                        versionName = cleanRemoteVersion,
                        versionTag = tagName,
                        downloadUrl = apkUrl,
                        apkName = apkName,
                        apkSizeBytes = apkSize,
                        releaseNotes = releaseNotes
                    )
                )
            } else {
                UpdateCheckResult.UpToDate(currentVersion)
            }
        } catch (e: Exception) {
            AppLogger.e("AppUpdater", "Error checking for update: ${e.message}", e)
            UpdateCheckResult.Error(e.localizedMessage ?: "Network error")
        }
    }

    /**
     * Compares semantic versioning (e.g. "0.3.0" vs "0.2.0")
     */
    fun isVersionNewer(remote: String, local: String): Boolean {
        if (remote == local) return false
        val rParts = remote.split('.').mapNotNull { it.toIntOrNull() }
        val lParts = local.split('.').mapNotNull { it.toIntOrNull() }
        val maxLen = maxOf(rParts.size, lParts.size)
        for (i in 0 until maxLen) {
            val r = rParts.getOrElse(i) { 0 }
            val l = lParts.getOrElse(i) { 0 }
            if (r > l) return true
            if (r < l) return false
        }
        return false
    }

    /**
     * Downloads APK from GitHub release and triggers native Android package installer
     */
    suspend fun downloadAndInstallApk(
        context: Context,
        update: UpdateInfo,
        onProgress: (Float) -> Unit,
        onError: (String) -> Unit
    ) = withContext(Dispatchers.IO) {
        val updatesDir = File(context.cacheDir, "updates").apply { mkdirs() }
        val targetApk = File(updatesDir, update.apkName)

        try {
            var downloadUrl = update.downloadUrl
            var connection: HttpURLConnection
            var redirects = 0

            // Follow HTTP redirects (GitHub release asset downloads redirect to S3 AWS)
            while (true) {
                connection = (URL(downloadUrl).openConnection() as HttpURLConnection).apply {
                    instanceFollowRedirects = true
                    connectTimeout = 15000
                    readTimeout = 30000
                    setRequestProperty("User-Agent", "DepthPaper-App")
                }
                val status = connection.responseCode
                if (status == HttpURLConnection.HTTP_MOVED_TEMP ||
                    status == HttpURLConnection.HTTP_MOVED_PERM ||
                    status == 307 || status == 308) {
                    val newUrl = connection.getHeaderField("Location")
                    if (newUrl != null && redirects < 5) {
                        downloadUrl = newUrl
                        redirects++
                        continue
                    }
                }
                break
            }

            val totalBytes = connection.contentLength.toLong().let { if (it > 0) it else update.apkSizeBytes }
            var downloadedBytes = 0L

            connection.inputStream.use { input ->
                FileOutputStream(targetApk).use { output ->
                    val buffer = ByteArray(32 * 1024)
                    var bytesRead: Int
                    while (input.read(buffer).also { bytesRead = it } != -1) {
                        output.write(buffer, 0, bytesRead)
                        downloadedBytes += bytesRead
                        if (totalBytes > 0) {
                            val prog = (downloadedBytes.toFloat() / totalBytes.toFloat()).coerceIn(0f, 1f)
                            withContext(Dispatchers.Main) {
                                onProgress(prog)
                            }
                        }
                    }
                    output.flush()
                }
            }

            withContext(Dispatchers.Main) {
                promptInstall(context, targetApk)
            }
        } catch (e: Exception) {
            e.printStackTrace()
            withContext(Dispatchers.Main) {
                onError("Download failed: ${e.localizedMessage ?: "Network error"}")
            }
        }
    }

    /**
     * Prompts Android package installer to install the downloaded APK
     */
    fun promptInstall(context: Context, apkFile: File) {
        // Android 8.0+ unknown sources permission check
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            if (!context.packageManager.canRequestPackageInstalls()) {
                val settingsIntent = Intent(Settings.ACTION_MANAGE_UNKNOWN_APP_SOURCES).apply {
                    data = Uri.parse("package:${context.packageName}")
                    addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)
                }
                context.startActivity(settingsIntent)
                return
            }
        }

        val apkUri = FileProvider.getUriForFile(
            context,
            "${context.packageName}.fileprovider",
            apkFile
        )

        val installIntent = Intent(Intent.ACTION_VIEW).apply {
            setDataAndType(apkUri, "application/vnd.android.package-archive")
            addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)
            addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
        }

        context.startActivity(installIntent)
    }
}

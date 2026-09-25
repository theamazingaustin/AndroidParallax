package com.example.depthpaper.data

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import org.json.JSONObject
import java.io.File
import java.io.FileOutputStream
import java.util.UUID

class ProjectRepository(val context: Context) {

    private val projectsDir = File(context.filesDir, "projects").apply { if (!exists()) mkdirs() }
    private val activeProjectFile = File(context.filesDir, "active_project.txt")

    fun getAllProjects(): List<WallpaperProject> {
        val projects = mutableListOf<WallpaperProject>()
        val dirs = projectsDir.listFiles { f -> f.isDirectory } ?: return emptyList()

        val activeId = getActiveProjectId()

        for (dir in dirs) {
            val metaFile = File(dir, "project.json")
            if (metaFile.exists()) {
                try {
                    val jsonStr = metaFile.readText()
                    val project = WallpaperProject.fromJson(JSONObject(jsonStr))
                    val withActiveFlag = project.copy(isActive = (project.id == activeId))
                    projects.add(withActiveFlag)
                } catch (e: Exception) {
                    e.printStackTrace()
                }
            }
        }

        // Sort: Active first, then Favorites, then newest to oldest
        return projects.sortedWith(
            compareByDescending<WallpaperProject> { it.isActive }
                .thenByDescending { it.isFavorite }
                .thenByDescending { it.createdTimestamp }
        )
    }

    fun getActiveProject(): WallpaperProject? {
        val id = getActiveProjectId() ?: return getAllProjects().firstOrNull()
        return getProjectById(id)
    }

    fun getProjectById(id: String): WallpaperProject? {
        val dir = File(projectsDir, id)
        val metaFile = File(dir, "project.json")
        if (!metaFile.exists()) return null
        return try {
            val jsonStr = metaFile.readText()
            val project = WallpaperProject.fromJson(JSONObject(jsonStr))
            project.copy(isActive = (project.id == getActiveProjectId()))
        } catch (_: Exception) {
            null
        }
    }

    fun getActiveProjectId(): String? {
        return if (activeProjectFile.exists()) activeProjectFile.readText().trim() else null
    }

    fun setActiveProject(id: String) {
        activeProjectFile.writeText(id)
        notifyWallpaperUpdated()
    }

    fun notifyWallpaperUpdated() {
        try {
            val intent = android.content.Intent("com.example.depthpaper.ACTION_WALLPAPER_UPDATED").apply {
                setPackage(context.packageName)
            }
            context.sendBroadcast(intent)
        } catch (_: Exception) {}
    }

    fun toggleFavorite(id: String): WallpaperProject? {
        val project = getProjectById(id) ?: return null
        val updated = project.copy(isFavorite = !project.isFavorite)
        saveProjectMetaOnly(updated)
        return updated
    }

    fun saveProject(
        project: WallpaperProject,
        sourceBmp: Bitmap? = null,
        cutoutBmp: Bitmap? = null,
        inpaintedBgBmp: Bitmap? = null,
        depthBmp: Bitmap? = null,
        rawDepthBmp: Bitmap? = null,
        thumbBmp: Bitmap? = null
    ): WallpaperProject {
        val dir = File(projectsDir, project.id).apply { if (!exists()) mkdirs() }

        var srcPath = project.sourceImagePath
        var cutPath = project.cutoutImagePath
        var bgPath = project.inpaintedBackgroundPath
        var depthPath = project.depthMapPath
        var thumbPath = project.thumbnailPath

        sourceBmp?.let {
            val f = File(dir, "source.png")
            saveBitmap(it, f)
            srcPath = f.absolutePath
        }
        cutoutBmp?.let {
            val f = File(dir, "cutout.png")
            saveBitmap(it, f)
            cutPath = f.absolutePath
        }
        inpaintedBgBmp?.let {
            val f = File(dir, "background.png")
            saveBitmap(it, f)
            bgPath = f.absolutePath
        }
        depthBmp?.let {
            val f = File(dir, "depth.png")
            saveBitmap(it, f)
            depthPath = f.absolutePath
        }
        rawDepthBmp?.let {
            val f = File(dir, "depth_raw.png")
            saveBitmap(it, f)
        }
        thumbBmp?.let {
            val f = File(dir, "thumbnail.png")
            saveBitmap(it, f)
            thumbPath = f.absolutePath
        }

        val updated = project.copy(
            sourceImagePath = srcPath,
            cutoutImagePath = cutPath,
            inpaintedBackgroundPath = bgPath,
            depthMapPath = depthPath,
            thumbnailPath = thumbPath
        )

        val metaFile = File(dir, "project.json")
        metaFile.writeText(updated.toJson().toString())

        // If no active project set yet, make this active
        if (getActiveProjectId() == null) {
            setActiveProject(updated.id)
        }

        return updated
    }

    fun saveCutoutOnly(projectId: String, cutoutBmp: Bitmap): String? {
        val dir = File(projectsDir, projectId)
        if (!dir.exists()) dir.mkdirs()
        val f = File(dir, "cutout.png")
        saveBitmap(cutoutBmp, f)
        if (getActiveProjectId() == projectId) {
            notifyWallpaperUpdated()
        }
        return f.absolutePath
    }

    fun saveRawDepthOnly(projectId: String, rawDepthBmp: Bitmap): String? {
        val dir = File(projectsDir, projectId)
        if (!dir.exists()) dir.mkdirs()
        val f = File(dir, "depth_raw.png")
        saveBitmap(rawDepthBmp, f)
        return f.absolutePath
    }

    fun loadRawDepthBitmap(projectId: String): Bitmap? {
        val f = File(File(projectsDir, projectId), "depth_raw.png")
        if (!f.exists()) return null
        return BitmapFactory.decodeFile(f.absolutePath)
    }

    fun saveProjectMetaOnly(project: WallpaperProject) {
        val dir = File(projectsDir, project.id)
        if (dir.exists()) {
            val metaFile = File(dir, "project.json")
            metaFile.writeText(project.toJson().toString())
            if (getActiveProjectId() == project.id) {
                notifyWallpaperUpdated()
            }
        }
    }

    fun deleteProject(id: String) {
        val dir = File(projectsDir, id)
        if (dir.exists()) {
            dir.deleteRecursively()
        }
        if (getActiveProjectId() == id) {
            val remaining = getAllProjects().firstOrNull()
            if (remaining != null) {
                setActiveProject(remaining.id)
            } else {
                activeProjectFile.delete()
            }
        }
    }

    fun duplicateProject(id: String): WallpaperProject? {
        val original = getProjectById(id) ?: return null
        val newId = UUID.randomUUID().toString()
        val origDir = File(projectsDir, id)
        val newDir = File(projectsDir, newId).apply { mkdirs() }

        origDir.listFiles()?.forEach { file ->
            if (file.name != "project.json") {
                file.copyTo(File(newDir, file.name), overwrite = true)
            }
        }

        val duplicated = original.copy(
            id = newId,
            title = "${original.title} (Copy)",
            createdTimestamp = System.currentTimeMillis(),
            isActive = false,
            isFavorite = false,
            sourceImagePath = File(newDir, "source.png").takeIf { it.exists() }?.absolutePath ?: "",
            cutoutImagePath = File(newDir, "cutout.png").takeIf { it.exists() }?.absolutePath ?: "",
            inpaintedBackgroundPath = File(newDir, "background.png").takeIf { it.exists() }?.absolutePath ?: "",
            depthMapPath = File(newDir, "depth.png").takeIf { it.exists() }?.absolutePath ?: "",
            thumbnailPath = File(newDir, "thumbnail.png").takeIf { it.exists() }?.absolutePath ?: ""
        )

        File(newDir, "project.json").writeText(duplicated.toJson().toString())
        return duplicated
    }

    fun loadBitmap(path: String): Bitmap? {
        if (path.isBlank()) return null
        val file = File(path)
        if (!file.exists()) return null
        return BitmapFactory.decodeFile(file.absolutePath)
    }

    private fun saveBitmap(bitmap: Bitmap, file: File) {
        FileOutputStream(file).use { out ->
            bitmap.compress(Bitmap.CompressFormat.PNG, 100, out)
        }
    }
}

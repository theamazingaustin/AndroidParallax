package com.example.depthpaper.core

import android.content.ClipData
import android.content.ClipboardManager
import android.content.Context
import android.os.Build
import android.util.Log
import android.widget.Toast
import com.example.depthpaper.BuildConfig
import java.io.File
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import java.util.concurrent.ConcurrentLinkedQueue

object AppLogger {
    private const val MAX_LOG_LINES = 600
    private val logBuffer = ConcurrentLinkedQueue<String>()
    private val timeFormat = SimpleDateFormat("HH:mm:ss.SSS", Locale.US)

    fun d(tag: String, message: String) = record("DEBUG", tag, message)
    fun i(tag: String, message: String) = record("INFO", tag, message)
    fun w(tag: String, message: String, tr: Throwable? = null) = record("WARN", tag, "$message ${tr?.stackTraceToString() ?: ""}")
    fun e(tag: String, message: String, tr: Throwable? = null) = record("ERROR", tag, "$message ${tr?.stackTraceToString() ?: ""}")

    private fun record(level: String, tag: String, message: String) {
        val timestamp = timeFormat.format(Date())
        val formatted = "[$timestamp] [$level/$tag] $message"

        when (level) {
            "DEBUG" -> Log.d(tag, message)
            "INFO" -> Log.i(tag, message)
            "WARN" -> Log.w(tag, message)
            "ERROR" -> Log.e(tag, message)
        }

        logBuffer.add(formatted)
        while (logBuffer.size > MAX_LOG_LINES) {
            logBuffer.poll()
        }
    }

    fun getDiagnosticReport(): String {
        return buildString {
            appendLine("=== DEPTHPAPER DIAGNOSTIC LOG ===")
            appendLine("Timestamp: ${Date()}")
            appendLine("App Version: ${BuildConfig.VERSION_NAME} (code ${BuildConfig.VERSION_CODE})")
            appendLine("Device: ${Build.MANUFACTURER} ${Build.MODEL} (${Build.PRODUCT})")
            appendLine("Android OS: SDK ${Build.VERSION.SDK_INT} (Android ${Build.VERSION.RELEASE})")
            appendLine("Hardware: ${Build.HARDWARE}, Board: ${Build.BOARD}")
            appendLine("=================================")
            for (line in logBuffer) {
                appendLine(line)
            }
            appendLine("=== END OF LOG ===")
        }
    }

    fun copyLogsToClipboard(context: Context) {
        val report = getDiagnosticReport()
        val clipboard = context.getSystemService(Context.CLIPBOARD_SERVICE) as? ClipboardManager
        if (clipboard != null) {
            val clip = ClipData.newPlainText("DepthPaper Diagnostic Logs", report)
            clipboard.setPrimaryClip(clip)
            Toast.makeText(context, "Diagnostic logs copied to clipboard!", Toast.LENGTH_SHORT).show()
        }

        try {
            val logDir = File(context.filesDir, "logs").apply { mkdirs() }
            File(logDir, "app.log").writeText(report)
        } catch (_: Exception) {}
    }
}

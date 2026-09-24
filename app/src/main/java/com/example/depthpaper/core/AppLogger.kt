package com.example.depthpaper.core

import android.util.Log

object AppLogger {
    fun d(tag: String, message: String) {
        Log.d(tag, message)
    }

    fun i(tag: String, message: String) {
        Log.i(tag, message)
    }

    fun w(tag: String, message: String, tr: Throwable? = null) {
        if (tr != null) Log.w(tag, message, tr) else Log.w(tag, message)
    }

    fun e(tag: String, message: String, tr: Throwable? = null) {
        if (tr != null) Log.e(tag, message, tr) else Log.e(tag, message)
    }
}

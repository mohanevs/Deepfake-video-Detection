package com.example.veritas

import android.app.Activity
import android.content.Context
import android.content.Intent
import android.content.pm.PackageManager
import android.media.projection.MediaProjectionManager
import androidx.annotation.NonNull
import io.flutter.embedding.android.FlutterActivity
import io.flutter.embedding.engine.FlutterEngine
import io.flutter.plugin.common.MethodChannel
import io.flutter.plugin.common.MethodChannel.Result

class MainActivity: FlutterActivity() {
    
    private val CHANNEL = "screen_record_channel"
    private lateinit var mediaProjectionManager: MediaProjectionManager
    private val REQUEST_CODE = 1000
    private var pendingResult: Result? = null
    private var recordServiceIntent: Intent? = null
    
    override fun configureFlutterEngine(@NonNull flutterEngine: FlutterEngine) {
        super.configureFlutterEngine(flutterEngine)
        
        mediaProjectionManager = getSystemService(Context.MEDIA_PROJECTION_SERVICE) as MediaProjectionManager
        
        MethodChannel(flutterEngine.dartExecutor.binaryMessenger, CHANNEL)
            .setMethodCallHandler { call, result ->
                when (call.method) {
                    "getInstalledApps" -> {
                        val apps = getInstalledApps()
                        result.success(apps)
                    }
                    "startRecording" -> {
                        pendingResult = result
                        startActivityForResult(
                            mediaProjectionManager.createScreenCaptureIntent(),
                            REQUEST_CODE
                        )
                    }
                    "stopRecording" -> {
                        stopRecordingService()
                        result.success(null)
                    }
                    "launchApp" -> {
                        val packageName = call.argument<String>("packageName")
                        launchApp(packageName, result)
                    }
                    else -> result.notImplemented()
                }
            }
    }
    
    private fun getInstalledApps(): List<Map<String, String>> {
        val packageManager = packageManager
        val packages = packageManager.getInstalledApplications(PackageManager.GET_META_DATA)
        val apps = mutableListOf<Map<String, String>>()
        
        for (app in packages) {
            if (packageManager.getLaunchIntentForPackage(app.packageName) != null) {
                val appName = packageManager.getApplicationLabel(app).toString()
                val packageName = app.packageName
                
                apps.add(mapOf(
                    "name" to appName,
                    "packageName" to packageName
                ))
            }
        }
        
        // Sort alphabetically
        apps.sortBy { it["name"] }
        
        return apps
    }
    
    private fun launchApp(packageName: String?, result: Result) {
        if (packageName == null) {
            result.error("INVALID", "Package name is null", null)
            return
        }
        
        try {
            val launchIntent = packageManager.getLaunchIntentForPackage(packageName)
            if (launchIntent != null) {
                startActivity(launchIntent)
                result.success(null)
            } else {
                result.error("NOT_FOUND", "App not found", null)
            }
        } catch (e: Exception) {
            result.error("ERROR", e.message, null)
        }
    }
    
    private fun stopRecordingService() {
        recordServiceIntent?.let {
            stopService(it)
        }
        val intent = Intent(this, ScreenRecordService::class.java)
        intent.action = "STOP_RECORDING"
        startService(intent)
    }
    
    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        if (requestCode == REQUEST_CODE && resultCode == Activity.RESULT_OK) {
            recordServiceIntent = Intent(this, ScreenRecordService::class.java).apply {
                putExtra("resultCode", resultCode)
                putExtra("data", data)
            }
            startForegroundService(recordServiceIntent)
            pendingResult?.success(null)
        } else {
            pendingResult?.error("CANCELLED", "User cancelled", null)
        }
        super.onActivityResult(requestCode, resultCode, data)
    }
}
package com.example.veritas

import android.app.Activity
import android.content.Context
import android.content.Intent
import android.media.projection.MediaProjectionManager
import androidx.annotation.NonNull
import io.flutter.embedding.android.FlutterActivity
import io.flutter.embedding.engine.FlutterEngine
import io.flutter.plugin.common.MethodChannel
import android.provider.Settings
import android.net.Uri

class MainActivity: FlutterActivity() {

    private val CHANNEL = "screen_record_channel"
    private lateinit var mediaProjectionManager: MediaProjectionManager
    private val REQUEST_CODE = 1000
    private fun requestOverlayPermission() {
    if (!Settings.canDrawOverlays(this)) {
        val intent = Intent(
            Settings.ACTION_MANAGE_OVERLAY_PERMISSION,
            Uri.parse("package:$packageName")
        )
        startActivity(intent)
    }
}

    override fun configureFlutterEngine(@NonNull flutterEngine: FlutterEngine) {
        super.configureFlutterEngine(flutterEngine)

        mediaProjectionManager =
            getSystemService(Context.MEDIA_PROJECTION_SERVICE) as MediaProjectionManager

        MethodChannel(flutterEngine.dartExecutor.binaryMessenger, CHANNEL)
            .setMethodCallHandler { call, result ->

                when (call.method) {
                    "startRecording" -> {
                        startActivityForResult(
                            mediaProjectionManager.createScreenCaptureIntent(),
                            REQUEST_CODE
                        )
                        result.success(null)
                    }

                    "stopRecording" -> {
                        stopService(Intent(this, ScreenRecordService::class.java))
                        result.success("Stopped")
                    }
                }
            }
    }

    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        if (requestCode == REQUEST_CODE && resultCode == Activity.RESULT_OK) {

            val serviceIntent = Intent(this, ScreenRecordService::class.java)
            serviceIntent.putExtra("resultCode", resultCode)
            serviceIntent.putExtra("data", data)

            startForegroundService(serviceIntent)
        }
        super.onActivityResult(requestCode, resultCode, data)
    }
}
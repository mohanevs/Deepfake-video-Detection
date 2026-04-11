package com.example.veritas

import android.app.*
import android.content.Context
import android.content.Intent
import android.graphics.Color
import android.graphics.PixelFormat
import android.graphics.drawable.GradientDrawable
import android.hardware.display.DisplayManager
import android.hardware.display.VirtualDisplay
import android.media.MediaRecorder
import android.media.projection.MediaProjection
import android.media.projection.MediaProjectionManager
import android.os.*
import android.util.Log
import android.view.*
import android.widget.*
import androidx.core.app.NotificationCompat
import okhttp3.*
import okhttp3.MediaType.Companion.toMediaTypeOrNull
import okhttp3.RequestBody.Companion.asRequestBody
import org.json.JSONObject
import java.io.File
import java.io.IOException
import java.text.SimpleDateFormat
import java.util.*
import java.util.concurrent.TimeUnit
import android.app.NotificationChannel
import android.app.NotificationManager

class ScreenRecordService : Service() {

    private var mediaProjection: MediaProjection? = null
    private var mediaRecorder: MediaRecorder? = null
    private var virtualDisplay: VirtualDisplay? = null
    private var videoPath: String = ""
    private var windowManager: WindowManager? = null
    private var liveOverlay: View? = null
    private var statusTextView: TextView? = null
    private var greenDotView: View? = null

    // ── DEBUG OVERLAY VIEWS ───────────────────────────────────────────────────
    private var debugOverlay: View? = null
    private var debugStateText: TextView? = null   // current state label
    private var debugTimerText: TextView? = null   // live countdown / elapsed
    private var debugCycleText: TextView? = null   // cycle number
    private var debugLogText: TextView? = null     // rolling 3-line log
    private var debugCycleCount = 0
    private val debugLogLines = ArrayDeque<String>(4)
    
private val stopRecordingRunnable = Runnable {
    if (isRecording) stopRecordingAndSendToAPI()
}


private fun fullStop() {
    stopEverything()
    virtualDisplay?.release()
    virtualDisplay = null
    mediaProjection?.stop()
    mediaProjection = null
}


    // Countdown ticker while recording (counts down from 20)
    private var recordSecondsLeft = (RECORDING_DURATION / 1000L).toInt()
    private val recordCountdownRunnable = object : Runnable {
        override fun run() {
            if (!isRecording) return
            debugTimerText?.text = "⏱ recording: ${recordSecondsLeft}s left"
            if (recordSecondsLeft > 0) {
                recordSecondsLeft--
                mainHandler.postDelayed(this, 1000)
            }
        }
    }

    // Elapsed ticker while waiting for API response
    private var apiElapsedSecs = 0
    private val apiElapsedRunnable = object : Runnable {
        override fun run() {
            if (!isProcessing) return
            apiElapsedSecs++
            debugTimerText?.text = "⏳ waiting API: ${apiElapsedSecs}s"
            mainHandler.postDelayed(this, 1000)
        }
    }

    // Countdown ticker while showing result (counts down from 3)
    private var resultSecondsLeft = (RESULT_DISPLAY_DURATION / 1000L).toInt()
    private val resultCountdownRunnable = object : Runnable {
        override fun run() {
            if (resultSecondsLeft <= 0) return
            debugTimerText?.text = "👁 showing result: ${resultSecondsLeft}s"
            resultSecondsLeft--
            mainHandler.postDelayed(this, 1000)
        }
    }



    // ─────────────────────────────────────────────────────────────────────────

    private val mainHandler = Handler(Looper.getMainLooper())

    private var isRecording = false
    private var isProcessing = false
    private var shouldContinueLoop = true

    companion object {
        private const val RECORDING_DURATION = 20000L
        private const val RESULT_DISPLAY_DURATION = 3000L
    }

    private val API_URL = "http://10.57.70.12:8000/predict"

    private val client = OkHttpClient.Builder()
        .connectTimeout(30, TimeUnit.SECONDS)
        .readTimeout(60, TimeUnit.SECONDS)
        .writeTimeout(60, TimeUnit.SECONDS)
        .build()

    private val projectionCallback = object : MediaProjection.Callback() {
        override fun onStop() {
            if (!shouldContinueLoop) { fullStop(); stopSelf() }
        }
    }

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        if (intent?.action == "STOP_RECORDING") {
            shouldContinueLoop = false
            fullStop()
            stopSelf()
            return START_NOT_STICKY
        }

        val resultCode = intent?.getIntExtra("resultCode", Activity.RESULT_CANCELED)
            ?: return START_NOT_STICKY
        val data = intent.getParcelableExtra<Intent>("data")
            ?: return START_NOT_STICKY

        val channelId = "ScreenRecordServiceChannel"
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            val channel = NotificationChannel(channelId, "Screen Recording Service", NotificationManager.IMPORTANCE_LOW)
            getSystemService(NotificationManager::class.java)?.createNotificationChannel(channel)
        }
        startForeground(1, NotificationCompat.Builder(this, channelId)
            .setContentTitle("Screen Recording Active")
            .setContentText("Recording screen for deepfake verification.")
            .setSmallIcon(android.R.drawable.ic_media_play)
            .setPriority(NotificationCompat.PRIORITY_LOW)
            .build())

        val projectionManager = getSystemService(Context.MEDIA_PROJECTION_SERVICE) as MediaProjectionManager
        mediaProjection = projectionManager.getMediaProjection(resultCode, data)
        mediaProjection?.registerCallback(projectionCallback, null)

        windowManager = getSystemService(Context.WINDOW_SERVICE) as WindowManager
        showLiveOverlay()

        startRecordingLoop()

        return START_STICKY
    }

    

    private fun startRecordingLoop() {
        if (!shouldContinueLoop) return
        if (isRecording || isProcessing) {
            Log.w("ScreenRecord", "startRecordingLoop called while busy — skipping")
            return
        }

        debugCycleCount++
        mainHandler.post { debugCycleText?.text = "CYCLE: $debugCycleCount" }


        startRecording()

        // start the recording countdown in debug panel
    

        // schedule automatic stop after RECORDING_DURATION
        mainHandler.removeCallbacks(stopRecordingRunnable)
    mainHandler.postDelayed(stopRecordingRunnable, RECORDING_DURATION)
    }

    // ── LIVE OVERLAY ──────────────────────────────────────────────────────────
    private fun showLiveOverlay() {
       

        val container = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
            gravity = Gravity.CENTER
            setPadding(24, 12, 24, 12)
            background = GradientDrawable().apply {
                shape = GradientDrawable.RECTANGLE
                cornerRadius = 30f
                setColor(Color.parseColor("#CC000000"))
            }
        }

        greenDotView = View(this).apply {
            layoutParams = LinearLayout.LayoutParams(16, 16).apply {
                gravity = Gravity.CENTER_VERTICAL
                setMargins(0, 0, 12, 0)
            }
            background = GradientDrawable().apply { shape = GradientDrawable.OVAL; setColor(Color.parseColor("#4CAF50")) }
        }

        statusTextView = TextView(this).apply {
            text = "LIVE"; textSize = 20f
            setTextColor(Color.parseColor("#4CAF50"))
            typeface = android.graphics.Typeface.DEFAULT_BOLD
            gravity = Gravity.CENTER
        }

        container.addView(greenDotView)
        container.addView(statusTextView)

        val params = WindowManager.LayoutParams(
            WindowManager.LayoutParams.WRAP_CONTENT, WindowManager.LayoutParams.WRAP_CONTENT,
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) WindowManager.LayoutParams.TYPE_APPLICATION_OVERLAY
            else @Suppress("DEPRECATION") WindowManager.LayoutParams.TYPE_PHONE,
            WindowManager.LayoutParams.FLAG_NOT_FOCUSABLE or WindowManager.LayoutParams.FLAG_NOT_TOUCHABLE,
            PixelFormat.TRANSLUCENT
        ).apply { gravity = Gravity.TOP or Gravity.CENTER_HORIZONTAL; y = 80 }

        windowManager?.addView(container, params)
        liveOverlay = container
        startBlinkingAnimation()
    }

    private val blinkRunnable = object : Runnable {
        override fun run() {
            val dot = greenDotView ?: return
            if (statusTextView?.text == "LIVE") {
                dot.animate().alpha(if (dot.alpha == 1.0f) 0.3f else 1.0f).setDuration(500).start()
                mainHandler.postDelayed(this, 500)
            }
        }
    }

    private fun startBlinkingAnimation() {
        mainHandler.removeCallbacks(blinkRunnable)
        greenDotView?.alpha = 1.0f
        mainHandler.post(blinkRunnable)
    }

    private fun updateLiveOverlayWithResult(result: String) {
        val isReal = result.equals("REAL", ignoreCase = true)
        mainHandler.removeCallbacks(blinkRunnable)
        greenDotView?.animate()?.cancel()
        greenDotView?.alpha = 1.0f
        if (isReal) {
            setDotColor("#4CAF50"); statusTextView?.text = "✓ REAL"
            statusTextView?.setTextColor(Color.parseColor("#4CAF50"))
        } else {
            setDotColor("#F44336"); statusTextView?.text = "✗ FAKE"
            statusTextView?.setTextColor(Color.parseColor("#F44336"))
        }
        statusTextView?.textSize = 20f
    }

    private fun resetOverlayToLive() {
        setDotColor("#4CAF50")
        statusTextView?.text = "LIVE"
        statusTextView?.setTextColor(Color.parseColor("#4CAF50"))
        statusTextView?.textSize = 20f
        startBlinkingAnimation()
    }

    private fun updateLiveOverlayWithProcessing() {
        mainHandler.removeCallbacks(blinkRunnable)
        greenDotView?.animate()?.cancel()
        greenDotView?.alpha = 1.0f
        setDotColor("#FF9800")
        statusTextView?.text = "⟳"
        statusTextView?.setTextColor(Color.parseColor("#FF9800"))
        statusTextView?.textSize = 24f
    }

    private fun setDotColor(hex: String) {
        greenDotView?.background = GradientDrawable().apply { shape = GradientDrawable.OVAL; setColor(Color.parseColor(hex)) }
    }
    // ─────────────────────────────────────────────────────────────────────────

    private fun startRecording() {
        if (isRecording) return
        isRecording = true
        val metrics = resources.displayMetrics
        mediaRecorder = MediaRecorder()
        val timestamp = SimpleDateFormat("yyyyMMdd_HHmmss", Locale.US).format(Date())
        videoPath = "${getExternalFilesDir(null)}/recorded_$timestamp.mp4"
        try {
            mediaRecorder?.apply {
                setVideoSource(MediaRecorder.VideoSource.SURFACE)
                setOutputFormat(MediaRecorder.OutputFormat.MPEG_4)
                setOutputFile(videoPath)
                setVideoSize(metrics.widthPixels, metrics.heightPixels)
                setVideoEncoder(MediaRecorder.VideoEncoder.H264)
                setVideoFrameRate(30)
                setVideoEncodingBitRate(4 * 1024 * 1024)
                prepare()
            }
            val surface = mediaRecorder!!.surface

        if (virtualDisplay == null) {
            // First cycle — create the VirtualDisplay
            virtualDisplay = mediaProjection?.createVirtualDisplay(
                "ScreenCapture", metrics.widthPixels, metrics.heightPixels, metrics.densityDpi,
                DisplayManager.VIRTUAL_DISPLAY_FLAG_AUTO_MIRROR, surface, null, null
            )
        } else {
            // Subsequent cycles — just swap the surface, keep the same VirtualDisplay
            virtualDisplay?.surface = surface
        }
            mediaRecorder?.start()
        } catch (e: Exception) {
            Log.e("ScreenRecord", "Recording start error: ${e.message}")
        
            isRecording = false
        }
    }

    private fun stopRecordingAndSendToAPI() {
        if (!isRecording) return
        isRecording = false
        isProcessing = true

        mainHandler.removeCallbacks(stopRecordingRunnable)
        mainHandler.removeCallbacks(recordCountdownRunnable)
      
        updateLiveOverlayWithProcessing()

        // start API elapsed timer in debug panel
       

        try {
            mediaRecorder?.apply { try { stop() } catch (e: Exception) {} ; release() }
            mediaRecorder = null
            
            sendVideoToAPI(videoPath)
        } catch (e: Exception) {
            Log.e("ScreenRecord", "Stop error: ${e.message}")
        
            isProcessing = false
            scheduleNextLoop("ERROR")
        }
    }

    private fun sendVideoToAPI(videoPath: String) {
        val file = File(videoPath)
        if (!file.exists()) {
      
            isProcessing = false; scheduleNextLoop("FAKE"); return
        }
       

        val request = Request.Builder().url(API_URL)
            .post(MultipartBody.Builder().setType(MultipartBody.FORM)
                .addFormDataPart("file", file.name, file.asRequestBody("video/mp4".toMediaTypeOrNull()))
                .build())
            .build()

        client.newCall(request).enqueue(object : Callback {
            override fun onFailure(call: Call, e: IOException) {
                mainHandler.post {
                    mainHandler.removeCallbacks(apiElapsedRunnable)
                   
                    updateLiveOverlayWithResult("FAKE")
                    isProcessing = false; file.delete()
                    scheduleNextLoop("FAKE")
                }
            }

            override fun onResponse(call: Call, response: Response) {
                val resultText = parseResponse(response)
                mainHandler.post {
                    mainHandler.removeCallbacks(apiElapsedRunnable)
                    
                    updateLiveOverlayWithResult(resultText)
                    isProcessing = false; file.delete()
                    scheduleNextLoop(resultText)
                }
            }
        })
    }

    private fun scheduleNextLoop(result: String) {
        if (!shouldContinueLoop) return


        mainHandler.postDelayed({
          
            if (shouldContinueLoop && !isRecording && !isProcessing) {
          
                resetOverlayToLive()
                mainHandler.postDelayed({
                    if (shouldContinueLoop) startRecordingLoop()
                }, 300)
            }
        }, RESULT_DISPLAY_DURATION)
    }

    private fun parseResponse(response: Response): String {
        return try {
            val body = response.body?.string()
            when {
                body == null -> "FAKE"
                body.trim().startsWith("{") -> {
                    val json = JSONObject(body)
                    when (json.optInt("prediction", -1)) {
                        0 -> "REAL"; 1 -> "FAKE"
                        else -> if (body.contains(": 0") || body.contains("\"0\"")) "REAL" else "FAKE"
                    }
                }
                else -> when (body.trim()) {
                    "0" -> "REAL"; "1" -> "FAKE"
                    else -> if (response.isSuccessful) "REAL" else "FAKE"
                }
            }
        } catch (e: Exception) { "FAKE" }
    }

    private fun stopEverything() {
        shouldContinueLoop = false; isRecording = false; isProcessing = false
        mainHandler.removeCallbacksAndMessages(null)
        try {
            mediaRecorder?.apply { try { stop() } catch (e: Exception) {}; release() }
            mediaRecorder = null
            
        } catch (e: Exception) { Log.e("ScreenRecord", "Cleanup error: ${e.message}") }
    }

    override fun onDestroy() {
        super.onDestroy()
        fullStop()
        liveOverlay?.let { try { windowManager?.removeView(it) } catch (e: Exception) {} }
        
        liveOverlay = null; 
        client.dispatcher.executorService.shutdown()
    }

    override fun onBind(intent: Intent?): IBinder? = null
}
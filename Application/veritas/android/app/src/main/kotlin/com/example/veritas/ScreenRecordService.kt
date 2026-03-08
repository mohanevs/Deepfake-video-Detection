package com.example.veritas

import android.app.*
import android.content.Context
import android.content.Intent
import android.graphics.PixelFormat
import android.hardware.display.DisplayManager
import android.hardware.display.VirtualDisplay
import android.media.MediaRecorder
import android.media.projection.MediaProjection
import android.media.projection.MediaProjectionManager
import android.os.Build
import android.os.Environment
import android.os.IBinder
import android.view.*
import android.widget.ImageView
import androidx.core.app.NotificationCompat

class ScreenRecordService : Service() {

    private var mediaProjection: MediaProjection? = null
    private var mediaRecorder: MediaRecorder? = null
    private var virtualDisplay: VirtualDisplay? = null
    private var videoPath: String = ""

    private var windowManager: WindowManager? = null
    private var floatingView: View? = null

    private val projectionCallback = object : MediaProjection.Callback() {
        override fun onStop() {
            stopRecording()
            stopSelf()
        }
    }

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {

        // 🔴 Stop from notification or overlay
        if (intent?.action == "STOP_RECORDING") {
            stopRecording()
            stopSelf()
            return START_NOT_STICKY
        }

        val resultCode =
            intent?.getIntExtra("resultCode", Activity.RESULT_CANCELED)
                ?: return START_NOT_STICKY

        val data =
            intent.getParcelableExtra<Intent>("data")
                ?: return START_NOT_STICKY

        startForegroundService()

        val projectionManager =
            getSystemService(Context.MEDIA_PROJECTION_SERVICE) as MediaProjectionManager

        mediaProjection = projectionManager.getMediaProjection(resultCode, data)
        mediaProjection?.registerCallback(projectionCallback, null)

        startRecording()

        return START_STICKY
    }

    private fun startForegroundService() {

        val channelId = "screen_record_channel"

        val channel = NotificationChannel(
            channelId,
            "Screen Recording",
            NotificationManager.IMPORTANCE_LOW
        )

        val manager = getSystemService(NotificationManager::class.java)
        manager.createNotificationChannel(channel)

        val stopIntent = Intent(this, ScreenRecordService::class.java)
        stopIntent.action = "STOP_RECORDING"

        val stopPendingIntent = PendingIntent.getService(
            this,
            0,
            stopIntent,
            PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE
        )

        val notification = NotificationCompat.Builder(this, channelId)
            .setContentTitle("Recording Screen")
            .setContentText("Tap Stop to end recording")
            .setSmallIcon(android.R.drawable.ic_btn_speak_now)
            .addAction(
                android.R.drawable.ic_media_pause,
                "Stop",
                stopPendingIntent
            )
            .setOngoing(true)
            .build()

        startForeground(1, notification)
    }

    private fun startRecording() {

        val metrics = resources.displayMetrics
        val width = metrics.widthPixels
        val height = metrics.heightPixels
        val density = metrics.densityDpi

        mediaRecorder = MediaRecorder()

        videoPath = Environment.getExternalStoragePublicDirectory(
            Environment.DIRECTORY_DOWNLOADS
        ).absolutePath + "/recorded_${System.currentTimeMillis()}.mp4"

        mediaRecorder?.apply {
            setVideoSource(MediaRecorder.VideoSource.SURFACE)
            setOutputFormat(MediaRecorder.OutputFormat.MPEG_4)
            setOutputFile(videoPath)
            setVideoSize(width, height)
            setVideoEncoder(MediaRecorder.VideoEncoder.H264)
            setVideoFrameRate(30)
            setVideoEncodingBitRate(8 * 1024 * 1024)
            prepare()
        }

        virtualDisplay = mediaProjection?.createVirtualDisplay(
            "ScreenCapture",
            width,
            height,
            density,
            DisplayManager.VIRTUAL_DISPLAY_FLAG_AUTO_MIRROR,
            mediaRecorder?.surface,
            null,
            null
        )

        mediaRecorder?.start()

        showFloatingBubble()
    }

    private fun showFloatingBubble() {

        windowManager = getSystemService(WINDOW_SERVICE) as WindowManager

       val imageView = ImageView(this)

imageView.setImageResource(android.R.drawable.ic_media_pause)

imageView.setBackgroundResource(android.R.drawable.btn_default)

imageView.setPadding(60,60,60,60)

imageView.elevation = 10f

        val params = WindowManager.LayoutParams(
            WindowManager.LayoutParams.WRAP_CONTENT,
            WindowManager.LayoutParams.WRAP_CONTENT,
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O)
                WindowManager.LayoutParams.TYPE_APPLICATION_OVERLAY
            else
                WindowManager.LayoutParams.TYPE_PHONE,
            WindowManager.LayoutParams.FLAG_NOT_FOCUSABLE,
            PixelFormat.TRANSLUCENT
        )

        params.gravity = Gravity.TOP or Gravity.START
       params.x = 900   // horizontal position
params.y = 1600

        imageView.setOnClickListener {
            stopRecording()
            stopSelf()
        }

imageView.setOnTouchListener(object : View.OnTouchListener {

    private var initialX = 0
    private var initialY = 0
    private var initialTouchX = 0f
    private var initialTouchY = 0f

    override fun onTouch(view: View, event: MotionEvent): Boolean {

        when (event.action) {

            MotionEvent.ACTION_DOWN -> {
                initialX = params.x
                initialY = params.y
                initialTouchX = event.rawX
                initialTouchY = event.rawY
                return true
            }

            MotionEvent.ACTION_MOVE -> {
                params.x = initialX + (event.rawX - initialTouchX).toInt()
                params.y = initialY + (event.rawY - initialTouchY).toInt()

                windowManager?.updateViewLayout(imageView, params)
                return true
            }

            MotionEvent.ACTION_UP -> {

                // Detect simple click (no movement)
                val diffX = Math.abs(event.rawX - initialTouchX)
                val diffY = Math.abs(event.rawY - initialTouchY)

                if (diffX < 10 && diffY < 10) {
                    // It's a click
                    stopRecording()
                    stopSelf()
                }

                return true
            }
        }
        return false
    }
})

        windowManager?.addView(imageView, params)
        floatingView = imageView
    }

    private fun stopRecording() {
        try {
            mediaRecorder?.stop()
            mediaRecorder?.release()
            virtualDisplay?.release()
            mediaProjection?.stop()

            floatingView?.let {
                windowManager?.removeView(it)
                floatingView = null
            }

        } catch (e: Exception) {
            e.printStackTrace()
        }
    }

    override fun onBind(intent: Intent?): IBinder? = null
}


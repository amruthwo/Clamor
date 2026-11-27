package io.github.amruthwo.Clamor

import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.app.Service
import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent
import android.content.IntentFilter
import android.content.pm.ServiceInfo
import android.net.wifi.WifiManager
import android.os.BatteryManager
import android.os.Build
import android.os.IBinder
import android.os.PowerManager
import java.io.File

/**
 * Keeps Clamor in the swarm while the app is in the background: a foreground service with a
 * notification (clamor-node itself runs inside the app's process). Also:
 *  - lets mDNS (finding the swarm on Wi-Fi) through: Android drops multicast without a lock;
 *  - keeps the CPU awake only while the phone charges, so it can help the swarm overnight
 *    without draining the battery;
 *  - tells clamor-node the battery state (power.txt): apps may not read the kernel's battery
 *    files on Android, and Clamor pauses a phone that is on battery unless told otherwise.
 */
class ClamorService : Service() {
  private var multicast: WifiManager.MulticastLock? = null
  private var wake: PowerManager.WakeLock? = null

  private val battery = object : BroadcastReceiver() {
    override fun onReceive(context: Context, intent: Intent) {
      val level = intent.getIntExtra(BatteryManager.EXTRA_LEVEL, -1)
      val scale = intent.getIntExtra(BatteryManager.EXTRA_SCALE, 100)
      val plugged = intent.getIntExtra(BatteryManager.EXTRA_PLUGGED, 0) != 0
      val pct = if (level >= 0 && scale > 0) level * 100 / scale else 0
      try {
        // Same place as the app's data folder on the Rust side (app_data_dir: dataDir/<app id>).
        val dir = File(applicationInfo.dataDir, packageName)
        dir.mkdirs()
        File(dir, "power.txt").writeText("$pct ${if (plugged) 1 else 0}\n")
      } catch (_: Exception) {}
      if (plugged) holdWake() else releaseWake()
    }
  }

  override fun onCreate() {
    super.onCreate()
    val nm = getSystemService(NotificationManager::class.java)
    nm.createNotificationChannel(NotificationChannel(CHANNEL, "Clamor in the background", NotificationManager.IMPORTANCE_LOW).apply {
      description = "Shown while Clamor keeps this device in your swarm"
    })
    val open = PendingIntent.getActivity(this, 0, Intent(this, MainActivity::class.java), PendingIntent.FLAG_IMMUTABLE)
    val n: Notification = Notification.Builder(this, CHANNEL)
      .setContentTitle("Clamor")
      .setContentText("Part of your swarm. It helps only while charging, unless you change that.")
      .setSmallIcon(R.mipmap.ic_launcher)
      .setContentIntent(open)
      .setOngoing(true)
      .build()
    if (Build.VERSION.SDK_INT >= 34) {
      startForeground(1, n, ServiceInfo.FOREGROUND_SERVICE_TYPE_SPECIAL_USE)
    } else {
      startForeground(1, n)
    }
    val wifi = applicationContext.getSystemService(Context.WIFI_SERVICE) as WifiManager
    multicast = wifi.createMulticastLock("clamor-mdns").apply { setReferenceCounted(false); acquire() }
    // The current state arrives at once (the battery broadcast is sticky), then on every change.
    registerReceiver(battery, IntentFilter(Intent.ACTION_BATTERY_CHANGED))
  }

  private fun holdWake() {
    if (wake?.isHeld == true) return
    val pm = getSystemService(Context.POWER_SERVICE) as PowerManager
    wake = pm.newWakeLock(PowerManager.PARTIAL_WAKE_LOCK, "clamor:charging").apply { setReferenceCounted(false); acquire() }
  }

  private fun releaseWake() {
    wake?.let { if (it.isHeld) it.release() }
    wake = null
  }

  override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int = START_STICKY

  override fun onDestroy() {
    try { unregisterReceiver(battery) } catch (_: Exception) {}
    multicast?.let { if (it.isHeld) it.release() }
    releaseWake()
    super.onDestroy()
  }

  override fun onBind(intent: Intent?): IBinder? = null

  companion object {
    const val CHANNEL = "clamor"
  }
}

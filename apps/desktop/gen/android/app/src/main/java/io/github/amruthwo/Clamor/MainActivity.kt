package io.github.amruthwo.Clamor

import android.Manifest
import android.content.Intent
import android.content.pm.PackageManager
import android.os.Build
import android.os.Bundle
import androidx.activity.enableEdgeToEdge
import androidx.core.content.ContextCompat

class MainActivity : TauriActivity() {
  override fun onCreate(savedInstanceState: Bundle?) {
    enableEdgeToEdge()
    super.onCreate(savedInstanceState)
    // Asked together on first start:
    //  - notifications (Android 13+): the notification that keeps Clamor in the swarm in the background;
    //  - local network (Android 17+, shown as "Nearby devices"): finding and talking to the swarm.
    val wanted = mutableListOf<String>()
    if (Build.VERSION.SDK_INT >= 33) wanted += Manifest.permission.POST_NOTIFICATIONS
    if (Build.VERSION.SDK_INT >= 37) wanted += "android.permission.ACCESS_LOCAL_NETWORK"
    val missing = wanted.filter { ContextCompat.checkSelfPermission(this, it) != PackageManager.PERMISSION_GRANTED }
    if (missing.isNotEmpty()) requestPermissions(missing.toTypedArray(), 1)
    ContextCompat.startForegroundService(this, Intent(this, ClamorService::class.java))
  }
}

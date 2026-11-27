package io.github.amruthwo.Clamor

import android.Manifest
import android.content.Intent
import android.content.pm.PackageManager
import android.os.Build
import android.os.Bundle
import android.webkit.JavascriptInterface
import android.webkit.WebView
import androidx.activity.enableEdgeToEdge
import androidx.core.content.ContextCompat
import androidx.core.content.FileProvider
import java.io.File

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

  override fun onWebViewCreate(webView: WebView) {
    webView.addJavascriptInterface(Updates(), "ClamorAndroid")
  }

  /** For the app's updater: hand a downloaded (and already verified) APK to Android's installer,
   *  which asks the person "Update this app?". Only files the app downloaded are accepted. */
  inner class Updates {
    @JavascriptInterface
    fun installApk(path: String): String {
      return try {
        val src = File(path).canonicalFile
        val app = File(applicationInfo.dataDir).canonicalFile
        if (!src.path.startsWith(app.path + "/") || !src.name.endsWith(".apk") || !src.isFile) return "not an update this app downloaded"
        // The FileProvider serves the cache directory.
        val dir = File(cacheDir, "updates").apply { mkdirs() }
        val apk = if (src.parentFile?.canonicalPath == dir.canonicalPath) src else File(dir, src.name).also { src.copyTo(it, overwrite = true) }
        val uri = FileProvider.getUriForFile(this@MainActivity, "$packageName.fileprovider", apk)
        val intent = Intent(Intent.ACTION_VIEW).setDataAndType(uri, "application/vnd.android.package-archive")
          .addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION or Intent.FLAG_ACTIVITY_NEW_TASK)
        runOnUiThread { startActivity(intent) }
        ""
      } catch (e: Exception) {
        e.toString()
      }
    }
  }
}

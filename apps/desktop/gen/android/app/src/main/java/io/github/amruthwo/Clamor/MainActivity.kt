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
import androidx.core.view.ViewCompat
import androidx.core.view.WindowCompat
import androidx.core.view.WindowInsetsCompat
import androidx.core.content.FileProvider
import java.io.File
import org.json.JSONObject

class MainActivity : TauriActivity() {
  override fun onCreate(savedInstanceState: Bundle?) {
    // The phone's name as the person set it ("Galaxy S24 Ultra"), for its name in the swarm
    // (Android's host name is "localhost" on every phone). Read by clamor-node at start.
    val name = android.provider.Settings.Global.getString(contentResolver, "device_name")
      ?: android.provider.Settings.Secure.getString(contentResolver, "bluetooth_name")
    if (!name.isNullOrBlank()) android.system.Os.setenv("CLAMOR_DEVICE_NAME", name, true)
    enableEdgeToEdge()
    super.onCreate(savedInstanceState)
    keepClearOfSystemBars()
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

  /** Android 15+ draws apps under the status and navigation bars. Clamor's pages are web pages
   *  (and the dashboard sits in a frame, where CSS safe-area insets do not reach), so leave
   *  room natively: pad the content by the bars (and a camera cutout), in the page's colour. */
  private fun keepClearOfSystemBars() {
    val content = findViewById<android.view.View>(android.R.id.content)
    val night = (resources.configuration.uiMode and android.content.res.Configuration.UI_MODE_NIGHT_MASK) == android.content.res.Configuration.UI_MODE_NIGHT_YES
    // The page background (Material 3 surface), so the strips look like part of the app.
    content.setBackgroundColor(if (night) 0xFF111318.toInt() else 0xFFF9F9FC.toInt())
    WindowCompat.getInsetsController(window, content).apply {
      isAppearanceLightStatusBars = !night
      isAppearanceLightNavigationBars = !night
    }
    ViewCompat.setOnApplyWindowInsetsListener(content) { v, insets ->
      val bars = insets.getInsets(WindowInsetsCompat.Type.systemBars() or WindowInsetsCompat.Type.displayCutout())
      // The keyboard: shrink the page above it, so the message box stays visible.
      val ime = insets.getInsets(WindowInsetsCompat.Type.ime())
      v.setPadding(bars.left, bars.top, bars.right, maxOf(bars.bottom, ime.bottom))
      WindowInsetsCompat.CONSUMED
    }
  }

  private var web: WebView? = null

  override fun onWebViewCreate(webView: WebView) {
    web = webView
    webView.addJavascriptInterface(Updates(), "ClamorAndroid")
  }

  /** Start the scanner (QrScanActivity), asking for the camera first if needed. */
  private fun startScanner() {
    QrScanActivity.onResult = { text -> scanned(if (text == null) JSONObject().put("cancelled", true) else JSONObject().put("content", text)) }
    startActivity(Intent(this, QrScanActivity::class.java))
  }

  @Deprecated("Deprecated in Java")
  override fun onRequestPermissionsResult(requestCode: Int, permissions: Array<out String>, grantResults: IntArray) {
    @Suppress("DEPRECATION")
    super.onRequestPermissionsResult(requestCode, permissions, grantResults)
    if (requestCode == CAMERA_REQUEST) {
      if (grantResults.isNotEmpty() && grantResults[0] == PackageManager.PERMISSION_GRANTED) startScanner()
      else scanned(JSONObject().put("error", "Clamor needs the camera to scan the code"))
    }
  }

  private val CAMERA_REQUEST = 2

  /** The result of a QR scan, to the page's window.clamorScanned(result). */
  private fun scanned(result: JSONObject) {
    runOnUiThread { web?.evaluateJavascript("window.clamorScanned && window.clamorScanned($result)", null) }
  }

  /** For the app's updater: hand a downloaded (and already verified) APK to Android's installer,
   *  which asks the person "Update this app?". Only files the app downloaded are accepted. */
  inner class Updates {
    /** "Scan a QR code" (joining a swarm): Clamor's own scanner (no Google Play services needed),
     *  after asking for the camera. Answers with {content}, {cancelled} or {error}. */
    @JavascriptInterface
    fun scanQr() {
      runOnUiThread {
        if (ContextCompat.checkSelfPermission(this@MainActivity, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED) startScanner()
        else requestPermissions(arrayOf(Manifest.permission.CAMERA), CAMERA_REQUEST)
      }
    }

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

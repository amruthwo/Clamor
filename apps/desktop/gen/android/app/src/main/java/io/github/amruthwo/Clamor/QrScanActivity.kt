package io.github.amruthwo.Clamor

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.RectF
import android.os.Bundle
import android.view.Gravity
import android.view.View
import android.widget.FrameLayout
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.content.ContextCompat
import com.google.zxing.BarcodeFormat
import com.google.zxing.BinaryBitmap
import com.google.zxing.DecodeHintType
import com.google.zxing.MultiFormatReader
import com.google.zxing.NotFoundException
import com.google.zxing.PlanarYUVLuminanceSource
import com.google.zxing.common.HybridBinarizer
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicBoolean

/** "Scan a QR code" (joining a swarm), without Google Play services: CameraX shows the camera and
 *  ZXing (Apache-2.0) reads the code, so it works on any Android phone with a camera. The result
 *  goes to [onResult]: the code's text, or null if the person closed the scanner. */
class QrScanActivity : AppCompatActivity() {
  companion object {
    var onResult: ((String?) -> Unit)? = null
  }

  private val done = AtomicBoolean(false)
  private val analysis = Executors.newSingleThreadExecutor()
  private val reader = MultiFormatReader().apply {
    setHints(mapOf(DecodeHintType.POSSIBLE_FORMATS to listOf(BarcodeFormat.QR_CODE), DecodeHintType.TRY_HARDER to true))
  }

  override fun onCreate(savedInstanceState: Bundle?) {
    super.onCreate(savedInstanceState)
    val root = FrameLayout(this).apply { setBackgroundColor(Color.BLACK) }
    val preview = PreviewView(this)
    root.addView(preview, FrameLayout.LayoutParams(FrameLayout.LayoutParams.MATCH_PARENT, FrameLayout.LayoutParams.MATCH_PARENT))
    root.addView(Frame(this), FrameLayout.LayoutParams(FrameLayout.LayoutParams.MATCH_PARENT, FrameLayout.LayoutParams.MATCH_PARENT))
    val dp = resources.displayMetrics.density
    root.addView(TextView(this).apply {
      text = "Point the camera at the QR code from Add device"
      setTextColor(Color.WHITE)
      textSize = 16f
      gravity = Gravity.CENTER
      setPadding((24 * dp).toInt(), 0, (24 * dp).toInt(), 0)
    }, FrameLayout.LayoutParams(FrameLayout.LayoutParams.MATCH_PARENT, FrameLayout.LayoutParams.WRAP_CONTENT, Gravity.TOP).apply { topMargin = (96 * dp).toInt() })
    root.addView(TextView(this).apply {
      text = "Cancel"
      setTextColor(Color.WHITE)
      textSize = 16f
      setPadding((28 * dp).toInt(), (14 * dp).toInt(), (28 * dp).toInt(), (14 * dp).toInt())
      setBackgroundColor(0x66000000)
      setOnClickListener { finishWith(null) }
    }, FrameLayout.LayoutParams(FrameLayout.LayoutParams.WRAP_CONTENT, FrameLayout.LayoutParams.WRAP_CONTENT, Gravity.BOTTOM or Gravity.CENTER_HORIZONTAL).apply { bottomMargin = (64 * dp).toInt() })
    setContentView(root)

    val future = ProcessCameraProvider.getInstance(this)
    future.addListener({
      try {
        val provider = future.get()
        val p = Preview.Builder().build()
        p.setSurfaceProvider(preview.surfaceProvider)
        val a = ImageAnalysis.Builder().setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST).build()
        a.setAnalyzer(analysis) { image -> decode(image) }
        provider.unbindAll()
        provider.bindToLifecycle(this, CameraSelector.DEFAULT_BACK_CAMERA, p, a)
      } catch (e: Exception) {
        finishWith(null)
      }
    }, ContextCompat.getMainExecutor(this))
  }

  /** Read a QR code from the camera frame's brightness (Y) plane. */
  private fun decode(image: ImageProxy) {
    try {
      if (done.get()) return
      val plane = image.planes[0]
      val buf = plane.buffer
      val data = ByteArray(buf.remaining())
      buf.get(data)
      val stride = plane.rowStride
      if (stride * (image.height - 1) + image.width > data.size) return
      val source = PlanarYUVLuminanceSource(data, stride, image.height, 0, 0, image.width, image.height, false)
      val text = reader.decodeWithState(BinaryBitmap(HybridBinarizer(source))).text
      runOnUiThread { finishWith(text) }
    } catch (_: NotFoundException) {
      // No code in this frame: try the next one.
    } catch (_: Exception) {
    } finally {
      reader.reset()
      image.close()
    }
  }

  private fun finishWith(text: String?) {
    if (!done.compareAndSet(false, true)) return
    onResult?.invoke(text)
    onResult = null
    finish()
  }

  @Deprecated("Deprecated in Java")
  override fun onBackPressed() {
    finishWith(null)
    @Suppress("DEPRECATION")
    super.onBackPressed()
  }

  override fun onDestroy() {
    super.onDestroy()
    analysis.shutdown()
    if (!done.get()) {
      onResult?.invoke(null)
      onResult = null
    }
  }

  /** Dims everything but a square in the middle, where the code goes. */
  private class Frame(context: Context) : View(context) {
    private val dim = Paint().apply { color = 0x88000000.toInt() }
    private val edge = Paint(Paint.ANTI_ALIAS_FLAG).apply {
      color = 0xFFFD5453.toInt()
      style = Paint.Style.STROKE
      strokeWidth = 4 * context.resources.displayMetrics.density
    }

    override fun onDraw(canvas: Canvas) {
      val side = minOf(width, height) * 0.7f
      val l = (width - side) / 2
      val t = (height - side) / 2
      val r = RectF(l, t, l + side, t + side)
      canvas.drawRect(0f, 0f, width.toFloat(), t, dim)
      canvas.drawRect(0f, t + side, width.toFloat(), height.toFloat(), dim)
      canvas.drawRect(0f, t, l, t + side, dim)
      canvas.drawRect(l + side, t, width.toFloat(), t + side, dim)
      val radius = 24 * resources.displayMetrics.density
      canvas.drawRoundRect(r, radius, radius, edge)
    }
  }
}

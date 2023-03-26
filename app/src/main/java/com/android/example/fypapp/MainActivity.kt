package com.android.example.fypapp

import android.Manifest
import android.content.Context
import android.content.pm.PackageManager
import android.content.res.AssetFileDescriptor
import android.graphics.Bitmap
import android.os.Build
import android.os.Bundle
import android.util.Log
import android.view.View
import android.widget.CompoundButton
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import com.android.example.fypapp.databinding.ActivityMainBinding
import com.android.example.fypapp.ml.MaskClassificationModel
import org.opencv.android.OpenCVLoader
import org.opencv.android.Utils
import org.opencv.core.MatOfRect
import org.opencv.core.Point
import org.opencv.core.Scalar
import org.opencv.core.Size
import org.opencv.imgproc.Imgproc
import org.opencv.imgproc.Imgproc.FONT_HERSHEY_PLAIN
import org.tensorflow.lite.DataType
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.TensorImage
//import org.tensorflow.lite.support.image.ops.Rot90Op
import org.tensorflow.lite.support.common.ops.NormalizeOp

import org.tensorflow.lite.support.tensorbuffer.TensorBufferFloat
import java.io.FileInputStream
import java.io.IOException
import java.lang.Double
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import java.util.*
import java.util.concurrent.Executor
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import kotlin.Array
import kotlin.Boolean
import kotlin.Exception
import kotlin.Int
import kotlin.IntArray
import kotlin.Long
import kotlin.String
import kotlin.Throws
import kotlin.also
import kotlin.apply
import kotlin.arrayOf
import kotlin.let
import kotlin.math.max


class MainActivity : AppCompatActivity(), ImageAnalysis.Analyzer {
    private lateinit var viewBinding: ActivityMainBinding
    private lateinit var cameraExecutor: ExecutorService
    private lateinit var context: Context
    //private lateinit var tflite: InterpreterApi
    private lateinit var facemaskmodel : MaskClassificationModel
        //.newInstance(context)
    private var classNames = arrayOf("Incorrectly Worn Mask", "With Mask", "Without Mask")
    private var classColors = arrayOf(Scalar(255.0, 255.0, 0.0), Scalar(0.0, 255.0, 0.0),  Scalar(255.0, 0.0, 0.0))
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        viewBinding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(viewBinding.root)
        // Request camera permissions
        if (allPermissionsGranted()) {
            startCamera()
        } else {
            ActivityCompat.requestPermissions(
                this, REQUIRED_PERMISSIONS, REQUEST_CODE_PERMISSIONS)
        }
        context = getApplicationContext()
        facemaskmodel = MaskClassificationModel.newInstance(context)
        viewBinding.detectSwitch.setOnCheckedChangeListener(
            object: CompoundButton.OnCheckedChangeListener {
                override fun onCheckedChanged(p0: CompoundButton?, isChecked: Boolean) {
                    if (isChecked){
                        viewBinding.detectView.visibility= View.VISIBLE
                    } else {
                        viewBinding.detectView.visibility= View.INVISIBLE
                    }
                }
            }

        )

        cameraExecutor = Executors.newSingleThreadExecutor()
    }

    override fun onRequestPermissionsResult(
        requestCode: Int, permissions: Array<String>, grantResults:
        IntArray) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (requestCode == REQUEST_CODE_PERMISSIONS) {
            if (allPermissionsGranted()) {
                startCamera()
            } else {
                Toast.makeText(this,
                    "Permissions not granted by the user.",
                    Toast.LENGTH_SHORT).show()
                finish()
            }
        }
    }

    override fun analyze(image: ImageProxy) {
        // image processing here for the current frame
        Log.d("TAG", "analyze: got the frame at: " + image.imageInfo.timestamp)
        val bitmap: Bitmap = viewBinding.previewView.getBitmap()?:return
        image.close()
        if (bitmap != null && OpenCVLoader.initDebug()) {
            val bitmap1: Bitmap = detect(bitmap)
        } else {
            return
        }
        val bitmap1 = detect(bitmap)
        runOnUiThread { viewBinding.detectView.setImageBitmap(bitmap1) }
    }

    @Throws(IOException::class)
    private fun loadModelFile(): MappedByteBuffer {
        //open the model usin an input stream, and momory map it to load
        val fileDescriptor: AssetFileDescriptor = this.assets.openFd("mask_classification_model.tflite")
        val inputStream: FileInputStream = FileInputStream(fileDescriptor.fileDescriptor)
        val fileChannel: FileChannel = inputStream.getChannel()
        val startOffset = fileDescriptor.startOffset
        val declaredLength = fileDescriptor.declaredLength
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength)

    }
    private fun detect(bmpOriginal: Bitmap): Bitmap {
        val bmp = bmpOriginal
        val mat = MatOfRect()
        Utils.bitmapToMat(bmp, mat)
        val facesArray = facedetect(mat.nativeObjAddr)
        Log.d("detect", "Face array detected")
        val FACE_RECT_COLOR = Scalar(255.0, 0.0, 0.0)
        val FACE_RECT_THICKNESS = 10
        val TEXT_SIZE = 4.0
        val SIZE = Size(50.0, 50.0)
        //initialize an array of face
        val imageProcessor = ImageProcessor.Builder().add(NormalizeOp(0.0f, 225.0f)).build()

        for (face in facesArray) {
            //steps to crop the bitmap for inference
            val x = face.faceRect.x
            val y = face.faceRect.y
            val height = face.faceRect.height
            val width = face.faceRect.width
            val xmin = x
            val ymin = y
            val croppedFaceBitmap = Bitmap.createBitmap(bmp, max(0,xmin), max(0, ymin), width, height)
            val resizedCroppedFaceBitmap = Bitmap.createScaledBitmap(croppedFaceBitmap, 224, 224, false)
            // convert cropped and resized bitmap to tfbuffer
            if (resizedCroppedFaceBitmap !== null){
//                val tensorImage = TensorImage(DataType.UINT8)
//                tensorImage.load(resizedCroppedFaceBitmap)
//                val tensorImageBuffer = tensorImage.tensorBuffer

//                val tfImage = TensorImage.fromBitmap(resizedCroppedFaceBitmap)
//                val tensorBuffer = tfImage.tensorBuffer
                val tensorImage = imageProcessor.process(TensorImage.fromBitmap(resizedCroppedFaceBitmap))
                val tensorBuffer = tensorImage.tensorBuffer
                val tensorBufferFloat = TensorBufferFloat.createFrom(tensorBuffer, DataType.FLOAT32)
                val outputs = facemaskmodel.process(tensorBufferFloat)
                outputs::class.simpleName?.let { Log.d("Type of output", it) }
                val outputArray = outputs.outputFeature0AsTensorBuffer.floatArray
                val maxIdx = outputArray.indices.maxByOrNull { outputArray[it] } ?: -1
                val className = classNames[maxIdx]
                val textRectColor = classColors[maxIdx]
                val text_pos = Point(Double.max(0.0, face.faceRect.x.toDouble() - FACE_RECT_THICKNESS), Double.max(0.0, face.faceRect.y - FACE_RECT_THICKNESS.toDouble()))
                Imgproc.putText(mat, className, text_pos, FONT_HERSHEY_PLAIN, TEXT_SIZE,textRectColor)
                Imgproc.rectangle(mat, face.faceRect, textRectColor, FACE_RECT_THICKNESS)

            }
        }
        val bmp2 = bmp.copy(bmp.config, true)
        Utils.matToBitmap(mat, bmp2)
        return bmp2
    }
    private fun startCamera() {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)

        cameraProviderFuture.addListener({
            // Used to bind the lifecycle of cameras to the lifecycle owner
            val cameraProvider: ProcessCameraProvider = cameraProviderFuture.get()

            // Preview
            val preview = Preview.Builder()
                .build()
                .also {
                    it.setSurfaceProvider(viewBinding.previewView.surfaceProvider)
                }
            // image analyzer builder
            val imageAnalyzer = ImageAnalysis.Builder()
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .build()

            imageAnalyzer.setAnalyzer(getExecutor(), this)

            // Select back camera as a default
            val cameraSelector = CameraSelector.DEFAULT_BACK_CAMERA

            try {
                // Unbind use cases before rebinding
                cameraProvider.unbindAll()

                // Bind use cases to camera
                cameraProvider.bindToLifecycle(
                    this, cameraSelector, preview, imageAnalyzer)

            } catch(exc: Exception) {
                Log.e(TAG, "Use case binding failed", exc)
            }

        }, ContextCompat.getMainExecutor(this))
    }

    private fun allPermissionsGranted() = REQUIRED_PERMISSIONS.all {
        ContextCompat.checkSelfPermission(
            baseContext, it) == PackageManager.PERMISSION_GRANTED
    }

    private fun getExecutor(): Executor {
        return ContextCompat.getMainExecutor(this)
    }


    override fun onDestroy() {
        super.onDestroy()
        cameraExecutor.shutdown()
    }

    external fun facedetect(matAddr: Long): Array<Face>

    companion object {
        private const val TAG = "CameraXApp"
        private const val FILENAME_FORMAT = "yyyy-MM-dd-HH-mm-ss-SSS"
        private const val REQUEST_CODE_PERMISSIONS = 10
        private val REQUIRED_PERMISSIONS =
            mutableListOf (
                Manifest.permission.CAMERA,
                Manifest.permission.RECORD_AUDIO
            ).apply {
                if (Build.VERSION.SDK_INT <= Build.VERSION_CODES.P) {
                    add(Manifest.permission.WRITE_EXTERNAL_STORAGE)
                }
            }.toTypedArray()
        init {
            System.loadLibrary("facedetection")
        }
    }

}
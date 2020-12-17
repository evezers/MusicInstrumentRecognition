package com.evezers.instrumentrecognition

import android.Manifest
import android.content.Intent
import android.content.pm.PackageManager
import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaRecorder
import android.os.Bundle
import android.util.Log
import android.view.Menu
import android.view.MenuItem
import androidx.appcompat.app.AppCompatActivity
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import com.google.android.material.floatingactionbutton.FloatingActionButton
import kotlinx.android.synthetic.main.fragment_first.*
import me.gommeantilegit.sonopy.Sonopy
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.label.TensorLabel
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer
import java.io.FileInputStream
import java.nio.channels.FileChannel


class MainActivity : AppCompatActivity() {
    val TAG = "myLogs"
    var audioRecord: AudioRecord? = null
    var isRecording = false
    var isReading = false
    private var recordingThread: Thread? = null
    var REQUEST_CODE_PERMISSION_RECORD_AUDIO = 1337
    var myBufferSize = 352800
    val sampleRate = 22050
    var labelsText: List<String>? = null

    /** Output probability TensorBuffer.  */
    private var outputProbabilityBuffer: TensorBuffer? = null

    var labels: Map<Int, Any>? = null
    var tflite: Interpreter? = null

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)
        setSupportActionBar(findViewById(R.id.toolbar))

        var permissionStatus =
            ContextCompat.checkSelfPermission(this, Manifest.permission.RECORD_AUDIO)

        if (permissionStatus == PackageManager.PERMISSION_GRANTED) {

        } else {
            ActivityCompat.requestPermissions(
                this, arrayOf(Manifest.permission.RECORD_AUDIO),
                REQUEST_CODE_PERMISSION_RECORD_AUDIO
            )
        }

        createAudioRecorder()

        loadModel()

        findViewById<FloatingActionButton>(R.id.fab).setOnClickListener {
            if (isRecording){
                isRecording = false
                stopRecord()
            } else {
                isRecording = true
                startRecord()
            }
        }
    }

    fun createAudioRecorder() {

        val channelConfig = AudioFormat.CHANNEL_IN_MONO
        val audioFormat = AudioFormat.ENCODING_PCM_FLOAT
        val minInternalBufferSize = AudioRecord.getMinBufferSize(
            sampleRate,
            channelConfig, audioFormat
        )
        val internalBufferSize = minInternalBufferSize * 4
        Log.d(
            TAG, "minInternalBufferSize = " + minInternalBufferSize
                    + ", internalBufferSize = " + internalBufferSize
                    + ", myBufferSize = " + myBufferSize
        )
        audioRecord = AudioRecord(
            MediaRecorder.AudioSource.MIC,
            sampleRate, channelConfig, audioFormat, internalBufferSize
        )
//        audioRecord!!.positionNotificationPeriod = 1000
//        audioRecord!!
//            .setRecordPositionUpdateListener(object : OnRecordPositionUpdateListener {
//                override fun onPeriodicNotification(recorder: AudioRecord) {
//                    Log.d(TAG, "onPeriodicNotification")
//                }
//
//                override fun onMarkerReached(recorder: AudioRecord) {
//                    Log.d(TAG, "onMarkerReached")
//                }
//            })
    }

    fun loadModel(){
        labelsText = assets.open("models/labels.txt").bufferedReader().use { it.readLines() }
        val labelsIndexes = IntArray(labelsText!!.lastIndex) { it - 1 }
//        labels!! = labelsText.map { it to it }




        val fileDescriptor = assets.openFd("models/model.tflite")
        val inputStream = FileInputStream(fileDescriptor.fileDescriptor)
        val fileChannel = inputStream.channel
        val startOffset = fileDescriptor.startOffset
        val declaredLength = fileDescriptor.declaredLength
        val tfliteModel = fileChannel.map(
            FileChannel.MapMode.READ_ONLY,
            startOffset,
            declaredLength
        )

        tflite = Interpreter(tfliteModel)


        // Reads type and shape of input and output tensors, respectively.

        val imageShape = tflite!!.getInputTensor(0).shape() // {1, 128, 44, 1}

        val imageDataType = tflite!!.getInputTensor(0).dataType()
        val probabilityTensorIndex = 0
        val probabilityShape =
            tflite!!.getOutputTensor(0).shape() // {1, NUM_CLASSES}

        val probabilityDataType = tflite!!.getOutputTensor(probabilityTensorIndex).dataType()

        // Creates the input tensor.
        val inputImageBuffer = TensorImage(imageDataType)

        // Creates the output tensor and its processor.

        // Creates the output tensor and its processor.
        outputProbabilityBuffer =
            TensorBuffer.createFixedSize(probabilityShape, probabilityDataType)
    }


    fun startRecord(){
        var fab = findViewById<FloatingActionButton>(R.id.fab)
        fab.setImageResource(R.drawable.ic_baseline_stop_64)
        fab.supportBackgroundTintList = ContextCompat.getColorStateList(this, R.color.colorRed)

        audioRecord!!.startRecording()
        val recordingState = audioRecord!!.recordingState
        Log.d(TAG, "recordingState = $recordingState")

        readStart()
    }


    fun stopRecord(){
        var fab = findViewById<FloatingActionButton>(R.id.fab)
        fab.setImageResource(R.drawable.ic_baseline_mic_64)
        fab.supportBackgroundTintList = ContextCompat.getColorStateList(this, R.color.colorAccent)

        Log.d(TAG, "record stop")
        audioRecord!!.stop()
        readStop()
    }

    fun readStart() {
        Log.d(TAG, "read start")
        isReading = true
        Thread(Runnable {
            if (audioRecord == null) return@Runnable
            val myBuffer = ByteArray(myBufferSize)
            val buf = FloatArray(sampleRate * 2)
            var readCount = 0
            var totalCount = 0
            while (isReading) {
                readCount = audioRecord!!.read(buf, 0, sampleRate * 2, AudioRecord.READ_BLOCKING)

                // Functions that depend on filterbanks values, are instance bound to avoid recalculation with same parameters
                val sonopy =
                    Sonopy(sampleRate, 22050, 512, 1024, 128)
                val mels = sonopy.melSpec(buf)

//                val testData = FloatArray(1)

                var testData: Array<Array<Array<FloatArray?>?>?> = Array(1, { Array(128,  { Array(44,  { floatArrayOf(0F) }) }) })

                for(i in 0..43) {
                    for(k in 0..127) {
                        val cell = mels[i][k]

                        testData[0]?.get(k)?.set(i, floatArrayOf(cell))
                    }
                }


//                var variable: Array<FloatArray?> = arrayOf(FloatArray(1))

                tflite!!.run(testData, outputProbabilityBuffer!!.getBuffer().rewind())

                // Gets the map of label and probability.
                val labeledProbability: Map<String, Float> =
                    TensorLabel(labelsText!!, outputProbabilityBuffer!!)
                        .getMapWithFloatValue()

                val result = labeledProbability.toList().sortedBy { (_, value) -> value}


                textview_first.text = result[10].toString() + ' ' + result[9].toString() + ' ' + result[8].toString()


                Log.d(
                    TAG, result[8].toString() + ' ' + result[9].toString() + ' ' + result[10].toString()
                )





                totalCount += readCount


//                Log.d(
//                    TAG, "readCount = " + readCount + ", totalCount = "
//                            + totalCount + ' ' + mels.size
//                )
////                Log.d(
////                    TAG, mels[0].size.toString()
////                )
            }
        }).start()
    }

    fun readStop() {
        Log.d(TAG, "read stop")
        isReading = false
    }

    override fun onCreateOptionsMenu(menu: Menu): Boolean {
        // Inflate the menu; this adds items to the action bar if it is present.
        menuInflater.inflate(R.menu.menu_main, menu)
        return true
    }

    override fun onOptionsItemSelected(item: MenuItem): Boolean {

        // Handle action bar item clicks here. The action bar will
        // automatically handle clicks on the Home/Up button, so long
        // as you specify a parent activity in AndroidManifest.xml.
        when (item.itemId) {
            R.id.action_settings -> {
                val settingsIntent = Intent(this, SettingsActivity::class.java)

                startActivity(settingsIntent)
                return true
            }

        }

        return super.onOptionsItemSelected(item)
    }
}
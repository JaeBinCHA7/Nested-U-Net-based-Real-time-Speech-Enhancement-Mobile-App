package com.example.speech_enhancement_rt_on_mobile.bottomnavi;

import android.Manifest;
import android.content.Context;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.media.AudioFormat;
import android.media.AudioManager;
import android.media.AudioRecord;
import android.media.AudioTrack;
import android.media.MediaRecorder;
import android.net.Uri;
import android.os.Bundle;

import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;
import androidx.fragment.app.Fragment;

import android.util.Log;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.ImageButton;
import android.widget.Toast;

import com.chibde.visualizer.CircleBarVisualizer;
import com.example.speech_enhancement_rt_on_mobile.FileUtils;
import com.example.speech_enhancement_rt_on_mobile.R;
import com.example.speech_enhancement_rt_on_mobile.SpeechEnhancement;

import java.io.DataInputStream;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.text.SimpleDateFormat;
import java.util.Date;

public class AudioFragment extends Fragment implements View.OnClickListener {

    // for audio recording
    private int SAMPLE_RATE = 16000;
    private int stride = 128; // recording per 128 samples
    private ImageButton btn_record;
    private ImageButton btn_record_refresh;
    private ImageButton btn_record_play;
    private boolean isRecording = false;
    private boolean isRefresh = false;
    private int rBufferSize =
            AudioRecord.getMinBufferSize(
                    SAMPLE_RATE, AudioFormat.CHANNEL_IN_MONO, AudioFormat.ENCODING_PCM_16BIT); // 1280
    public AudioRecord mAudioRecord = null;
    private Thread recordingThread = null;
    private String audioFileName;
    private FileOutputStream fos = null;
    private int record_audio_len = 0;

    // for audio playing
    private Boolean isPlaying = false;
    protected static AudioTrack mAudioTrack;
    private int tBufferSize = AudioTrack.getMinBufferSize(SAMPLE_RATE, AudioFormat.CHANNEL_OUT_MONO, AudioFormat.ENCODING_PCM_16BIT);
    public Thread mPlayThread = null;

    // for audio loading
    private ImageButton btn_upload;
    Context ct;
    private Uri selectedMediaUri = null;

    // for audio permission
    private String recordPermission = Manifest.permission.RECORD_AUDIO;
    private int PERMISSION_CODE = 21;

    // for audio visualiizer
    public CircleBarVisualizer circleBarVisualizer;

    // for real-time speech enhancement
    private ImageButton btn_se;
    private SpeechEnhancement speechEnhancement;
    private String tflitePath = "nunet_lstm_e67.tflite"; /** Enter tflite path made in python (Tensorflow) */
    boolean btn_se_on = false;


    @Override
    public View onCreateView(LayoutInflater inflater, ViewGroup container,
                             Bundle savedInstanceState) {
        initSE();

        View v = inflater.inflate(R.layout.fragment_audio, container, false);
        ct = container.getContext();

        btn_record = v.findViewById(R.id.btn_record);
        btn_record_refresh = v.findViewById(R.id.btn_record_refresh);
        btn_record_play = v.findViewById(R.id.btn_record_play);
        btn_upload = v.findViewById(R.id.btn_upload);
        btn_se = v.findViewById(R.id.btn_se);

        btn_record.setOnClickListener(this);
        btn_record_refresh.setOnClickListener(this);
        btn_record_play.setOnClickListener(this);
        btn_upload.setOnClickListener(this);
        btn_se.setOnClickListener(this);

        return v;
    }

    @Override
    public void onClick(View v) {
        switch (v.getId()) {
            /** Button for audio recording */
            case R.id.btn_record:
                if (!isRefresh) {
                    mAudioRecord =
                            new AudioRecord(
                                    MediaRecorder.AudioSource.DEFAULT,
                                    SAMPLE_RATE,
                                    AudioFormat.CHANNEL_IN_MONO,
                                    AudioFormat.ENCODING_PCM_16BIT,
                                    rBufferSize);
                    String recordPath = getActivity().getExternalFilesDir("/").getAbsolutePath();
                    String timeStamp = new SimpleDateFormat("yyyyMMdd_HHmmss").format(new Date());
                    audioFileName = recordPath + "/" + "RecordFile_" + timeStamp + "_" + "audio.pcm";

                    try {
                        fos = new FileOutputStream(audioFileName);
                    } catch (FileNotFoundException e) {
                        e.printStackTrace();
                    }

                    isRefresh = true;
                }

                if (isRecording) {
                    btn_record.setImageDrawable(getResources().getDrawable(R.drawable.btn_audio_record, null));
                    stopRecording();
                    Toast.makeText(getActivity().getApplicationContext(), "Complete recoding audio", Toast.LENGTH_SHORT).show();
                } else {
                    if (checkAudioPermission()) {
                        isRecording = true;
                        btn_record.setImageDrawable(getResources().getDrawable(R.drawable.btn_record_clicked, null));
                        mAudioRecord.startRecording();
                        startRecording();
                    }
                }
                break;
            /** Refresh about audio recoding */
            case R.id.btn_record_refresh:
                if (isRecording) {
                    Toast.makeText(getActivity().getApplicationContext(), "Stop recording audio", Toast.LENGTH_SHORT).show();
                } else {
                    refreshRecording();
                    Toast.makeText(getActivity().getApplicationContext(), "Complete refreshing audio recording", Toast.LENGTH_SHORT).show();
                    mAudioRecord =
                            new AudioRecord(
                                    MediaRecorder.AudioSource.DEFAULT,
                                    SAMPLE_RATE,
                                    AudioFormat.CHANNEL_IN_MONO,
                                    AudioFormat.ENCODING_PCM_16BIT,
                                    rBufferSize);
                    fos = null;
                    try {
                        fos = new FileOutputStream(audioFileName);
                    } catch (FileNotFoundException e) {
                        e.printStackTrace();
                    }
                }
                break;


            /** Button for uploading the audio file */
            case R.id.btn_upload:
                Intent intent = new Intent();
                intent.setType("audio/*");
                intent.setAction("android.intent.action.GET_CONTENT");

                this.startActivityForResult(intent, 101);
                break;

            /** Button for speech enhancement */
            case R.id.btn_se:
                Log.d("", "btn: btn_se_clicked");
                if(!btn_se_on) {
                    btn_se_on = true;
                    Toast.makeText(getActivity().getApplicationContext(), "Real-time speech enhancement in progress.", Toast.LENGTH_SHORT).show();
                    btn_se.setImageDrawable(getResources().getDrawable(R.drawable.btn_se_clicked));
                }else{
                    Toast.makeText(getActivity().getApplicationContext(), "Exit speech enhancement", Toast.LENGTH_SHORT).show();
                    btn_se_on = false;
                    btn_se.setImageDrawable(getResources().getDrawable(R.drawable.btn_se_unclicked));
                }
                break;


            /** Button for playing audio */
            case R.id.btn_record_play:
                if (isPlaying) {
                    stopAudio();
                } else {
                    playAudio(audioFileName);
                }
                break;

        }
    }


    /**
     * About audio recording
     */
    // Audio recording (Start)
    private void startRecording() {
        recordingThread = new Thread(new Runnable() {
            public void run() {
                writeAudioDataToFile();
            }
        }, "AudioRecorder Thread");
        recordingThread.start();
    }

    // Audio recording (Exit)
    private void stopRecording() {
        if (null != mAudioRecord) {
            isRecording = false;
            mAudioRecord.stop(); // 녹음 종료
        }
    }

    // Audio recording (Refresh)
    private void refreshRecording() {
        try {
            fos.close();
        } catch (IOException e) {
            e.printStackTrace();
        }

        String recordPath = getActivity().getExternalFilesDir("/").getAbsolutePath();
        String timeStamp = new SimpleDateFormat("yyyyMMdd_HHmmss").format(new Date());
        audioFileName = recordPath + "/" + "RecordFile_" + timeStamp + "_" + "audio.pcm";

        if (null != mAudioRecord) {
            mAudioRecord.release();
            mAudioRecord = null;
            recordingThread = null;
        }
    }

    // Audio recording (Write to file)
    private void writeAudioDataToFile() {
        short sData[] = new short[stride];


        while (isRecording) {
            mAudioRecord.read(sData, 0, sData.length); // 마이크를 통해 short 타입의 데이터를 실시간으로 리드함.
            try {
                byte bData[] = short2byte(sData); // 파일 쓰기를 위해 Byte 타입으로 변환
                fos.write(bData, 0, bData.length); // 파일 쓰기
            } catch (IOException e) {
                e.printStackTrace();
            }
        }

    }

    /* Audio recording - short type
    File input/output - byte type */
    private byte[] short2byte(short[] sData) {
        byte[] bytes = new byte[sData.length * 2];
        for (int i = 0; i < sData.length; i++) {
            bytes[i * 2] = (byte) (sData[i] & 0x00FF);
            bytes[(i * 2) + 1] = (byte) (sData[i] >> 8);
        }
        return bytes;
    }

    // about audio file permission
    private boolean checkAudioPermission() {
        if (ActivityCompat.checkSelfPermission(getActivity().getApplicationContext(), recordPermission) == PackageManager.PERMISSION_GRANTED) {
            return true;
        } else {
            ActivityCompat.requestPermissions(getActivity(), new String[]{recordPermission}, PERMISSION_CODE);
            return false;
        }
    }

    /**
     * About audio play
     */
    // Play recording file
    private void playAudio(String clickedURL) {
        /** AUdio visualizer */
        CircleBarVisualizer circleBarVisualizer = getActivity().findViewById(R.id.visualizer);
        circleBarVisualizer.setColor(ContextCompat.getColor(getActivity(), R.color.circle_visualizer));

        // Stream type, Sampling rate, Channel, Audio format, Buffer size
        mAudioTrack =
                new AudioTrack(
                        AudioManager.STREAM_MUSIC,
                        SAMPLE_RATE,
                        AudioFormat.CHANNEL_OUT_MONO,
                        AudioFormat.ENCODING_PCM_16BIT,
                        tBufferSize,
                        AudioTrack.MODE_STREAM);

        mPlayThread = new Thread(new Runnable() {
            @Override
            public void run() {
                byte[] writeData = new byte[512];

                FileInputStream fis = null;
                try {
                    fis = new FileInputStream(clickedURL);
                } catch (FileNotFoundException e) {
                    e.printStackTrace();
                }

                DataInputStream dis = new DataInputStream(fis);
                mAudioTrack.play();
                circleBarVisualizer.setPlayer(mAudioTrack.getAudioSessionId());
                record_audio_len = 0;
                while (isPlaying) {
                    try {
                        int ret = dis.read(writeData, 0, 512); // return Data length

                        /**
                         * Real-time Audio Speech Enhancement
                         * */
                        if(btn_se_on){
                            short[] shortData = byteArrayToShortArray(writeData); // 128
                            double[] doubleData = shortArrayToDoubleArray(shortData); // 128

                            double[] se_out = speechEnhancement.audioSE(doubleData); // 128

                            shortData = doubleArrayToShortArray(se_out);
                            writeData = shortArrayToByteArray(shortData);
                        }

                        if (ret <= 0) {
                            getActivity().runOnUiThread(new Runnable() {
                                @Override
                                public void run() {
                                    isPlaying = false;
                                }
                            });
                            break;
                        }
                        mAudioTrack.write(writeData, 0, ret);
                    } catch (IOException e) {
                        e.printStackTrace();
                    }

                }
                mAudioTrack.stop();
                mAudioTrack.release();
                mAudioTrack = null;

                try {
                    dis.close();
                    fis.close();
                } catch (IOException e) {
                    e.printStackTrace();
                }
            }
        });

        mPlayThread.start();
        btn_record_play.setImageDrawable(getResources().getDrawable(R.drawable.btn_play_clicked, null));
        isPlaying = true;

        mAudioTrack.setPlaybackPositionUpdateListener(new AudioTrack.OnPlaybackPositionUpdateListener() {
            @Override
            public void onMarkerReached(AudioTrack audioTrack) {
                /*btn_record_play.setImageDrawable(getResources().getDrawable(R.drawable.btn_original_play, null));*/
//                stopAudio();
            }

            @Override
            public void onPeriodicNotification(AudioTrack audioTrack) {
                // nothing to do
            }
        });
    }

    // Stop audio recording
    private void stopAudio() {
        btn_record_play.setImageDrawable(getResources().getDrawable(R.drawable.btn_original_play, null));
        isPlaying = false;
        mAudioTrack.stop();
    }

    /**
     * About loading audio
     * */
    public void onActivityResult(int requestCode, int resultCode, Intent data) {
        super.onActivityResult(requestCode, resultCode, data);

        if (requestCode == 101 && resultCode == -1)
        {
            selectedMediaUri = data.getData();
            audioFileName = FileUtils.getPath(ct, selectedMediaUri);
            Toast.makeText(getActivity().getApplicationContext(), "Complete uploading audio", Toast.LENGTH_SHORT).show();
        }
    }


    /**
     * About speech enhancement
     */
    void initSE(){
        try {
            speechEnhancement = new SpeechEnhancement(getActivity(), tflitePath);
        } catch (IOException e) {
            Log.d("class", "Failed to create noise reduction");
        }

    }


    public short[] byteArrayToShortArray(byte[] bytes) {
        short[] shorts = new short[bytes.length/2];
        ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN).asShortBuffer().get(shorts);

        return shorts;
    }

    private double[] shortArrayToDoubleArray(short[] shortData) {
        int size = shortData.length;
        double[] doubleData = new double[size];
        for (int i = 0; i < size; i++) {
            doubleData[i] = shortData[i] / 32768.0;
        }
        return doubleData;
    }

    private short[] doubleArrayToShortArray(double[] doubleArray) {
        short[] shortArray = new short[doubleArray.length];
        for (int i = 0; i < doubleArray.length; i++) {
            shortArray[i] = (short) (doubleArray[i] * 32768.0);
        }
        return shortArray;
    }

    //convert short to byte
    private byte[] shortArrayToByteArray(short[] sData) {
        byte[] bytes = new byte[sData.length * 2];
        for (int i = 0; i < sData.length; i++) {
            bytes[i * 2] = (byte) (sData[i] & 0x00FF);
            bytes[(i * 2) + 1] = (byte) (sData[i] >> 8);
        }
        return bytes;
    }

}
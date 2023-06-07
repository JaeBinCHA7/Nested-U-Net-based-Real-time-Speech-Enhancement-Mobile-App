package com.example.real_time_speech_enhancer_app.bottomnavi;

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
import androidx.fragment.app.Fragment;

import android.os.Environment;
import android.util.Log;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.ImageButton;
import android.widget.Toast;

import com.example.real_time_speech_enhancer_app.FileUtils;
import com.example.real_time_speech_enhancer_app.R;
import com.example.real_time_speech_enhancer_app.RTSE_NUTLS;

import java.io.DataInputStream;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.text.SimpleDateFormat;
import java.util.Date;

public class BaselineFragment extends Fragment implements View.OnClickListener {

    // Declare audio recording variable
    private int SAMPLE_RATE = 16000;
    private int frame_len = 512;
    private int hop_len = 256; // Real-time recording of audio as much as 256 samples
    private ImageButton btn_record;
    private ImageButton btn_record_refresh;
    private ImageButton btn_record_play;
    private boolean isRecording = false;
    private boolean isRefresh = false;
    private int rBufferSize =
            AudioRecord.getMinBufferSize(
                    SAMPLE_RATE, AudioFormat.CHANNEL_IN_MONO, AudioFormat.ENCODING_PCM_16BIT); // 1280
    public AudioRecord mAudioRecord = null;
    private Thread recordingThread = null; // recording thread
    private String audioFileName; // Audio recording creation file name
    private FileOutputStream fos = null;
    private int totalSamples = 0;

    // Declare audio playback variable
    private Boolean isPlaying = false;
    protected static AudioTrack mAudioTrack;
    private int tBufferSize = AudioTrack.getMinBufferSize(SAMPLE_RATE, AudioFormat.CHANNEL_OUT_MONO, AudioFormat.ENCODING_PCM_16BIT);
    public Thread mPlayThread = null;

    // Declare audio load variable
    private ImageButton btn_upload;
    Context ct;
    private Uri audioUri = null; // audio file uri
    private Uri selectedMediaUri = null;

    // audio permission
    private String recordPermission = Manifest.permission.RECORD_AUDIO;
    private int PERMISSION_CODE = 21;

    // speech enhancement
    private ImageButton btn_se;
    private RTSE_NUTLS rtse_nutls;
    private String tflitePath = "nutls.tflite";

    // real-time speech enhancement
    boolean btn_se_on = false;

    // download audio
    private ImageButton btn_download;
    private byte[] out_audio;

    @Override
    public View onCreateView(LayoutInflater inflater, ViewGroup container,
                             Bundle savedInstanceState) {
        initSE();

        View v = inflater.inflate(R.layout.fragment_baseline, container, false);
        ct = container.getContext();
        // 버튼 선언
        btn_record = v.findViewById(R.id.btn_record);
        btn_record_refresh = v.findViewById(R.id.btn_record_refresh);
        btn_record_play = v.findViewById(R.id.btn_record_play);
        btn_upload = v.findViewById(R.id.btn_upload);
        btn_se = v.findViewById(R.id.btn_se);
        btn_download = v.findViewById(R.id.btn_download);

        btn_record.setOnClickListener(this);
        btn_record_refresh.setOnClickListener(this);
        btn_record_play.setOnClickListener(this);
        btn_upload.setOnClickListener(this);
        btn_se.setOnClickListener(this);
        btn_download.setOnClickListener(this);


        return v;
    }

    @Override
    public void onClick(View v) {
        switch (v.getId()) {
            /** Audio recording button */
            case R.id.btn_record:
                if (!isRefresh) { // Create an AudioRecord for the first time if you haven't refreshed it even once.
                    // AudioRecord creation and initialization
                    mAudioRecord =
                            new AudioRecord(
                                    MediaRecorder.AudioSource.DEFAULT,
                                    SAMPLE_RATE,
                                    AudioFormat.CHANNEL_IN_MONO,
                                    AudioFormat.ENCODING_PCM_16BIT,
                                    rBufferSize);

                    String recordPath = getActivity().getExternalFilesDir("/").getAbsolutePath(); // Check the external path of the file
                    String timeStamp = new SimpleDateFormat("yyyyMMdd_HHmmss").format(new Date());
                    audioFileName = recordPath + "/" + "RecordFile_" + timeStamp + "_" + "audio.pcm";

                    try {
                        fos = new FileOutputStream(audioFileName);
                    } catch (FileNotFoundException e) {
                        e.printStackTrace();
                    }

                    isRefresh = true;
                }

                if (isRecording) { // recording
                    btn_record.setImageDrawable(getResources().getDrawable(R.drawable.record_off, null)); // Change the recording icon
                    stopRecording();
                    Toast.makeText(getActivity().getApplicationContext(), "Recording is complete.", Toast.LENGTH_SHORT).show();
                } else {/*
                 * 1. Audio permission check
                 * 2. Check whether recording is performed for the first time */
                    if (checkAudioPermission()) {
                        isRecording = true; // Recording state change
                        btn_record.setImageDrawable(getResources().getDrawable(R.drawable.record_on, null)); // Recording icon change

                        mAudioRecord.startRecording();
                        startRecording();
                    }
                }
                break;
            /** Audio refresh button */
            case R.id.btn_record_refresh:
                if (isRecording) { // Toast to stop recording when the refresh button is pressed during recording.
                    Toast.makeText(getActivity().getApplicationContext(), "Stop recording.", Toast.LENGTH_SHORT).show();
                } else { // Audio record initialization + file output stream initialization on refresh
                    refreshRecording(); // Initialize audio record and file name
                    Toast.makeText(getActivity().getApplicationContext(), "Recording initialized.", Toast.LENGTH_SHORT).show();
                    // AudioRecord creation and initialization
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


            /** Audio upload button */
            case R.id.btn_upload:
                Intent intent = new Intent();
                intent.setType("audio/*");
                intent.setAction("android.intent.action.GET_CONTENT");

                this.startActivityForResult(intent, 101);
                break;

            /** Audio speech enhancement button */
            case R.id.btn_se:
                Log.d("", "btn: btn_se_clicked");
                if (!btn_se_on) {
                    btn_se_on = true;
                    Toast.makeText(getActivity().getApplicationContext(), "Real-time speech enhancement in progress", Toast.LENGTH_SHORT).show();
                    btn_se.setImageDrawable(getResources().getDrawable(R.drawable.speech_enhancement_on));
                } else {
                    Toast.makeText(getActivity().getApplicationContext(), "Turn off real-time speech enhancement.", Toast.LENGTH_SHORT).show();
                    btn_se_on = false;
                    btn_se.setImageDrawable(getResources().getDrawable(R.drawable.speech_enhancement_off));
                }
                break;


            /** Audio play button */
            case R.id.btn_record_play:
                if (isPlaying) { // Pause if playing when clicking the play button
                    stopAudio();
                } else {
                    playAudio(audioFileName);
                }
                break;

            /** Audio download button */
            case R.id.btn_download:
                // Set file save path and file name
                String fileName = "app_enhanced.wav";
                String filePath = Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_DOWNLOADS).getPath() + "/" + fileName;

                // download file
                try {
                    FileOutputStream fos = new FileOutputStream(filePath);
                    fos.write(out_audio);
                    fos.close();
                    Toast.makeText(getActivity().getApplicationContext(), "Download is complete.", Toast.LENGTH_SHORT).show();
                } catch (IOException e) {
                    e.printStackTrace();
                    Toast.makeText(getActivity().getApplicationContext(), "Download failed.", Toast.LENGTH_SHORT).show();
                }

        }
    }

    /**
     * Functions related to audio speech enhancement
     */
    void initSE() {
        try {
            rtse_nutls = new RTSE_NUTLS(getActivity(), tflitePath, frame_len, hop_len);
        } catch (IOException e) {
            Log.d("class", "Failed to create noise reduction");
        }

    }

    /**
     * Functions related to audio recording
     */
    // record audio - start
    private void startRecording() {
        recordingThread = new Thread(new Runnable() {
            public void run() {
                writeAudioDataToFile();
            }
        }, "AudioRecorder Thread");
        recordingThread.start();
    }

    // record audio - end
    private void stopRecording() {
        if (null != mAudioRecord) {
            isRecording = false;
            mAudioRecord.stop(); // 녹음 종료
        }
    }

    // record audio - refresh
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
            mAudioRecord.release(); // 오디오 레코드 초기화
            mAudioRecord = null;
            recordingThread = null;
        }
    }

    // record audio - write file
    private void writeAudioDataToFile() {
        short sData[] = new short[hop_len];


        while (isRecording) {
            mAudioRecord.read(sData, 0, sData.length); // Read short-type data through the microphone in real time.
            try {
                byte bData[] = short2byte(sData); // Convert to Byte type for file writing
                fos.write(bData, 0, bData.length); // write file
            } catch (IOException e) {
                e.printStackTrace();
            }
        }

    }

    /* Audio recording is short, file writing is byte type.
     * To write audio recording data into a file, it must be converted into a byte type. */
    private byte[] short2byte(short[] sData) {
        byte[] bytes = new byte[sData.length * 2];
        for (int i = 0; i < sData.length; i++) {
            bytes[i * 2] = (byte) (sData[i] & 0x00FF);
            bytes[(i * 2) + 1] = (byte) (sData[i] >> 8);
        }
        return bytes;
    }

    // Check audio file permissions
    private boolean checkAudioPermission() {
        if (ActivityCompat.checkSelfPermission(getActivity().getApplicationContext(), recordPermission) == PackageManager.PERMISSION_GRANTED) {
            return true;
        } else {
            ActivityCompat.requestPermissions(getActivity(), new String[]{recordPermission}, PERMISSION_CODE);
            return false;
        }
    }

    /**
     * Functions related to audio playback
     */
    // play the recording file
    private void playAudio(String clickedURL) {
        // stream type / sampling rate / channel / audio format / buffer size
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
                byte[] writeData = new byte[frame_len];

                FileInputStream fis = null;
                try {
                    fis = new FileInputStream(clickedURL);
                } catch (FileNotFoundException e) {
                    e.printStackTrace();
                }

                int totalBytes = 0; // 파일의 전체 바이트 수
                try {
                    totalBytes = fis.available();
                } catch (IOException e) {
                    throw new RuntimeException(e);
                }
                totalSamples = totalBytes / 2; // 샘플의 총 개수
                DataInputStream dis = new DataInputStream(fis);

                mAudioTrack.setNotificationMarkerPosition(totalSamples-1000);
                mAudioTrack.setPlaybackPositionUpdateListener(new AudioTrack.OnPlaybackPositionUpdateListener() {
                    @Override
                    public void onMarkerReached(AudioTrack audioTrack) {
                        getActivity().runOnUiThread(new Runnable() {
                            @Override
                            public void run() {
                                btn_record_play.setImageDrawable(getResources().getDrawable(R.drawable.play_off, null));
                                isPlaying = false;
                            }
                        });
                    }

                    @Override
                    public void onPeriodicNotification(AudioTrack audioTrack) {
                        // nothing to do
                    }
                });


                mAudioTrack.play();  // Play must be performed before writing

                while (isPlaying) {
                    try {
                        int ret = dis.read(writeData, 0, hop_len * 2); // byte 512

                        /**
                         * Real-time Audio Speech Enhancement
                         * */
                        if (btn_se_on) {
                            short[] shortData = byteArrayToShortArray(writeData); // short 256
                            double[] doubleData = shortArrayToDoubleArray(shortData); // double 256
                            double[] se_out = rtse_nutls.audioSE(doubleData); // 256
                            shortData = doubleArrayToShortArray(se_out); //
                            writeData = shortArrayToByteArray(shortData);
                        }

                        if (ret <= 0) {
                            getActivity().runOnUiThread(new Runnable() { // for UI
                                @Override
                                public void run() {
                                    isPlaying = false;
                                }
                            });
                            break;
                        }
                        mAudioTrack.write(writeData, 0, ret); // When writing to AudioTrack, it is transmitted to the speaker.
                        out_audio = concatenateByteArrays(out_audio, writeData);
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
        btn_record_play.setImageDrawable(getResources().getDrawable(R.drawable.play_on, null));
        isPlaying = true;
    }

    // stop playing the recording
    private void stopAudio() {
        btn_record_play.setImageDrawable(getResources().getDrawable(R.drawable.play_off, null));
        isPlaying = false;
        mAudioTrack.stop();
    }

    /**
     * Functions related to audio loading
     */
    public void onActivityResult(int requestCode, int resultCode, Intent data) {
        super.onActivityResult(requestCode, resultCode, data);

        if (requestCode == 101 && resultCode == -1) {
            selectedMediaUri = data.getData();
            audioFileName = FileUtils.getPath(ct, selectedMediaUri);
            Toast.makeText(getActivity().getApplicationContext(), "Upload is complete.", Toast.LENGTH_SHORT).show();
        }
    }

    public short[] byteArrayToShortArray(byte[] bytes) {
        short[] shorts = new short[bytes.length / 2];
        ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN).asShortBuffer().get(shorts);

        return shorts;
    }

    private double[] shortArrayToDoubleArray(short[] shortData) {
        int size = shortData.length;
        double[] doubleData = new double[size];
        double scalar = 1.0 / 32768.0;

        for (int i = 0; i < size; i++) {
            doubleData[i] = shortData[i] * scalar;
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


    /**
     * Audio download related functions
     */
    public static byte[] concatenateByteArrays(byte[] a, byte[] b) {
        byte[] result;
        if (a == null) {
            result = new byte[b.length];
            System.arraycopy(b, 0, result, 0, b.length);
        } else {
            result = new byte[a.length + b.length];
            System.arraycopy(a, 0, result, 0, a.length);
            System.arraycopy(b, 0, result, a.length, b.length);
        }

        return result;
    }

}
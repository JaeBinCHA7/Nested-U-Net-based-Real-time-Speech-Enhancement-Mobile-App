package com.example.speech_enhancement_rt_on_mobile;

import android.app.Activity;
import android.os.Build;
import android.util.Log;

import org.apache.commons.math3.complex.Complex;
import org.apache.commons.math3.transform.DftNormalization;
import org.apache.commons.math3.transform.FastFourierTransformer;
import org.apache.commons.math3.transform.TransformType;
import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.nnapi.NnApiDelegate;
import org.tensorflow.lite.support.common.FileUtil;

import java.io.IOException;
import java.nio.ByteOrder;
import java.nio.MappedByteBuffer;
import java.util.HashMap;
import java.util.Map;

public class SpeechEnhancement {
    private Interpreter.Options tfOptions = new Interpreter.Options();
    private NnApiDelegate nnApiDelegate = null;
    private MappedByteBuffer tfModel;
    private Interpreter tflite;
//    private int stride = 128;
    private int stride = 256;
    private int frame_len = 512;
    private double[] frameBuffer = new double[frame_len];
    private Complex[] realTranformedBuffer;
    private double[] in_buffer = new double[frame_len];
    private double[] out_buffer = new double[frame_len];

    // MSFE6 - Encoder
    private float[][][][] msfe6_en_prev1 = new float[1][1][256][64];
    private float[][][][] msfe6_en_prev2 = new float[1][1][128][32];
    private float[][][][] msfe6_en_prev3 = new float[1][1][64][32];
    private float[][][][] msfe6_en_prev4 = new float[1][1][32][32];
    private float[][][][] msfe6_en_prev5 = new float[1][1][16][32];
    private float[][][][] msfe6_en_prev6 = new float[1][1][8][32];
    private float[][] msfe6_en_h = new float[1][64];
    private float[][] msfe6_en_c = new float[1][64];
    // MSFE5 - Encoder
    private float[][][][] msfe5_en_prev1 = new float[1][1][128][64];
    private float[][][][] msfe5_en_prev2 = new float[1][1][64][32];
    private float[][][][] msfe5_en_prev3 = new float[1][1][32][32];
    private float[][][][] msfe5_en_prev4 = new float[1][1][16][32];
    private float[][][][] msfe5_en_prev5 = new float[1][1][8][32];
    private float[][] msfe5_en_h = new float[1][64];
    private float[][] msfe5_en_c = new float[1][64];
    // MSFE4 - Encoder
    private float[][][][] msfe4_en_prev1 = new float[1][1][64][64];
    private float[][][][] msfe4_en_prev2 = new float[1][1][32][32];
    private float[][][][] msfe4_en_prev3 = new float[1][1][16][32];
    private float[][][][] msfe4_en_prev4 = new float[1][1][8][32];
    private float[][] msfe4_en_h = new float[1][64];
    private float[][] msfe4_en_c = new float[1][64];
    // MSFE4(2) - Encoder
    private float[][][][] msfe4_en2_prev1 = new float[1][1][32][64];
    private float[][][][] msfe4_en2_prev2 = new float[1][1][16][32];
    private float[][][][] msfe4_en2_prev3 = new float[1][1][8][32];
    private float[][][][] msfe4_en2_prev4 = new float[1][1][4][32];
    private float[][] msfe4_en2_h = new float[1][32];
    private float[][] msfe4_en2_c = new float[1][32];
    // MSFE4(3) - Encoder
    private float[][][][] msfe4_en3_prev1 = new float[1][1][16][64];
    private float[][][][] msfe4_en3_prev2 = new float[1][1][8][32];
    private float[][][][] msfe4_en3_prev3 = new float[1][1][4][32];
    private float[][][][] msfe4_en3_prev4 = new float[1][1][2][32];
    private float[][] msfe4_en3_h = new float[1][16];
    private float[][] msfe4_en3_c = new float[1][16];
    // MSFE3 - Encoder
    private float[][][][] msfe3_en_prev1 = new float[1][1][8][64];
    private float[][][][] msfe3_en_prev2 = new float[1][1][4][32];
    private float[][][][] msfe3_en_prev3 = new float[1][1][2][32];
    private float[][] msfe3_en_h = new float[1][16];
    private float[][] msfe3_en_c = new float[1][16];
    // LSTM
    private float[][] h_state = new float[1][128];
    private float[][] c_state = new float[1][128];
    // MSFE3 - Decoder
    private float[][][][] msfe3_de_prev1 = new float[1][1][8][64];
    private float[][][][] msfe3_de_prev2 = new float[1][1][4][32];
    private float[][][][] msfe3_de_prev3 = new float[1][1][2][32];
    private float[][] msfe3_de_h = new float[1][16];
    private float[][] msfe3_de_c = new float[1][16];
    // MSFE4 - Decoder
    private float[][][][] msfe4_de_prev1 = new float[1][1][16][64];
    private float[][][][] msfe4_de_prev2 = new float[1][1][8][32];
    private float[][][][] msfe4_de_prev3 = new float[1][1][4][32];
    private float[][][][] msfe4_de_prev4 = new float[1][1][2][32];
    private float[][] msfe4_de_h = new float[1][16];
    private float[][] msfe4_de_c = new float[1][16];
    // MSFE4(2) - Decoder
    private float[][][][] msfe4_de2_prev1 = new float[1][1][32][64];
    private float[][][][] msfe4_de2_prev2 = new float[1][1][16][32];
    private float[][][][] msfe4_de2_prev3 = new float[1][1][8][32];
    private float[][][][] msfe4_de2_prev4 = new float[1][1][4][32];
    private float[][] msfe4_de2_h = new float[1][32];
    private float[][] msfe4_de2_c = new float[1][32];
    // MSFE4(3) - Decoder
    private float[][][][] msfe4_de3_prev1 = new float[1][1][64][64];
    private float[][][][] msfe4_de3_prev2 = new float[1][1][32][32];
    private float[][][][] msfe4_de3_prev3 = new float[1][1][16][32];
    private float[][][][] msfe4_de3_prev4 = new float[1][1][8][32];
    private float[][] msfe4_de3_h = new float[1][64];
    private float[][] msfe4_de3_c = new float[1][64];
    // MSFE5 - Decoder
    private float[][][][] msfe5_de_prev1 = new float[1][1][128][64];
    private float[][][][] msfe5_de_prev2 = new float[1][1][64][32];
    private float[][][][] msfe5_de_prev3 = new float[1][1][32][32];
    private float[][][][] msfe5_de_prev4 = new float[1][1][16][32];
    private float[][][][] msfe5_de_prev5 = new float[1][1][8][32];
    private float[][] msfe5_de_h = new float[1][64];
    private float[][] msfe5_de_c = new float[1][64];
    // MSFE6 - Decoder
    private float[][][][] msfe6_de_prev1 = new float[1][1][256][64];
    private float[][][][] msfe6_de_prev2 = new float[1][1][128][32];
    private float[][][][] msfe6_de_prev3 = new float[1][1][64][32];
    private float[][][][] msfe6_de_prev4 = new float[1][1][32][32];
    private float[][][][] msfe6_de_prev5 = new float[1][1][16][32];
    private float[][][][] msfe6_de_prev6 = new float[1][1][8][32];
    private float[][] msfe6_de_h = new float[1][64];
    private float[][] msfe6_de_c = new float[1][64];

    private float[][][][] tflite_out = new float[1][1][256][1];

    public SpeechEnhancement(Activity activity, String tflitePath) throws IOException {
        tfOptions.setNumThreads(-1);

        tfModel = FileUtil.loadMappedFile(activity, tflitePath);
        try {
            tflite = new Interpreter(tfModel, tfOptions);
        }catch (Exception e){
            throw new RuntimeException(e);
        }

        Log.d("", "Init interpreter of tflite");
    }

    // CRN - Conv1
    public float[][][][] runningTFLite(float[][][][] inputData) {

        /** TFLite Input */
        Map<String, Object> inputs = new HashMap<>();
        // MSFE 6 - Encoder
        inputs.put("input", inputData);
        inputs.put("msfe6_en_prev1", msfe6_en_prev1);
        inputs.put("msfe6_en_prev2", msfe6_en_prev2);
        inputs.put("msfe6_en_prev3", msfe6_en_prev3);
        inputs.put("msfe6_en_prev4", msfe6_en_prev4);
        inputs.put("msfe6_en_prev5", msfe6_en_prev5);
        inputs.put("msfe6_en_prev6", msfe6_en_prev6);
        inputs.put("msfe6_en_h", msfe6_en_h);
        inputs.put("msfe6_en_c", msfe6_en_c);
        // MSFE 5 Encoder
        inputs.put("msfe5_en_prev1", msfe5_en_prev1);
        inputs.put("msfe5_en_prev2", msfe5_en_prev2);
        inputs.put("msfe5_en_prev3", msfe5_en_prev3);
        inputs.put("msfe5_en_prev4", msfe5_en_prev4);
        inputs.put("msfe5_en_prev5", msfe5_en_prev5);
        inputs.put("msfe5_en_h", msfe5_en_h);
        inputs.put("msfe5_en_c", msfe5_en_c);
        // MSFE 4 Encoder
        inputs.put("msfe4_en_prev1", msfe4_en_prev1);
        inputs.put("msfe4_en_prev2", msfe4_en_prev2);
        inputs.put("msfe4_en_prev3", msfe4_en_prev3);
        inputs.put("msfe4_en_prev4", msfe4_en_prev4);
        inputs.put("msfe4_en_h", msfe4_en_h);
        inputs.put("msfe4_en_c", msfe4_en_c);
        // MSFE 4(2) Encoder
        inputs.put("msfe4_en2_prev1", msfe4_en2_prev1);
        inputs.put("msfe4_en2_prev2", msfe4_en2_prev2);
        inputs.put("msfe4_en2_prev3", msfe4_en2_prev3);
        inputs.put("msfe4_en2_prev4", msfe4_en2_prev4);
        inputs.put("msfe4_en2_h", msfe4_en2_h);
        inputs.put("msfe4_en2_c", msfe4_en2_c);
        // MSFE 4(3) Encoder
        inputs.put("msfe4_en3_prev1", msfe4_en3_prev1);
        inputs.put("msfe4_en3_prev2", msfe4_en3_prev2);
        inputs.put("msfe4_en3_prev3", msfe4_en3_prev3);
        inputs.put("msfe4_en3_prev4", msfe4_en3_prev4);
        inputs.put("msfe4_en3_h", msfe4_en3_h);
        inputs.put("msfe4_en3_c", msfe4_en3_c);
        // MSFE 3 Encoder
        inputs.put("msfe3_en_prev1", msfe3_en_prev1);
        inputs.put("msfe3_en_prev2", msfe3_en_prev2);
        inputs.put("msfe3_en_prev3", msfe3_en_prev3);
        inputs.put("msfe3_en_h", msfe3_en_h);
        inputs.put("msfe3_en_c", msfe3_en_c);
        // LSTM
        inputs.put("h_state", h_state);
        inputs.put("c_state", c_state);
        // MSFE 3 Decoder
        inputs.put("msfe3_de_prev1", msfe3_de_prev1);
        inputs.put("msfe3_de_prev2", msfe3_de_prev2);
        inputs.put("msfe3_de_prev3", msfe3_de_prev3);
        inputs.put("msfe3_de_h", msfe3_de_h);
        inputs.put("msfe3_de_c", msfe3_de_c);
        // MSFE 4 Decoder
        inputs.put("msfe4_de_prev1", msfe4_de_prev1);
        inputs.put("msfe4_de_prev2", msfe4_de_prev2);
        inputs.put("msfe4_de_prev3", msfe4_de_prev3);
        inputs.put("msfe4_de_prev4", msfe4_de_prev4);
        inputs.put("msfe4_de_h", msfe4_de_h);
        inputs.put("msfe4_de_c", msfe4_de_c);
        // MSFE 4(2) Decoder
        inputs.put("msfe4_de2_prev1", msfe4_de2_prev1);
        inputs.put("msfe4_de2_prev2", msfe4_de2_prev2);
        inputs.put("msfe4_de2_prev3", msfe4_de2_prev3);
        inputs.put("msfe4_de2_prev4", msfe4_de2_prev4);
        inputs.put("msfe4_de2_h", msfe4_de2_h);
        inputs.put("msfe4_de2_c", msfe4_de2_c);
        // MSFE 4(3) Decoder
        inputs.put("msfe4_de3_prev1", msfe4_de3_prev1);
        inputs.put("msfe4_de3_prev2", msfe4_de3_prev2);
        inputs.put("msfe4_de3_prev3", msfe4_de3_prev3);
        inputs.put("msfe4_de3_prev4", msfe4_de3_prev4);
        inputs.put("msfe4_de3_h", msfe4_de3_h);
        inputs.put("msfe4_de3_c", msfe4_de3_c);
        // MSFE 5 Decoder
        inputs.put("msfe5_de_prev1", msfe5_de_prev1);
        inputs.put("msfe5_de_prev2", msfe5_de_prev2);
        inputs.put("msfe5_de_prev3", msfe5_de_prev3);
        inputs.put("msfe5_de_prev4", msfe5_de_prev4);
        inputs.put("msfe5_de_prev5", msfe5_de_prev5);
        inputs.put("msfe5_de_h", msfe5_de_h);
        inputs.put("msfe5_de_c", msfe5_de_c);
        // MSFE 6 Decoder
        inputs.put("msfe6_de_prev1", msfe6_de_prev1);
        inputs.put("msfe6_de_prev2", msfe6_de_prev2);
        inputs.put("msfe6_de_prev3", msfe6_de_prev3);
        inputs.put("msfe6_de_prev4", msfe6_de_prev4);
        inputs.put("msfe6_de_prev5", msfe6_de_prev5);
        inputs.put("msfe6_de_prev6", msfe6_de_prev6);
        inputs.put("msfe6_de_h", msfe6_de_h);
        inputs.put("msfe6_de_c", msfe6_de_c);

        /** TFLite Output */
        Map<String, Object> outputs = new HashMap<>();

        outputs.put("msfe6_en_cur1", msfe6_en_prev1);
        outputs.put("msfe6_en_cur2", msfe6_en_prev2);
        outputs.put("msfe6_en_cur3", msfe6_en_prev3);
        outputs.put("msfe6_en_cur4", msfe6_en_prev4);
        outputs.put("msfe6_en_cur5", msfe6_en_prev5);
        outputs.put("msfe6_en_cur6", msfe6_en_prev6);
        outputs.put("msfe6_en_h", msfe6_en_h);
        outputs.put("msfe6_en_c", msfe6_en_c);
        // MSFE 5 Encoder
        outputs.put("msfe5_en_cur1", msfe5_en_prev1);
        outputs.put("msfe5_en_cur2", msfe5_en_prev2);
        outputs.put("msfe5_en_cur3", msfe5_en_prev3);
        outputs.put("msfe5_en_cur4", msfe5_en_prev4);
        outputs.put("msfe5_en_cur5", msfe5_en_prev5);
        outputs.put("msfe5_en_h", msfe5_en_h);
        outputs.put("msfe5_en_c", msfe5_en_c);
        // MSFE 4 Encoder
        outputs.put("msfe4_en_cur1", msfe4_en_prev1);
        outputs.put("msfe4_en_cur2", msfe4_en_prev2);
        outputs.put("msfe4_en_cur3", msfe4_en_prev3);
        outputs.put("msfe4_en_cur4", msfe4_en_prev4);
        outputs.put("msfe4_en_h", msfe4_en_h);
        outputs.put("msfe4_en_c", msfe4_en_c);
        // MSFE 4(2) Encoder
        outputs.put("msfe4_en2_cur1", msfe4_en2_prev1);
        outputs.put("msfe4_en2_cur2", msfe4_en2_prev2);
        outputs.put("msfe4_en2_cur3", msfe4_en2_prev3);
        outputs.put("msfe4_en2_cur4", msfe4_en2_prev4);
        outputs.put("msfe4_en2_h", msfe4_en2_h);
        outputs.put("msfe4_en2_c", msfe4_en2_c);
        // MSFE 4(3) Encoder
        outputs.put("msfe4_en3_cur1", msfe4_en3_prev1);
        outputs.put("msfe4_en3_cur2", msfe4_en3_prev2);
        outputs.put("msfe4_en3_cur3", msfe4_en3_prev3);
        outputs.put("msfe4_en3_cur4", msfe4_en3_prev4);
        outputs.put("msfe4_en3_h", msfe4_en3_h);
        outputs.put("msfe4_en3_c", msfe4_en3_c);
        // MSFE 3 Encoder
        outputs.put("msfe3_en_cur1", msfe3_en_prev1);
        outputs.put("msfe3_en_cur2", msfe3_en_prev2);
        outputs.put("msfe3_en_cur3", msfe3_en_prev3);
        outputs.put("msfe3_en_h", msfe3_en_h);
        outputs.put("msfe3_en_c", msfe3_en_c);
        // LSTM
        outputs.put("h_state", h_state);
        outputs.put("c_state", c_state);
        // MSFE 3 Decoder
        outputs.put("msfe3_de_cur1", msfe3_de_prev1);
        outputs.put("msfe3_de_cur2", msfe3_de_prev2);
        outputs.put("msfe3_de_cur3", msfe3_de_prev3);
        outputs.put("msfe3_de_h", msfe3_de_h);
        outputs.put("msfe3_de_c", msfe3_de_c);
        // MSFE 4 Decoder
        outputs.put("msfe4_de_cur1", msfe4_de_prev1);
        outputs.put("msfe4_de_cur2", msfe4_de_prev2);
        outputs.put("msfe4_de_cur3", msfe4_de_prev3);
        outputs.put("msfe4_de_cur4", msfe4_de_prev4);
        outputs.put("msfe4_de_h", msfe4_de_h);
        outputs.put("msfe4_de_c", msfe4_de_c);
        // MSFE 4(2) Decoder
        outputs.put("msfe4_de2_cur1", msfe4_de2_prev1);
        outputs.put("msfe4_de2_cur2", msfe4_de2_prev2);
        outputs.put("msfe4_de2_cur3", msfe4_de2_prev3);
        outputs.put("msfe4_de2_cur4", msfe4_de2_prev4);
        outputs.put("msfe4_de2_h", msfe4_de2_h);
        outputs.put("msfe4_de2_c", msfe4_de2_c);
        // MSFE 4(3) Decoder
        outputs.put("msfe4_de3_cur1", msfe4_de3_prev1);
        outputs.put("msfe4_de3_cur2", msfe4_de3_prev2);
        outputs.put("msfe4_de3_cur3", msfe4_de3_prev3);
        outputs.put("msfe4_de3_cur4", msfe4_de3_prev4);
        outputs.put("msfe4_de3_h", msfe4_de3_h);
        outputs.put("msfe4_de3_c", msfe4_de3_c);
        // MSFE 5 Decoder
        outputs.put("msfe5_de_cur1", msfe5_de_prev1);
        outputs.put("msfe5_de_cur2", msfe5_de_prev2);
        outputs.put("msfe5_de_cur3", msfe5_de_prev3);
        outputs.put("msfe5_de_cur4", msfe5_de_prev4);
        outputs.put("msfe5_de_cur5", msfe5_de_prev5);
        outputs.put("msfe5_de_h", msfe5_de_h);
        outputs.put("msfe5_de_c", msfe5_de_c);
        // MSFE 6 Decoder
        outputs.put("msfe6_de_cur1", msfe6_de_prev1);
        outputs.put("msfe6_de_cur2", msfe6_de_prev2);
        outputs.put("msfe6_de_cur3", msfe6_de_prev3);
        outputs.put("msfe6_de_cur4", msfe6_de_prev4);
        outputs.put("msfe6_de_cur5", msfe6_de_prev5);
        outputs.put("msfe6_de_cur6", msfe6_de_prev6);
        outputs.put("msfe6_de_h", msfe6_de_h);
        outputs.put("msfe6_de_c", msfe6_de_c);
        outputs.put("model_out", tflite_out);

        double startTime = System.currentTimeMillis();
        tflite.runSignature(inputs, outputs, "nunet_lstm");
        double endTime = System.currentTimeMillis();
        System.out.println("TFLite processing time : " + (endTime - startTime) / 1000 + "s\n");

        return tflite_out;
    }

    public double[] audioSE(double[] doubleData) {
        double startTime = System.currentTimeMillis();

        // Shift left and Input audio
        in_buffer = shiftLeft(in_buffer, stride);
        in_buffer = audioToBuffer(in_buffer, doubleData);

        // FFT
        FastFourierTransformer fft = new FastFourierTransformer(DftNormalization.STANDARD);
        Complex[] transformedBuffer = fft.transform(in_buffer, TransformType.FORWARD);

        realTranformedBuffer = getComplexSlice(transformedBuffer, 0, 257);

        // get magnitude/phase
        float[] mags = new float[realTranformedBuffer.length];
        float[] phase = new float[realTranformedBuffer.length];
        for (int i = 0; i < realTranformedBuffer.length; i++) {
            double real = realTranformedBuffer[i].getReal();
            double imag = realTranformedBuffer[i].getImaginary();
            mags[i] = (float) Math.sqrt(Math.pow(real, 2) + Math.pow(imag, 2));
            phase[i] = (float) Math.atan2(imag, real);
        } // [257]

        // Reshape [257] -> [1,1,256,1]
        float[][][][] remags = reshapeBeforeEncoder(mags);

        /** TFLite */
        float[][][][] tflite_out = runningTFLite(remags);

        // Reshape and Padding [1,1,256,1] -> [1,1,257]
        float[][][] out_mask = reshapeAndPadding(tflite_out);
//        float[][][] out_mask = reshapeAndPadding(remags);


        // Reshape [1,1,257] -> [257]
        float[] maskOneDim = reshapeThreeToOne(out_mask);

        // TF-Masking
        double[] outReal = getReal(mags, phase, maskOneDim);
        double[] outImag = getImag(mags, phase, maskOneDim);

        // IFFT
        Complex[] outComplex = createComplex(outReal, outImag);
        Complex[] fullComplex = reconstructComplex(outComplex);
        FastFourierTransformer ifft = new FastFourierTransformer(DftNormalization.STANDARD);
        Complex[] detransBuffer = ifft.transform(fullComplex, TransformType.INVERSE);

        // Get real
        double[] real = new double[detransBuffer.length];
        for (int i = 0; i < detransBuffer.length; i++) {
            real[i] = detransBuffer[i].getReal();
        }
        // Out-buffer : Shift Left and Input zero
        out_buffer = shiftLeft(out_buffer, stride);

        // Out-buffer : Overlap-add
        out_buffer = overlapAdd(out_buffer, real);

        double[] se_out = getSlice(out_buffer, 0, stride);

        for(int i=0; i <se_out.length;i++){
            se_out[i] /= 2;
        }

        double endTime = System.currentTimeMillis();
        System.out.println("1 loop's processing time : " + (endTime - startTime) / 1000 + "s\n");

        double loss = getLoss(doubleData, se_out);
        System.out.println("Loss : " + loss);


        return se_out;
    }

    private double[] overlapAdd(double[] buffer, double[] audio) {
        for (int i = 0; i < buffer.length; i++) {
            buffer[i] = buffer[i] + audio[i];
        }

        return buffer;
    }

    // Reshape [257] -> [1,1,256,1]
    private float[][][][] reshapeBeforeEncoder(float[] input) {
        float[][][][] matrix = new float[1][1][input.length - 1][1];
        for (int i = 1; i < input.length; i++) {
            matrix[0][0][i - 1][0] = input[i];
        }
        return matrix;
    }

    public Complex[] getComplexSlice(Complex[] array, int startIndex, int endIndex) {
        // Get the slice of the Array
        Complex[] slicedArray = new Complex[endIndex - startIndex];
        //copying array elements from the original array to the newly created sliced array
        for (int i = 0; i < slicedArray.length; i++) {
            slicedArray[i] = array[startIndex + i];
        }
        //returns the slice of an array
        return slicedArray;
    }


    private Complex[] reconstructComplex(Complex[] half) {
        Complex[] fullComplex = new Complex[(half.length - 1) * 2];
        Complex[] conjugateComplex = new Complex[half.length - 1];
        for (int i = 0; i < half.length; i++) { // 0~256
            fullComplex[i] = half[i];
        }

        for (int i = 1; i < half.length - 1; i++) { // 1~255
            conjugateComplex[i] = half[i].conjugate();
            fullComplex[fullComplex.length - i] = conjugateComplex[i];
        }
        return fullComplex;
    }

    private Complex[] createComplex(double[] real, double[] imag) {
        Complex[] complexArray = new Complex[real.length];
        for (int i = 0; i < real.length; i++) {
            complexArray[i] = new Complex(real[i], imag[i]);
        }

        return complexArray;
    }

    // Reshape and Padding [1,1,256,1] -> [1,1,257]
    private float[][][] reshapeAndPadding(float[][][][] input) {
        float[][][] matrix = new float[1][1][257];
        matrix[0][0][0] = 0;
        for (int i = 1; i < 257; i++) {
            matrix[0][0][i] = input[0][0][i - 1][0];
        }

        return matrix;
    }

    private double[] getReal(float[] mag, float[] phase, float[] mask) {
        float[] enhancedMag = new float[mag.length];
        double[] real = new double[mag.length];

        for (int k = 0; k < enhancedMag.length; k++) {
            enhancedMag[k] = mag[k] * mask[k];
            real[k] = enhancedMag[k] * (float) Math.cos(phase[k]);

        }

        return real;
    }

    private double[] getImag(float[] mag, float[] phase, float[] mask) {
        float[] enhancedMag = new float[mag.length];
        double[] imag = new double[mag.length];

        for (int k = 0; k < enhancedMag.length; k++) {
            enhancedMag[k] = mag[k] * mask[k];
            imag[k] = enhancedMag[k] * (float) Math.sin(phase[k]);
        }

        return imag;
    }

    private double[] shiftLeft(double[] input, int stride) {
        for (int i = 0; i < (frame_len - stride); i++) {
            input[i] = input[stride + i];
        }
        for (int i = (frame_len - stride); i < frame_len; i++) {
            input[i] = 0;
        }
        return input;
    }

    private double[] audioToBuffer(double[] buffer, double[] audio) {
        for (int i = 0; i < audio.length; i++) {
            buffer[buffer.length - audio.length + i] = audio[i];
        }

        return buffer;
    }

    private float[] reshapeThreeToOne(float[][][] matrix) {
        float[] array = new float[matrix[0][0].length];
        for (int i = 0; i < matrix[0][0].length; i++) {
            array[i] = matrix[0][0][i];
        }

        return array;
    }

    public double[] getSlice(double[] array, int startIndex, int endIndex) {
        // Get the slice of the Array
        double[] slicedArray = new double[endIndex - startIndex];
        //copying array elements from the original array to the newly created sliced array
        for (int i = 0; i < slicedArray.length; i++) {
            slicedArray[i] = array[startIndex + i];
        }
        //returns the slice of an array
        return slicedArray;
    }

    private double getLoss(double[] in_audio, double[] est_audio) {
        double[] loss = new double[in_audio.length];
        double sum = 0;

        for (int i = 0; i < in_audio.length; i++) {
            loss[i] = est_audio[i] - in_audio[i];
        }

        for (int i = 0; i < loss.length; i++) {
            loss[i] = Math.pow(loss[i], 2);
            sum += loss[i];
        }
        double avg = sum / loss.length;

        return avg;
    }
}

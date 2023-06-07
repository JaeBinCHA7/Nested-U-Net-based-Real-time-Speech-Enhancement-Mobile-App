import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppressiong of information
import soundfile as sf
import numpy as np
import tensorflow as tf
import time
from pesq import pesq as get_pesq
from numpy import concatenate

tf.debugging.set_log_device_placement(True)
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


def real_time_speech_enhancer(noisy_speech):
    # Configuration
    frame_len = 512
    frame_step = 256

    tf_window = tf.signal.hann_window(frame_len)
    tf_window = tf_window.numpy()
    tf_window[0], tf_window[-1] = 1e-7, 1e-7

    inverse_window_fn = tf.signal.inverse_stft_window_fn(frame_step, forward_window_fn=tf.signal.hann_window, name=None)
    inverse_window = inverse_window_fn(frame_len, dtype=tf.float32)
    inverse_window = inverse_window.numpy()
    ############################################################################
    # Define buffer for input frames
    ############################################################################
    in_buffer = np.zeros((frame_len)).astype('float32')  # [512,]
    out_buffer = np.zeros((frame_len)).astype('float32')  # [512,]
    num_blocks = (audio.shape[0] - (frame_len - frame_step)) // frame_step
    out_file = np.zeros((len(audio) + (frame_len - frame_step)))  # [48000,]
    time_array = []

    tflite_out = {
        'input': np.zeros((1, 1, 256, 1), dtype=np.float32),
        # Encoder MSFE6 encoder
        'msfe6_ee_cur1': np.zeros((1, 1, 256, 64), dtype=np.float32),
        'msfe6_ee_cur2': np.zeros((1, 1, 128, 32), dtype=np.float32),
        'msfe6_ee_cur3': np.zeros((1, 1, 64, 32), dtype=np.float32),
        'msfe6_ee_cur4': np.zeros((1, 1, 32, 32), dtype=np.float32),
        'msfe6_ee_cur5': np.zeros((1, 1, 16, 32), dtype=np.float32),
        'msfe6_ee_cur6': np.zeros((1, 1, 8, 32), dtype=np.float32),
        # Encoder MSFE5 encoder
        'msfe5_ee_cur1': np.zeros((1, 1, 128, 64), dtype=np.float32),
        'msfe5_ee_cur2': np.zeros((1, 1, 64, 32), dtype=np.float32),
        'msfe5_ee_cur3': np.zeros((1, 1, 32, 32), dtype=np.float32),
        'msfe5_ee_cur4': np.zeros((1, 1, 16, 32), dtype=np.float32),
        'msfe5_ee_cur5': np.zeros((1, 1, 8, 32), dtype=np.float32),
        # Encoder MSFE4 encoder
        'msfe4_ee_cur1': np.zeros((1, 1, 64, 64), dtype=np.float32),
        'msfe4_ee_cur2': np.zeros((1, 1, 32, 32), dtype=np.float32),
        'msfe4_ee_cur3': np.zeros((1, 1, 16, 32), dtype=np.float32),
        'msfe4_ee_cur4': np.zeros((1, 1, 8, 32), dtype=np.float32),
        # Encoder MSFE4 (2) encoder
        'msfe4_ee2_cur1': np.zeros((1, 1, 32, 64), dtype=np.float32),
        'msfe4_ee2_cur2': np.zeros((1, 1, 16, 32), dtype=np.float32),
        'msfe4_ee2_cur3': np.zeros((1, 1, 8, 32), dtype=np.float32),
        'msfe4_ee2_cur4': np.zeros((1, 1, 4, 32), dtype=np.float32),
        # Encoder MSFE4 (3) encoder
        'msfe4_ee3_cur1': np.zeros((1, 1, 16, 64), dtype=np.float32),
        'msfe4_ee3_cur2': np.zeros((1, 1, 8, 32), dtype=np.float32),
        'msfe4_ee3_cur3': np.zeros((1, 1, 4, 32), dtype=np.float32),
        'msfe4_ee3_cur4': np.zeros((1, 1, 2, 32), dtype=np.float32),
        # Encoder MSFE3 encoder
        'msfe3_ee_cur1': np.zeros((1, 1, 8, 64), dtype=np.float32),
        'msfe3_ee_cur2': np.zeros((1, 1, 4, 32), dtype=np.float32),
        'msfe3_ee_cur3': np.zeros((1, 1, 2, 32), dtype=np.float32),

        # Encoder MSFE6 decoder
        'msfe6_ed_cur1': np.zeros((1, 1, 4, 64), dtype=np.float32),
        'msfe6_ed_cur2': np.zeros((1, 1, 8, 64), dtype=np.float32),
        'msfe6_ed_cur3': np.zeros((1, 1, 16, 64), dtype=np.float32),
        'msfe6_ed_cur4': np.zeros((1, 1, 32, 64), dtype=np.float32),
        'msfe6_ed_cur5': np.zeros((1, 1, 64, 64), dtype=np.float32),
        'msfe6_ed_cur6': np.zeros((1, 1, 128, 64), dtype=np.float32),
        # Encoder MSFE5 decoder
        'msfe5_ed_cur1': np.zeros((1, 1, 4, 64), dtype=np.float32),
        'msfe5_ed_cur2': np.zeros((1, 1, 8, 64), dtype=np.float32),
        'msfe5_ed_cur3': np.zeros((1, 1, 16, 64), dtype=np.float32),
        'msfe5_ed_cur4': np.zeros((1, 1, 32, 64), dtype=np.float32),
        'msfe5_ed_cur5': np.zeros((1, 1, 64, 64), dtype=np.float32),
        # Encoder MSFE4 decoder
        'msfe4_ed_cur1': np.zeros((1, 1, 4, 64), dtype=np.float32),
        'msfe4_ed_cur2': np.zeros((1, 1, 8, 64), dtype=np.float32),
        'msfe4_ed_cur3': np.zeros((1, 1, 16, 64), dtype=np.float32),
        'msfe4_ed_cur4': np.zeros((1, 1, 32, 64), dtype=np.float32),
        # Encoder MSFE4 (2) decoder
        'msfe4_ed2_cur1': np.zeros((1, 1, 2, 64), dtype=np.float32),
        'msfe4_ed2_cur2': np.zeros((1, 1, 4, 64), dtype=np.float32),
        'msfe4_ed2_cur3': np.zeros((1, 1, 8, 64), dtype=np.float32),
        'msfe4_ed2_cur4': np.zeros((1, 1, 16, 64), dtype=np.float32),
        # Encoder MSFE4 (3) decoder
        'msfe4_ed3_cur1': np.zeros((1, 1, 1, 64), dtype=np.float32),
        'msfe4_ed3_cur2': np.zeros((1, 1, 2, 64), dtype=np.float32),
        'msfe4_ed3_cur3': np.zeros((1, 1, 4, 64), dtype=np.float32),
        'msfe4_ed3_cur4': np.zeros((1, 1, 8, 64), dtype=np.float32),
        # Encoder MSFE3 decoder
        'msfe3_ed_cur1': np.zeros((1, 1, 1, 64), dtype=np.float32),
        'msfe3_ed_cur2': np.zeros((1, 1, 2, 64), dtype=np.float32),
        'msfe3_ed_cur3': np.zeros((1, 1, 4, 64), dtype=np.float32),

        # Decoder MSFE3 encoder
        'msfe3_de_cur1': np.zeros((1, 1, 8, 128), dtype=np.float32),
        'msfe3_de_cur2': np.zeros((1, 1, 4, 64), dtype=np.float32),
        'msfe3_de_cur3': np.zeros((1, 1, 2, 64), dtype=np.float32),
        # Decoder MSFE4  encoder
        'msfe4_de_cur1': np.zeros((1, 1, 16, 128), dtype=np.float32),
        'msfe4_de_cur2': np.zeros((1, 1, 8, 64), dtype=np.float32),
        'msfe4_de_cur3': np.zeros((1, 1, 4, 64), dtype=np.float32),
        'msfe4_de_cur4': np.zeros((1, 1, 2, 64), dtype=np.float32),
        # Decoder MSFE4 (2) encoder
        'msfe4_de2_cur1': np.zeros((1, 1, 32, 128), dtype=np.float32),
        'msfe4_de2_cur2': np.zeros((1, 1, 16, 64), dtype=np.float32),
        'msfe4_de2_cur3': np.zeros((1, 1, 8, 64), dtype=np.float32),
        'msfe4_de2_cur4': np.zeros((1, 1, 4, 64), dtype=np.float32),
        # Decoder MSFE4 (3) encoder
        'msfe4_de3_cur1': np.zeros((1, 1, 64, 128), dtype=np.float32),
        'msfe4_de3_cur2': np.zeros((1, 1, 32, 64), dtype=np.float32),
        'msfe4_de3_cur3': np.zeros((1, 1, 16, 64), dtype=np.float32),
        'msfe4_de3_cur4': np.zeros((1, 1, 8, 64), dtype=np.float32),
        # Decoder MSFE5 encoder
        'msfe5_de_cur1': np.zeros((1, 1, 128, 128), dtype=np.float32),
        'msfe5_de_cur2': np.zeros((1, 1, 64, 64), dtype=np.float32),
        'msfe5_de_cur3': np.zeros((1, 1, 32, 64), dtype=np.float32),
        'msfe5_de_cur4': np.zeros((1, 1, 16, 64), dtype=np.float32),
        'msfe5_de_cur5': np.zeros((1, 1, 8, 64), dtype=np.float32),
        # Decoder MSFE6 Encoder
        'msfe6_de_cur1': np.zeros((1, 1, 256, 128), dtype=np.float32),
        'msfe6_de_cur2': np.zeros((1, 1, 128, 64), dtype=np.float32),
        'msfe6_de_cur3': np.zeros((1, 1, 64, 64), dtype=np.float32),
        'msfe6_de_cur4': np.zeros((1, 1, 32, 64), dtype=np.float32),
        'msfe6_de_cur5': np.zeros((1, 1, 16, 64), dtype=np.float32),
        'msfe6_de_cur6': np.zeros((1, 1, 8, 64), dtype=np.float32),

        # Decoder MSFE3 decoder
        'msfe3_dd_cur1': np.zeros((1, 1, 1, 64), dtype=np.float32),
        'msfe3_dd_cur2': np.zeros((1, 1, 2, 64), dtype=np.float32),
        'msfe3_dd_cur3': np.zeros((1, 1, 4, 64), dtype=np.float32),
        # Decoder MSFE4 decoder
        'msfe4_dd_cur1': np.zeros((1, 1, 1, 64), dtype=np.float32),
        'msfe4_dd_cur2': np.zeros((1, 1, 2, 64), dtype=np.float32),
        'msfe4_dd_cur3': np.zeros((1, 1, 4, 64), dtype=np.float32),
        'msfe4_dd_cur4': np.zeros((1, 1, 8, 64), dtype=np.float32),
        # Decoder MSFE4 (2) decoder
        'msfe4_dd2_cur1': np.zeros((1, 1, 2, 64), dtype=np.float32),
        'msfe4_dd2_cur2': np.zeros((1, 1, 4, 64), dtype=np.float32),
        'msfe4_dd2_cur3': np.zeros((1, 1, 8, 64), dtype=np.float32),
        'msfe4_dd2_cur4': np.zeros((1, 1, 16, 64), dtype=np.float32),
        # Decoder MSFE4 (3) decoder
        'msfe4_dd3_cur1': np.zeros((1, 1, 4, 64), dtype=np.float32),
        'msfe4_dd3_cur2': np.zeros((1, 1, 8, 64), dtype=np.float32),
        'msfe4_dd3_cur3': np.zeros((1, 1, 16, 64), dtype=np.float32),
        'msfe4_dd3_cur4': np.zeros((1, 1, 32, 64), dtype=np.float32),
        # Decoder MSFE5 decoder
        'msfe5_dd_cur1': np.zeros((1, 1, 4, 64), dtype=np.float32),
        'msfe5_dd_cur2': np.zeros((1, 1, 8, 64), dtype=np.float32),
        'msfe5_dd_cur3': np.zeros((1, 1, 16, 64), dtype=np.float32),
        'msfe5_dd_cur4': np.zeros((1, 1, 32, 64), dtype=np.float32),
        'msfe5_dd_cur5': np.zeros((1, 1, 64, 64), dtype=np.float32),
        # Decoder MSFE6 decoder
        'msfe6_dd_cur1': np.zeros((1, 1, 4, 64), dtype=np.float32),
        'msfe6_dd_cur2': np.zeros((1, 1, 8, 64), dtype=np.float32),
        'msfe6_dd_cur3': np.zeros((1, 1, 16, 64), dtype=np.float32),
        'msfe6_dd_cur4': np.zeros((1, 1, 32, 64), dtype=np.float32),
        'msfe6_dd_cur5': np.zeros((1, 1, 64, 64), dtype=np.float32),
        'msfe6_dd_cur6': np.zeros((1, 1, 128, 64), dtype=np.float32),

        # State for LSTM
        'msfe6_en_h': np.zeros((1, 21), dtype=np.float32),
        'msfe6_en_c': np.zeros((1, 21), dtype=np.float32),
        'msfe5_en_h': np.zeros((1, 21), dtype=np.float32),
        'msfe5_en_c': np.zeros((1, 21), dtype=np.float32),
        'msfe4_en_h': np.zeros((1, 21), dtype=np.float32),
        'msfe4_en_c': np.zeros((1, 21), dtype=np.float32),
        'msfe4_en2_h': np.zeros((1, 21), dtype=np.float32),
        'msfe4_en2_c': np.zeros((1, 21), dtype=np.float32),
        'msfe4_en3_h': np.zeros((1, 21), dtype=np.float32),
        'msfe4_en3_c': np.zeros((1, 21), dtype=np.float32),
        'msfe3_en_h': np.zeros((1, 21), dtype=np.float32),
        'msfe3_en_c': np.zeros((1, 21), dtype=np.float32),
        'state_h': np.zeros((1, 21), dtype=np.float32),
        'state_c': np.zeros((1, 21), dtype=np.float32),
        'msfe3_de_h': np.zeros((1, 21), dtype=np.float32),
        'msfe3_de_c': np.zeros((1, 21), dtype=np.float32),
        'msfe4_de_h': np.zeros((1, 21), dtype=np.float32),
        'msfe4_de_c': np.zeros((1, 21), dtype=np.float32),
        'msfe4_de2_h': np.zeros((1, 21), dtype=np.float32),
        'msfe4_de2_c': np.zeros((1, 21), dtype=np.float32),
        'msfe4_de3_h': np.zeros((1, 21), dtype=np.float32),
        'msfe4_de3_c': np.zeros((1, 21), dtype=np.float32),
        'msfe5_de_h': np.zeros((1, 21), dtype=np.float32),
        'msfe5_de_c': np.zeros((1, 21), dtype=np.float32),
        'msfe6_de_h': np.zeros((1, 21), dtype=np.float32),
        'msfe6_de_c': np.zeros((1, 21), dtype=np.float32),
        "model_out": np.zeros((1, 1, 256, 1), dtype=np.float32)
    }

    for idx in range(num_blocks):
        start_time = time.time()

        in_buffer[:-frame_step] = in_buffer[frame_step:]
        in_buffer[-frame_step:] = noisy_speech[idx * frame_step:(idx * frame_step) + frame_step]
        windowed_buffer = in_buffer * tf_window

        in_block_fft = np.fft.rfft(windowed_buffer)  # [257,]

        in_mag = np.abs(in_block_fft)  # [257,]
        in_phase = np.angle(in_block_fft)  # [257,]

        expand_mag = np.reshape(in_mag, (1, 1, -1, 1)).astype('float32')
        sliced_mag = expand_mag[:, :, 1:]  # 검증 완료

        tflite_out = nutls_lstm_sm(input=sliced_mag,
                                   msfe6_ee_prev1=tflite_out['msfe6_ee_cur1'],
                                   msfe6_ee_prev2=tflite_out['msfe6_ee_cur2'],
                                   msfe6_ee_prev3=tflite_out['msfe6_ee_cur3'],
                                   msfe6_ee_prev4=tflite_out['msfe6_ee_cur4'],
                                   msfe6_ee_prev5=tflite_out['msfe6_ee_cur5'],
                                   msfe6_ee_prev6=tflite_out['msfe6_ee_cur6'],
                                   msfe5_ee_prev1=tflite_out['msfe5_ee_cur1'],
                                   msfe5_ee_prev2=tflite_out['msfe5_ee_cur2'],
                                   msfe5_ee_prev3=tflite_out['msfe5_ee_cur3'],
                                   msfe5_ee_prev4=tflite_out['msfe5_ee_cur4'],
                                   msfe5_ee_prev5=tflite_out['msfe5_ee_cur5'],
                                   msfe4_ee_prev1=tflite_out['msfe4_ee_cur1'],
                                   msfe4_ee_prev2=tflite_out['msfe4_ee_cur2'],
                                   msfe4_ee_prev3=tflite_out['msfe4_ee_cur3'],
                                   msfe4_ee_prev4=tflite_out['msfe4_ee_cur4'],
                                   msfe4_ee2_prev1=tflite_out['msfe4_ee2_cur1'],
                                   msfe4_ee2_prev2=tflite_out['msfe4_ee2_cur2'],
                                   msfe4_ee2_prev3=tflite_out['msfe4_ee2_cur3'],
                                   msfe4_ee2_prev4=tflite_out['msfe4_ee2_cur4'],
                                   msfe4_ee3_prev1=tflite_out['msfe4_ee3_cur1'],
                                   msfe4_ee3_prev2=tflite_out['msfe4_ee3_cur2'],
                                   msfe4_ee3_prev3=tflite_out['msfe4_ee3_cur3'],
                                   msfe4_ee3_prev4=tflite_out['msfe4_ee3_cur4'],
                                   msfe3_ee_prev1=tflite_out['msfe3_ee_cur1'],
                                   msfe3_ee_prev2=tflite_out['msfe3_ee_cur2'],
                                   msfe3_ee_prev3=tflite_out['msfe3_ee_cur3'],

                                   msfe6_ed_prev1=tflite_out['msfe6_ed_cur1'],
                                   msfe6_ed_prev2=tflite_out['msfe6_ed_cur2'],
                                   msfe6_ed_prev3=tflite_out['msfe6_ed_cur3'],
                                   msfe6_ed_prev4=tflite_out['msfe6_ed_cur4'],
                                   msfe6_ed_prev5=tflite_out['msfe6_ed_cur5'],
                                   msfe6_ed_prev6=tflite_out['msfe6_ed_cur6'],
                                   msfe5_ed_prev1=tflite_out['msfe5_ed_cur1'],
                                   msfe5_ed_prev2=tflite_out['msfe5_ed_cur2'],
                                   msfe5_ed_prev3=tflite_out['msfe5_ed_cur3'],
                                   msfe5_ed_prev4=tflite_out['msfe5_ed_cur4'],
                                   msfe5_ed_prev5=tflite_out['msfe5_ed_cur5'],
                                   msfe4_ed_prev1=tflite_out['msfe4_ed_cur1'],
                                   msfe4_ed_prev2=tflite_out['msfe4_ed_cur2'],
                                   msfe4_ed_prev3=tflite_out['msfe4_ed_cur3'],
                                   msfe4_ed_prev4=tflite_out['msfe4_ed_cur4'],
                                   msfe4_ed2_prev1=tflite_out['msfe4_ed2_cur1'],
                                   msfe4_ed2_prev2=tflite_out['msfe4_ed2_cur2'],
                                   msfe4_ed2_prev3=tflite_out['msfe4_ed2_cur3'],
                                   msfe4_ed2_prev4=tflite_out['msfe4_ed2_cur4'],
                                   msfe4_ed3_prev1=tflite_out['msfe4_ed3_cur1'],
                                   msfe4_ed3_prev2=tflite_out['msfe4_ed3_cur2'],
                                   msfe4_ed3_prev3=tflite_out['msfe4_ed3_cur3'],
                                   msfe4_ed3_prev4=tflite_out['msfe4_ed3_cur4'],
                                   msfe3_ed_prev1=tflite_out['msfe3_ed_cur1'],
                                   msfe3_ed_prev2=tflite_out['msfe3_ed_cur2'],
                                   msfe3_ed_prev3=tflite_out['msfe3_ed_cur3'],

                                   msfe3_de_prev1=tflite_out['msfe3_de_cur1'],
                                   msfe3_de_prev2=tflite_out['msfe3_de_cur2'],
                                   msfe3_de_prev3=tflite_out['msfe3_de_cur3'],
                                   msfe4_de_prev1=tflite_out['msfe4_de_cur1'],
                                   msfe4_de_prev2=tflite_out['msfe4_de_cur2'],
                                   msfe4_de_prev3=tflite_out['msfe4_de_cur3'],
                                   msfe4_de_prev4=tflite_out['msfe4_de_cur4'],
                                   msfe4_de2_prev1=tflite_out['msfe4_de2_cur1'],
                                   msfe4_de2_prev2=tflite_out['msfe4_de2_cur2'],
                                   msfe4_de2_prev3=tflite_out['msfe4_de2_cur3'],
                                   msfe4_de2_prev4=tflite_out['msfe4_de2_cur4'],
                                   msfe4_de3_prev1=tflite_out['msfe4_de3_cur1'],
                                   msfe4_de3_prev2=tflite_out['msfe4_de3_cur2'],
                                   msfe4_de3_prev3=tflite_out['msfe4_de3_cur3'],
                                   msfe4_de3_prev4=tflite_out['msfe4_de3_cur4'],
                                   msfe5_de_prev1=tflite_out['msfe5_de_cur1'],
                                   msfe5_de_prev2=tflite_out['msfe5_de_cur2'],
                                   msfe5_de_prev3=tflite_out['msfe5_de_cur3'],
                                   msfe5_de_prev4=tflite_out['msfe5_de_cur4'],
                                   msfe5_de_prev5=tflite_out['msfe5_de_cur5'],
                                   msfe6_de_prev1=tflite_out['msfe6_de_cur1'],
                                   msfe6_de_prev2=tflite_out['msfe6_de_cur2'],
                                   msfe6_de_prev3=tflite_out['msfe6_de_cur3'],
                                   msfe6_de_prev4=tflite_out['msfe6_de_cur4'],
                                   msfe6_de_prev5=tflite_out['msfe6_de_cur5'],
                                   msfe6_de_prev6=tflite_out['msfe6_de_cur6'],

                                   msfe3_dd_prev1=tflite_out['msfe3_dd_cur1'],
                                   msfe3_dd_prev2=tflite_out['msfe3_dd_cur2'],
                                   msfe3_dd_prev3=tflite_out['msfe3_dd_cur3'],
                                   msfe4_dd_prev1=tflite_out['msfe4_dd_cur1'],
                                   msfe4_dd_prev2=tflite_out['msfe4_dd_cur2'],
                                   msfe4_dd_prev3=tflite_out['msfe4_dd_cur3'],
                                   msfe4_dd_prev4=tflite_out['msfe4_dd_cur4'],
                                   msfe4_dd2_prev1=tflite_out['msfe4_dd2_cur1'],
                                   msfe4_dd2_prev2=tflite_out['msfe4_dd2_cur2'],
                                   msfe4_dd2_prev3=tflite_out['msfe4_dd2_cur3'],
                                   msfe4_dd2_prev4=tflite_out['msfe4_dd2_cur4'],
                                   msfe4_dd3_prev1=tflite_out['msfe4_dd3_cur1'],
                                   msfe4_dd3_prev2=tflite_out['msfe4_dd3_cur2'],
                                   msfe4_dd3_prev3=tflite_out['msfe4_dd3_cur3'],
                                   msfe4_dd3_prev4=tflite_out['msfe4_dd3_cur4'],
                                   msfe5_dd_prev1=tflite_out['msfe5_dd_cur1'],
                                   msfe5_dd_prev2=tflite_out['msfe5_dd_cur2'],
                                   msfe5_dd_prev3=tflite_out['msfe5_dd_cur3'],
                                   msfe5_dd_prev4=tflite_out['msfe5_dd_cur4'],
                                   msfe5_dd_prev5=tflite_out['msfe5_dd_cur5'],
                                   msfe6_dd_prev1=tflite_out['msfe6_dd_cur1'],
                                   msfe6_dd_prev2=tflite_out['msfe6_dd_cur2'],
                                   msfe6_dd_prev3=tflite_out['msfe6_dd_cur3'],
                                   msfe6_dd_prev4=tflite_out['msfe6_dd_cur4'],
                                   msfe6_dd_prev5=tflite_out['msfe6_dd_cur5'],
                                   msfe6_dd_prev6=tflite_out['msfe6_dd_cur6'],

                                   msfe6_en_h=tflite_out['msfe6_en_h'],
                                   msfe6_en_c=tflite_out['msfe6_en_c'],
                                   msfe5_en_h=tflite_out['msfe5_en_h'],
                                   msfe5_en_c=tflite_out['msfe5_en_c'],
                                   msfe4_en_h=tflite_out['msfe4_en_h'],
                                   msfe4_en_c=tflite_out['msfe4_en_c'],
                                   msfe4_en2_h=tflite_out['msfe4_en2_h'],
                                   msfe4_en2_c=tflite_out['msfe4_en2_c'],
                                   msfe4_en3_h=tflite_out['msfe4_en3_h'],
                                   msfe4_en3_c=tflite_out['msfe4_en3_c'],
                                   msfe3_en_h=tflite_out['msfe3_en_h'],
                                   msfe3_en_c=tflite_out['msfe3_en_c'],
                                   state_h=tflite_out['state_h'],
                                   state_c=tflite_out['state_c'],
                                   msfe3_de_h=tflite_out['msfe3_de_h'],
                                   msfe3_de_c=tflite_out['msfe3_de_c'],
                                   msfe4_de_h=tflite_out['msfe4_de_h'],
                                   msfe4_de_c=tflite_out['msfe4_de_c'],
                                   msfe4_de2_h=tflite_out['msfe4_de2_h'],
                                   msfe4_de2_c=tflite_out['msfe4_de2_c'],
                                   msfe4_de3_h=tflite_out['msfe4_de3_h'],
                                   msfe4_de3_c=tflite_out['msfe4_de3_c'],
                                   msfe5_de_h=tflite_out['msfe5_de_h'],
                                   msfe5_de_c=tflite_out['msfe5_de_c'],
                                   msfe6_de_h=tflite_out['msfe6_de_h'],
                                   msfe6_de_c=tflite_out['msfe6_de_c'],
                                   )

        est_mag = np.pad(tflite_out['model_out'], ((0, 0), (0, 0), (1, 0), (0, 0)),
                         mode='edge')  # Best performance when mode is 'edge'
        est_mag = np.squeeze(est_mag)

        estimated_complex = est_mag * np.exp(1j * in_phase)  # [1,1,257]
        estimated_block = np.fft.irfft(estimated_complex).astype('float32')  # [1,1,512]

        estimated_block *= inverse_window
        # Method 1
        out_buffer[:-frame_step] = out_buffer[frame_step:]  # [512,]
        out_buffer[-frame_step:] = 0
        out_buffer += np.squeeze(estimated_block)  # Overlap-Add

        out_file[idx * frame_step:(idx * frame_step) + frame_step] = out_buffer[:frame_step]
        time_array.append(time.time() - start_time)

    out_file = out_file[frame_len - frame_step:]

    return (out_file), time_array


# load models
interpreter = tf.lite.Interpreter(model_path='./tflite/nutls_lstm.tflite')
interpreter.allocate_tensors()

# Get signature of tflite
signature = interpreter.get_signature_list()
print('Signature', signature)
nutls_lstm_sm = interpreter.get_signature_runner('nutls_lstm_sm')

# Load the noisy test sample
audio, fs = sf.read('./data/-.wav')
if fs != 16000:
    raise ValueError('This model only supports 16k sampling rate.')

# Real-time processing
out_file, time_array = real_time_speech_enhancer(audio)
sf.write('./data/-.wav', out_file, fs)
processing_time_per_frame = np.mean(np.stack(time_array))

# Measuring speech enhancement performance
clean, fs = sf.read("./data/-.wav")
noisy, fs = sf.read("./data/-.wav")
enhanced, fs = sf.read("./data/-.wav")


def cal_pesq(y_true, y_pred):
    sr = 16000
    mode = "wb"
    pesq_score = get_pesq(sr, y_true, y_pred, mode)
    return pesq_score


print("##########################################################################")
print("Speech enhancement performance")
print("PESQ (Before real-time voice enhancement) : ", cal_pesq(clean, noisy))
print("--->>>")
print("PESQ (After real-time voice enhancement) : ", cal_pesq(clean, enhanced))
print("##########################################################################")
print("Real-time processing speed (RTF (Real-time Factor), if it is greater than 1, real-time processing is not possible) : ")
print("RTF : ", processing_time_per_frame / 0.016)
print("##########################################################################")

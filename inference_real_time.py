import soundfile as sf
import numpy as np
import tensorflow as tf
import time
from pesq import pesq as get_pesq
import config as cfg

# Load models
interpreter = tf.lite.Interpreter(model_path=None)  # tflite Path
interpreter.allocate_tensors()
tensor_details = interpreter.get_tensor_details()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

############################################################################
# Get signature of tflite
############################################################################
signature = interpreter.get_signature_list()
print('Signature', signature)
nunet_lstm = interpreter.get_signature_runner('nunet_lstm')

############################################################################
# Load Audio sample
############################################################################
audio, fs = sf.read('Input path of audio sample')
if fs != 16000:
    raise ValueError('This model only supports 16k sampling rate.')
in_buffer = np.zeros((cfg.win_len)).astype('float32')  # [512,]
out_buffer = np.zeros((cfg.win_len)).astype('float32')  # [512,]
num_blocks = (audio.shape[0] - (cfg.win_len - cfg.stride)) // cfg.stride
out_file = np.zeros((len(audio) + (cfg.win_len - cfg.stride)))  # [48000,]

time_array = []
padding = np.array([[0, 0], [0, 0], [1, 0], [0, 0]])

############################################################################
# Define buffer for input frames
############################################################################
nunet_out = {
    'input': np.zeros((1, 1, 256, 1), dtype=np.float32),
    # MSFE6 - Encoder
    "msfe6_en_cur1": np.zeros((1, 1, 256, 64), dtype=np.float32),
    "msfe6_en_cur2": np.zeros((1, 1, 128, 32), dtype=np.float32),
    "msfe6_en_cur3": np.zeros((1, 1, 64, 32), dtype=np.float32),
    "msfe6_en_cur4": np.zeros((1, 1, 32, 32), dtype=np.float32),
    "msfe6_en_cur5": np.zeros((1, 1, 16, 32), dtype=np.float32),
    "msfe6_en_cur6": np.zeros((1, 1, 8, 32), dtype=np.float32),
    "msfe6_en_h": np.zeros((1, 64), dtype=np.float32),
    "msfe6_en_c": np.zeros((1, 64), dtype=np.float32),
    # MSFE5 - Encoder
    "msfe5_en_cur1": np.zeros((1, 1, 128, 64), dtype=np.float32),
    "msfe5_en_cur2": np.zeros((1, 1, 64, 32), dtype=np.float32),
    "msfe5_en_cur3": np.zeros((1, 1, 32, 32), dtype=np.float32),
    "msfe5_en_cur4": np.zeros((1, 1, 16, 32), dtype=np.float32),
    "msfe5_en_cur5": np.zeros((1, 1, 8, 32), dtype=np.float32),
    "msfe5_en_h": np.zeros((1, 64), dtype=np.float32),
    "msfe5_en_c": np.zeros((1, 64), dtype=np.float32),
    # MSFE4 - Encoder
    "msfe4_en_cur1": np.zeros((1, 1, 64, 64), dtype=np.float32),
    "msfe4_en_cur2": np.zeros((1, 1, 32, 32), dtype=np.float32),
    "msfe4_en_cur3": np.zeros((1, 1, 16, 32), dtype=np.float32),
    "msfe4_en_cur4": np.zeros((1, 1, 8, 32), dtype=np.float32),
    "msfe4_en_h": np.zeros((1, 64), dtype=np.float32),
    "msfe4_en_c": np.zeros((1, 64), dtype=np.float32),
    # MSFE4(2) - Encoder
    "msfe4_en2_cur1": np.zeros((1, 1, 32, 64), dtype=np.float32),
    "msfe4_en2_cur2": np.zeros((1, 1, 16, 32), dtype=np.float32),
    "msfe4_en2_cur3": np.zeros((1, 1, 8, 32), dtype=np.float32),
    "msfe4_en2_cur4": np.zeros((1, 1, 4, 32), dtype=np.float32),
    "msfe4_en2_h": np.zeros((1, 32), dtype=np.float32),
    "msfe4_en2_c": np.zeros((1, 32), dtype=np.float32),
    # MSFE4(3) - Encoder
    "msfe4_en3_cur1": np.zeros((1, 1, 16, 64), dtype=np.float32),
    "msfe4_en3_cur2": np.zeros((1, 1, 8, 32), dtype=np.float32),
    "msfe4_en3_cur3": np.zeros((1, 1, 4, 32), dtype=np.float32),
    "msfe4_en3_cur4": np.zeros((1, 1, 2, 32), dtype=np.float32),
    "msfe4_en3_h": np.zeros((1, 16), dtype=np.float32),
    "msfe4_en3_c": np.zeros((1, 16), dtype=np.float32),
    # MSFE3 - Encoder
    "msfe3_en_cur1": np.zeros((1, 1, 8, 64), dtype=np.float32),
    "msfe3_en_cur2": np.zeros((1, 1, 4, 32), dtype=np.float32),
    "msfe3_en_cur3": np.zeros((1, 1, 2, 32), dtype=np.float32),
    "msfe3_en_h": np.zeros((1, 16), dtype=np.float32),
    "msfe3_en_c": np.zeros((1, 16), dtype=np.float32),
    # LSTM
    "h_state": np.zeros((1, 128), dtype=np.float32),
    "c_state": np.zeros((1, 128), dtype=np.float32),
    # MSFE3 - Decoder
    "msfe3_de_cur1": np.zeros((1, 1, 8, 64), dtype=np.float32),
    "msfe3_de_cur2": np.zeros((1, 1, 4, 32), dtype=np.float32),
    "msfe3_de_cur3": np.zeros((1, 1, 2, 32), dtype=np.float32),
    "msfe3_de_h": np.zeros((1, 16), dtype=np.float32),
    "msfe3_de_c": np.zeros((1, 16), dtype=np.float32),
    # MSFE4 - Decoder
    "msfe4_de_cur1": np.zeros((1, 1, 16, 64), dtype=np.float32),
    "msfe4_de_cur2": np.zeros((1, 1, 8, 32), dtype=np.float32),
    "msfe4_de_cur3": np.zeros((1, 1, 4, 32), dtype=np.float32),
    "msfe4_de_cur4": np.zeros((1, 1, 2, 32), dtype=np.float32),
    "msfe4_de_h": np.zeros((1, 16), dtype=np.float32),
    "msfe4_de_c": np.zeros((1, 16), dtype=np.float32),
    # MSFE4(2) - Decoder
    "msfe4_de2_cur1": np.zeros((1, 1, 32, 64), dtype=np.float32),
    "msfe4_de2_cur2": np.zeros((1, 1, 16, 32), dtype=np.float32),
    "msfe4_de2_cur3": np.zeros((1, 1, 8, 32), dtype=np.float32),
    "msfe4_de2_cur4": np.zeros((1, 1, 4, 32), dtype=np.float32),
    "msfe4_de2_h": np.zeros((1, 32), dtype=np.float32),
    "msfe4_de2_c": np.zeros((1, 32), dtype=np.float32),
    # MSFE4(3) - Decoder
    "msfe4_de3_cur1": np.zeros((1, 1, 64, 64), dtype=np.float32),
    "msfe4_de3_cur2": np.zeros((1, 1, 32, 32), dtype=np.float32),
    "msfe4_de3_cur3": np.zeros((1, 1, 16, 32), dtype=np.float32),
    "msfe4_de3_cur4": np.zeros((1, 1, 8, 32), dtype=np.float32),
    "msfe4_de3_h": np.zeros((1, 64), dtype=np.float32),
    "msfe4_de3_c": np.zeros((1, 64), dtype=np.float32),
    # MSFE5 - Decoder
    "msfe5_de_cur1": np.zeros((1, 1, 128, 64), dtype=np.float32),
    "msfe5_de_cur2": np.zeros((1, 1, 64, 32), dtype=np.float32),
    "msfe5_de_cur3": np.zeros((1, 1, 32, 32), dtype=np.float32),
    "msfe5_de_cur4": np.zeros((1, 1, 16, 32), dtype=np.float32),
    "msfe5_de_cur5": np.zeros((1, 1, 8, 32), dtype=np.float32),
    "msfe5_de_h": np.zeros((1, 64), dtype=np.float32),
    "msfe5_de_c": np.zeros((1, 64), dtype=np.float32),
    # MSFE6 - Decoder
    "msfe6_de_cur1": np.zeros((1, 1, 256, 64), dtype=np.float32),
    "msfe6_de_cur2": np.zeros((1, 1, 128, 32), dtype=np.float32),
    "msfe6_de_cur3": np.zeros((1, 1, 64, 32), dtype=np.float32),
    "msfe6_de_cur4": np.zeros((1, 1, 32, 32), dtype=np.float32),
    "msfe6_de_cur5": np.zeros((1, 1, 16, 32), dtype=np.float32),
    "msfe6_de_cur6": np.zeros((1, 1, 8, 32), dtype=np.float32),
    "msfe6_de_h": np.zeros((1, 64), dtype=np.float32),
    "msfe6_de_c": np.zeros((1, 64), dtype=np.float32),
    "model_out": np.zeros((1, 1, 256, 1), dtype=np.float32)
}

############################################################################
# Real-time processing
############################################################################
for idx in range(num_blocks):
    start_time = time.time()
    in_buffer[:-cfg.stride] = in_buffer[cfg.stride:]
    in_buffer[-cfg.stride:] = audio[idx * cfg.stride:(idx * cfg.stride) + cfg.stride]

    in_block_fft = np.fft.rfft(in_buffer)  # [257,]

    in_mag = np.abs(in_block_fft)  # [257,]
    in_phase = np.angle(in_block_fft)  # [257,]

    expand_mag = np.reshape(in_mag, (1, 1, -1, 1)).astype('float32')  # [1,1,257,1]
    sliced_mag = expand_mag[:, :, 1:]  # [1, 1, 256, 1]

    ############################################################################
    # NUNET LSTM
    ############################################################################
    nunet_out = nunet_lstm(input=sliced_mag,
                           msfe6_en_prev1=nunet_out['msfe6_en_cur1'],
                           msfe6_en_prev2=nunet_out['msfe6_en_cur2'],
                           msfe6_en_prev3=nunet_out['msfe6_en_cur3'],
                           msfe6_en_prev4=nunet_out['msfe6_en_cur4'],
                           msfe6_en_prev5=nunet_out['msfe6_en_cur5'],
                           msfe6_en_prev6=nunet_out['msfe6_en_cur6'],
                           msfe6_en_h=nunet_out['msfe6_en_h'],
                           msfe6_en_c=nunet_out['msfe6_en_c'],
                           msfe5_en_prev1=nunet_out['msfe5_en_cur1'],
                           msfe5_en_prev2=nunet_out['msfe5_en_cur2'],
                           msfe5_en_prev3=nunet_out['msfe5_en_cur3'],
                           msfe5_en_prev4=nunet_out['msfe5_en_cur4'],
                           msfe5_en_prev5=nunet_out['msfe5_en_cur5'],
                           msfe5_en_h=nunet_out['msfe5_en_h'],
                           msfe5_en_c=nunet_out['msfe5_en_c'],
                           msfe4_en_prev1=nunet_out['msfe4_en_cur1'],
                           msfe4_en_prev2=nunet_out['msfe4_en_cur2'],
                           msfe4_en_prev3=nunet_out['msfe4_en_cur3'],
                           msfe4_en_prev4=nunet_out['msfe4_en_cur4'],
                           msfe4_en_h=nunet_out['msfe4_en_h'],
                           msfe4_en_c=nunet_out['msfe4_en_c'],
                           msfe4_en2_prev1=nunet_out['msfe4_en2_cur1'],
                           msfe4_en2_prev2=nunet_out['msfe4_en2_cur2'],
                           msfe4_en2_prev3=nunet_out['msfe4_en2_cur3'],
                           msfe4_en2_prev4=nunet_out['msfe4_en2_cur4'],
                           msfe4_en2_h=nunet_out['msfe4_en2_h'],
                           msfe4_en2_c=nunet_out['msfe4_en2_c'],
                           msfe4_en3_prev1=nunet_out['msfe4_en3_cur1'],
                           msfe4_en3_prev2=nunet_out['msfe4_en3_cur2'],
                           msfe4_en3_prev3=nunet_out['msfe4_en3_cur3'],
                           msfe4_en3_prev4=nunet_out['msfe4_en3_cur4'],
                           msfe4_en3_h=nunet_out['msfe4_en3_h'],
                           msfe4_en3_c=nunet_out['msfe4_en3_c'],
                           msfe3_en_prev1=nunet_out['msfe3_en_cur1'],
                           msfe3_en_prev2=nunet_out['msfe3_en_cur2'],
                           msfe3_en_prev3=nunet_out['msfe3_en_cur3'],
                           msfe3_en_h=nunet_out['msfe3_en_h'],
                           msfe3_en_c=nunet_out['msfe3_en_c'],
                           h_state=nunet_out['h_state'],
                           c_state=nunet_out['c_state'],
                           msfe3_de_prev1=nunet_out['msfe3_de_cur1'],
                           msfe3_de_prev2=nunet_out['msfe3_de_cur2'],
                           msfe3_de_prev3=nunet_out['msfe3_de_cur3'],
                           msfe3_de_h=nunet_out['msfe3_de_h'],
                           msfe3_de_c=nunet_out['msfe3_de_c'],
                           msfe4_de_prev1=nunet_out['msfe4_de_cur1'],
                           msfe4_de_prev2=nunet_out['msfe4_de_cur2'],
                           msfe4_de_prev3=nunet_out['msfe4_de_cur3'],
                           msfe4_de_prev4=nunet_out['msfe4_de_cur4'],
                           msfe4_de_h=nunet_out['msfe4_de_h'],
                           msfe4_de_c=nunet_out['msfe4_de_c'],
                           msfe4_de2_prev1=nunet_out['msfe4_de2_cur1'],
                           msfe4_de2_prev2=nunet_out['msfe4_de2_cur2'],
                           msfe4_de2_prev3=nunet_out['msfe4_de2_cur3'],
                           msfe4_de2_prev4=nunet_out['msfe4_de2_cur4'],
                           msfe4_de2_h=nunet_out['msfe4_de2_h'],
                           msfe4_de2_c=nunet_out['msfe4_de2_c'],
                           msfe4_de3_prev1=nunet_out['msfe4_de3_cur1'],
                           msfe4_de3_prev2=nunet_out['msfe4_de3_cur2'],
                           msfe4_de3_prev3=nunet_out['msfe4_de3_cur3'],
                           msfe4_de3_prev4=nunet_out['msfe4_de3_cur4'],
                           msfe4_de3_h=nunet_out['msfe4_de3_h'],
                           msfe4_de3_c=nunet_out['msfe4_de3_c'],
                           msfe5_de_prev1=nunet_out['msfe5_de_cur1'],
                           msfe5_de_prev2=nunet_out['msfe5_de_cur2'],
                           msfe5_de_prev3=nunet_out['msfe5_de_cur3'],
                           msfe5_de_prev4=nunet_out['msfe5_de_cur4'],
                           msfe5_de_prev5=nunet_out['msfe5_de_cur5'],
                           msfe5_de_h=nunet_out['msfe5_de_h'],
                           msfe5_de_c=nunet_out['msfe5_de_c'],
                           msfe6_de_prev1=nunet_out['msfe6_de_cur1'],
                           msfe6_de_prev2=nunet_out['msfe6_de_cur2'],
                           msfe6_de_prev3=nunet_out['msfe6_de_cur3'],
                           msfe6_de_prev4=nunet_out['msfe6_de_cur4'],
                           msfe6_de_prev5=nunet_out['msfe6_de_cur5'],
                           msfe6_de_prev6=nunet_out['msfe6_de_cur6'],
                           msfe6_de_h=nunet_out['msfe6_de_h'],
                           msfe6_de_c=nunet_out['msfe6_de_c']
                           )

    ############################################################################
    # T-F masking
    ############################################################################
    mag_padding = np.array([[0, 0], [0, 0], [1, 0], [0, 0]])
    out_mask = np.pad(nunet_out['model_out'], mag_padding)
    out_mask = np.squeeze(out_mask)
    out_mask = np.tanh(out_mask)
    estimated_complex = in_mag * out_mask * np.exp(1j * in_phase)
    estimated_block = np.fft.irfft(estimated_complex)
    estimated_block = estimated_block.astype('float32')
    out_buffer[:-cfg.stride] = out_buffer[cfg.stride:]
    out_buffer[-cfg.stride:] = np.zeros((cfg.stride))
    out_buffer += np.squeeze(estimated_block)  # Overlap-Add
    out_file[idx * cfg.stride:(idx * cfg.stride) + cfg.stride] = out_buffer[:cfg.stride]
    time_array.append(time.time() - start_time)

out_file = out_file[384:]
out_file = out_file / 4

print("1 loop's processing time", np.mean(np.stack(time_array)))
print("Total processing time", np.sum(np.stack(time_array)))

sf.write('Input path of enhanced audio', out_file, fs)

enhanced, fs = sf.read("Input path of enhanced audio")
clean, fs = sf.read("Input path of clean audio")
noisy, fs = sf.read("Input path of noisy audio")


def cal_pesq(y_true, y_pred):
    sr = 16000
    mode = "wb"
    pesq_score = get_pesq(sr, y_true, y_pred, mode)
    return pesq_score


print("PESQ of Enhanced audio : ", cal_pesq(clean, enhanced))
print("PESQ of Noisy audio : ", cal_pesq(clean, noisy))

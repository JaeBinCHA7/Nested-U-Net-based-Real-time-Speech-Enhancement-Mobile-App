from keras.layers import Concatenate,ZeroPadding2D, Reshape
import tensorflow as tf
import numpy as np
from model import NUNET_LSTM

class TFL_SIGNITURE(tf.Module):
    def __init__(self, TFL_MODEL, weight_path):
        self.model = TFL_MODEL
        self.model.load_weights(weight_path)
        layers = self.model.layers

        ############################################################################
        # MSFE6 - Encoder
        ############################################################################
        self.in_conv = layers[1]
        self.msfe6_en_in = layers[2]
        self.msfe6_en_conv1 = layers[4]
        self.msfe6_en_conv2 = layers[6]
        self.msfe6_en_conv3 = layers[8]
        self.msfe6_en_conv4 = layers[10]
        self.msfe6_en_conv5 = layers[12]
        self.msfe6_en_conv6 = layers[14]
        self.msfe6_en_lstm = layers[16]
        self.msfe6_en_dense = layers[17]
        self.msfe6_en_spconv1 = layers[20]
        self.msfe6_en_spconv2 = layers[22]
        self.msfe6_en_spconv3 = layers[24]
        self.msfe6_en_spconv4 = layers[26]
        self.msfe6_en_spconv5 = layers[28]
        self.msfe6_en_spconv6 = layers[30]
        self.msfe6_down_sampling = layers[32]
        ############################################################################
        # MSFE5 - Encoder
        ############################################################################
        self.msfe5_en_in = layers[33]
        self.msfe5_en_conv1 = layers[35]
        self.msfe5_en_conv2 = layers[37]
        self.msfe5_en_conv3 = layers[39]
        self.msfe5_en_conv4 = layers[41]
        self.msfe5_en_conv5 = layers[43]
        self.msfe5_en_lstm = layers[45]
        self.msfe5_en_dense = layers[46]
        self.msfe5_en_spconv1 = layers[49]
        self.msfe5_en_spconv2 = layers[51]
        self.msfe5_en_spconv3 = layers[53]
        self.msfe5_en_spconv4 = layers[55]
        self.msfe5_en_spconv5 = layers[57]
        self.msfe5_down_sampling = layers[59]

        ############################################################################
        # MSFE4 - Encoder
        ############################################################################
        self.msfe4_en_in = layers[60]
        self.msfe4_en_conv1 = layers[62]
        self.msfe4_en_conv2 = layers[64]
        self.msfe4_en_conv3 = layers[66]
        self.msfe4_en_conv4 = layers[68]
        self.msfe4_en_lstm = layers[70]
        self.msfe4_en_dense = layers[71]
        self.msfe4_en_spconv1 = layers[74]
        self.msfe4_en_spconv2 = layers[76]
        self.msfe4_en_spconv3 = layers[78]
        self.msfe4_en_spconv4 = layers[80]
        self.msfe4_down_sampling = layers[82]

        ############################################################################
        # MSFE4(2) - Encoder
        ############################################################################
        self.msfe4_en2_in = layers[83]
        self.msfe4_en2_conv1 = layers[85]
        self.msfe4_en2_conv2 = layers[87]
        self.msfe4_en2_conv3 = layers[89]
        self.msfe4_en2_conv4 = layers[91]
        self.msfe4_en2_lstm = layers[93]
        self.msfe4_en2_dense = layers[94]
        self.msfe4_en2_spconv1 = layers[97]
        self.msfe4_en2_spconv2 = layers[99]
        self.msfe4_en2_spconv3 = layers[101]
        self.msfe4_en2_spconv4 = layers[103]
        self.msfe4_down_sampling2 = layers[105]

        ############################################################################
        # MSFE4(3) - Encoder
        ############################################################################
        self.msfe4_en3_in = layers[106]
        self.msfe4_en3_conv1 = layers[108]
        self.msfe4_en3_conv2 = layers[110]
        self.msfe4_en3_conv3 = layers[112]
        self.msfe4_en3_conv4 = layers[114]
        self.msfe4_en3_lstm = layers[116]
        self.msfe4_en3_dense = layers[117]
        self.msfe4_en3_spconv1 = layers[120]
        self.msfe4_en3_spconv2 = layers[122]
        self.msfe4_en3_spconv3 = layers[124]
        self.msfe4_en3_spconv4 = layers[126]
        self.msfe4_down_sampling3 = layers[128]

        ############################################################################
        # MSFE3 - Encoder
        ############################################################################
        self.msfe3_en_in = layers[129]
        self.msfe3_en_conv1 = layers[131]
        self.msfe3_en_conv2 = layers[133]
        self.msfe3_en_conv3 = layers[135]
        self.msfe3_en_lstm = layers[137]
        self.msfe3_en_dense = layers[138]
        self.msfe3_en_spconv1 = layers[141]
        self.msfe3_en_spconv2 = layers[143]
        self.msfe3_en_spconv3 = layers[145]
        self.msfe3_down_sampling = layers[147]

        ############################################################################
        # LSTM
        ############################################################################
        self.lstm = layers[149]
        self.dense = layers[150]

        ############################################################################
        # MSFE3 - Decoder
        ############################################################################
        self.msfe3_de_upsampling = layers[153]
        self.msfe3_de_in = layers[154]
        self.msfe3_de_conv1 = layers[156]
        self.msfe3_de_conv2 = layers[158]
        self.msfe3_de_conv3 = layers[160]
        self.msfe3_de_lstm = layers[162]
        self.msfe3_de_dense = layers[163]
        self.msfe3_de_spconv1 = layers[166]
        self.msfe3_de_spconv2 = layers[168]
        self.msfe3_de_spconv3 = layers[170]
        ############################################################################
        # MSFE4 - Decoder
        ############################################################################
        self.msfe4_de_upsampling = layers[173]
        self.msfe4_de_in = layers[174]
        self.msfe4_de_conv1 = layers[176]
        self.msfe4_de_conv2 = layers[178]
        self.msfe4_de_conv3 = layers[180]
        self.msfe4_de_conv4 = layers[182]
        self.msfe4_de_lstm = layers[184]
        self.msfe4_de_dense = layers[185]
        self.msfe4_de_spconv1 = layers[188]
        self.msfe4_de_spconv2 = layers[190]
        self.msfe4_de_spconv3 = layers[192]
        self.msfe4_de_spconv4 = layers[194]
        ############################################################################
        # MSFE4(2) - Decoder
        ############################################################################
        self.msfe4_de2_upsampling = layers[197]
        self.msfe4_de2_in = layers[198]
        self.msfe4_de2_conv1 = layers[200]
        self.msfe4_de2_conv2 = layers[202]
        self.msfe4_de2_conv3 = layers[204]
        self.msfe4_de2_conv4 = layers[206]
        self.msfe4_de2_lstm = layers[208]
        self.msfe4_de2_dense = layers[209]
        self.msfe4_de2_spconv1 = layers[212]
        self.msfe4_de2_spconv2 = layers[214]
        self.msfe4_de2_spconv3 = layers[216]
        self.msfe4_de2_spconv4 = layers[218]
        ############################################################################
        # MSFE4(3) - Decoder
        ############################################################################
        self.msfe4_de3_upsampling = layers[221]
        self.msfe4_de3_in = layers[222]
        self.msfe4_de3_conv1 = layers[224]
        self.msfe4_de3_conv2 = layers[226]
        self.msfe4_de3_conv3 = layers[228]
        self.msfe4_de3_conv4 = layers[230]
        self.msfe4_de3_lstm = layers[232]
        self.msfe4_de3_dense = layers[233]
        self.msfe4_de3_spconv1 = layers[236]
        self.msfe4_de3_spconv2 = layers[238]
        self.msfe4_de3_spconv3 = layers[240]
        self.msfe4_de3_spconv4 = layers[242]
        ############################################################################
        # MSFE5 - Decoder
        ############################################################################
        self.msfe5_de_upsampling = layers[245]
        self.msfe5_de_in = layers[246]
        self.msfe5_de_conv1 = layers[248]
        self.msfe5_de_conv2 = layers[250]
        self.msfe5_de_conv3 = layers[252]
        self.msfe5_de_conv4 = layers[254]
        self.msfe5_de_conv5 = layers[256]
        self.msfe5_de_lstm = layers[258]
        self.msfe5_de_dense = layers[259]
        self.msfe5_de_spconv1 = layers[262]
        self.msfe5_de_spconv2 = layers[264]
        self.msfe5_de_spconv3 = layers[266]
        self.msfe5_de_spconv4 = layers[268]
        self.msfe5_de_spconv5 = layers[270]
        ############################################################################
        # MSFE6 - Decoder
        ############################################################################
        self.msfe6_de_upsampling = layers[273]
        self.msfe6_de_in = layers[274]
        self.msfe6_de_conv1 = layers[276]
        self.msfe6_de_conv2 = layers[278]
        self.msfe6_de_conv3 = layers[280]
        self.msfe6_de_conv4 = layers[282]
        self.msfe6_de_conv5 = layers[284]
        self.msfe6_de_conv6 = layers[286]
        self.msfe6_de_lstm = layers[288]
        self.msfe6_de_dense = layers[289]
        self.msfe6_de_spconv1 = layers[292]
        self.msfe6_de_spconv2 = layers[294]
        self.msfe6_de_spconv3 = layers[296]
        self.msfe6_de_spconv4 = layers[298]
        self.msfe6_de_spconv5 = layers[300]
        self.msfe6_de_spconv6 = layers[302]

        self.out_conv = layers[304]

    # Reshape (Before LSTM)
    def reshape_before_lstm(self, x):
        shape = np.shape(x)
        reshape_out = Reshape((shape[1], shape[2] * shape[3]))(x)

        return reshape_out, shape

    # Reshape (After LSTM)
    def reshape_after_lstm(self, x, shape):
        lstm_out = Reshape((shape[1], shape[2], shape[3]))(x)

        return lstm_out


    @tf.function(
        input_signature=[tf.TensorSpec(shape=[None, 1, 256, 1], dtype=tf.float32, name='input'),  # Input
                         # MSFE6 - Encoder
                         tf.TensorSpec(shape=[None, 1, 256, 64], dtype=tf.float32, name='msfe6_en_prev1'),
                         tf.TensorSpec(shape=[None, 1, 128, 32], dtype=tf.float32, name='msfe6_en_prev2'),
                         tf.TensorSpec(shape=[None, 1, 64, 32], dtype=tf.float32, name='msfe6_en_prev3'),
                         tf.TensorSpec(shape=[None, 1, 32, 32], dtype=tf.float32, name='msfe6_en_prev4'),
                         tf.TensorSpec(shape=[None, 1, 16, 32], dtype=tf.float32, name='msfe6_en_prev5'),
                         tf.TensorSpec(shape=[None, 1, 8, 32], dtype=tf.float32, name='msfe6_en_prev6'),
                         tf.TensorSpec(shape=[1, 64], dtype=tf.float32, name='msfe6_en_h'),
                         tf.TensorSpec(shape=[1, 64], dtype=tf.float32, name='msfe6_en_c'),
                         # MSFE5 - Encoder
                         tf.TensorSpec(shape=[None, 1, 128, 64], dtype=tf.float32, name='msfe5_en_prev1'),
                         tf.TensorSpec(shape=[None, 1, 64, 32], dtype=tf.float32, name='msfe5_en_prev2'),
                         tf.TensorSpec(shape=[None, 1, 32, 32], dtype=tf.float32, name='msfe5_en_prev3'),
                         tf.TensorSpec(shape=[None, 1, 16, 32], dtype=tf.float32, name='msfe5_en_prev4'),
                         tf.TensorSpec(shape=[None, 1, 8, 32], dtype=tf.float32, name='msfe5_en_prev5'),
                         tf.TensorSpec(shape=[1, 64], dtype=tf.float32, name='msfe5_en_h'),
                         tf.TensorSpec(shape=[1, 64], dtype=tf.float32, name='msfe5_en_c'),
                         # MSFE4 - Encoder
                         tf.TensorSpec(shape=[None, 1, 64, 64], dtype=tf.float32, name='msfe4_en_prev1'),
                         tf.TensorSpec(shape=[None, 1, 32, 32], dtype=tf.float32, name='msfe4_en_prev2'),
                         tf.TensorSpec(shape=[None, 1, 16, 32], dtype=tf.float32, name='msfe4_en_prev3'),
                         tf.TensorSpec(shape=[None, 1, 8, 32], dtype=tf.float32, name='msfe4_en_prev4'),
                         tf.TensorSpec(shape=[1, 64], dtype=tf.float32, name='msfe4_en_h'),
                         tf.TensorSpec(shape=[1, 64], dtype=tf.float32, name='msfe4_en_c'),
                         # MSFE4(2) - Encoder
                         tf.TensorSpec(shape=[None, 1, 32, 64], dtype=tf.float32, name='msfe4_en2_prev1'),
                         tf.TensorSpec(shape=[None, 1, 16, 32], dtype=tf.float32, name='msfe4_en2_prev2'),
                         tf.TensorSpec(shape=[None, 1, 8, 32], dtype=tf.float32, name='msfe4_en2_prev3'),
                         tf.TensorSpec(shape=[None, 1, 4, 32], dtype=tf.float32, name='msfe4_en2_prev4'),
                         tf.TensorSpec(shape=[1, 32], dtype=tf.float32, name='msfe4_en2_h'),
                         tf.TensorSpec(shape=[1, 32], dtype=tf.float32, name='msfe4_en2_c'),
                         # MSFE4(3) - Encoder
                         tf.TensorSpec(shape=[None, 1, 16, 64], dtype=tf.float32, name='msfe4_en3_prev1'),
                         tf.TensorSpec(shape=[None, 1, 8, 32], dtype=tf.float32, name='msfe4_en3_prev2'),
                         tf.TensorSpec(shape=[None, 1, 4, 32], dtype=tf.float32, name='msfe4_en3_prev3'),
                         tf.TensorSpec(shape=[None, 1, 2, 32], dtype=tf.float32, name='msfe4_en3_prev4'),
                         tf.TensorSpec(shape=[1, 16], dtype=tf.float32, name='msfe4_en3_h'),
                         tf.TensorSpec(shape=[1, 16], dtype=tf.float32, name='msfe4_en3_c'),
                         # MSFE3 - Encoder
                         tf.TensorSpec(shape=[None, 1, 8, 64], dtype=tf.float32, name='msfe3_en_prev1'),
                         tf.TensorSpec(shape=[None, 1, 4, 32], dtype=tf.float32, name='msfe3_en_prev2'),
                         tf.TensorSpec(shape=[None, 1, 2, 32], dtype=tf.float32, name='msfe3_en_prev3'),
                         tf.TensorSpec(shape=[1, 16], dtype=tf.float32, name='msfe3_en_h'),
                         tf.TensorSpec(shape=[1, 16], dtype=tf.float32, name='msfe3_en_c'),
                         # LSTM
                         tf.TensorSpec(shape=[1, 128], dtype=tf.float32),
                         tf.TensorSpec(shape=[1, 128], dtype=tf.float32),
                         # MSFE3 - Decoder
                         tf.TensorSpec(shape=[None, 1, 8, 64], dtype=tf.float32, name='msfe3_de_prev1'),
                         tf.TensorSpec(shape=[None, 1, 4, 32], dtype=tf.float32, name='msfe3_de_prev2'),
                         tf.TensorSpec(shape=[None, 1, 2, 32], dtype=tf.float32, name='msfe3_de_prev3'),
                         tf.TensorSpec(shape=[1, 16], dtype=tf.float32, name='msfe3_de_h'),
                         tf.TensorSpec(shape=[1, 16], dtype=tf.float32, name='msfe3_de_c'),
                         # MSFE4 - Decoder
                         tf.TensorSpec(shape=[None, 1, 16, 64], dtype=tf.float32, name='msfe4_de_prev1'),
                         tf.TensorSpec(shape=[None, 1, 8, 32], dtype=tf.float32, name='msfe4_de_prev2'),
                         tf.TensorSpec(shape=[None, 1, 4, 32], dtype=tf.float32, name='msfe4_de_prev3'),
                         tf.TensorSpec(shape=[None, 1, 2, 32], dtype=tf.float32, name='msfe4_de_prev4'),
                         tf.TensorSpec(shape=[1, 16], dtype=tf.float32, name='msfe4_de_h'),
                         tf.TensorSpec(shape=[1, 16], dtype=tf.float32, name='msfe4_de_c'),
                         # MSFE4(2) - Decoder
                         tf.TensorSpec(shape=[None, 1, 32, 64], dtype=tf.float32, name='msfe4_de2_prev1'),
                         tf.TensorSpec(shape=[None, 1, 16, 32], dtype=tf.float32, name='msfe4_de2_prev2'),
                         tf.TensorSpec(shape=[None, 1, 8, 32], dtype=tf.float32, name='msfe4_de2_prev3'),
                         tf.TensorSpec(shape=[None, 1, 4, 32], dtype=tf.float32, name='msfe4_de2_prev4'),
                         tf.TensorSpec(shape=[1, 32], dtype=tf.float32, name='msfe4_de2_h'),
                         tf.TensorSpec(shape=[1, 32], dtype=tf.float32, name='msfe4_de2_c'),
                         # MSFE4(3) - Decoder
                         tf.TensorSpec(shape=[None, 1, 64, 64], dtype=tf.float32, name='msfe4_de3_prev1'),
                         tf.TensorSpec(shape=[None, 1, 32, 32], dtype=tf.float32, name='msfe4_de3_prev2'),
                         tf.TensorSpec(shape=[None, 1, 16, 32], dtype=tf.float32, name='msfe4_de3_prev3'),
                         tf.TensorSpec(shape=[None, 1, 8, 32], dtype=tf.float32, name='msfe4_de3_prev4'),
                         tf.TensorSpec(shape=[1, 64], dtype=tf.float32, name='msfe4_de3_h'),
                         tf.TensorSpec(shape=[1, 64], dtype=tf.float32, name='msfe4_de3_c'),
                         # MSFE5 - Decoder
                         tf.TensorSpec(shape=[None, 1, 128, 64], dtype=tf.float32, name='msfe5_de_prev1'),
                         tf.TensorSpec(shape=[None, 1, 64, 32], dtype=tf.float32, name='msfe5_de_prev2'),
                         tf.TensorSpec(shape=[None, 1, 32, 32], dtype=tf.float32, name='msfe5_de_prev3'),
                         tf.TensorSpec(shape=[None, 1, 16, 32], dtype=tf.float32, name='msfe5_de_prev4'),
                         tf.TensorSpec(shape=[None, 1, 8, 32], dtype=tf.float32, name='msfe5_de_prev5'),
                         tf.TensorSpec(shape=[1, 64], dtype=tf.float32, name='msfe5_de_h'),
                         tf.TensorSpec(shape=[1, 64], dtype=tf.float32, name='msfe5_de_c'),
                         # MSFE6 - Decoder
                         tf.TensorSpec(shape=[None, 1, 256, 64], dtype=tf.float32, name='msfe6_de_prev1'),
                         tf.TensorSpec(shape=[None, 1, 128, 32], dtype=tf.float32, name='msfe6_de_prev2'),
                         tf.TensorSpec(shape=[None, 1, 64, 32], dtype=tf.float32, name='msfe6_de_prev3'),
                         tf.TensorSpec(shape=[None, 1, 32, 32], dtype=tf.float32, name='msfe6_de_prev4'),
                         tf.TensorSpec(shape=[None, 1, 16, 32], dtype=tf.float32, name='msfe6_de_prev5'),
                         tf.TensorSpec(shape=[None, 1, 8, 32], dtype=tf.float32, name='msfe6_de_prev6'),
                         tf.TensorSpec(shape=[1, 64], dtype=tf.float32, name='msfe6_de_h'),
                         tf.TensorSpec(shape=[1, 64], dtype=tf.float32, name='msfe6_de_c'),
                         ])
    def nunet_lstm(self, input,
                   msfe6_en_prev1, msfe6_en_prev2, msfe6_en_prev3, msfe6_en_prev4, msfe6_en_prev5, msfe6_en_prev6,
                   msfe6_en_h, msfe6_en_c,
                   msfe5_en_prev1, msfe5_en_prev2, msfe5_en_prev3, msfe5_en_prev4, msfe5_en_prev5, msfe5_en_h,
                   msfe5_en_c,
                   msfe4_en_prev1, msfe4_en_prev2, msfe4_en_prev3, msfe4_en_prev4, msfe4_en_h, msfe4_en_c,
                   msfe4_en2_prev1, msfe4_en2_prev2, msfe4_en2_prev3, msfe4_en2_prev4, msfe4_en2_h, msfe4_en2_c,
                   msfe4_en3_prev1, msfe4_en3_prev2, msfe4_en3_prev3, msfe4_en3_prev4, msfe4_en3_h, msfe4_en3_c,
                   msfe3_en_prev1, msfe3_en_prev2, msfe3_en_prev3, msfe3_en_h, msfe3_en_c,
                   h_state, c_state,
                   msfe3_de_prev1, msfe3_de_prev2, msfe3_de_prev3, msfe3_de_h, msfe3_de_c,
                   msfe4_de_prev1, msfe4_de_prev2, msfe4_de_prev3, msfe4_de_prev4, msfe4_de_h, msfe4_de_c,
                   msfe4_de2_prev1, msfe4_de2_prev2, msfe4_de2_prev3, msfe4_de2_prev4, msfe4_de2_h, msfe4_de2_c,
                   msfe4_de3_prev1, msfe4_de3_prev2, msfe4_de3_prev3, msfe4_de3_prev4, msfe4_de3_h, msfe4_de3_c,
                   msfe5_de_prev1, msfe5_de_prev2, msfe5_de_prev3, msfe5_de_prev4, msfe5_de_prev5, msfe5_de_h,
                   msfe5_de_c,
                   msfe6_de_prev1, msfe6_de_prev2, msfe6_de_prev3, msfe6_de_prev4, msfe6_de_prev5, msfe6_de_prev6,
                   msfe6_de_h, msfe6_de_c):
        in_out = self.in_conv(input)
        ############################################################################
        # MSFE6 - Encoder
        ############################################################################
        msfe6_en_cur1 = self.msfe6_en_in(in_out)
        msfe6_en_cur2 = self.msfe6_en_conv1(Concatenate(axis=1)([msfe6_en_prev1, msfe6_en_cur1]))
        msfe6_en_cur3 = self.msfe6_en_conv2(Concatenate(axis=1)([msfe6_en_prev2, msfe6_en_cur2]))
        msfe6_en_cur4 = self.msfe6_en_conv3(Concatenate(axis=1)([msfe6_en_prev3, msfe6_en_cur3]))
        msfe6_en_cur5 = self.msfe6_en_conv4(Concatenate(axis=1)([msfe6_en_prev4, msfe6_en_cur4]))
        msfe6_en_cur6 = self.msfe6_en_conv5(Concatenate(axis=1)([msfe6_en_prev5, msfe6_en_cur5]))
        en6_out = self.msfe6_en_conv6(Concatenate(axis=1)([msfe6_en_prev6, msfe6_en_cur6]))

        # lstm
        reshape_out, shape = self.reshape_before_lstm(en6_out)
        en_lstm, msfe6_en_h, msfe6_en_c = self.msfe6_en_lstm(reshape_out, initial_state=[msfe6_en_h, msfe6_en_c])
        en_lstm = self.msfe6_en_dense(en_lstm)
        en_lstm = self.reshape_after_lstm(en_lstm, shape)

        # Decoder
        de_out = self.msfe6_en_spconv1(Concatenate(axis=3)([en_lstm, en6_out]))
        de_out = self.msfe6_en_spconv2(Concatenate(axis=3)([de_out, msfe6_en_cur6]))
        de_out = self.msfe6_en_spconv3(Concatenate(axis=3)([de_out, msfe6_en_cur5]))
        de_out = self.msfe6_en_spconv4(Concatenate(axis=3)([de_out, msfe6_en_cur4]))
        de_out = self.msfe6_en_spconv5(Concatenate(axis=3)([de_out, msfe6_en_cur3]))
        de_out = self.msfe6_en_spconv6(Concatenate(axis=3)([de_out, msfe6_en_cur2]))

        de_out += msfe6_en_cur1  # [None, 372, 4 ,32]

        # Down-sampling
        msfe6_out = self.msfe6_down_sampling(de_out)  # [None, 372, 128 ,64]

        ############################################################################
        # MSFE5 - Encoder
        ############################################################################
        msfe5_en_cur1 = self.msfe5_en_in(msfe6_out)
        msfe5_en_cur2 = self.msfe5_en_conv1(Concatenate(axis=1)([msfe5_en_prev1, msfe5_en_cur1]))
        msfe5_en_cur3 = self.msfe5_en_conv2(Concatenate(axis=1)([msfe5_en_prev2, msfe5_en_cur2]))
        msfe5_en_cur4 = self.msfe5_en_conv3(Concatenate(axis=1)([msfe5_en_prev3, msfe5_en_cur3]))
        msfe5_en_cur5 = self.msfe5_en_conv4(Concatenate(axis=1)([msfe5_en_prev4, msfe5_en_cur4]))
        en5_out = self.msfe5_en_conv5(Concatenate(axis=1)([msfe5_en_prev5, msfe5_en_cur5]))

        # lstm
        reshape_out, shape = self.reshape_before_lstm(en5_out)
        en_lstm, msfe5_en_h, msfe5_en_c = self.msfe5_en_lstm(reshape_out, initial_state=[msfe5_en_h, msfe5_en_c])
        en_lstm = self.msfe5_en_dense(en_lstm)
        en_lstm = self.reshape_after_lstm(en_lstm, shape)

        # Decoder
        de_out = self.msfe5_en_spconv1(Concatenate(axis=3)([en_lstm, en5_out]))
        de_out = self.msfe5_en_spconv2(Concatenate(axis=3)([de_out, msfe5_en_cur5]))
        de_out = self.msfe5_en_spconv3(Concatenate(axis=3)([de_out, msfe5_en_cur4]))
        de_out = self.msfe5_en_spconv4(Concatenate(axis=3)([de_out, msfe5_en_cur3]))
        de_out = self.msfe5_en_spconv5(Concatenate(axis=3)([de_out, msfe5_en_cur2]))

        de_out += msfe5_en_cur1  # [None, 372, 4 ,32]

        # Down-sampling
        msfe5_out = self.msfe5_down_sampling(de_out)

        ############################################################################
        # MSFE4 - Encoder
        ############################################################################
        msfe4_en_cur1 = self.msfe4_en_in(msfe5_out)
        msfe4_en_cur2 = self.msfe4_en_conv1(Concatenate(axis=1)([msfe4_en_prev1, msfe4_en_cur1]))
        msfe4_en_cur3 = self.msfe4_en_conv2(Concatenate(axis=1)([msfe4_en_prev2, msfe4_en_cur2]))
        msfe4_en_cur4 = self.msfe4_en_conv3(Concatenate(axis=1)([msfe4_en_prev3, msfe4_en_cur3]))
        en4_out = self.msfe4_en_conv4(Concatenate(axis=1)([msfe4_en_prev4, msfe4_en_cur4]))

        # lstm
        reshape_out, shape = self.reshape_before_lstm(en4_out)
        en_lstm, msfe4_en_h, msfe4_en_c = self.msfe4_en_lstm(reshape_out, initial_state=[msfe4_en_h, msfe4_en_c])
        en_lstm = self.msfe4_en_dense(en_lstm)
        en_lstm = self.reshape_after_lstm(en_lstm, shape)

        # Decoder
        de_out = self.msfe4_en_spconv1(Concatenate(axis=3)([en_lstm, en4_out]))
        de_out = self.msfe4_en_spconv2(Concatenate(axis=3)([de_out, msfe4_en_cur4]))
        de_out = self.msfe4_en_spconv3(Concatenate(axis=3)([de_out, msfe4_en_cur3]))
        de_out = self.msfe4_en_spconv4(Concatenate(axis=3)([de_out, msfe4_en_cur2]))

        de_out += msfe4_en_cur1  # [None, 372, 4 ,32]

        # Down-sampling
        msfe4_out = self.msfe4_down_sampling(de_out)  # [None, 372, 128 ,64]

        ############################################################################
        # MSFE4(2) - Encoder
        ############################################################################
        msfe4_en2_cur1 = self.msfe4_en2_in(msfe4_out)
        msfe4_en2_cur2 = self.msfe4_en2_conv1(Concatenate(axis=1)([msfe4_en2_prev1, msfe4_en2_cur1]))
        msfe4_en2_cur3 = self.msfe4_en2_conv2(Concatenate(axis=1)([msfe4_en2_prev2, msfe4_en2_cur2]))
        msfe4_en2_cur4 = self.msfe4_en2_conv3(Concatenate(axis=1)([msfe4_en2_prev3, msfe4_en2_cur3]))
        en4_out = self.msfe4_en2_conv4(Concatenate(axis=1)([msfe4_en2_prev4, msfe4_en2_cur4]))

        # lstm
        reshape_out, shape = self.reshape_before_lstm(en4_out)
        en_lstm, msfe4_en2_h, msfe4_en2_c = self.msfe4_en2_lstm(reshape_out,
                                                                     initial_state=[msfe4_en2_h, msfe4_en2_c])
        en_lstm = self.msfe4_en2_dense(en_lstm)
        en_lstm = self.reshape_after_lstm(en_lstm, shape)

        # Decoder
        de_out = self.msfe4_en2_spconv1(Concatenate(axis=3)([en_lstm, en4_out]))
        de_out = self.msfe4_en2_spconv2(Concatenate(axis=3)([de_out, msfe4_en2_cur4]))
        de_out = self.msfe4_en2_spconv3(Concatenate(axis=3)([de_out, msfe4_en2_cur3]))
        de_out = self.msfe4_en2_spconv4(Concatenate(axis=3)([de_out, msfe4_en2_cur2]))

        de_out += msfe4_en2_cur1  # [None, 372, 4 ,32]

        # Down-sampling
        msfe4_out2 = self.msfe4_down_sampling2(de_out)  # [None, 372, 128 ,64]

        ############################################################################
        # MSFE4(3) - Encoder
        ############################################################################
        msfe4_en3_cur1 = self.msfe4_en3_in(msfe4_out2)
        msfe4_en3_cur2 = self.msfe4_en3_conv1(Concatenate(axis=1)([msfe4_en3_prev1, msfe4_en3_cur1]))
        msfe4_en3_cur3 = self.msfe4_en3_conv2(Concatenate(axis=1)([msfe4_en3_prev2, msfe4_en3_cur2]))
        msfe4_en3_cur4 = self.msfe4_en3_conv3(Concatenate(axis=1)([msfe4_en3_prev3, msfe4_en3_cur3]))
        en4_out = self.msfe4_en3_conv4(Concatenate(axis=1)([msfe4_en3_prev4, msfe4_en3_cur4]))

        # lstm
        reshape_out, shape = self.reshape_before_lstm(en4_out)
        en_lstm, msfe4_en3_h, msfe4_en3_c = self.msfe4_en3_lstm(reshape_out,
                                                                     initial_state=[msfe4_en3_h, msfe4_en3_c])
        en_lstm = self.msfe4_en3_dense(en_lstm)
        en_lstm = self.reshape_after_lstm(en_lstm, shape)

        # Decoder
        de_out = self.msfe4_en3_spconv1(Concatenate(axis=3)([en_lstm, en4_out]))
        de_out = self.msfe4_en3_spconv2(Concatenate(axis=3)([de_out, msfe4_en3_cur4]))
        de_out = self.msfe4_en3_spconv3(Concatenate(axis=3)([de_out, msfe4_en3_cur3]))
        de_out = self.msfe4_en3_spconv4(Concatenate(axis=3)([de_out, msfe4_en3_cur2]))

        de_out += msfe4_en3_cur1  # [None, 372, 4 ,32]

        # Down-sampling
        msfe4_out3 = self.msfe4_down_sampling3(de_out)  # [None, 372, 128 ,64]

        ############################################################################
        # MSFE3 - Encoder
        ############################################################################
        msfe3_en_cur1 = self.msfe3_en_in(msfe4_out3)
        msfe3_en_cur2 = self.msfe3_en_conv1(Concatenate(axis=1)([msfe3_en_prev1, msfe3_en_cur1]))
        msfe3_en_cur3 = self.msfe3_en_conv2(Concatenate(axis=1)([msfe3_en_prev2, msfe3_en_cur2]))
        en3_out = self.msfe3_en_conv3(Concatenate(axis=1)([msfe3_en_prev3, msfe3_en_cur3]))

        # lstm
        reshape_out, shape = self.reshape_before_lstm(en3_out)
        en_lstm, msfe3_en_h, msfe3_en_c = self.msfe3_en_lstm(reshape_out, initial_state=[msfe3_en_h, msfe3_en_c])
        en_lstm = self.msfe3_en_dense(en_lstm)
        en_lstm = self.reshape_after_lstm(en_lstm, shape)

        # Decoder
        de_out = self.msfe3_en_spconv1(Concatenate(axis=3)([en_lstm, en3_out]))
        de_out = self.msfe3_en_spconv2(Concatenate(axis=3)([de_out, msfe3_en_cur3]))
        de_out = self.msfe3_en_spconv3(Concatenate(axis=3)([de_out, msfe3_en_cur2]))

        de_out += msfe3_en_cur1  # [None, 372, 4 ,32]

        # Down-sampling
        msfe3_out = self.msfe3_down_sampling(de_out)  # [None, 372, 128 ,64]

        ############################################################################
        # LSTM
        ############################################################################
        reshape_out, shape = self.reshape_before_lstm(msfe3_out)
        lstm_out, h_state, c_state = self.lstm(reshape_out, initial_state=[h_state, c_state])
        lstm_out = self.dense(lstm_out)
        lstm_out = self.reshape_after_lstm(lstm_out, shape)

        ############################################################################
        # MSFE3 - Decoder
        ############################################################################
        en_in = self.msfe3_de_upsampling(Concatenate(axis=3)([lstm_out, msfe3_out]))
        msfe3_de_cur1 = self.msfe3_de_in(en_in)
        msfe3_de_cur2 = self.msfe3_de_conv1(Concatenate(axis=1)([msfe3_de_prev1, msfe3_de_cur1]))
        msfe3_de_cur3 = self.msfe3_de_conv2(Concatenate(axis=1)([msfe3_de_prev2, msfe3_de_cur2]))
        en3_out = self.msfe3_de_conv3(Concatenate(axis=1)([msfe3_de_prev3, msfe3_de_cur3]))

        # lstm
        reshape_out, shape = self.reshape_before_lstm(en3_out)
        de_lstm, msfe3_de_h, msfe3_de_c = self.msfe3_de_lstm(reshape_out, initial_state=[msfe3_de_h, msfe3_de_c])
        de_lstm = self.msfe3_de_dense(de_lstm)
        de_lstm = self.reshape_after_lstm(de_lstm, shape)

        # Decoder
        de_out = self.msfe3_de_spconv1(Concatenate(axis=3)([de_lstm, en3_out]))
        de_out = self.msfe3_de_spconv2(Concatenate(axis=3)([de_out, msfe3_de_cur3]))
        de_out = self.msfe3_de_spconv3(Concatenate(axis=3)([de_out, msfe3_de_cur2]))

        de_out += msfe3_de_cur1

        ############################################################################
        # MSFE4 - Decoder
        ############################################################################
        en_in = self.msfe4_de_upsampling(Concatenate(axis=3)([de_out, msfe4_out3]))
        msfe4_de_cur1 = self.msfe4_de_in(en_in)
        msfe4_de_cur2 = self.msfe4_de_conv1(Concatenate(axis=1)([msfe4_de_prev1, msfe4_de_cur1]))
        msfe4_de_cur3 = self.msfe4_de_conv2(Concatenate(axis=1)([msfe4_de_prev2, msfe4_de_cur2]))
        msfe4_de_cur4 = self.msfe4_de_conv3(Concatenate(axis=1)([msfe4_de_prev3, msfe4_de_cur3]))
        en4_out = self.msfe4_de_conv4(Concatenate(axis=1)([msfe4_de_prev4, msfe4_de_cur4]))

        # lstm
        reshape_out, shape = self.reshape_before_lstm(en4_out)
        de_lstm, msfe4_de_h, msfe4_de_c = self.msfe4_de_lstm(reshape_out, initial_state=[msfe4_de_h, msfe4_de_c])
        de_lstm = self.msfe4_de_dense(de_lstm)
        de_lstm = self.reshape_after_lstm(de_lstm, shape)

        # Decoder
        de_out = self.msfe4_de_spconv1(Concatenate(axis=3)([de_lstm, en4_out]))
        de_out = self.msfe4_de_spconv2(Concatenate(axis=3)([de_out, msfe4_de_cur4]))
        de_out = self.msfe4_de_spconv3(Concatenate(axis=3)([de_out, msfe4_de_cur3]))
        de_out = self.msfe4_de_spconv4(Concatenate(axis=3)([de_out, msfe4_de_cur2]))

        de_out += msfe4_de_cur1
        ############################################################################
        # MSFE4(2) - Decoder
        ############################################################################
        en_in = self.msfe4_de2_upsampling(Concatenate(axis=3)([de_out, msfe4_out2]))
        msfe4_de2_cur1 = self.msfe4_de2_in(en_in)
        msfe4_de2_cur2 = self.msfe4_de2_conv1(Concatenate(axis=1)([msfe4_de2_prev1, msfe4_de2_cur1]))
        msfe4_de2_cur3 = self.msfe4_de2_conv2(Concatenate(axis=1)([msfe4_de2_prev2, msfe4_de2_cur2]))
        msfe4_de2_cur4 = self.msfe4_de2_conv3(Concatenate(axis=1)([msfe4_de2_prev3, msfe4_de2_cur3]))
        en4_out = self.msfe4_de2_conv4(Concatenate(axis=1)([msfe4_de2_prev4, msfe4_de2_cur4]))

        # lstm
        reshape_out, shape = self.reshape_before_lstm(en4_out)
        de_lstm, msfe4_de2_h, msfe4_de2_c = self.msfe4_de2_lstm(reshape_out,
                                                                     initial_state=[msfe4_de2_h, msfe4_de2_c])
        de_lstm = self.msfe4_de2_dense(de_lstm)
        de_lstm = self.reshape_after_lstm(de_lstm, shape)

        # Decoder
        de_out = self.msfe4_de2_spconv1(Concatenate(axis=3)([de_lstm, en4_out]))
        de_out = self.msfe4_de2_spconv2(Concatenate(axis=3)([de_out, msfe4_de2_cur4]))
        de_out = self.msfe4_de2_spconv3(Concatenate(axis=3)([de_out, msfe4_de2_cur3]))
        de_out = self.msfe4_de2_spconv4(Concatenate(axis=3)([de_out, msfe4_de2_cur2]))

        de_out += msfe4_de2_cur1

        ############################################################################
        # MSFE4(3) - Decoder
        ############################################################################
        en_in = self.msfe4_de3_upsampling(Concatenate(axis=3)([de_out, msfe4_out]))
        msfe4_de3_cur1 = self.msfe4_de3_in(en_in)
        msfe4_de3_cur2 = self.msfe4_de3_conv1(Concatenate(axis=1)([msfe4_de3_prev1, msfe4_de3_cur1]))
        msfe4_de3_cur3 = self.msfe4_de3_conv2(Concatenate(axis=1)([msfe4_de3_prev2, msfe4_de3_cur2]))
        msfe4_de3_cur4 = self.msfe4_de3_conv3(Concatenate(axis=1)([msfe4_de3_prev3, msfe4_de3_cur3]))
        en4_out = self.msfe4_de3_conv4(Concatenate(axis=1)([msfe4_de3_prev4, msfe4_de3_cur4]))

        # lstm
        reshape_out, shape = self.reshape_before_lstm(en4_out)
        de_lstm, msfe4_de3_h, msfe4_de3_c = self.msfe4_de3_lstm(reshape_out,
                                                                     initial_state=[msfe4_de3_h, msfe4_de3_c])
        de_lstm = self.msfe4_de3_dense(de_lstm)
        de_lstm = self.reshape_after_lstm(de_lstm, shape)

        # Decoder
        de_out = self.msfe4_de3_spconv1(Concatenate(axis=3)([de_lstm, en4_out]))
        de_out = self.msfe4_de3_spconv2(Concatenate(axis=3)([de_out, msfe4_de3_cur4]))
        de_out = self.msfe4_de3_spconv3(Concatenate(axis=3)([de_out, msfe4_de3_cur3]))
        de_out = self.msfe4_de3_spconv4(Concatenate(axis=3)([de_out, msfe4_de3_cur2]))

        de_out += msfe4_de3_cur1

        ############################################################################
        # MSFE5 - Decoder
        ############################################################################
        en_in = self.msfe5_de_upsampling(Concatenate(axis=3)([de_out, msfe5_out]))
        msfe5_de_cur1 = self.msfe5_de_in(en_in)
        msfe5_de_cur2 = self.msfe5_de_conv1(Concatenate(axis=1)([msfe5_de_prev1, msfe5_de_cur1]))
        msfe5_de_cur3 = self.msfe5_de_conv2(Concatenate(axis=1)([msfe5_de_prev2, msfe5_de_cur2]))
        msfe5_de_cur4 = self.msfe5_de_conv3(Concatenate(axis=1)([msfe5_de_prev3, msfe5_de_cur3]))
        msfe5_de_cur5 = self.msfe5_de_conv4(Concatenate(axis=1)([msfe5_de_prev4, msfe5_de_cur4]))
        en5_out = self.msfe5_de_conv5(Concatenate(axis=1)([msfe5_de_prev5, msfe5_de_cur5]))

        # lstm
        reshape_out, shape = self.reshape_before_lstm(en5_out)
        de_lstm, msfe5_de_h, msfe5_de_c = self.msfe5_de_lstm(reshape_out, initial_state=[msfe5_de_h, msfe5_de_c])
        de_lstm = self.msfe5_de_dense(de_lstm)
        de_lstm = self.reshape_after_lstm(de_lstm, shape)

        # Decoder
        de_out = self.msfe5_de_spconv1(Concatenate(axis=3)([de_lstm, en5_out]))
        de_out = self.msfe5_de_spconv2(Concatenate(axis=3)([de_out, msfe5_de_cur5]))
        de_out = self.msfe5_de_spconv3(Concatenate(axis=3)([de_out, msfe5_de_cur4]))
        de_out = self.msfe5_de_spconv4(Concatenate(axis=3)([de_out, msfe5_de_cur3]))
        de_out = self.msfe5_de_spconv5(Concatenate(axis=3)([de_out, msfe5_de_cur2]))

        de_out += msfe5_de_cur1

        ############################################################################
        # MSFE6 - Decoder
        ############################################################################
        en_in = self.msfe6_de_upsampling(Concatenate(axis=3)([de_out, msfe6_out]))
        msfe6_de_cur1 = self.msfe6_de_in(en_in)
        msfe6_de_cur2 = self.msfe6_de_conv1(Concatenate(axis=1)([msfe6_de_prev1, msfe6_de_cur1]))
        msfe6_de_cur3 = self.msfe6_de_conv2(Concatenate(axis=1)([msfe6_de_prev2, msfe6_de_cur2]))
        msfe6_de_cur4 = self.msfe6_de_conv3(Concatenate(axis=1)([msfe6_de_prev3, msfe6_de_cur3]))
        msfe6_de_cur5 = self.msfe6_de_conv4(Concatenate(axis=1)([msfe6_de_prev4, msfe6_de_cur4]))
        msfe6_de_cur6 = self.msfe6_de_conv5(Concatenate(axis=1)([msfe6_de_prev5, msfe6_de_cur5]))
        en6_out = self.msfe6_de_conv6(Concatenate(axis=1)([msfe6_de_prev6, msfe6_de_cur6]))

        # lstm
        reshape_out, shape = self.reshape_before_lstm(en6_out)
        de_lstm, msfe6_de_h, msfe6_de_c = self.msfe6_de_lstm(reshape_out, initial_state=[msfe6_de_h, msfe6_de_c])
        de_lstm = self.msfe6_de_dense(de_lstm)
        de_lstm = self.reshape_after_lstm(de_lstm, shape)

        # Decoder
        de_out = self.msfe6_de_spconv1(Concatenate(axis=3)([de_lstm, en6_out]))
        de_out = self.msfe6_de_spconv2(Concatenate(axis=3)([de_out, msfe6_de_cur6]))
        de_out = self.msfe6_de_spconv3(Concatenate(axis=3)([de_out, msfe6_de_cur5]))
        de_out = self.msfe6_de_spconv4(Concatenate(axis=3)([de_out, msfe6_de_cur4]))
        de_out = self.msfe6_de_spconv5(Concatenate(axis=3)([de_out, msfe6_de_cur3]))
        de_out = self.msfe6_de_spconv6(Concatenate(axis=3)([de_out, msfe6_de_cur2]))

        de_out += msfe6_de_cur1

        model_out = self.out_conv(de_out)

        return {
            "msfe6_en_cur1": msfe6_en_cur1,  # MSFE6 - Encoder
            "msfe6_en_cur2": msfe6_en_cur2,
            "msfe6_en_cur3": msfe6_en_cur3,
            "msfe6_en_cur4": msfe6_en_cur4,
            "msfe6_en_cur5": msfe6_en_cur5,
            "msfe6_en_cur6": msfe6_en_cur6,
            "msfe6_en_h": msfe6_en_h,
            "msfe6_en_c": msfe6_en_c,
            "msfe5_en_cur1": msfe5_en_cur1,  # MSFE5 - Encoder
            "msfe5_en_cur2": msfe5_en_cur2,
            "msfe5_en_cur3": msfe5_en_cur3,
            "msfe5_en_cur4": msfe5_en_cur4,
            "msfe5_en_cur5": msfe5_en_cur5,
            "msfe5_en_h": msfe5_en_h,
            "msfe5_en_c": msfe5_en_c,
            "msfe4_en_cur1": msfe4_en_cur1,  # MSFE4 - Encoder
            "msfe4_en_cur2": msfe4_en_cur2,
            "msfe4_en_cur3": msfe4_en_cur3,
            "msfe4_en_cur4": msfe4_en_cur4,
            "msfe4_en_h": msfe4_en_h,
            "msfe4_en_c": msfe4_en_c,
            "msfe4_en2_cur1": msfe4_en2_cur1,  # MSFE4(2) - Encoder
            "msfe4_en2_cur2": msfe4_en2_cur2,
            "msfe4_en2_cur3": msfe4_en2_cur3,
            "msfe4_en2_cur4": msfe4_en2_cur4,
            "msfe4_en2_h": msfe4_en2_h,
            "msfe4_en2_c": msfe4_en2_c,
            "msfe4_en3_cur1": msfe4_en3_cur1,  # MSFE4(3) - Encoder
            "msfe4_en3_cur2": msfe4_en3_cur2,
            "msfe4_en3_cur3": msfe4_en3_cur3,
            "msfe4_en3_cur4": msfe4_en3_cur4,
            "msfe4_en3_h": msfe4_en3_h,
            "msfe4_en3_c": msfe4_en3_c,
            "msfe3_en_cur1": msfe3_en_cur1,  # MSFE3 - Encoder
            "msfe3_en_cur2": msfe3_en_cur2,
            "msfe3_en_cur3": msfe3_en_cur3,
            "msfe3_en_h": msfe3_en_h,
            "msfe3_en_c": msfe3_en_c,
            "h_state": h_state,  # LSTM
            "c_state": c_state,
            "msfe3_de_cur1": msfe3_de_cur1,  # MSFE3 - Decoder
            "msfe3_de_cur2": msfe3_de_cur2,
            "msfe3_de_cur3": msfe3_de_cur3,
            "msfe3_de_h": msfe3_de_h,
            "msfe3_de_c": msfe3_de_c,
            "msfe4_de_cur1": msfe4_de_cur1,  # MSFE4 - Decoder
            "msfe4_de_cur2": msfe4_de_cur2,
            "msfe4_de_cur3": msfe4_de_cur3,
            "msfe4_de_cur4": msfe4_de_cur4,
            "msfe4_de_h": msfe4_de_h,
            "msfe4_de_c": msfe4_de_c,
            "msfe4_de2_cur1": msfe4_de2_cur1,  # MSFE4(2) - Decoder
            "msfe4_de2_cur2": msfe4_de2_cur2,
            "msfe4_de2_cur3": msfe4_de2_cur3,
            "msfe4_de2_cur4": msfe4_de2_cur4,
            "msfe4_de2_h": msfe4_de2_h,
            "msfe4_de2_c": msfe4_de2_c,
            "msfe4_de3_cur1": msfe4_de3_cur1,  # MSFE4(3) - Decoder
            "msfe4_de3_cur2": msfe4_de3_cur2,
            "msfe4_de3_cur3": msfe4_de3_cur3,
            "msfe4_de3_cur4": msfe4_de3_cur4,
            "msfe4_de3_h": msfe4_de3_h,
            "msfe4_de3_c": msfe4_de3_c,
            "msfe5_de_cur1": msfe5_de_cur1,  # MSFE5 - Decoder
            "msfe5_de_cur2": msfe5_de_cur2,
            "msfe5_de_cur3": msfe5_de_cur3,
            "msfe5_de_cur4": msfe5_de_cur4,
            "msfe5_de_cur5": msfe5_de_cur5,
            "msfe5_de_h": msfe5_de_h,
            "msfe5_de_c": msfe5_de_c,
            "msfe6_de_cur1": msfe6_de_cur1,  # MSFE6 - Decoder
            "msfe6_de_cur2": msfe6_de_cur2,
            "msfe6_de_cur3": msfe6_de_cur3,
            "msfe6_de_cur4": msfe6_de_cur4,
            "msfe6_de_cur5": msfe6_de_cur5,
            "msfe6_de_cur6": msfe6_de_cur6,
            "msfe6_de_h": msfe6_de_h,
            "msfe6_de_c": msfe6_de_c,
            "model_out": model_out,
        }


if __name__ == "__main__":
    win_len = 512
    stride = 128
    fft_len = 512

    weights_path = "./saved_model/nunet_lstm.h5"
    tflite_path = "./tflite/nunet_lstm.tflite"
    saved_model = "./saved_model/nunet_lstm"

    m = NUNET_LSTM()
    tflite_model = m.tflite_model()

    model_signature = TFL_SIGNITURE(tflite_model, weights_path)

    tf.saved_model.save(
        model_signature, saved_model,
        signatures={
            'nunet_lstm': model_signature.nunet_lstm.get_concrete_function(),
        }
    )

    # Convert the saved model using TFLiteConverter
    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model)
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,  # enable TensorFlow Lite ops.
        tf.lite.OpsSet.SELECT_TF_OPS  # enable TensorFlow ops.
    ]
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.experimental_new_converter = True
    tflite_model = converter.convert()

    # Print the signatures from the converted model
    interpreter = tf.lite.Interpreter(model_content=tflite_model)
    signatures = interpreter.get_signature_list()
    print(signatures)

    with open(tflite_path, 'wb') as f:
        f.write(tflite_model)

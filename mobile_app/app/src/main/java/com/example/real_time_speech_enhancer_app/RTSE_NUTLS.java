package com.example.real_time_speech_enhancer_app;

import android.app.Activity;
import android.util.Log;

import org.apache.commons.math3.complex.Complex;
import org.apache.commons.math3.transform.DftNormalization;
import org.apache.commons.math3.transform.FastFourierTransformer;
import org.apache.commons.math3.transform.TransformType;
import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.nnapi.NnApiDelegate;
import org.tensorflow.lite.support.common.FileUtil;

import java.io.IOException;
import java.nio.MappedByteBuffer;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;

public class RTSE_NUTLS {

    private final int frame_len;
    private final int stride;
    private final double[] window;
    private final double[] inverse_window;

    private Interpreter.Options tfOptions = new Interpreter.Options();
    private NnApiDelegate nnApiDelegate = null;
    private MappedByteBuffer tfModel;
    private Interpreter tflite;
    private Complex[] realTranformedBuffer;
    private double[] in_buffer;
    private double[] out_buffer;
    private double[] estimated_block;

    private FastFourierTransformer fft;
    private FastFourierTransformer ifft;
    private int fft_bins;
    private float[] mags;
    private float[] phase;

    double half = 0.5;

    public RTSE_NUTLS(Activity activity, String tflitePath, int frame_len, int stride) throws IOException {
        this.frame_len = frame_len;
        this.stride = stride;
        this.in_buffer = new double[this.frame_len];
        this.out_buffer = new double[this.frame_len];
        this.estimated_block = new double[this.frame_len];
        this.fft_bins = frame_len / 2 + 1;
        this.mags = new float[this.fft_bins];
        this.phase = new float[this.fft_bins];

        this.window = new double[]{1e-07, 3.7640333e-05, 0.00015059114, 0.0003387928, 0.00060227513, 0.0009409487, 0.0013547838, 0.0018436909, 0.0024076402, 0.0030465126, 0.0037602186, 0.004548669, 0.005411744, 0.0063492954, 0.0073611736, 0.00844726, 0.009607375, 0.01084131, 0.0121489465, 0.013530016, 0.014984369, 0.016511768, 0.018111974, 0.019784749, 0.021529824, 0.02334699, 0.025235921, 0.027196348, 0.029227972, 0.031330496, 0.033503592, 0.03574696, 0.038060248, 0.040443063, 0.04289514, 0.045415998, 0.048005342, 0.050662756, 0.05338785, 0.05618018, 0.059039384, 0.06196496, 0.064956516, 0.06801358, 0.0711357, 0.0743224, 0.07757321, 0.080887675, 0.0842652, 0.087705374, 0.09120759, 0.094771415, 0.09839624, 0.10208154, 0.105826795, 0.10963139, 0.11349478, 0.11741638, 0.12139559, 0.1254318, 0.12952444, 0.13367286, 0.13787645, 0.1421346, 0.14644665, 0.15081188, 0.15522975, 0.1596995, 0.16422054, 0.16879213, 0.1734136, 0.17808425, 0.18280336, 0.18757027, 0.19238421, 0.1972445, 0.20215037, 0.20710108, 0.21209592, 0.21713412, 0.2222149, 0.22733751, 0.23250121, 0.23770517, 0.24294865, 0.24823081, 0.25355092, 0.25890815, 0.26430166, 0.26973063, 0.27519435, 0.28069192, 0.28622246, 0.29178524, 0.29737937, 0.30300403, 0.30865827, 0.31434143, 0.3200525, 0.32579064, 0.33155507, 0.33734488, 0.34315917, 0.34899703, 0.35485768, 0.36074018, 0.3666436, 0.37256718, 0.37850994, 0.38447094, 0.3904494, 0.39644432, 0.40245488, 0.40848005, 0.41451907, 0.42057097, 0.42663476, 0.43270966, 0.4387947, 0.44488895, 0.45099142, 0.45710137, 0.46321777, 0.46933964, 0.47546616, 0.4815964, 0.48772943, 0.49386424, 0.5, 0.5061358, 0.5122706, 0.51840365, 0.52453387, 0.5306604, 0.53678226, 0.54289865, 0.5490086, 0.5551111, 0.5612053, 0.56729037, 0.5733653, 0.5794291, 0.585481, 0.59152, 0.59754515, 0.6035557, 0.60955065, 0.6155291, 0.6214901, 0.6274328, 0.63335645, 0.6392598, 0.6451423, 0.651003, 0.65684086, 0.6626552, 0.66844493, 0.67420936, 0.6799475, 0.68565863, 0.69134176, 0.69699603, 0.7026207, 0.70821476, 0.71377754, 0.71930814, 0.7248057, 0.7302694, 0.7356984, 0.74109197, 0.7464491, 0.7517692, 0.75705135, 0.7622949, 0.76749885, 0.7726625, 0.7777852, 0.7828659, 0.78790414, 0.79289895, 0.79784966, 0.8027556, 0.8076159, 0.8124298, 0.8171966, 0.82191575, 0.8265864, 0.83120793, 0.83577955, 0.84030056, 0.8447703, 0.8491881, 0.85355335, 0.85786545, 0.86212355, 0.86632717, 0.8704756, 0.8745682, 0.8786044, 0.8825836, 0.88650525, 0.89036864, 0.89417326, 0.89791846, 0.9016038, 0.9052286, 0.9087924, 0.9122946, 0.9157348, 0.9191124, 0.9224268, 0.92567766, 0.9288643, 0.93198645, 0.9350435, 0.9380351, 0.94096065, 0.9438199, 0.9466121, 0.94933724, 0.95199466, 0.954584, 0.9571049, 0.95955694, 0.9619398, 0.96425307, 0.9664964, 0.96866953, 0.970772, 0.9728037, 0.9747641, 0.97665304, 0.9784702, 0.98021525, 0.98188806, 0.9834882, 0.98501563, 0.98647, 0.9878511, 0.9891587, 0.9903927, 0.9915527, 0.9926388, 0.9936507, 0.99458826, 0.99545133, 0.9962398, 0.9969535, 0.99759233, 0.9981563, 0.99864525, 0.9990591, 0.99939775, 0.9996612, 0.99984944, 0.99996233, 1.0, 0.99996233, 0.99984944, 0.9996612, 0.99939775, 0.9990591, 0.9986452, 0.9981563, 0.99759233, 0.9969535, 0.9962398, 0.99545133, 0.99458826, 0.9936507, 0.9926388, 0.9915527, 0.9903926, 0.98915863, 0.987851, 0.98647, 0.98501563, 0.9834882, 0.98188806, 0.9802152, 0.97847015, 0.976653, 0.9747641, 0.97280365, 0.970772, 0.9686695, 0.96649635, 0.964253, 0.96193975, 0.95955694, 0.95710486, 0.954584, 0.95199466, 0.9493372, 0.9466121, 0.9438198, 0.94096065, 0.938035, 0.93504345, 0.9319864, 0.92886424, 0.9256776, 0.9224268, 0.9191123, 0.91573477, 0.9122946, 0.9087924, 0.90522856, 0.9016038, 0.89791846, 0.8941732, 0.8903686, 0.8865052, 0.8825836, 0.8786044, 0.8745682, 0.87047553, 0.86632717, 0.8621235, 0.8578654, 0.85355335, 0.849188, 0.84477025, 0.8403005, 0.8357794, 0.8312079, 0.8265865, 0.8219158, 0.81719667, 0.8124298, 0.80761576, 0.8027555, 0.79784966, 0.7928989, 0.787904, 0.7828658, 0.777785, 0.7726624, 0.7674987, 0.7622947, 0.7570514, 0.7517692, 0.7464491, 0.74109185, 0.73569834, 0.7302693, 0.7248056, 0.719308, 0.7137775, 0.7082147, 0.7026205, 0.69699585, 0.6913416, 0.68565863, 0.67994756, 0.67420936, 0.66844493, 0.6626551, 0.65684086, 0.65100294, 0.6451423, 0.63925976, 0.6333563, 0.6274327, 0.62148994, 0.6155289, 0.6095505, 0.60355574, 0.5975452, 0.59151995, 0.5854809, 0.57942903, 0.5733652, 0.5672903, 0.56120527, 0.555111, 0.5490085, 0.54289854, 0.53678215, 0.5306602, 0.5245337, 0.51840365, 0.5122706, 0.50613576, 0.5, 0.4938642, 0.48772934, 0.48159632, 0.47546607, 0.46933955, 0.46321762, 0.45710123, 0.45099127, 0.44488874, 0.4387945, 0.4327097, 0.4266348, 0.42057094, 0.41451904, 0.40848002, 0.4024548, 0.39644426, 0.3904493, 0.38447085, 0.3785098, 0.37256706, 0.3666435, 0.36074, 0.3548575, 0.34899706, 0.34315914, 0.33734486, 0.33155507, 0.32579064, 0.32005244, 0.31434137, 0.3086582, 0.3030039, 0.29737926, 0.29178512, 0.28622234, 0.28069174, 0.27519417, 0.2697307, 0.26430166, 0.25890812, 0.2535509, 0.24823079, 0.24294859, 0.23770511, 0.23250112, 0.22733742, 0.22221479, 0.21713397, 0.21209577, 0.20710093, 0.20215037, 0.1972445, 0.19238421, 0.18757024, 0.18280333, 0.1780842, 0.17341354, 0.16879207, 0.16422045, 0.15969944, 0.15522963, 0.15081179, 0.14644647, 0.14213446, 0.13787648, 0.13367286, 0.12952444, 0.12543178, 0.12139556, 0.11741632, 0.113494724, 0.10963133, 0.105826735, 0.10208148, 0.09839615, 0.094771326, 0.091207504, 0.087705255, 0.0842652, 0.080887675, 0.07757321, 0.0743224, 0.07113567, 0.06801355, 0.064956486, 0.0619649, 0.059039325, 0.05618012, 0.05338779, 0.050662696, 0.048005283, 0.045415938, 0.04289514, 0.040443063, 0.038060218, 0.03574696, 0.033503592, 0.031330466, 0.029227942, 0.027196318, 0.025235891, 0.02334693, 0.021529794, 0.019784689, 0.018111914, 0.016511708, 0.014984369, 0.013530016, 0.012148917, 0.01084131, 0.009607345, 0.00844723, 0.0073611736, 0.0063492656, 0.0054117143, 0.004548669, 0.0037602186, 0.0030464828, 0.0024076104, 0.0018436909, 0.0013547838, 0.0009409487, 0.00060227513, 0.0003387928, 0.00015059114, 1e-07};
        this.inverse_window = new double[]{0.0, 3.764317e-05, 0.0001506365, 0.00033902243, 0.000603001, 0.0009427211, 0.0013584597, 0.0018505018, 0.0024192617, 0.0030651316, 0.0037886035, 0.004590238, 0.0054706354, 0.006430435, 0.007470345, 0.008591178, 0.009793751, 0.011078927, 0.012447727, 0.013901091, 0.015440158, 0.017066045, 0.018779933, 0.0205831, 0.022476831, 0.024462579, 0.026541721, 0.028715799, 0.030986369, 0.033355076, 0.035823613, 0.038393762, 0.041067336, 0.043846175, 0.046732344, 0.049727727, 0.052834507, 0.056054793, 0.059390787, 0.06284474, 0.066419035, 0.07011598, 0.0739381, 0.07788785, 0.08196782, 0.08618061, 0.0905289, 0.0950155, 0.09964303, 0.10441443, 0.109332465, 0.11440015, 0.11962032, 0.124996044, 0.13053031, 0.1362261, 0.14208649, 0.14811453, 0.1543133, 0.16068585, 0.1672353, 0.17396459, 0.1808769, 0.18797512, 0.1952622, 0.20274106, 0.21041453, 0.21828535, 0.22635634, 0.23462991, 0.24310857, 0.25179476, 0.26069054, 0.26979804, 0.2791191, 0.2886554, 0.29840827, 0.3083791, 0.3185688, 0.32897806, 0.3396073, 0.35045657, 0.36152574, 0.37281412, 0.38432068, 0.39604422, 0.4079829, 0.42013457, 0.4324962, 0.4450649, 0.45783693, 0.47080794, 0.48397312, 0.49732727, 0.5108645, 0.52457815, 0.53846097, 0.5525053, 0.566703, 0.581045, 0.59552157, 0.6101225, 0.62483674, 0.6396529, 0.6545589, 0.66954195, 0.68458873, 0.69968545, 0.7148177, 0.72997063, 0.7451288, 0.7602763, 0.7753978, 0.79047626, 0.80549514, 0.8204375, 0.83528596, 0.85002375, 0.86463344, 0.87909794, 0.8933998, 0.90752256, 0.921449, 0.9351632, 0.94864863, 0.9618895, 0.97487164, 0.98757976, 1.0, 1.0121192, 1.0239246, 1.0354047, 1.0465481, 1.057345, 1.0677862, 1.0778632, 1.0875689, 1.0968964, 1.1058407, 1.1143967, 1.1225619, 1.1303332, 1.137709, 1.1446886, 1.1512728, 1.1574621, 1.1632587, 1.1686656, 1.1736866, 1.1783259, 1.1825887, 1.1864809, 1.1900088, 1.1931789, 1.1959999, 1.198479, 1.2006252, 1.2024469, 1.2039536, 1.2051548, 1.20606, 1.2066796, 1.2070234, 1.207102, 1.2069252, 1.2065041, 1.2058488, 1.2049695, 1.2038772, 1.202582, 1.2010939, 1.1994234, 1.1975803, 1.1955744, 1.1934154, 1.1911126, 1.1886756, 1.1861135, 1.1834345, 1.1806479, 1.1777616, 1.1747838, 1.1717228, 1.1685858, 1.1653804, 1.1621134, 1.1587919, 1.1554224, 1.1520115, 1.1485652, 1.1450893, 1.1415896, 1.1380712, 1.1345396, 1.1309996, 1.1274558, 1.1239132, 1.1203756, 1.1168474, 1.1133324, 1.1098343, 1.1063567, 1.102903, 1.0994765, 1.0960798, 1.0927163, 1.0893886, 1.086099, 1.0828501, 1.0796443, 1.0764836, 1.0733702, 1.0703062, 1.0672929, 1.0643325, 1.0614264, 1.0585763, 1.0557835, 1.0530493, 1.0503751, 1.0477619, 1.0452108, 1.0427231, 1.0402995, 1.0379412, 1.0356488, 1.0334232, 1.031265, 1.029175, 1.0271538, 1.025202, 1.0233203, 1.0215089, 1.0197687, 1.0180995, 1.0165025, 1.0149775, 1.013525, 1.0121453, 1.0108387, 1.0096054, 1.0084461, 1.0073603, 1.0063488, 1.0054115, 1.0045485, 1.0037601, 1.0030464, 1.0024077, 1.0018437, 1.0013547, 1.0009409, 1.0006022, 1.0003388, 1.0001506, 1.0000377, 1.0, 1.0000377, 1.0001506, 1.0003388, 1.0006022, 1.0009409, 1.0013548, 1.0018437, 1.0024077, 1.0030464, 1.0037601, 1.0045485, 1.0054115, 1.0063488, 1.0073603, 1.0084461, 1.0096055, 1.0108387, 1.0121453, 1.013525, 1.0149775, 1.0165025, 1.0180995, 1.0197687, 1.021509, 1.0233203, 1.025202, 1.0271538, 1.029175, 1.031265, 1.0334233, 1.0356488, 1.0379413, 1.0402995, 1.0427232, 1.0452108, 1.0477619, 1.0503751, 1.0530493, 1.0557835, 1.0585763, 1.0614265, 1.0643326, 1.067293, 1.0703062, 1.0733703, 1.0764836, 1.0796443, 1.0828501, 1.086099, 1.0893886, 1.0927165, 1.0960798, 1.0994765, 1.1029031, 1.1063569, 1.1098343, 1.1133324, 1.1168474, 1.1203756, 1.1239133, 1.1274558, 1.1309997, 1.1345396, 1.1380712, 1.1415896, 1.1450894, 1.1485652, 1.1520116, 1.1554226, 1.1587918, 1.1621133, 1.1653804, 1.1685858, 1.1717229, 1.1747841, 1.1777616, 1.1806479, 1.1834345, 1.1861134, 1.1886758, 1.1911128, 1.1934155, 1.1955744, 1.1975802, 1.1994233, 1.2010939, 1.2025821, 1.2038772, 1.2049696, 1.2058487, 1.206504, 1.2069253, 1.2071018, 1.2070234, 1.2066796, 1.2060602, 1.2051547, 1.2039535, 1.2024469, 1.2006252, 1.198479, 1.1959997, 1.193179, 1.1900084, 1.1864805, 1.1825886, 1.1783259, 1.1736865, 1.1686658, 1.1632586, 1.1574618, 1.1512725, 1.1446886, 1.1377089, 1.1303331, 1.1225619, 1.1143967, 1.1058402, 1.096896, 1.0875685, 1.077863, 1.0677859, 1.0573449, 1.046548, 1.0354044, 1.0239245, 1.0121192, 1.0, 0.9875796, 0.9748716, 0.96188956, 0.9486482, 0.9351628, 0.9214488, 0.90752226, 0.8933996, 0.87909764, 0.8646333, 0.8500237, 0.83528596, 0.82043743, 0.8054951, 0.7904761, 0.7753979, 0.7602765, 0.74512845, 0.72997016, 0.7148173, 0.69968516, 0.6845884, 0.6695417, 0.6545587, 0.63965285, 0.62483674, 0.6101224, 0.59552157, 0.581045, 0.566703, 0.55250525, 0.5384606, 0.5245778, 0.51086414, 0.49732706, 0.48397285, 0.47080758, 0.45783657, 0.4450649, 0.43249616, 0.42013443, 0.40798286, 0.3960442, 0.38432065, 0.3728139, 0.3615255, 0.35045636, 0.33960703, 0.32897782, 0.3185685, 0.30837885, 0.29840827, 0.28865528, 0.27911904, 0.26979798, 0.26069054, 0.2517947, 0.24310853, 0.2346298, 0.22635616, 0.21828523, 0.21041434, 0.20274092, 0.19526197, 0.18797489, 0.1808769, 0.17396459, 0.16723527, 0.16068581, 0.15431327, 0.14811446, 0.1420864, 0.136226, 0.13053022, 0.12499597, 0.11962021, 0.11440003, 0.10933236, 0.10441429, 0.099643014, 0.095015496, 0.0905289, 0.086180605, 0.08196778, 0.0778878, 0.07393806, 0.07011591, 0.06641897, 0.062844664, 0.059390724, 0.05605472, 0.05283444, 0.049727663, 0.04673234, 0.043846175, 0.0410673, 0.038393755, 0.03582361, 0.033355042, 0.030986337, 0.028715763, 0.02654169, 0.024462514, 0.022476798, 0.020583035, 0.018779872, 0.017065983, 0.015440158, 0.013901091, 0.012447694, 0.011078926, 0.009793719, 0.008591148, 0.007470345, 0.0064304047, 0.005470605, 0.004590238, 0.0037886035, 0.0030651016, 0.002419232, 0.0018505018, 0.0013584595, 0.0009427211, 0.000603001, 0.00033902243, 0.0001506365, 3.764317e-05};
        this.window[0] = 1e-7;
        this.window[this.frame_len - 1] = 1e-7;

        fft = new FastFourierTransformer(DftNormalization.STANDARD);
        ifft = new FastFourierTransformer(DftNormalization.STANDARD);

        tfOptions.setNumThreads(-1);

        tfModel = FileUtil.loadMappedFile(activity, tflitePath);
        try {
            tflite = new Interpreter(tfModel, tfOptions);
        } catch (Exception e) {
            throw new RuntimeException(e);
        }

        Log.d("", "Init interpreter of tflite");
    }

    // Encoder MSFE6 encoder
    private float[][][][] msfe6_ee_prev1 = new float[1][1][256][64];
    private float[][][][] msfe6_ee_prev2 = new float[1][1][128][32];
    private float[][][][] msfe6_ee_prev3 = new float[1][1][64][32];
    private float[][][][] msfe6_ee_prev4 = new float[1][1][32][32];
    private float[][][][] msfe6_ee_prev5 = new float[1][1][16][32];
    private float[][][][] msfe6_ee_prev6 = new float[1][1][8][32];
    // Encoder MSFE5 encoder
    private float[][][][] msfe5_ee_prev1 = new float[1][1][128][64];
    private float[][][][] msfe5_ee_prev2 = new float[1][1][64][32];
    private float[][][][] msfe5_ee_prev3 = new float[1][1][32][32];
    private float[][][][] msfe5_ee_prev4 = new float[1][1][16][32];
    private float[][][][] msfe5_ee_prev5 = new float[1][1][8][32];
    // Encoder MSFE4 encoder
    private float[][][][] msfe4_ee_prev1 = new float[1][1][64][64];
    private float[][][][] msfe4_ee_prev2 = new float[1][1][32][32];
    private float[][][][] msfe4_ee_prev3 = new float[1][1][16][32];
    private float[][][][] msfe4_ee_prev4 = new float[1][1][8][32];
    // Encoder MSFE4(2) encoder
    private float[][][][] msfe4_ee2_prev1 = new float[1][1][32][64];
    private float[][][][] msfe4_ee2_prev2 = new float[1][1][16][32];
    private float[][][][] msfe4_ee2_prev3 = new float[1][1][8][32];
    private float[][][][] msfe4_ee2_prev4 = new float[1][1][4][32];
    // Encoder MSFE4(3) encoder
    private float[][][][] msfe4_ee3_prev1 = new float[1][1][16][64];
    private float[][][][] msfe4_ee3_prev2 = new float[1][1][8][32];
    private float[][][][] msfe4_ee3_prev3 = new float[1][1][4][32];
    private float[][][][] msfe4_ee3_prev4 = new float[1][1][2][32];
    // Encoder MSFE3 encoder
    private float[][][][] msfe3_ee_prev1 = new float[1][1][8][64];
    private float[][][][] msfe3_ee_prev2 = new float[1][1][4][32];
    private float[][][][] msfe3_ee_prev3 = new float[1][1][2][32];

    // Encoder MSFE6 decoder
    private float[][][][] msfe6_ed_prev1 = new float[1][1][4][64];
    private float[][][][] msfe6_ed_prev2 = new float[1][1][8][64];
    private float[][][][] msfe6_ed_prev3 = new float[1][1][16][64];
    private float[][][][] msfe6_ed_prev4 = new float[1][1][32][64];
    private float[][][][] msfe6_ed_prev5 = new float[1][1][64][64];
    private float[][][][] msfe6_ed_prev6 = new float[1][1][128][64];
    // Encoder MSFE5 decoder
    private float[][][][] msfe5_ed_prev1 = new float[1][1][4][64];
    private float[][][][] msfe5_ed_prev2 = new float[1][1][8][64];
    private float[][][][] msfe5_ed_prev3 = new float[1][1][16][64];
    private float[][][][] msfe5_ed_prev4 = new float[1][1][32][64];
    private float[][][][] msfe5_ed_prev5 = new float[1][1][64][64];
    // Encoder MSFE4 decoder
    private float[][][][] msfe4_ed_prev1 = new float[1][1][4][64];
    private float[][][][] msfe4_ed_prev2 = new float[1][1][8][64];
    private float[][][][] msfe4_ed_prev3 = new float[1][1][16][64];
    private float[][][][] msfe4_ed_prev4 = new float[1][1][32][64];
    // Encoder MSFE4(2) decoder
    private float[][][][] msfe4_ed2_prev1 = new float[1][1][2][64];
    private float[][][][] msfe4_ed2_prev2 = new float[1][1][4][64];
    private float[][][][] msfe4_ed2_prev3 = new float[1][1][8][64];
    private float[][][][] msfe4_ed2_prev4 = new float[1][1][16][64];
    // Encoder MSFE4(3) decoder
    private float[][][][] msfe4_ed3_prev1 = new float[1][1][1][64];
    private float[][][][] msfe4_ed3_prev2 = new float[1][1][2][64];
    private float[][][][] msfe4_ed3_prev3 = new float[1][1][4][64];
    private float[][][][] msfe4_ed3_prev4 = new float[1][1][8][64];
    // Encoder MSFE3 decoder
    private float[][][][] msfe3_ed_prev1 = new float[1][1][1][64];
    private float[][][][] msfe3_ed_prev2 = new float[1][1][2][64];
    private float[][][][] msfe3_ed_prev3 = new float[1][1][4][64];

    // Decoder MSFE3 encoder
    private float[][][][] msfe3_de_prev1 = new float[1][1][8][128];
    private float[][][][] msfe3_de_prev2 = new float[1][1][4][64];
    private float[][][][] msfe3_de_prev3 = new float[1][1][2][64];
    // Decoder MSFE4 encoder
    private float[][][][] msfe4_de_prev1 = new float[1][1][16][128];
    private float[][][][] msfe4_de_prev2 = new float[1][1][8][64];
    private float[][][][] msfe4_de_prev3 = new float[1][1][4][64];
    private float[][][][] msfe4_de_prev4 = new float[1][1][2][64];
    // Decoder MSFE4(2) encoder
    private float[][][][] msfe4_de2_prev1 = new float[1][1][32][128];
    private float[][][][] msfe4_de2_prev2 = new float[1][1][16][64];
    private float[][][][] msfe4_de2_prev3 = new float[1][1][8][64];
    private float[][][][] msfe4_de2_prev4 = new float[1][1][4][64];
    // Decoder MSFE4(3) encoder
    private float[][][][] msfe4_de3_prev1 = new float[1][1][64][128];
    private float[][][][] msfe4_de3_prev2 = new float[1][1][32][64];
    private float[][][][] msfe4_de3_prev3 = new float[1][1][16][64];
    private float[][][][] msfe4_de3_prev4 = new float[1][1][8][64];
    // Decoder MSFE5 encoder
    private float[][][][] msfe5_de_prev1 = new float[1][1][128][128];
    private float[][][][] msfe5_de_prev2 = new float[1][1][64][64];
    private float[][][][] msfe5_de_prev3 = new float[1][1][32][64];
    private float[][][][] msfe5_de_prev4 = new float[1][1][16][64];
    private float[][][][] msfe5_de_prev5 = new float[1][1][8][64];
    // Decoder MSFE6 encoder
    private float[][][][] msfe6_de_prev1 = new float[1][1][256][128];
    private float[][][][] msfe6_de_prev2 = new float[1][1][128][64];
    private float[][][][] msfe6_de_prev3 = new float[1][1][64][64];
    private float[][][][] msfe6_de_prev4 = new float[1][1][32][64];
    private float[][][][] msfe6_de_prev5 = new float[1][1][16][64];
    private float[][][][] msfe6_de_prev6 = new float[1][1][8][64];

    // Decoder MSFE3 decoder
    private float[][][][] msfe3_dd_prev1 = new float[1][1][1][64];
    private float[][][][] msfe3_dd_prev2 = new float[1][1][2][64];
    private float[][][][] msfe3_dd_prev3 = new float[1][1][4][64];
    // Decoder MSFE4 decoder
    private float[][][][] msfe4_dd_prev1 = new float[1][1][1][64];
    private float[][][][] msfe4_dd_prev2 = new float[1][1][2][64];
    private float[][][][] msfe4_dd_prev3 = new float[1][1][4][64];
    private float[][][][] msfe4_dd_prev4 = new float[1][1][8][64];
    // Decoder MSFE4(2) decoder
    private float[][][][] msfe4_dd2_prev1 = new float[1][1][2][64];
    private float[][][][] msfe4_dd2_prev2 = new float[1][1][4][64];
    private float[][][][] msfe4_dd2_prev3 = new float[1][1][8][64];
    private float[][][][] msfe4_dd2_prev4 = new float[1][1][16][64];
    // Decoder MSFE4(3) decoder
    private float[][][][] msfe4_dd3_prev1 = new float[1][1][4][64];
    private float[][][][] msfe4_dd3_prev2 = new float[1][1][8][64];
    private float[][][][] msfe4_dd3_prev3 = new float[1][1][16][64];
    private float[][][][] msfe4_dd3_prev4 = new float[1][1][32][64];
    // Decoder MSFE5 decoder
    private float[][][][] msfe5_dd_prev1 = new float[1][1][4][64];
    private float[][][][] msfe5_dd_prev2 = new float[1][1][8][64];
    private float[][][][] msfe5_dd_prev3 = new float[1][1][16][64];
    private float[][][][] msfe5_dd_prev4 = new float[1][1][32][64];
    private float[][][][] msfe5_dd_prev5 = new float[1][1][64][64];
    // Decoder MSFE6 decoder
    private float[][][][] msfe6_dd_prev1 = new float[1][1][4][64];
    private float[][][][] msfe6_dd_prev2 = new float[1][1][8][64];
    private float[][][][] msfe6_dd_prev3 = new float[1][1][16][64];
    private float[][][][] msfe6_dd_prev4 = new float[1][1][32][64];
    private float[][][][] msfe6_dd_prev5 = new float[1][1][64][64];
    private float[][][][] msfe6_dd_prev6 = new float[1][1][128][64];

    // Encoder MSFE6 Dilated Dense Block
    private float[][][][] msfe6_en_ddb_prev_in = new float[1][1][4][32];
    private float[][][][] msfe6_en_ddb_prev1 = new float[1][1][4][16];
    private float[][][][] msfe6_en_ddb_prev2 = new float[1][2][4][32];
    private float[][][][] msfe6_en_ddb_prev3 = new float[1][4][4][48];
    private float[][][][] msfe6_en_ddb_prev4 = new float[1][8][4][64];
    private float[][][][] msfe6_en_ddb_prev5 = new float[1][16][4][80];
    private float[][][][] msfe6_en_ddb_prev6 = new float[1][32][4][96];
    private float[][][][] msfe6_en_ddb_prev_out = new float[1][1][4][16];
    // Encoder MSFE5 Dilated Dense Block
    private float[][][][] msfe5_en_ddb_prev_in = new float[1][1][4][32];
    private float[][][][] msfe5_en_ddb_prev1 = new float[1][1][4][16];
    private float[][][][] msfe5_en_ddb_prev2 = new float[1][2][4][32];
    private float[][][][] msfe5_en_ddb_prev3 = new float[1][4][4][48];
    private float[][][][] msfe5_en_ddb_prev4 = new float[1][8][4][64];
    private float[][][][] msfe5_en_ddb_prev5 = new float[1][16][4][80];
    private float[][][][] msfe5_en_ddb_prev6 = new float[1][32][4][96];
    private float[][][][] msfe5_en_ddb_prev_out = new float[1][1][4][16];
    // Encoder MSFE4 Dilated Dense Block
    private float[][][][] msfe4_en_ddb_prev_in = new float[1][1][4][32];
    private float[][][][] msfe4_en_ddb_prev1 = new float[1][1][4][16];
    private float[][][][] msfe4_en_ddb_prev2 = new float[1][2][4][32];
    private float[][][][] msfe4_en_ddb_prev3 = new float[1][4][4][48];
    private float[][][][] msfe4_en_ddb_prev4 = new float[1][8][4][64];
    private float[][][][] msfe4_en_ddb_prev5 = new float[1][16][4][80];
    private float[][][][] msfe4_en_ddb_prev6 = new float[1][32][4][96];
    private float[][][][] msfe4_en_ddb_prev_out = new float[1][1][4][16];
    // Encoder MSFE4(2) Dilated Dense Block
    private float[][][][] msfe4_en2_ddb_prev_in = new float[1][1][2][32];
    private float[][][][] msfe4_en2_ddb_prev1 = new float[1][1][2][16];
    private float[][][][] msfe4_en2_ddb_prev2 = new float[1][2][2][32];
    private float[][][][] msfe4_en2_ddb_prev3 = new float[1][4][2][48];
    private float[][][][] msfe4_en2_ddb_prev4 = new float[1][8][2][64];
    private float[][][][] msfe4_en2_ddb_prev5 = new float[1][16][2][80];
    private float[][][][] msfe4_en2_ddb_prev6 = new float[1][32][2][96];
    private float[][][][] msfe4_en2_ddb_prev_out = new float[1][1][2][16];
    // Encoder MSFE4(3) Dilated Dense Block
    private float[][][][] msfe4_en3_ddb_prev_in = new float[1][1][1][32];
    private float[][][][] msfe4_en3_ddb_prev1 = new float[1][1][1][16];
    private float[][][][] msfe4_en3_ddb_prev2 = new float[1][2][1][32];
    private float[][][][] msfe4_en3_ddb_prev3 = new float[1][4][1][48];
    private float[][][][] msfe4_en3_ddb_prev4 = new float[1][8][1][64];
    private float[][][][] msfe4_en3_ddb_prev5 = new float[1][16][1][80];
    private float[][][][] msfe4_en3_ddb_prev6 = new float[1][32][1][96];
    private float[][][][] msfe4_en3_ddb_prev_out = new float[1][1][1][16];
    // Encoder MSFE3 Dilated Dense Block
    private float[][][][] msfe3_en_ddb_prev_in = new float[1][1][1][32];
    private float[][][][] msfe3_en_ddb_prev1 = new float[1][1][1][16];
    private float[][][][] msfe3_en_ddb_prev2 = new float[1][2][1][32];
    private float[][][][] msfe3_en_ddb_prev3 = new float[1][4][1][48];
    private float[][][][] msfe3_en_ddb_prev4 = new float[1][8][1][64];
    private float[][][][] msfe3_en_ddb_prev5 = new float[1][16][1][80];
    private float[][][][] msfe3_en_ddb_prev6 = new float[1][32][1][96];
    private float[][][][] msfe3_en_ddb_prev_out = new float[1][1][1][16];

    // Dilated Dense Block
    private float[][][][] ddb_prev_in = new float[1][1][4][64];
    private float[][][][] ddb_prev1 = new float[1][1][4][32];
    private float[][][][] ddb_prev2 = new float[1][2][4][64];
    private float[][][][] ddb_prev3 = new float[1][4][4][96];
    private float[][][][] ddb_prev4 = new float[1][8][4][128];
    private float[][][][] ddb_prev5 = new float[1][16][4][160];
    private float[][][][] ddb_prev6 = new float[1][32][4][192];
    private float[][][][] ddb_prev_out = new float[1][1][4][32];

    // Decoder MSFE6 Dilated Dense Block
    private float[][][][] msfe6_de_ddb_prev_in = new float[1][1][4][32];
    private float[][][][] msfe6_de_ddb_prev1 = new float[1][1][4][16];
    private float[][][][] msfe6_de_ddb_prev2 = new float[1][2][4][32];
    private float[][][][] msfe6_de_ddb_prev3 = new float[1][4][4][48];
    private float[][][][] msfe6_de_ddb_prev4 = new float[1][8][4][64];
    private float[][][][] msfe6_de_ddb_prev5 = new float[1][16][4][80];
    private float[][][][] msfe6_de_ddb_prev6 = new float[1][32][4][96];
    private float[][][][] msfe6_de_ddb_prev_out = new float[1][1][4][16];
    // Decoder MSFE5 Dilated Dense Block
    private float[][][][] msfe5_de_ddb_prev_in = new float[1][1][4][32];
    private float[][][][] msfe5_de_ddb_prev1 = new float[1][1][4][16];
    private float[][][][] msfe5_de_ddb_prev2 = new float[1][2][4][32];
    private float[][][][] msfe5_de_ddb_prev3 = new float[1][4][4][48];
    private float[][][][] msfe5_de_ddb_prev4 = new float[1][8][4][64];
    private float[][][][] msfe5_de_ddb_prev5 = new float[1][16][4][80];
    private float[][][][] msfe5_de_ddb_prev6 = new float[1][32][4][96];
    private float[][][][] msfe5_de_ddb_prev_out = new float[1][1][4][16];
    // Decoder MSFE4 Dilated Dense Block
    private float[][][][] msfe4_de_ddb_prev_in = new float[1][1][1][32];
    private float[][][][] msfe4_de_ddb_prev1 = new float[1][1][1][16];
    private float[][][][] msfe4_de_ddb_prev2 = new float[1][2][1][32];
    private float[][][][] msfe4_de_ddb_prev3 = new float[1][4][1][48];
    private float[][][][] msfe4_de_ddb_prev4 = new float[1][8][1][64];
    private float[][][][] msfe4_de_ddb_prev5 = new float[1][16][1][80];
    private float[][][][] msfe4_de_ddb_prev6 = new float[1][32][1][96];
    private float[][][][] msfe4_de_ddb_prev_out = new float[1][1][1][16];
    // Decoder MSFE4(2) Dilated Dense Block
    private float[][][][] msfe4_de2_ddb_prev_in = new float[1][1][2][32];
    private float[][][][] msfe4_de2_ddb_prev1 = new float[1][1][2][16];
    private float[][][][] msfe4_de2_ddb_prev2 = new float[1][2][2][32];
    private float[][][][] msfe4_de2_ddb_prev3 = new float[1][4][2][48];
    private float[][][][] msfe4_de2_ddb_prev4 = new float[1][8][2][64];
    private float[][][][] msfe4_de2_ddb_prev5 = new float[1][16][2][80];
    private float[][][][] msfe4_de2_ddb_prev6 = new float[1][32][2][96];
    private float[][][][] msfe4_de2_ddb_prev_out = new float[1][1][2][16];
    // Decoder MSFE4(3) Dilated Dense Block
    private float[][][][] msfe4_de3_ddb_prev_in = new float[1][1][4][32];
    private float[][][][] msfe4_de3_ddb_prev1 = new float[1][1][4][16];
    private float[][][][] msfe4_de3_ddb_prev2 = new float[1][2][4][32];
    private float[][][][] msfe4_de3_ddb_prev3 = new float[1][4][4][48];
    private float[][][][] msfe4_de3_ddb_prev4 = new float[1][8][4][64];
    private float[][][][] msfe4_de3_ddb_prev5 = new float[1][16][4][80];
    private float[][][][] msfe4_de3_ddb_prev6 = new float[1][32][4][96];
    private float[][][][] msfe4_de3_ddb_prev_out = new float[1][1][4][16];
    // Decoder MSFE3 Dilated Dense Block
    private float[][][][] msfe3_de_ddb_prev_in = new float[1][1][1][32];
    private float[][][][] msfe3_de_ddb_prev1 = new float[1][1][1][16];
    private float[][][][] msfe3_de_ddb_prev2 = new float[1][2][1][32];
    private float[][][][] msfe3_de_ddb_prev3 = new float[1][4][1][48];
    private float[][][][] msfe3_de_ddb_prev4 = new float[1][8][1][64];
    private float[][][][] msfe3_de_ddb_prev5 = new float[1][16][1][80];
    private float[][][][] msfe3_de_ddb_prev6 = new float[1][32][1][96];
    private float[][][][] msfe3_de_ddb_prev_out = new float[1][1][1][16];

    private float[][][][] tflite_out = new float[1][1][256][1];


    public float[][][][] runningTFLite(float[][][][] inputData) {
        /** TFLite Input */
        Map<String, Object> inputs = new HashMap<>();

        inputs.put("input", inputData);
        // Encoder MSFE 6 encoder
        inputs.put("msfe6_ee_prev1", msfe6_ee_prev1);
        inputs.put("msfe6_ee_prev2", msfe6_ee_prev2);
        inputs.put("msfe6_ee_prev3", msfe6_ee_prev3);
        inputs.put("msfe6_ee_prev4", msfe6_ee_prev4);
        inputs.put("msfe6_ee_prev5", msfe6_ee_prev5);
        inputs.put("msfe6_ee_prev6", msfe6_ee_prev6);
        // Encoder MSFE 5 encoder
        inputs.put("msfe5_ee_prev1", msfe5_ee_prev1);
        inputs.put("msfe5_ee_prev2", msfe5_ee_prev2);
        inputs.put("msfe5_ee_prev3", msfe5_ee_prev3);
        inputs.put("msfe5_ee_prev4", msfe5_ee_prev4);
        inputs.put("msfe5_ee_prev5", msfe5_ee_prev5);
        // Encoder MSFE 4 encoder
        inputs.put("msfe4_ee_prev1", msfe4_ee_prev1);
        inputs.put("msfe4_ee_prev2", msfe4_ee_prev2);
        inputs.put("msfe4_ee_prev3", msfe4_ee_prev3);
        inputs.put("msfe4_ee_prev4", msfe4_ee_prev4);
        // Encoder MSFE 4(2) encoder
        inputs.put("msfe4_ee2_prev1", msfe4_ee2_prev1);
        inputs.put("msfe4_ee2_prev2", msfe4_ee2_prev2);
        inputs.put("msfe4_ee2_prev3", msfe4_ee2_prev3);
        inputs.put("msfe4_ee2_prev4", msfe4_ee2_prev4);
        // Encoder MSFE 4(3) encoder
        inputs.put("msfe4_ee3_prev1", msfe4_ee3_prev1);
        inputs.put("msfe4_ee3_prev2", msfe4_ee3_prev2);
        inputs.put("msfe4_ee3_prev3", msfe4_ee3_prev3);
        inputs.put("msfe4_ee3_prev4", msfe4_ee3_prev4);
        // Encoder MSFE 3 encoder
        inputs.put("msfe3_ee_prev1", msfe3_ee_prev1);
        inputs.put("msfe3_ee_prev2", msfe3_ee_prev2);
        inputs.put("msfe3_ee_prev3", msfe3_ee_prev3);

        // Encoder MSFE 6 decoder
        inputs.put("msfe6_ed_prev1", msfe6_ed_prev1);
        inputs.put("msfe6_ed_prev2", msfe6_ed_prev2);
        inputs.put("msfe6_ed_prev3", msfe6_ed_prev3);
        inputs.put("msfe6_ed_prev4", msfe6_ed_prev4);
        inputs.put("msfe6_ed_prev5", msfe6_ed_prev5);
        inputs.put("msfe6_ed_prev6", msfe6_ed_prev6);
        // Encoder MSFE 5 decoder
        inputs.put("msfe5_ed_prev1", msfe5_ed_prev1);
        inputs.put("msfe5_ed_prev2", msfe5_ed_prev2);
        inputs.put("msfe5_ed_prev3", msfe5_ed_prev3);
        inputs.put("msfe5_ed_prev4", msfe5_ed_prev4);
        inputs.put("msfe5_ed_prev5", msfe5_ed_prev5);
        // Encoder MSFE 4 decoder
        inputs.put("msfe4_ed_prev1", msfe4_ed_prev1);
        inputs.put("msfe4_ed_prev2", msfe4_ed_prev2);
        inputs.put("msfe4_ed_prev3", msfe4_ed_prev3);
        inputs.put("msfe4_ed_prev4", msfe4_ed_prev4);
        // Encoder MSFE 4(2) decoder
        inputs.put("msfe4_ed2_prev1", msfe4_ed2_prev1);
        inputs.put("msfe4_ed2_prev2", msfe4_ed2_prev2);
        inputs.put("msfe4_ed2_prev3", msfe4_ed2_prev3);
        inputs.put("msfe4_ed2_prev4", msfe4_ed2_prev4);
        // Encoder MSFE 4(3) decoder
        inputs.put("msfe4_ed3_prev1", msfe4_ed3_prev1);
        inputs.put("msfe4_ed3_prev2", msfe4_ed3_prev2);
        inputs.put("msfe4_ed3_prev3", msfe4_ed3_prev3);
        inputs.put("msfe4_ed3_prev4", msfe4_ed3_prev4);
        // Encoder MSFE 3 decoder
        inputs.put("msfe3_ed_prev1", msfe3_ed_prev1);
        inputs.put("msfe3_ed_prev2", msfe3_ed_prev2);
        inputs.put("msfe3_ed_prev3", msfe3_ed_prev3);

        // Decoder MSFE 6 encoder
        inputs.put("msfe6_de_prev1", msfe6_de_prev1);
        inputs.put("msfe6_de_prev2", msfe6_de_prev2);
        inputs.put("msfe6_de_prev3", msfe6_de_prev3);
        inputs.put("msfe6_de_prev4", msfe6_de_prev4);
        inputs.put("msfe6_de_prev5", msfe6_de_prev5);
        inputs.put("msfe6_de_prev6", msfe6_de_prev6);
        // Decoder MSFE 5 encoder
        inputs.put("msfe5_de_prev1", msfe5_de_prev1);
        inputs.put("msfe5_de_prev2", msfe5_de_prev2);
        inputs.put("msfe5_de_prev3", msfe5_de_prev3);
        inputs.put("msfe5_de_prev4", msfe5_de_prev4);
        inputs.put("msfe5_de_prev5", msfe5_de_prev5);
        // Decoder MSFE 4 encoder
        inputs.put("msfe4_de_prev1", msfe4_de_prev1);
        inputs.put("msfe4_de_prev2", msfe4_de_prev2);
        inputs.put("msfe4_de_prev3", msfe4_de_prev3);
        inputs.put("msfe4_de_prev4", msfe4_de_prev4);
        // Decoder MSFE 4(2) encoder
        inputs.put("msfe4_de2_prev1", msfe4_de2_prev1);
        inputs.put("msfe4_de2_prev2", msfe4_de2_prev2);
        inputs.put("msfe4_de2_prev3", msfe4_de2_prev3);
        inputs.put("msfe4_de2_prev4", msfe4_de2_prev4);
        // Decoder MSFE 4(3) encoder
        inputs.put("msfe4_de3_prev1", msfe4_de3_prev1);
        inputs.put("msfe4_de3_prev2", msfe4_de3_prev2);
        inputs.put("msfe4_de3_prev3", msfe4_de3_prev3);
        inputs.put("msfe4_de3_prev4", msfe4_de3_prev4);
        // Decoder MSFE 3 encoder
        inputs.put("msfe3_de_prev1", msfe3_de_prev1);
        inputs.put("msfe3_de_prev2", msfe3_de_prev2);
        inputs.put("msfe3_de_prev3", msfe3_de_prev3);

        // Decoder MSFE 6 decoder
        inputs.put("msfe6_dd_prev1", msfe6_dd_prev1);
        inputs.put("msfe6_dd_prev2", msfe6_dd_prev2);
        inputs.put("msfe6_dd_prev3", msfe6_dd_prev3);
        inputs.put("msfe6_dd_prev4", msfe6_dd_prev4);
        inputs.put("msfe6_dd_prev5", msfe6_dd_prev5);
        inputs.put("msfe6_dd_prev6", msfe6_dd_prev6);
        // Decoder MSFE 5 decoder
        inputs.put("msfe5_dd_prev1", msfe5_dd_prev1);
        inputs.put("msfe5_dd_prev2", msfe5_dd_prev2);
        inputs.put("msfe5_dd_prev3", msfe5_dd_prev3);
        inputs.put("msfe5_dd_prev4", msfe5_dd_prev4);
        inputs.put("msfe5_dd_prev5", msfe5_dd_prev5);
        // Decoder MSFE 4 decoder
        inputs.put("msfe4_dd_prev1", msfe4_dd_prev1);
        inputs.put("msfe4_dd_prev2", msfe4_dd_prev2);
        inputs.put("msfe4_dd_prev3", msfe4_dd_prev3);
        inputs.put("msfe4_dd_prev4", msfe4_dd_prev4);
        // Decoder MSFE 4(2) decoder
        inputs.put("msfe4_dd2_prev1", msfe4_dd2_prev1);
        inputs.put("msfe4_dd2_prev2", msfe4_dd2_prev2);
        inputs.put("msfe4_dd2_prev3", msfe4_dd2_prev3);
        inputs.put("msfe4_dd2_prev4", msfe4_dd2_prev4);
        // Decoder MSFE 4(3) decoder
        inputs.put("msfe4_dd3_prev1", msfe4_dd3_prev1);
        inputs.put("msfe4_dd3_prev2", msfe4_dd3_prev2);
        inputs.put("msfe4_dd3_prev3", msfe4_dd3_prev3);
        inputs.put("msfe4_dd3_prev4", msfe4_dd3_prev4);
        // Decoder MSFE 3 decoder
        inputs.put("msfe3_dd_prev1", msfe3_dd_prev1);
        inputs.put("msfe3_dd_prev2", msfe3_dd_prev2);
        inputs.put("msfe3_dd_prev3", msfe3_dd_prev3);

        // Encoder MSFE6 Dilated Dense Block
        inputs.put("msfe6_en_ddb_prev_in", msfe6_en_ddb_prev_in);
        inputs.put("msfe6_en_ddb_prev1", msfe6_en_ddb_prev1);
        inputs.put("msfe6_en_ddb_prev2", msfe6_en_ddb_prev2);
        inputs.put("msfe6_en_ddb_prev3", msfe6_en_ddb_prev3);
        inputs.put("msfe6_en_ddb_prev4", msfe6_en_ddb_prev4);
        inputs.put("msfe6_en_ddb_prev5", msfe6_en_ddb_prev5);
        inputs.put("msfe6_en_ddb_prev6", msfe6_en_ddb_prev6);
        inputs.put("msfe6_en_ddb_prev_out", msfe6_en_ddb_prev_out);
        // Encoder MSFE5 Dilated Dense Block
        inputs.put("msfe5_en_ddb_prev_in", msfe5_en_ddb_prev_in);
        inputs.put("msfe5_en_ddb_prev1", msfe5_en_ddb_prev1);
        inputs.put("msfe5_en_ddb_prev2", msfe5_en_ddb_prev2);
        inputs.put("msfe5_en_ddb_prev3", msfe5_en_ddb_prev3);
        inputs.put("msfe5_en_ddb_prev4", msfe5_en_ddb_prev4);
        inputs.put("msfe5_en_ddb_prev5", msfe5_en_ddb_prev5);
        inputs.put("msfe5_en_ddb_prev6", msfe5_en_ddb_prev6);
        inputs.put("msfe5_en_ddb_prev_out", msfe5_en_ddb_prev_out);
        // Encoder MSFE4 Dilated Dense Block
        inputs.put("msfe4_en_ddb_prev_in", msfe4_en_ddb_prev_in);
        inputs.put("msfe4_en_ddb_prev1", msfe4_en_ddb_prev1);
        inputs.put("msfe4_en_ddb_prev2", msfe4_en_ddb_prev2);
        inputs.put("msfe4_en_ddb_prev3", msfe4_en_ddb_prev3);
        inputs.put("msfe4_en_ddb_prev4", msfe4_en_ddb_prev4);
        inputs.put("msfe4_en_ddb_prev5", msfe4_en_ddb_prev5);
        inputs.put("msfe4_en_ddb_prev6", msfe4_en_ddb_prev6);
        inputs.put("msfe4_en_ddb_prev_out", msfe4_en_ddb_prev_out);
        // Encoder MSFE4(2) Dilated Dense Block
        inputs.put("msfe4_en2_ddb_prev_in", msfe4_en2_ddb_prev_in);
        inputs.put("msfe4_en2_ddb_prev1", msfe4_en2_ddb_prev1);
        inputs.put("msfe4_en2_ddb_prev2", msfe4_en2_ddb_prev2);
        inputs.put("msfe4_en2_ddb_prev3", msfe4_en2_ddb_prev3);
        inputs.put("msfe4_en2_ddb_prev4", msfe4_en2_ddb_prev4);
        inputs.put("msfe4_en2_ddb_prev5", msfe4_en2_ddb_prev5);
        inputs.put("msfe4_en2_ddb_prev6", msfe4_en2_ddb_prev6);
        inputs.put("msfe4_en2_ddb_prev_out", msfe4_en2_ddb_prev_out);
        // Encoder MSFE4(3) Dilated Dense Block
        inputs.put("msfe4_en3_ddb_prev_in", msfe4_en3_ddb_prev_in);
        inputs.put("msfe4_en3_ddb_prev1", msfe4_en3_ddb_prev1);
        inputs.put("msfe4_en3_ddb_prev2", msfe4_en3_ddb_prev2);
        inputs.put("msfe4_en3_ddb_prev3", msfe4_en3_ddb_prev3);
        inputs.put("msfe4_en3_ddb_prev4", msfe4_en3_ddb_prev4);
        inputs.put("msfe4_en3_ddb_prev5", msfe4_en3_ddb_prev5);
        inputs.put("msfe4_en3_ddb_prev6", msfe4_en3_ddb_prev6);
        inputs.put("msfe4_en3_ddb_prev_out", msfe4_en3_ddb_prev_out);
        // Encoder MSFE3 Dilated Dense Block
        inputs.put("msfe3_en_ddb_prev_in", msfe3_en_ddb_prev_in);
        inputs.put("msfe3_en_ddb_prev1", msfe3_en_ddb_prev1);
        inputs.put("msfe3_en_ddb_prev2", msfe3_en_ddb_prev2);
        inputs.put("msfe3_en_ddb_prev3", msfe3_en_ddb_prev3);
        inputs.put("msfe3_en_ddb_prev4", msfe3_en_ddb_prev4);
        inputs.put("msfe3_en_ddb_prev5", msfe3_en_ddb_prev5);
        inputs.put("msfe3_en_ddb_prev6", msfe3_en_ddb_prev6);
        inputs.put("msfe3_en_ddb_prev_out", msfe3_en_ddb_prev_out);

        // Dilated Dense Block
        inputs.put("ddb_prev_in", ddb_prev_in);
        inputs.put("ddb_prev1", ddb_prev1);
        inputs.put("ddb_prev2", ddb_prev2);
        inputs.put("ddb_prev3", ddb_prev3);
        inputs.put("ddb_prev4", ddb_prev4);
        inputs.put("ddb_prev5", ddb_prev5);
        inputs.put("ddb_prev6", ddb_prev6);
        inputs.put("ddb_prev_out", ddb_prev_out);

        // Decoder MSFE6 Dilated Dense Block
        inputs.put("msfe6_de_ddb_prev_in", msfe6_de_ddb_prev_in);
        inputs.put("msfe6_de_ddb_prev1", msfe6_de_ddb_prev1);
        inputs.put("msfe6_de_ddb_prev2", msfe6_de_ddb_prev2);
        inputs.put("msfe6_de_ddb_prev3", msfe6_de_ddb_prev3);
        inputs.put("msfe6_de_ddb_prev4", msfe6_de_ddb_prev4);
        inputs.put("msfe6_de_ddb_prev5", msfe6_de_ddb_prev5);
        inputs.put("msfe6_de_ddb_prev6", msfe6_de_ddb_prev6);
        inputs.put("msfe6_de_ddb_prev_out", msfe6_de_ddb_prev_out);
        // Decoder MSFE5 Dilated Dense Block
        inputs.put("msfe5_de_ddb_prev_in", msfe5_de_ddb_prev_in);
        inputs.put("msfe5_de_ddb_prev1", msfe5_de_ddb_prev1);
        inputs.put("msfe5_de_ddb_prev2", msfe5_de_ddb_prev2);
        inputs.put("msfe5_de_ddb_prev3", msfe5_de_ddb_prev3);
        inputs.put("msfe5_de_ddb_prev4", msfe5_de_ddb_prev4);
        inputs.put("msfe5_de_ddb_prev5", msfe5_de_ddb_prev5);
        inputs.put("msfe5_de_ddb_prev6", msfe5_de_ddb_prev6);
        inputs.put("msfe5_de_ddb_prev_out", msfe5_de_ddb_prev_out);
        // Decoder MSFE4 Dilated Dense Block
        inputs.put("msfe4_de_ddb_prev_in", msfe4_de_ddb_prev_in);
        inputs.put("msfe4_de_ddb_prev1", msfe4_de_ddb_prev1);
        inputs.put("msfe4_de_ddb_prev2", msfe4_de_ddb_prev2);
        inputs.put("msfe4_de_ddb_prev3", msfe4_de_ddb_prev3);
        inputs.put("msfe4_de_ddb_prev4", msfe4_de_ddb_prev4);
        inputs.put("msfe4_de_ddb_prev5", msfe4_de_ddb_prev5);
        inputs.put("msfe4_de_ddb_prev6", msfe4_de_ddb_prev6);
        inputs.put("msfe4_de_ddb_prev_out", msfe4_de_ddb_prev_out);
        // Decoder MSFE4(2) Dilated Dense Block
        inputs.put("msfe4_de2_ddb_prev_in", msfe4_de2_ddb_prev_in);
        inputs.put("msfe4_de2_ddb_prev1", msfe4_de2_ddb_prev1);
        inputs.put("msfe4_de2_ddb_prev2", msfe4_de2_ddb_prev2);
        inputs.put("msfe4_de2_ddb_prev3", msfe4_de2_ddb_prev3);
        inputs.put("msfe4_de2_ddb_prev4", msfe4_de2_ddb_prev4);
        inputs.put("msfe4_de2_ddb_prev5", msfe4_de2_ddb_prev5);
        inputs.put("msfe4_de2_ddb_prev6", msfe4_de2_ddb_prev6);
        inputs.put("msfe4_de2_ddb_prev_out", msfe4_de2_ddb_prev_out);
        // Decoder MSFE4(3) Dilated Dense Block
        inputs.put("msfe4_de3_ddb_prev_in", msfe4_de3_ddb_prev_in);
        inputs.put("msfe4_de3_ddb_prev1", msfe4_de3_ddb_prev1);
        inputs.put("msfe4_de3_ddb_prev2", msfe4_de3_ddb_prev2);
        inputs.put("msfe4_de3_ddb_prev3", msfe4_de3_ddb_prev3);
        inputs.put("msfe4_de3_ddb_prev4", msfe4_de3_ddb_prev4);
        inputs.put("msfe4_de3_ddb_prev5", msfe4_de3_ddb_prev5);
        inputs.put("msfe4_de3_ddb_prev6", msfe4_de3_ddb_prev6);
        inputs.put("msfe4_de3_ddb_prev_out", msfe4_de3_ddb_prev_out);
        // Decoder MSFE3 Dilated Dense Block
        inputs.put("msfe3_de_ddb_prev_in", msfe3_de_ddb_prev_in);
        inputs.put("msfe3_de_ddb_prev1", msfe3_de_ddb_prev1);
        inputs.put("msfe3_de_ddb_prev2", msfe3_de_ddb_prev2);
        inputs.put("msfe3_de_ddb_prev3", msfe3_de_ddb_prev3);
        inputs.put("msfe3_de_ddb_prev4", msfe3_de_ddb_prev4);
        inputs.put("msfe3_de_ddb_prev5", msfe3_de_ddb_prev5);
        inputs.put("msfe3_de_ddb_prev6", msfe3_de_ddb_prev6);
        inputs.put("msfe3_de_ddb_prev_out", msfe3_de_ddb_prev_out);

        /** TFLite Output */
        Map<String, Object> outputs = new HashMap<>();

        // Encoder MSFE6 encoder
        outputs.put("msfe6_ee_cur1", msfe6_ee_prev1);
        outputs.put("msfe6_ee_cur2", msfe6_ee_prev2);
        outputs.put("msfe6_ee_cur3", msfe6_ee_prev3);
        outputs.put("msfe6_ee_cur4", msfe6_ee_prev4);
        outputs.put("msfe6_ee_cur5", msfe6_ee_prev5);
        outputs.put("msfe6_ee_cur6", msfe6_ee_prev6);
        // Encoder MSFE5 encoder
        outputs.put("msfe5_ee_cur1", msfe5_ee_prev1);
        outputs.put("msfe5_ee_cur2", msfe5_ee_prev2);
        outputs.put("msfe5_ee_cur3", msfe5_ee_prev3);
        outputs.put("msfe5_ee_cur4", msfe5_ee_prev4);
        outputs.put("msfe5_ee_cur5", msfe5_ee_prev5);
        // Encoder MSFE4 encoder
        outputs.put("msfe4_ee_cur1", msfe4_ee_prev1);
        outputs.put("msfe4_ee_cur2", msfe4_ee_prev2);
        outputs.put("msfe4_ee_cur3", msfe4_ee_prev3);
        outputs.put("msfe4_ee_cur4", msfe4_ee_prev4);
        // Encoder MSFE4(2) encoder
        outputs.put("msfe4_ee2_cur1", msfe4_ee2_prev1);
        outputs.put("msfe4_ee2_cur2", msfe4_ee2_prev2);
        outputs.put("msfe4_ee2_cur3", msfe4_ee2_prev3);
        outputs.put("msfe4_ee2_cur4", msfe4_ee2_prev4);
        // Encoder MSFE4(3) encoder
        outputs.put("msfe4_ee3_cur1", msfe4_ee3_prev1);
        outputs.put("msfe4_ee3_cur2", msfe4_ee3_prev2);
        outputs.put("msfe4_ee3_cur3", msfe4_ee3_prev3);
        outputs.put("msfe4_ee3_cur4", msfe4_ee3_prev4);
        // Encoder MSFE3 encoder
        outputs.put("msfe3_ee_cur1", msfe3_ee_prev1);
        outputs.put("msfe3_ee_cur2", msfe3_ee_prev2);
        outputs.put("msfe3_ee_cur3", msfe3_ee_prev3);

        // Encoder MSFE6 decoder
        outputs.put("msfe6_ed_cur1", msfe6_ed_prev1);
        outputs.put("msfe6_ed_cur2", msfe6_ed_prev2);
        outputs.put("msfe6_ed_cur3", msfe6_ed_prev3);
        outputs.put("msfe6_ed_cur4", msfe6_ed_prev4);
        outputs.put("msfe6_ed_cur5", msfe6_ed_prev5);
        outputs.put("msfe6_ed_cur6", msfe6_ed_prev6);
        // Encoder MSFE5 decoder
        outputs.put("msfe5_ed_cur1", msfe5_ed_prev1);
        outputs.put("msfe5_ed_cur2", msfe5_ed_prev2);
        outputs.put("msfe5_ed_cur3", msfe5_ed_prev3);
        outputs.put("msfe5_ed_cur4", msfe5_ed_prev4);
        outputs.put("msfe5_ed_cur5", msfe5_ed_prev5);
        // Encoder MSFE4 decoder
        outputs.put("msfe4_ed_cur1", msfe4_ed_prev1);
        outputs.put("msfe4_ed_cur2", msfe4_ed_prev2);
        outputs.put("msfe4_ed_cur3", msfe4_ed_prev3);
        outputs.put("msfe4_ed_cur4", msfe4_ed_prev4);
        // Encoder MSFE4(2) decoder
        outputs.put("msfe4_ed2_cur1", msfe4_ed2_prev1);
        outputs.put("msfe4_ed2_cur2", msfe4_ed2_prev2);
        outputs.put("msfe4_ed2_cur3", msfe4_ed2_prev3);
        outputs.put("msfe4_ed2_cur4", msfe4_ed2_prev4);
        // Encoder MSFE4(3) decoder
        outputs.put("msfe4_ed3_cur1", msfe4_ed3_prev1);
        outputs.put("msfe4_ed3_cur2", msfe4_ed3_prev2);
        outputs.put("msfe4_ed3_cur3", msfe4_ed3_prev3);
        outputs.put("msfe4_ed3_cur4", msfe4_ed3_prev4);
        // Encoder MSFE3 decoder
        outputs.put("msfe3_ed_cur1", msfe3_ed_prev1);
        outputs.put("msfe3_ed_cur2", msfe3_ed_prev2);
        outputs.put("msfe3_ed_cur3", msfe3_ed_prev3);

        // Decoder MSFE6 encoder
        outputs.put("msfe6_de_cur1", msfe6_de_prev1);
        outputs.put("msfe6_de_cur2", msfe6_de_prev2);
        outputs.put("msfe6_de_cur3", msfe6_de_prev3);
        outputs.put("msfe6_de_cur4", msfe6_de_prev4);
        outputs.put("msfe6_de_cur5", msfe6_de_prev5);
        outputs.put("msfe6_de_cur6", msfe6_de_prev6);
        // Decoder MSFE5 encoder
        outputs.put("msfe5_de_cur1", msfe5_de_prev1);
        outputs.put("msfe5_de_cur2", msfe5_de_prev2);
        outputs.put("msfe5_de_cur3", msfe5_de_prev3);
        outputs.put("msfe5_de_cur4", msfe5_de_prev4);
        outputs.put("msfe5_de_cur5", msfe5_de_prev5);
        // Decoder MSFE4 encoder
        outputs.put("msfe4_de_cur1", msfe4_de_prev1);
        outputs.put("msfe4_de_cur2", msfe4_de_prev2);
        outputs.put("msfe4_de_cur3", msfe4_de_prev3);
        outputs.put("msfe4_de_cur4", msfe4_de_prev4);
        // Decoder MSFE4(2) encoder
        outputs.put("msfe4_de2_cur1", msfe4_de2_prev1);
        outputs.put("msfe4_de2_cur2", msfe4_de2_prev2);
        outputs.put("msfe4_de2_cur3", msfe4_de2_prev3);
        outputs.put("msfe4_de2_cur4", msfe4_de2_prev4);
        // Decoder MSFE4(3) encoder
        outputs.put("msfe4_de3_cur1", msfe4_de3_prev1);
        outputs.put("msfe4_de3_cur2", msfe4_de3_prev2);
        outputs.put("msfe4_de3_cur3", msfe4_de3_prev3);
        outputs.put("msfe4_de3_cur4", msfe4_de3_prev4);
        // Decoder MSFE3 encoder
        outputs.put("msfe3_de_cur1", msfe3_de_prev1);
        outputs.put("msfe3_de_cur2", msfe3_de_prev2);
        outputs.put("msfe3_de_cur3", msfe3_de_prev3);

        // Decoder MSFE6 decoder
        outputs.put("msfe6_dd_cur1", msfe6_dd_prev1);
        outputs.put("msfe6_dd_cur2", msfe6_dd_prev2);
        outputs.put("msfe6_dd_cur3", msfe6_dd_prev3);
        outputs.put("msfe6_dd_cur4", msfe6_dd_prev4);
        outputs.put("msfe6_dd_cur5", msfe6_dd_prev5);
        outputs.put("msfe6_dd_cur6", msfe6_dd_prev6);
        // Decoder MSFE5 decoder
        outputs.put("msfe5_dd_cur1", msfe5_dd_prev1);
        outputs.put("msfe5_dd_cur2", msfe5_dd_prev2);
        outputs.put("msfe5_dd_cur3", msfe5_dd_prev3);
        outputs.put("msfe5_dd_cur4", msfe5_dd_prev4);
        outputs.put("msfe5_dd_cur5", msfe5_dd_prev5);
        // Decoder MSFE4 decoder
        outputs.put("msfe4_dd_cur1", msfe4_dd_prev1);
        outputs.put("msfe4_dd_cur2", msfe4_dd_prev2);
        outputs.put("msfe4_dd_cur3", msfe4_dd_prev3);
        outputs.put("msfe4_dd_cur4", msfe4_dd_prev4);
        // Decoder MSFE4(2) decoder
        outputs.put("msfe4_dd2_cur1", msfe4_dd2_prev1);
        outputs.put("msfe4_dd2_cur2", msfe4_dd2_prev2);
        outputs.put("msfe4_dd2_cur3", msfe4_dd2_prev3);
        outputs.put("msfe4_dd2_cur4", msfe4_dd2_prev4);
        // Decoder MSFE4(3) decoder
        outputs.put("msfe4_dd3_cur1", msfe4_dd3_prev1);
        outputs.put("msfe4_dd3_cur2", msfe4_dd3_prev2);
        outputs.put("msfe4_dd3_cur3", msfe4_dd3_prev3);
        outputs.put("msfe4_dd3_cur4", msfe4_dd3_prev4);
        // Decoder MSFE3 decoder
        outputs.put("msfe3_dd_cur1", msfe3_dd_prev1);
        outputs.put("msfe3_dd_cur2", msfe3_dd_prev2);
        outputs.put("msfe3_dd_cur3", msfe3_dd_prev3);

        // Encoder MSFE6 Dilated Dense Block
        outputs.put("msfe6_en_ddb_cur_in", msfe6_en_ddb_prev_in);
        outputs.put("msfe6_en_ddb_cur1", msfe6_en_ddb_prev1);
        outputs.put("msfe6_en_ddb_cur2", msfe6_en_ddb_prev2);
        outputs.put("msfe6_en_ddb_cur3", msfe6_en_ddb_prev3);
        outputs.put("msfe6_en_ddb_cur4", msfe6_en_ddb_prev4);
        outputs.put("msfe6_en_ddb_cur5", msfe6_en_ddb_prev5);
        outputs.put("msfe6_en_ddb_cur6", msfe6_en_ddb_prev6);
        outputs.put("msfe6_en_ddb_cur_out", msfe6_en_ddb_prev_out);
        // Encoder MSFE5 Dilated Dense Block
        outputs.put("msfe5_en_ddb_cur_in", msfe5_en_ddb_prev_in);
        outputs.put("msfe5_en_ddb_cur1", msfe5_en_ddb_prev1);
        outputs.put("msfe5_en_ddb_cur2", msfe5_en_ddb_prev2);
        outputs.put("msfe5_en_ddb_cur3", msfe5_en_ddb_prev3);
        outputs.put("msfe5_en_ddb_cur4", msfe5_en_ddb_prev4);
        outputs.put("msfe5_en_ddb_cur5", msfe5_en_ddb_prev5);
        outputs.put("msfe5_en_ddb_cur6", msfe5_en_ddb_prev6);
        outputs.put("msfe5_en_ddb_cur_out", msfe5_en_ddb_prev_out);
        // Encoder MSFE4 Dilated Dense Block
        outputs.put("msfe4_en_ddb_cur_in", msfe4_en_ddb_prev_in);
        outputs.put("msfe4_en_ddb_cur1", msfe4_en_ddb_prev1);
        outputs.put("msfe4_en_ddb_cur2", msfe4_en_ddb_prev2);
        outputs.put("msfe4_en_ddb_cur3", msfe4_en_ddb_prev3);
        outputs.put("msfe4_en_ddb_cur4", msfe4_en_ddb_prev4);
        outputs.put("msfe4_en_ddb_cur5", msfe4_en_ddb_prev5);
        outputs.put("msfe4_en_ddb_cur6", msfe4_en_ddb_prev6);
        outputs.put("msfe4_en_ddb_cur_out", msfe4_en_ddb_prev_out);
        // Encoder MSFE4(2) Dilated Dense Block
        outputs.put("msfe4_en2_ddb_cur_in", msfe4_en2_ddb_prev_in);
        outputs.put("msfe4_en2_ddb_cur1", msfe4_en2_ddb_prev1);
        outputs.put("msfe4_en2_ddb_cur2", msfe4_en2_ddb_prev2);
        outputs.put("msfe4_en2_ddb_cur3", msfe4_en2_ddb_prev3);
        outputs.put("msfe4_en2_ddb_cur4", msfe4_en2_ddb_prev4);
        outputs.put("msfe4_en2_ddb_cur5", msfe4_en2_ddb_prev5);
        outputs.put("msfe4_en2_ddb_cur6", msfe4_en2_ddb_prev6);
        outputs.put("msfe4_en2_ddb_cur_out", msfe4_en2_ddb_prev_out);
        // Encoder MSFE4(3) Dilated Dense Block
        outputs.put("msfe4_en3_ddb_cur_in", msfe4_en3_ddb_prev_in);
        outputs.put("msfe4_en3_ddb_cur1", msfe4_en3_ddb_prev1);
        outputs.put("msfe4_en3_ddb_cur2", msfe4_en3_ddb_prev2);
        outputs.put("msfe4_en3_ddb_cur3", msfe4_en3_ddb_prev3);
        outputs.put("msfe4_en3_ddb_cur4", msfe4_en3_ddb_prev4);
        outputs.put("msfe4_en3_ddb_cur5", msfe4_en3_ddb_prev5);
        outputs.put("msfe4_en3_ddb_cur6", msfe4_en3_ddb_prev6);
        outputs.put("msfe4_en3_ddb_cur_out", msfe4_en3_ddb_prev_out);
        // Encoder MSFE3 Dilated Dense Block
        outputs.put("msfe3_en_ddb_cur_in", msfe3_en_ddb_prev_in);
        outputs.put("msfe3_en_ddb_cur1", msfe3_en_ddb_prev1);
        outputs.put("msfe3_en_ddb_cur2", msfe3_en_ddb_prev2);
        outputs.put("msfe3_en_ddb_cur3", msfe3_en_ddb_prev3);
        outputs.put("msfe3_en_ddb_cur4", msfe3_en_ddb_prev4);
        outputs.put("msfe3_en_ddb_cur5", msfe3_en_ddb_prev5);
        outputs.put("msfe3_en_ddb_cur6", msfe3_en_ddb_prev6);
        outputs.put("msfe3_en_ddb_cur_out", msfe3_en_ddb_prev_out);

        // Dilated Dense Block
        outputs.put("ddb_cur_in", ddb_prev_in);
        outputs.put("ddb_cur1", ddb_prev1);
        outputs.put("ddb_cur2", ddb_prev2);
        outputs.put("ddb_cur3", ddb_prev3);
        outputs.put("ddb_cur4", ddb_prev4);
        outputs.put("ddb_cur5", ddb_prev5);
        outputs.put("ddb_cur6", ddb_prev6);
        outputs.put("ddb_cur_out", ddb_prev_out);

        // Decoder MSFE6 Dilated Dense Block
        outputs.put("msfe6_de_ddb_cur_in", msfe6_de_ddb_prev_in);
        outputs.put("msfe6_de_ddb_cur1", msfe6_de_ddb_prev1);
        outputs.put("msfe6_de_ddb_cur2", msfe6_de_ddb_prev2);
        outputs.put("msfe6_de_ddb_cur3", msfe6_de_ddb_prev3);
        outputs.put("msfe6_de_ddb_cur4", msfe6_de_ddb_prev4);
        outputs.put("msfe6_de_ddb_cur5", msfe6_de_ddb_prev5);
        outputs.put("msfe6_de_ddb_cur6", msfe6_de_ddb_prev6);
        outputs.put("msfe6_de_ddb_cur_out", msfe6_de_ddb_prev_out);
        // Decoder MSFE5 Dilated Dense Block
        outputs.put("msfe5_de_ddb_cur_in", msfe5_de_ddb_prev_in);
        outputs.put("msfe5_de_ddb_cur1", msfe5_de_ddb_prev1);
        outputs.put("msfe5_de_ddb_cur2", msfe5_de_ddb_prev2);
        outputs.put("msfe5_de_ddb_cur3", msfe5_de_ddb_prev3);
        outputs.put("msfe5_de_ddb_cur4", msfe5_de_ddb_prev4);
        outputs.put("msfe5_de_ddb_cur5", msfe5_de_ddb_prev5);
        outputs.put("msfe5_de_ddb_cur6", msfe5_de_ddb_prev6);
        outputs.put("msfe5_de_ddb_cur_out", msfe5_de_ddb_prev_out);
        // Decoder MSFE4 Dilated Dense Block
        outputs.put("msfe4_de_ddb_cur_in", msfe4_de_ddb_prev_in);
        outputs.put("msfe4_de_ddb_cur1", msfe4_de_ddb_prev1);
        outputs.put("msfe4_de_ddb_cur2", msfe4_de_ddb_prev2);
        outputs.put("msfe4_de_ddb_cur3", msfe4_de_ddb_prev3);
        outputs.put("msfe4_de_ddb_cur4", msfe4_de_ddb_prev4);
        outputs.put("msfe4_de_ddb_cur5", msfe4_de_ddb_prev5);
        outputs.put("msfe4_de_ddb_cur6", msfe4_de_ddb_prev6);
        outputs.put("msfe4_de_ddb_cur_out", msfe4_de_ddb_prev_out);
        // Decoder MSFE4(2) Dilated Dense Block
        outputs.put("msfe4_de2_ddb_cur_in", msfe4_de2_ddb_prev_in);
        outputs.put("msfe4_de2_ddb_cur1", msfe4_de2_ddb_prev1);
        outputs.put("msfe4_de2_ddb_cur2", msfe4_de2_ddb_prev2);
        outputs.put("msfe4_de2_ddb_cur3", msfe4_de2_ddb_prev3);
        outputs.put("msfe4_de2_ddb_cur4", msfe4_de2_ddb_prev4);
        outputs.put("msfe4_de2_ddb_cur5", msfe4_de2_ddb_prev5);
        outputs.put("msfe4_de2_ddb_cur6", msfe4_de2_ddb_prev6);
        outputs.put("msfe4_de2_ddb_cur_out", msfe4_de2_ddb_prev_out);
        // Decoder MSFE4(3) Dilated Dense Block
        outputs.put("msfe4_de3_ddb_cur_in", msfe4_de3_ddb_prev_in);
        outputs.put("msfe4_de3_ddb_cur1", msfe4_de3_ddb_prev1);
        outputs.put("msfe4_de3_ddb_cur2", msfe4_de3_ddb_prev2);
        outputs.put("msfe4_de3_ddb_cur3", msfe4_de3_ddb_prev3);
        outputs.put("msfe4_de3_ddb_cur4", msfe4_de3_ddb_prev4);
        outputs.put("msfe4_de3_ddb_cur5", msfe4_de3_ddb_prev5);
        outputs.put("msfe4_de3_ddb_cur6", msfe4_de3_ddb_prev6);
        outputs.put("msfe4_de3_ddb_cur_out", msfe4_de3_ddb_prev_out);
        // Decoder MSFE3 Dilated Dense Block
        outputs.put("msfe3_de_ddb_cur_in", msfe3_de_ddb_prev_in);
        outputs.put("msfe3_de_ddb_cur1", msfe3_de_ddb_prev1);
        outputs.put("msfe3_de_ddb_cur2", msfe3_de_ddb_prev2);
        outputs.put("msfe3_de_ddb_cur3", msfe3_de_ddb_prev3);
        outputs.put("msfe3_de_ddb_cur4", msfe3_de_ddb_prev4);
        outputs.put("msfe3_de_ddb_cur5", msfe3_de_ddb_prev5);
        outputs.put("msfe3_de_ddb_cur6", msfe3_de_ddb_prev6);
        outputs.put("msfe3_de_ddb_cur_out", msfe3_de_ddb_prev_out);

        outputs.put("model_out", tflite_out);
        tflite.runSignature(inputs, outputs, "nutls");

        return tflite_out;
    }

    public double[] audioSE(double[] doubleData) {
        // Shift left and Input audio
        this.in_buffer = shiftLeft(this.in_buffer, stride);
        this.in_buffer = audioToBuffer(this.in_buffer, doubleData);

        double[] win_buffer = hannWindow(this.in_buffer);

        // FFT
        Complex[] transformedBuffer = fft.transform(win_buffer, TransformType.FORWARD);
        realTranformedBuffer = getComplexSlice(transformedBuffer, 0, 257);

        // get magnitude/phase
        for (int i = 0; i < this.fft_bins; i++) {
            double real = realTranformedBuffer[i].getReal();
            double imag = realTranformedBuffer[i].getImaginary();
            mags[i] = (float) Math.sqrt(real * real + imag * imag);
            phase[i] = (float) Math.atan2(imag, real);
        } // [257]

        // Reshape [257] -> [1,1,256,1]
        float[][][][] remags = reshapeBeforeEncoder(mags);

        /** TFLite */
        float[][][][] tflite_out = runningTFLite(remags);

        // Reshape and Padding [1,1,256,1] -> [1,1,257]
        float[][][] out_mask = reshapeAndPadding(tflite_out);

        // Reshape [1,1,257] -> [257]
        float[] maskOneDim = reshapeThreeToOne(out_mask);

//        Complex[] outComplex = tfMasking(mags, phase, maskOneDim); // TF-Masking
        Complex[] outComplex = spectralMapping(maskOneDim, phase); // Spectral-Mapping

        // IFFT
        Complex[] fullComplex = reconstructComplex(outComplex);
        Complex[] detransBuffer = ifft.transform(fullComplex, TransformType.INVERSE);

        // Get real
        for (int i = 0; i < detransBuffer.length; i++) {
            estimated_block[i] = detransBuffer[i].getReal();
        }

        // Windowing
        double[] windowed_samples = inverseWindow(estimated_block);

        // Out-buffer : Shift Left and Input zero
        this.out_buffer = shiftLeft(this.out_buffer, stride);

        // Out-buffer : Overlap-add
        for (int i = 0; i < this.out_buffer.length; i++) {
            this.out_buffer[i] += estimated_block[i];
        }

        // Slicing to get out_audio
        double[] se_out = new double[stride];
        System.arraycopy(this.out_buffer, 0, se_out, 0, stride);

        for (int i = 0; i < se_out.length; i++) {
            se_out[i] *= half;
        }

        return se_out;
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
        Complex[] slicedArray = Arrays.copyOfRange(array, startIndex, endIndex);

        //returns the slice of an array
        return slicedArray;
    }


    private Complex[] reconstructComplex(Complex[] half) { // 257
        int halfLen = half.length;
        Complex[] fullComplex = new Complex[(half.length - 1) * 2]; // 512
        Complex[] conjugateComplex = new Complex[half.length - 1]; // 256

        System.arraycopy(half, 0, fullComplex, 0, halfLen);

        for (int i = 1; i < half.length - 1; i++) { // 1~255
            conjugateComplex[i] = half[i].conjugate();
            fullComplex[fullComplex.length - i] = conjugateComplex[i];
        }
        return fullComplex;
    }

    // Reshape and Padding [1,1,256,1] -> [1,1,257]
    private float[][][] reshapeAndPadding(float[][][][] input) {
        float[][][] matrix = new float[1][1][257];
        matrix[0][0][0] = 0;
        for (int i = 0; i < 256; i++) {
            matrix[0][0][i + 1] = input[0][0][i][0];
        }
        return matrix;
    }

    private Complex[] tfMasking(float[] mag, float[] phase, float[] mask) {
        float[] enhancedMag = new float[mag.length];
        double[] real = new double[mag.length];
        double[] imag = new double[mag.length];
        Complex[] complexArray = new Complex[real.length];

        for (int i = 0; i < enhancedMag.length; i++) {
            real[i] = mag[i] * mask[i] * (float) Math.cos(phase[i]);
            imag[i] = mag[i] * mask[i] * (float) Math.sin(phase[i]);

            complexArray[i] = new Complex(real[i], imag[i]);
        }

        return complexArray;
    }

    private Complex[] spectralMapping(float[] mask, float[] phase) {
        float[] enhancedMag = new float[mask.length];
        double[] real = new double[mask.length];
        double[] imag = new double[mask.length];
        Complex[] complexArray = new Complex[real.length];

        for (int i = 0; i < enhancedMag.length; i++) {
            real[i] = mask[i] * (float) Math.cos(phase[i]);
            imag[i] = mask[i] * (float) Math.sin(phase[i]);

            complexArray[i] = new Complex(real[i], imag[i]);
        }

        return complexArray;
    }

    private double[] shiftLeft(double[] input, int stride) {
        int len = input.length;
        double[] shifted = new double[len];
        System.arraycopy(input, stride, shifted, 0, len - stride);
        Arrays.fill(shifted, len - stride, len, 0);
        return shifted;
    }

    private double[] audioToBuffer(double[] buffer, double[] audio) {
        for (int i = 0; i < audio.length; i++) {
            buffer[buffer.length - audio.length + i] = audio[i];
        }

        return buffer;
    }

    private float[] reshapeThreeToOne(float[][][] matrix) {
        float[] array = new float[matrix[0][0].length];
        System.arraycopy(matrix[0][0], 0, array, 0, matrix[0][0].length);

        return array;
    }


    private double[] hannWindow(double[] in_audio) {
        int length = in_audio.length;
        double[] windowed_audio = new double[length];

        for (int i = 0; i < length; i++) {
            windowed_audio[i] = in_audio[i] * this.window[i];
        }

        return windowed_audio;
    }

    private double[] inverseWindow(double[] in_audio) {
        int length = in_audio.length;
        double[] windowed_audio = new double[length];

        for (int i = 0; i < length; i++) {
            windowed_audio[i] = in_audio[i] * this.inverse_window[i];
        }

        return windowed_audio;
    }
}

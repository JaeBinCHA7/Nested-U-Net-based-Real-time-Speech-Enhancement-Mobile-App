# A Real-time Implementation Of a Nested U-NET based Speech Enhancement
https://doi.org/10.7840/kics.2023.48.9.1064

**abstract** : Speech enhancement is a technology that removes noise and enhances speech intelligibility and is used in many fields, such as voice recognition, video conferencing, etc. Recently, DNN-based speech enhancement technology has been actively researched, and a nested U-Net-based speech enhancement model that can effectively utilize local and global information of a speech signal shows excellent performance. To extend the usage of speech enhancement, real-time execution should be possible on various edge devices such as smartphones. In this paper, NUNet-TLS, one of the latest models based on nested U-Net, was implemented in real-time in a smartphone app environment. We first analyzed the model's operation time according to the blocks used, indicating that the dilated convolution block at the bottleneck of NUNet-TLS requires a relatively long operation time because of frequent memory usage. Based on this, we replaced dilated convolution with Long-Short Term Memory (LSTM), significantly reducing the operation time. We implemented the proposed model as a smartphone app after changing the data input/output to a form of frame-basis; the implementation app showed that, compared to the previous model, it used less memory when executed in real-time but achieved equivalent speech enhancement performance.

## Demo
### Real-time speech enhancement app UI and description.
<center><img src = "https://github.com/JaeBinCHA7/Nested-U-Net-based-Real-time-Speech-Enhancement-Mobile-App/assets/87358781/5695a530-55a4-489d-872d-55ce9c6cba25" width="70%" height="70%"></center>

The app provides recording, audio upload, audio download, audio play, and real-time speech enhancement on/off. speech input in real time through recording or audio upload, and enhanced voice is output in real time when the speech enhancement button is activated. Later, the output speech can be downloaded and compared with the input speech.

**# version 1 (2022.11.23)**

[https://user-images.githubusercontent.com/87358781/203532969-44e8c9e1-9632-43e9-b6c0-e4cd64ea4003.mp4](https://user-images.githubusercontent.com/87358781/203690680-cc53a848-c6db-4a5e-8db7-dd397ccbd784.mp4)

**# version 2 (2023.06.07)**

[https://github.com/JaeBinCHA7/Nested-U-Net-based-Real-time-Speech-Enhancement-Mobile-App/assets/87358781/a586bd6b-e414-4ad9-97d9-17ee8cb77a2a](https://github.com/JaeBinCHA7/Nested-U-Net-based-Real-time-Speech-Enhancement-Mobile-App/assets/87358781/a586bd6b-e414-4ad9-97d9-17ee8cb77a2a)

[https://github.com/JaeBinCHA7/Nested-U-Net-based-Real-time-Speech-Enhancement-Mobile-App/assets/87358781/b8c3bec3-7014-45ae-a161-eaeedc20ab28](https://github.com/JaeBinCHA7/Nested-U-Net-based-Real-time-Speech-Enhancement-Mobile-App/assets/87358781/b8c3bec3-7014-45ae-a161-eaeedc20ab28)


## Real-time Implementation 
<center><img src = "https://github.com/JaeBinCHA7/Nested-U-Net-based-Real-time-Speech-Enhancement-Mobile-App/assets/87358781/c1005256-a0e9-47c2-b8df-3bac8268b4ad" width="60%" height="60%"></center>

A general DNN-based speech enhancement system uses a method of batch processing by receiving speech clips of a specific chunk. However, since this method is not suitable for real-time implementation, this paper uses the data input/output method as shown in Figure. The real-time system processes continuous speech input by dividing it into short frame units obtained through overlapping windows. From the frames output in this process, the final enhanced speech signal can be obtained through the inter-frame overlap-sum process. Frame-to-frame overlap is allowed around the input frame at time t. The number of overlapping samples between frames is directly related to the amount of computation in a real-time system. In this paper, a Hann window with a length of 512 samples was used and 50% overlap between frames was allowed. Therefore, when the sampling frequency is 16 kHz, the permissible calculation time per input frame is 16 msec (256 samples). Meanwhile, in the input/output structure shown in Figure, a time delay of half frame (16msec) occurs between the input and output of the model, which is an allowable range in a general speech communication system.

## Nested U-Net TLS using LSTM 
<center><img src = "https://github.com/JaeBinCHA7/Nested-U-Net-based-Real-time-Speech-Enhancement-Mobile-App/assets/87358781/5bbf0994-22a0-407f-b583-4816df130d8e" width="60%" height="60%"></center>

NUNet-TLS achieves excellent speech enhancement performance by using an additional skip connection and CTFA (Causal Time-Frequency Attention) for nested U-Nets. Figure shows the structure of NUNet-TLS. Downsampling (DS) and upsampling (US) layers exist between the encoder and decoder stages, and the bottleneck block is the Dilated Dense block. The U-Net nested in the encoder-decoder stage consists of an input layer (IL), an encoder layer (EL), a bottleneck block (BB), and a decoder layer (DL).

### Real-time processing limitations of Nested U-Net
<center><img src = "https://github.com/JaeBinCHA7/Nested-U-Net-based-Real-time-Speech-Enhancement-Mobile-App/assets/87358781/937e6b59-967a-4a61-8600-a14ff69383d7" width="50%" height="50%"></center>

Figure shows the operation of dilated convolution used in the bottleneck block of the model. Dilated convolution uses (2, 3) convolutional modules just like the encoder and decoder blocks. However, unlike the conventional convolution module, the filter operation is performed at intervals equal to the dilation rate, so the acceptance range is very wide. Therefore, dilated convolution further delays processing time compared to conventional convolution. Since the dilated dense block used in the baseline uses a large number of dilated convolution layers, the processing time increases in proportion to the dilation rate and the number of layers.

### Optimization 
On the other hand, there is LSTM (Long-Short Term Memory) as a module that utilizes data information over time, such as dilated convolution. LSTM determines storage and deletion according to the importance of data through the forget gate, and since only one state needs to be passed to pass the previous data to the next input, the time required to buffer and process a large amount of past data can be reduced. That is, using LSTM can operate more efficiently in terms of real-time processing than dilated convolution.
In fact, we compared the processing time by changing the dilated convolution of the bottleneck block of the baseline model to LSTM. In this paper, we experimented by fixing the unit of LSTM to 21 to keep the number of parameters similar to that of the baseline model.

## Experimental Setting
* **Dataset**
<center><img src = "https://github.com/JaeBinCHA7/Nested-U-Net-based-Real-time-Speech-Enhancement-Mobile-App/assets/87358781/6ddaf380-6017-47b2-ba8f-0072c3eb523c" width="50%" height="50%"></center>

speech data for training used Wall Street Journal (WSJ0) data of 22 hours and 30 minutes in which 81 English speakers read news articles. The SNR of the input speech was composed of SNR of 0, 5, 10, and 15 dB. At this time, Chime2, Chime3, and NoiseX-92 data were used as noise data. The sampling rate is 16kHz. As test data, speech and ETSI  noise data that were not used for training were used. The SNR of the test speech was also set to 0, 5, 10, and 15 dB.

* **Loss**
<center><img src = "https://user-images.githubusercontent.com/87358781/203689803-587d4b4e-3929-40f0-bf41-29c68c5afd8c.png" width="60%" height="60%"></center>

* **Hyperparameter**

The deep learning model was trained in the TensorFlow framework, learning rate was 0.001, batch size was 3, epoch was 100, and Adam was used as optimizer. The FFT length of the STFT used during training was 32 ms, and the Hann window was used as the window.

* **Settings**
<center><img src = "https://github.com/JaeBinCHA7/Nested-U-Net-based-Real-time-Speech-Enhancement-Mobile-App/assets/87358781/218167c3-6258-4e37-879d-36af7d4472dc" width="50%" height="50%"></center>


## Experimental Results
* **Performance**
<center><img src = "https://github.com/JaeBinCHA7/Nested-U-Net-based-Real-time-Speech-Enhancement-Mobile-App/assets/87358781/f850efa5-3315-4253-8d4a-b495f4451450" width="80%" height="80%"></center>


* **RTF(Real-time Factor) and Memory Usage**
<center><img src = "https://github.com/JaeBinCHA7/Nested-U-Net-based-Real-time-Speech-Enhancement-Mobile-App/assets/87358781/d9b752eb-9e04-482c-8515-02f8f0d189fc" width="50%" height="50%"></center>

Real-time factor (RTF) was measured to check whether the speech enhancement system was implemented in real time. RTF is the value obtained by dividing the inference time of the system by the input time, and if RTF is less than 1, real-time processing is possible. RTF can be seen in Figure (a). When the NUNet-TLS set as the baseline model was implemented in real time without modification, the RTF was 2.708, significantly exceeding 1, making real-time processing impossible, whereas the proposed model with the bottleneck block changed to LSTM showed that the RTF was 0.897, which was processed within 1. You can check.

Figure (b) compares the memory usage of the TFLite model installed in the app. Memory usage was divided into model loading memory and real-time execution memory. The model loading memory refers to the memory increased after the speech enhancement model is loaded into the device, and the real-time execution memory refers to the memory increased when the loaded model is executed in real time. At this time, the memory usage was measured using the CPU profiler provided by Android Studio, and was measured based on the maximum memory usage.
   Looking at (b) of Figure 7, it can be seen that the loading memory of the baseline and the proposed model is almost the same, but the baseline model uses about 10MB more in real-time execution memory. Memory usage varies depending on the amount of computation and parameters of the model. The reason for the difference in memory usage during real-time execution, even though the parameters and MACs of the two models are similar, is due to memory allocation and access by dilated convolution. The reduction of RTF by about 3 times is also possible because memory usage is reduced by replacing dilated convolution with LSTM.
   
* **Spectrogram**
<center><img src = "https://github.com/JaeBinCHA7/Nested-U-Net-based-Real-time-Speech-Enhancement-Mobile-App/assets/87358781/623bc252-46b1-4915-a986-7f4f37df7266" width="50%" height="50%"></center>

(a) Spectrogram of clean target speech without noise (b) Spectrogram of input speech mixed with noise (c) Spectrogram of enhanced speech through file-to-file inference in PC (d) Spectrogram of enhanced speech obtained through frame-by-frame inference of the TFLite model in the app 
  

## Requirements
 <img src="https://img.shields.io/badge/TensorFlow2.9-FF6F00?style=flat&logo=TensorFlow&logoColor=white"/> <img src="https://img.shields.io/badge/Python3.7-3776AB?style=flat&logo=Python&logoColor=white"/> <img src="https://img.shields.io/badge/Android Studio-3DDC84?style=flat&logo=Android Studio&logoColor=white"/> <img src="https://img.shields.io/badge/Ubuntu20.04-E95420?style=flat&logo=Ubuntu&logoColor=white"/>
 
This repo is tested with Ubuntu 20.04, Tensorflow 2.9, Python3.7, CUDA11.2. For package dependencies

## Reference
**Monoaural Speech Enhancement Using a Nested U-Net withTwo-Level Skip Connections**   
S. Hwang, S. W. Park, and Y. Park   
[[paper]](https://www.isca-speech.org/archive/pdfs/interspeech_2022/hwang22b_interspeech.pdf)  [[code]](https://github.com/seorim0/NUNet-TLS)   

## Update
* **2023.06.07** upload codes

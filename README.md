# Real time Speech Enhancement in Mobile
**abstract** : Recently, a sound quality enhancement algorithm using nested U-Net, which has shown excellent performance in the field of deep learning-based sound quality enhancement, has been developed.
implemented. At this time, it is confirmed that the real-time system cannot be implemented because the RTF value of the existing superimposed U-Net exceeds 1,
Through model optimization and lightweight work, while maintaining PESQ and STOI performance, which are objective evaluation indicators of voice, similar
It reduced the RTF value by about 30% or more. Through this, finally, superimposed U-Net-based sound quality in the smartphone environment
The enhancement algorithm has been successfully implemented. Real-time sound quality enhancement technology is used in many fields such as AI voice recognition and video conferencing.
It can be used as a preprocessing function.

## Demo
This smartphone app allows intuitive observation of real-time speech enhancement technology.

[https://user-images.githubusercontent.com/87358781/203532969-44e8c9e1-9632-43e9-b6c0-e4cd64ea4003.mp4](https://user-images.githubusercontent.com/87358781/203690680-cc53a848-c6db-4a5e-8db7-dd397ccbd784.mp4)

## Nested U-Net in Tensorflow
![nunet_lstm_kernel1_process](https://user-images.githubusercontent.com/87358781/203689186-da1804e7-4b8c-47f9-945c-ccc68d109546.png)
In the nested U-Net, since the time axis kernel size of the convolution layer is 2 in the encoder-decoder stage, the current information and the past information are delivered together to the next layer. In addition, the bottleneck block of the nested U-Net uses a convolutional layer with various dilations to pass the information of the short and far past to the next layer together. However, in real-time processing, delay is inevitable because an additional buffer to store past information is required according to the kernel size and dilation of the convolution layer. Therefore, in order to prevent delay due to the use of an additional buffer, the time axis kernel size of the convolution layer used in the encoder-decoder stage was changed from 2 to 1, and the bottleneck block was replaced with LSTM to experiment.

## Requirements
 <img src="https://img.shields.io/badge/TensorFlow2.9-FF6F00?style=flat&logo=TensorFlow&logoColor=white"/> <img src="https://img.shields.io/badge/Python3.7-3776AB?style=flat&logo=Python&logoColor=white"/> <img src="https://img.shields.io/badge/Android Studio-3DDC84?style=flat&logo=Android Studio&logoColor=white"/> <img src="https://img.shields.io/badge/Ubuntu20.04-E95420?style=flat&logo=Ubuntu&logoColor=white"/>
 
This repo is tested with Ubuntu 20.04, Tensorflow 2.9, Python3.7, CUDA11.2. For package dependencies

## Experimental Setting
* **Dataset**
![dataset](https://user-images.githubusercontent.com/87358781/203690279-706b1b9e-5022-4552-91c1-a3f826e432fa.png)

* **Loss**
![loss](https://user-images.githubusercontent.com/87358781/203689803-587d4b4e-3929-40f0-bf41-29c68c5afd8c.png)

* **Hyperparameter**
![hyperparam](https://user-images.githubusercontent.com/87358781/203690392-b77320dc-1489-4c1a-a618-fc3aadbf661d.png)

* **Settings**
![settings](https://user-images.githubusercontent.com/87358781/203690421-4c3dd8c1-432e-4871-8e5d-f6840dc2a9a2.png)

## Experimental Results
* **Performance**
![perf](https://user-images.githubusercontent.com/87358781/203691482-1e15dfb3-56df-4be9-8772-9e8fd841dd02.png)

* **RTF(Real-time Factor)**
![rtf](https://user-images.githubusercontent.com/87358781/203691615-b13aa002-df0f-40dc-9860-baf859e84cba.png)

RTF is a value obtained by dividing the voice inference time through the model by the voice length. If the value is 1 or less, real-time processing is possible, and the lower the value, the better the real-time system. As a result of the experiment, it was confirmed that real-time processing in the mobile environment (Android) was not possible with the existing nested U-Net, and when the bottleneck block was replaced with LSTM, the RTF was reduced by about 30% compared to the existing model. It can be seen that processing is possible. In addition, compared to the existing model (Nested U-Net), the optimization model shows a slight difference in PESQ value within the range of 0.1 (Nested U-Net: 2.64, optimized structure: 2.60) and has the same STOI value, so it has great performance. It can be seen that the optimization for real-time processing was well done without any drop.

## Reference
**Monoaural Speech Enhancement Using a Nested U-Net withTwo-Level Skip Connections**   
S. Hwang, S. W. Park, and Y. Park   
[[paper]](https://www.isca-speech.org/archive/pdfs/interspeech_2022/hwang22b_interspeech.pdf)  [[code]](https://github.com/seorim0/NUNet-TLS)   

## Update
* **2022.11.23** upload codes

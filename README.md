# Real time Speech Enhancement in Mobile
**abstract** : Recently, a sound quality enhancement algorithm using nested U-Net, which has shown excellent performance in the field of deep learning-based sound quality enhancement, has been developed.
implemented. At this time, it is confirmed that the real-time system cannot be implemented because the RTF value of the existing superimposed U-Net exceeds 1,
Through model optimization and lightweight work, while maintaining PESQ and STOI performance, which are objective evaluation indicators of voice, similar
It reduced the RTF value by about 30% or more. Through this, finally, superimposed U-Net-based sound quality in the smartphone environment
The enhancement algorithm has been successfully implemented. Real-time sound quality enhancement technology is used in many fields such as AI voice recognition and video conferencing.
It can be used as a preprocessing function.

## Demo
This smartphone app allows intuitive observation of real-time speech enhancement technology.

https://user-images.githubusercontent.com/87358781/203532969-44e8c9e1-9632-43e9-b6c0-e4cd64ea4003.mp4

## Requirements
 <img src="https://img.shields.io/badge/TensorFlow2.9-FF6F00?style=flat&logo=TensorFlow&logoColor=white"/> <img src="https://img.shields.io/badge/Python3.7-3776AB?style=flat&logo=Python&logoColor=white"/> 
 
This repo is tested with Ubuntu 20.04, Tensorflow 2.9, Python3.7, CUDA11.2. For package dependencies

## Update
* **2022.11.23** upload codes

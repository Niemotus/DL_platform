Medical Image Segmentation Platform

Convolution Neural Network based algorithm training and inference platform for segmenting medical image volumes (.nii). 

Prerequisites:
- CUDA
- Theano
- Lasagne
- Numpy
- Nibabel

Data:
Data should be preprocessed registered image volumes in .nii format.

Configuration:
conf/training_dataset1.conf

Training:
src/train_seg_net.py

Inference:
src/test_seg_net.py

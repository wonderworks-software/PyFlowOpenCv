Face Detection by Convolutional Neural Network (CNN) detectors
=============================

Starting from version 3.3, OpenCV supports the Caffe, TensorFlow, and Torch/PyTorch frameworks. OpenCV can load pre-trained CNN model directly.

In this example, we are going to build a computer vision diagram can detect face using Deep Learning Neural Network. We are going to use a new node named 'face_detection_dnn'.

..  image:: res/face_dnn_node.png

Similar to our previous face Detection example, we are going to create the following diagram:

..  image:: res/face_detection_dnn_diagram.png


The Detection result will be much stable than our previous example using Haar feature.

..  image:: res/face_dnn_detection_result.gif

Here I have tried below face detectors for finding landmarks:


1. Dlib's cnn_face_detection_model_v1 : CNN architecture trained model mmod_human_face_detector.dat.
2. OpenCV's DNN module : Pre-trained deep learning caffe model with SSD method.



To run this project:

1.you should install dlib library.

2. to run using cnn_face_detection_model_v: run face_part_detector_cnn.py.
   to run using re-trained deep learning caffe model with SSD method:  run face_part_detector_dl.py.


Notice:
 these two algorithms perform the same task but you will notice that cnn has a better performance than deep learnng model
  but it consume a lot of time to perfom this task than deep learnng model.

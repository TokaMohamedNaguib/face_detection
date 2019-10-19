"""

    Created on Mon Dec 03 11:15:45 2018
    @author: keyur-r

    python facial_landmarks.py -i <image> -l <> -w <> -d <> -p <> -m <> -t <>

    l -> hog or cnn or dl
    w -> model path for facial landmarks (shape_predictor_68_face_landmarks.dat)
    d -> cnn trained model path (mmod_human_face_detector.dat)
    p -> Caffe prototype file for dnn module (deploy.prototxt.txt)
    m -> Caffe trained model weights path (res10_300x300_ssd_iter_140000.caffemodel)
    t -> Thresold to filter weak face in dnn

"""

import numpy as np
import dlib
import cv2
from image_utility import save_image, generate_random_color, draw_border
from imutils import face_utils




def dl_landmarks(image, gray, h, w):
    # # This is based on SSD deep learning pretrained model

    inputBlob = cv2.dnn.blobFromImage(cv2.resize(
        image, (300, 300)), 1, (300, 300), (104, 177, 123))

    face_detector.setInput(inputBlob)
    detections = face_detector.forward()

    for i in range(0, detections.shape[2]):

        # Probability of prediction
        prediction_score = detections[0, 0, i, 2]
        if prediction_score < 0.6:
            continue

        # compute the (x, y)-coordinates of the bounding box for the
        # object
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (x1, y1, x2, y2) = box.astype("int")

        # For better landmark detection
        y1, x2 = int(y1 * 1.15), int(x2 * 1.05)

        # Make the prediction and transfom it to numpy array
        shape = predictor(gray, dlib.rectangle(int(x1), int(y1), int(x2), int(y2)))
        shape = face_utils.shape_to_np(shape)
        cv2.rectangle(image, (x1, y1), (x2, y2), generate_random_color(), 2)

        # Draw on our image, all the finded cordinate points (x,y)


        for (x, y) in shape[37:42]:
            cv2.circle(image, (x, y), 2, (0, 255, 0), -1)
            (a, b, c, d) = cv2.boundingRect(np.array([shape[37:42]]))
            cv2.rectangle(image, (a, b), (a + c, b + d), (0, 0, 255), 2)

        for (x, y) in shape[43:48]:
            cv2.circle(image, (x, y), 2, (0, 255, 0), -1)
            (a, b, c, d) = cv2.boundingRect(np.array([shape[43:48]]))
            cv2.rectangle(image, (a, b), (a + c, b + d), (0, 0, 255), 2)

        for (x, y) in shape[17:22]:
            cv2.circle(image, (x, y), 2, (0, 255, 0), -1)
            (a, b, c, d) = cv2.boundingRect(np.array([shape[17:22]]))
            cv2.rectangle(image, (a, b), (a + c, b + d), (0, 0, 255), 2)

        for (x, y) in shape[23:27]:
            cv2.circle(image, (x, y), 2, (0, 255, 0), -1)
            (a, b, c, d) = cv2.boundingRect(np.array([shape[23:27]]))
            cv2.rectangle(image, (a, b), (a + c, b + d), (0, 0, 255), 2)


def face_detection(image):
    # Converting the image to gray scale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # The 1 in the second argument indicates that we should upsample the image
    # 1 time. This will make everything bigger and allow us to detect more
    # faces.

    # write at the top left corner of the image
    img_height, img_width = image.shape[:2]

    dl_landmarks(image, gray, img_height, img_width)

    cv2.putText(image, "68 Pts - {}".format("dl"), (img_width - 200, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                generate_random_color(), 2)

    save_image(image)

    # Show the image
    cv2.imshow("Facial Landmarks", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":



    # Pre-trained caffe deep learning face detection model (SSD)
    face_detector = cv2.dnn.readNetFromCaffe("models/deploy.prototxt.txt", "models/res10_300x300_ssd_iter_140000.caffemodel")


    # landmark predictor
    predictor = dlib.shape_predictor("models/shape_predictor_68_face_landmarks.dat")
    image = cv2.imread("test_images/testimage.jpeg")
    face_detection(image)



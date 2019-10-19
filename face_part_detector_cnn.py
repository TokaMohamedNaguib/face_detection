

import numpy as np
import dlib
import cv2
from image_utility import save_image, generate_random_color, draw_border
from imutils import face_utils





def cnn_landmarks(image, gray):
    faces_cnn = face_detector(gray, 1)

    # CNN
    for (i, face) in enumerate(faces_cnn):
        # Finding points for rectangle to draw on face
        x, y, w, h = face.rect.left(), face.rect.top(), face.rect.width(), face.rect.height()

        # Drawing simple rectangle around found faces
        cv2.rectangle(image, (x, y), (x + w, y + h), generate_random_color(), 2)

        # Make the prediction and transfom it to numpy array
        shape = predictor(gray, face.rect)
        shape = face_utils.shape_to_np(shape)

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

    cnn_landmarks(image, gray)


    cv2.putText(image, "68 Pts - {}".format("cnn"), (img_width - 200, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                generate_random_color(), 2)

    save_image(image)

    # Show the image
    cv2.imshow("Facial Landmarks", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":



    face_detector = dlib.cnn_face_detection_model_v1("models/mmod_human_face_detector.dat")


    # landmark predictor
    predictor = dlib.shape_predictor("models/shape_predictor_68_face_landmarks.dat")
    image = cv2.imread("test_images/testimage.jpeg")
    face_detection(image)



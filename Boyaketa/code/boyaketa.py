# -*- coding: utf-8 -*-

#Importing Library
import cv2 

img = cv2.imread('../asset/images/img6.jpg');


gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

faces = face_cascade.detectMultiScale(gray, 1.3, 5)


if len(faces) != 0:         # If there are faces in the images
    for f in faces:         # For each face in the image

        # Get the origin co-ordinates and the length and width till where the face extends
        x, y, w, h = [ v for v in f ]

        # get the rectangle img around all the faces
        sub_face = img[y:y+h, x:x+w]
        # apply a gaussian blur on this new recangle image
        sub_face = cv2.GaussianBlur(sub_face,(23, 23), 30)
        # merge this blurry rectangle to our final image
        img[y:y+sub_face.shape[0], x:x+sub_face.shape[1]] = sub_face
        face_file_name = "./face_" + str(y) + ".jpg"

# cv2.imshow("Detected face", result_image)
cv2.imshow('img',img)

cv2.waitKey(0)  
  
#closing all open windows  
cv2.destroyAllWindows()
cv2.imwrite("../asset/images/result.png", img)
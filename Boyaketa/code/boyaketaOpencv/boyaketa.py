# -*- coding: utf-8 -*-

import cv2

img = cv2.imread('../../asset/images/img.jpg');

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

face_cascade = cv2.CascadeClassifier('cascade.xml')

faces = face_cascade.detectMultiScale(gray, 1.3, 5)

if len(faces) != 0:         
    for f in faces:         

        x, y, w, h = [ v for v in f ]

        sub_face = img[y:y+h, x:x+w]
        sub_face = cv2.GaussianBlur(sub_face,(23, 23), 30)
        img[y:y+sub_face.shape[0], x:x+sub_face.shape[1]] = sub_face
        face_file_name = "./face_" + str(y) + ".jpg"

cv2.imshow('img',img)

cv2.waitKey(0)

cv2.destroyAllWindows()
cv2.imwrite("../../asset/images/result.png", img)
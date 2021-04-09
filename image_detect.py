import cv2
from datetime import datetime
import numpy as np
import logging as log

log.basicConfig(filename='ligma.log', level=log.INFO)
time = datetime.now()
casecade_file = "haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(casecade_file)
font = cv2.FONT_HERSHEY_PLAIN

img = cv2.imread('test.jpg', 1)
blnk = np.zeros(img.shape, np.uint8)

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img_to_show = img
faces = face_cascade.detectMultiScale(
    gray,
    scaleFactor=1.1,
    minNeighbors=5,
    minSize=(30, 30)
)

for (x, y, w, h) in faces:
    if len(faces) > 1:
        cv2.putText(img_to_show, f"No.of Faces = {len(faces)}", (10, 20), font, 1.3, (0, 0, 255), 2, cv2.LINE_AA)
    else:
        cv2.putText(img_to_show, f"No.of Faces = {len(faces)}", (10, 20), font, 1.3, (255, 255, 255), 2,
                    cv2.LINE_AA)
    cv2.rectangle(img_to_show, pt1=(x, y), pt2=(x + w, y + h), thickness=3, color=(181, 114, 232))
    cv2.rectangle(img_to_show, pt1=(x - 1, y), pt2=(x + 70, y - 25), thickness=-1, color=(181, 114, 232))
    cv2.putText(img_to_show, "Face", (x, y - 5), font, 1.5, (255, 255, 255), 2, cv2.LINE_AA)


cv2.imshow('Anime Waifu', img_to_show)

k = cv2.waitKey(0)
if k == ord('s'):
    log.info(str(faces)+"Time : "+ str(time))
    print("logged")
    cv2.imwrite('aadhavan.png', img)
    print("Pic saved.s")
elif k == 27:
    cv2.destroyAllWindows()





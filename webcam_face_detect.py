import cv2
from datetime import datetime
from time import sleep
import logging as log

log.basicConfig(filename='logfile.log', level=log.INFO)
time = datetime.now()
casecade_file = "haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(casecade_file)
font = cv2.FONT_HERSHEY_PLAIN

video_cam = cv2.VideoCapture(0)
print(video_cam.get(3))
print(video_cam.get(4))

while True:
    if not video_cam.isOpened():
        print("Unable to Open Video Camera.")
        sleep(5)
        pass
    ret, frame = video_cam.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    img_to_show = frame
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
        cv2.rectangle(img_to_show, pt1=(x-1, y), pt2=(x + 70, y - 25), thickness=-1, color=(181, 114, 232))
        cv2.putText(img_to_show, "Face", (x, y - 5), font, 1.5, (255, 255, 255), 2, cv2.LINE_AA)

    cv2.imshow('FrontFace Demo', img_to_show)

    k = cv2.waitKey(1)
    if k == ord('s'):
        log.info(str(faces)+"Time : " + str(time))
        print("logged")
        cv2.imwrite('frontface.png', frame)
        print("Pic saved.")
    elif k == 27:
        break

video_cam.release()
cv2.destroyAllWindows()





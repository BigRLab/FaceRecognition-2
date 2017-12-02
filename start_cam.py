import cv2
import numpy as np

win_name = "preview"
cv2.namedWindow(win_name, cv2.WINDOW_AUTOSIZE)
vc = cv2.VideoCapture(0)
img_shape = (480, 640, 3)

face_cascade = cv2.CascadeClassifier('/usr/local/share/OpenCV/haarcascades/haarcascade_frontalface_alt2.xml')  # ubuntu

if vc.isOpened():  # try to get the first frame
    rval, frame = vc.read()
else:
    rval = False

fms_counter = 0
fms_rate = 3
image = None
detec = np.zeros(img_shape, dtype='unit8')

while rval:

    rval, frame = vc.read()
    key = cv2.waitKey(20)

    if image is None:
        image = frame.copy()

    if fms_counter % fms_rate == 0:
        image = frame.copy()  # cv2.resize(frame.copy(), (240, 320))
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray, 1.05, 5)
        if len(faces > 0):
            detec = image.copy()
        for (x, y, w, h) in faces:
            cv2.rectangle(detec, (x, y), (x + w, y + h), (255, 0, 0), 2)

    show = np.hstack((image, detec))
    cv2.imshow(win_name, show)

    fms_counter += 1
    if key == 27: # exit on ESC
        break

    fms_counter = fms_counter%100

cv2.destroyWindow(win_name)

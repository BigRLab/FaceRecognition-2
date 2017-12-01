import cv2

cv2.namedWindow("preview")
vc = cv2.VideoCapture(0)

face_cascade = cv2.CascadeClassifier('/usr/local/share/OpenCV/haarcascades/haarcascade_frontalface_alt2.xml')  # ubuntu

if vc.isOpened(): # try to get the first frame
    rval, frame = vc.read()
else:
    rval = False

while rval:

    rval, frame = vc.read()
    key = cv2.waitKey(20)

    image = cv2.resize(frame.copy(), (240, 320))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.02, 5)
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)

    cv2.imshow("preview", image)

    if key == 27: # exit on ESC
        break

cv2.destroyWindow("preview")

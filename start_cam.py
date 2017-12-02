import cv2
import numpy as np
from keras.models import load_model

clf_g = load_model('models/gender_model')
clf_a = load_model('models/age_model')

win_name = "preview"
img_shape = (480, 640, 3)
gender = {0: 'Female', 1: 'Male'}
ages = {0: 'Young', 1: 'Adult', 2: 'Senior'}
font = cv2.FONT_HERSHEY_PLAIN

product = {0: 'caffe', 1: 'cappuccino', 2: 'te', 3: 'ginseng'}
sugar = {0: 'zero', 1: 'poco', 2: 'molto', 3: 'massimo'}

cv2.namedWindow(win_name, cv2.WINDOW_AUTOSIZE)
vc = cv2.VideoCapture(0)

face_cascade = cv2.CascadeClassifier('/usr/local/share/OpenCV/haarcascades/haarcascade_frontalface_alt2.xml')  # ubuntu

if vc.isOpened():  # try to get the first frame
    rval, frame = vc.read()
else:
    rval = False

fms_counter = 0
fms_rate = 120
image = None
detec = np.zeros(img_shape, dtype='uint8')

while rval:

    rval, frame = vc.read()
    key = cv2.waitKey(20)

    image = frame.copy()  # cv2.resize(frame.copy(), (240, 320))

    if fms_counter % fms_rate == 0:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray, 1.075, 5)
        if len(faces) > 0:
            detec = image.copy()
        for (x, y, w, h) in faces:
            # predicting
            x_input = cv2.resize(gray[y0: y0 + h, x0: x0 + w], clf_g.input_shape)
            fg = clf_g.predict(x_input)
            fa = clf_a.predict(x_input)
            g = np.argmax(fg)
            pg = np.max(fg)
            a = np.argmax(fa)
            pa = np.max(fa)
            
            sg = gender[g] + ": {:.3}".format(pg)
            sa = ages[a] + ": {:.3}".format(pa)
            cv2.rectangle(detec, (x, y), (x + w, y + h), (255, 200, 0), 3)
            cv2.putText(detec, sg, (x, y - 30), font, 2, (255, 200, 0), 2, cv2.LINE_AA)
            cv2.putText(detec, sa, (x, y - 5), font, 2, (255, 200, 0), 2, cv2.LINE_AA)
            sug = sugar[np.random.randint(4)]
            caf = product[np.random.randint(4)]
            print "ordinato " + caf + " con " + sug + " zucchero"
            
    show = np.hstack((image, detec))
    cv2.imshow(win_name, show)

    fms_counter += 1
    if key == 27: # exit on ESC
        break

    fms_counter = fms_counter%100

cv2.destroyWindow(win_name)

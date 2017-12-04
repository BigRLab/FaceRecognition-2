import cv2
import numpy as np
from keras.models import load_model
import os

clf_g = load_model('models/cnn_mauri/gender_model')
clf_a = load_model('models/cnn_mauri/age_model')

win_name = "preview"
img_shape = (480, 640, 3)
gender = {0: 'Female', 1: 'Male'}
ages = {0: 'Young', 1: 'Adult', 2: 'Senior'}
font = cv2.FONT_HERSHEY_PLAIN

product = {0: 'Caffe', 1: 'Cappuccino', 2: 'Te', 3: 'Ginseng'}
sugar = {0: ' senza zucchero', 1: ' con poco zucchero', 2: ' con molto zucchero'}

cv2.namedWindow(win_name, cv2.WINDOW_AUTOSIZE)
vc = cv2.VideoCapture(0)

face_cascade = cv2.CascadeClassifier('/usr/local/share/OpenCV/haarcascades/haarcascade_frontalface_alt2.xml')  # ubuntu

if vc.isOpened():  # try to get the first frame
    rval, frame = vc.read()
else:
    rval = False

fms_counter = 0
fms_rate = 600
image = None
detec = np.zeros(img_shape, dtype='uint8')
scale_coef = 0.8

wrs = 1

while rval:

    rval, frame = vc.read()
    key = cv2.waitKey(20)

    image = frame.copy()  # cv2.resize(frame.copy(), (240, 320))

    if fms_counter % fms_rate == 0:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray, 1.075, 5)
        if len(faces) > 0:
            detec = image.copy()

            ind = 0
            maxw = 0
            maxi = 0
            for (x, y, w, h) in faces:
                if w > maxw:
                    maxw = w
                    maxi = ind
                    ind += 1

            (x, y, w, h) = faces[maxi]
            # predicting
            if min(w, h) < 60:
                continue

            x_input = cv2.resize(gray[y: y + h, x: x + w] / 255.0, clf_g.input_shape[1: 3])
            x_input = x_input.reshape((1, ) + x_input.shape + (1, ))
            fg = clf_g.predict(x_input)
            fa = clf_a.predict(x_input)
            gb = int(x_input.shape[1] / 4.0)
            up_mask = np.mean(x_input[0, 2 *gb: -gb, gb: -gb])
            low_mask = np.mean(x_input[0, -gb:, gb: -gb])
            print "\n GENDER PREDICTION"
            print "original pred", fg
            print "mask", up_mask, low_mask
            mf = 2.0 * min(max(up_mask  - low_mask, 0), 1) / up_mask
            print mf
            fg = fg + np.array([-mf, mf])  # (1 - scale_coef) * fg + scale_coef * np.array([-mf, mf])
            fg = np.exp(fg) / np.sum(np.exp(fg))
            print "scaled pred", fg

            print "\n AGE PREDICTION"
            print "original pred", fa
            sf = max(np.median(x_input[0, :gb, :gb]), np.median(x_input[0, :gb, :-gb])) / 4.0
            print sf
            fa = fa + np.array([-sf, -sf, sf])  # (1 - scale_coef) * fg + scale_coef * np.array([-mf, mf])
            fa = np.exp(fa) / np.sum(np.exp(fa))
            print "scaled pred", fa

            g = np.argmax(fg)
            pg = np.max(fg)
            a = np.argmax(fa)
            pa = np.max(fa)

            sg = gender[g] + ": {:.3}".format(pg)
            sa = ages[a] + ": {:.3}".format(pa)
            cv2.rectangle(detec, (x, y), (x + w, y + h), (255, 200, 0), 3)
            cv2.putText(detec, sg, (x, y - 30), font, 2, (255, 200, 0), 2, cv2.LINE_AA)
            cv2.putText(detec, sa, (x, y - 5), font, 2, (255, 200, 0), 2, cv2.LINE_AA)
            sug = sugar[np.random.randint(len(sugar))]
            caf = product[np.random.randint(len(product))]
            testo = str(wrs) + ", " + gender[g] + ", " + ages[a] + ", " + caf + sug
            wrs += 1
            wrs = wrs % 2
            print testo
            cmd = "echo '{}' > /sensors/gender/data".format(testo)
            os.system(cmd)

    show = np.hstack((image, detec))
    cv2.imshow(win_name, show)

    fms_counter += 1
    if key == 27: # exit on ESC
        break

    fms_counter = fms_counter%100

cv2.destroyWindow(win_name)

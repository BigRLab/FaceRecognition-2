import speech_recognition as sr
# import librosa
import numpy as np
# import matplotlib.pyplot as plt
import cv2

# obtain audio from the microphone
r = sr.Recognizer()
# r.energy_threshold = 4000.0
win_name = "Vending Machine"

# with sr.AudioFile('microphone-results.wav') as source:
#    audio = r.record(source)

benv = "Benvenuto!"
sel = "Seleziona un prodotto"
scel = "Hai selezionato: "

cv2.namedWindow(win_name, cv2.WINDOW_AUTOSIZE)
im_size = (480, 640)

sc_benv = '/home/udoo/Facerecognition/data/images/F1.jpg'
sc_1 = '/home/udoo/Facerecognition/data/images/F1.jpg'
# sc_benv = '/Users/Alessandro/Desktop/Adoor_Gopalakarishnan_0001.jpg'
# sc_1 = '/Users/Alessandro/Desktop/prova.jpg'

im_benv = cv2.imread(sc_benv)
p = cv2.resize(im_benv, im_size)

with sr.Microphone() as source:
    #print "initial thre: ", r.energy_threshold
    r.adjust_for_ambient_noise(source)  # listen for 1 second to calibrate the energy threshold for ambient noise levels
    #print "adjusted thre: ", r.energy_threshold
    und = False
    cv2.imshow(win_name, p)
    cv2.waitKey(0)

    while not und:
        print benv + sel
        audio = r.listen(source)
        try:
            print scel + r.recognize_google(audio, language="it-IT")
            und = True
        except sr.UnknownValueError:
            print "ripeti per favore"

    # selezionare la schermata del prodotto
    im_1 = cv2.imread(sc_1)
    p1 = cv2.resize(im_1, im_size)
    cv2.imshow(win_name, p1)
    cv2.waitKey(0)
    # chiedere lo zucchero
    # grazie e arrivederci
# recognize speech using Sphinx
# language="it-IT" , keyword_entries=[("hello", 1.0), ("coffee", 1.0)], grammar='counting.gram'
"""
try:
    #  print("Sphinx thinks you said: " + r.recognize_sphinx(audio, keyword_entries=[("cappuccino", 1.0), ("caffe", 1.0)], language="it-IT"))
    print scel + r.recognize_google(audio, language="it-IT")
except sr.UnknownValueError:
    print("Could not understand audio")
"""
#except sr.RequestError as e:
#    print("error; {0}".format(e))


# with open("microphone-results.wav", "wb") as f:
#    f.write(audio.get_wav_data())

cv2.destroyWindow(win_name)

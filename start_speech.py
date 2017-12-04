import speech_recognition as sr
# import librosa
import numpy as np
# import matplotlib.pyplot as plt

# obtain audio from the microphone
r = sr.Recognizer()
# r.energy_threshold = 4000.0

# with sr.AudioFile('microphone-results.wav') as source:
#    audio = r.record(source)

benv = "Benvenuto!"
sel = "Seleziona un prodotto"
scel = "Hai selezionato: "

with sr.Microphone() as source:
    #print "initial thre: ", r.energy_threshold
    r.adjust_for_ambient_noise(source)  # listen for 1 second to calibrate the energy threshold for ambient noise levels
    #print "adjusted thre: ", r.energy_threshold
    und = False
    while not und:
        print benv + sel
        audio = r.listen(source)
        try:
            print scel + r.recognize_google(audio, language="it-IT")
            und = True
        except sr.UnknownValueError:
            print "ripeti per favore"


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

import numpy as np
# import matplotlib.pyplot as plt
from os import listdir
import time
# import cv2.face as fc
import cv2
from os import listdir
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split, GridSearchCV
import sklearn.preprocessing as prep
from myMLmodules import balance_set
from keras.preprocessing import image
from sklearn.decomposition import PCA
from sklearn.svm import SVC
import hickle as hkl


# dir_path = '/media/udoo/HD/Faces/faces/'  # ubuntu
# dir_path = '/Volumes/HD/Faces/faces_clean/'  # mac
dir_path = '/Volumes/HD/Faces/dataset/faces64/'
# dir_path = '/home/users/fmontefoschi/scratch-local/FaceRecognition/data/faces512/' # BSC
#dir_path = '/Users/Alessandro/Desktop/Faces/'

im_size = (64, 64)
n_components = 200
data_aug = 0
classifier = 'cnn'

categories = {'female': {'young': 0, 'adult': 1, 'senior': 2},
              'male': {'young': 3, 'adult': 4, 'senior': 5}}
# age_categories = {'young': 0, 'adult': 1, 'senior': 2}

# face_cascade = cv2.CascadeClassifier('/opt/local/share/OpenCV/haarcascades/haarcascade_frontalface_default.xml')
# face_cascade = cv2.CascadeClassifier('/opt/local/share/OpenCV/haarcascades/haarcascade_frontalface_alt.xml')
face_cascade = cv2.CascadeClassifier('/opt/local/share/OpenCV/haarcascades/haarcascade_frontalface_alt2.xml')  # mac
# face_cascade = cv2.CascadeClassifier('/usr/local/share/OpenCV/haarcascades/haarcascade_frontalface_alt2.xml')  # ubuntu
# face_cascade = cv2.CascadeClassifier('/opt/local/share/OpenCV/haarcascades/haarcascade_frontalface_alt_tree.xml')


def load_images(dir_path):
    data = []
    labels = []

    gender = listdir(dir_path)

    for g in gender:

        if g.startswith('.'):
            continue

        print "Gender: ", g
        ages = listdir(dir_path + g)
        print ages
        for a in ages:

            if a.startswith('.'):
                continue
            print "   Ages: ", a
            subjects = listdir(dir_path + g + '/' + a)
            for sub in subjects:

                if sub.startswith('.'):
                    continue

                # print "      subj: ", sub
                images = listdir(dir_path + g + '/' + a + '/' + sub)
                img = images[0]

                if img.startswith('.'):
                    if len(images) > 0:
                        img = images[1]
                    else:
                        continue

                p = cv2.imread(dir_path + g + '/' + a + '/' + sub + '/' + img)
                # p = cv2.resize(p, (256, 256))

                data.append(cv2.cvtColor(cv2.resize(p, im_size), cv2.COLOR_BGR2GRAY))
                labels.append(categories[g][a])

    return np.array(data), np.array(labels)


def load_face(dir_path):
    data = []
    labels = []
    minsiz = (1000, 1000)

    gender = listdir(dir_path)

    for g in gender:

        if g.startswith('.'):
            continue

        print "Gender: ", g
        ages = listdir(dir_path + g)
        print ages
        for a in ages:

            if a.startswith('.'):
                continue
            print "   Ages: ", a
            subjects = listdir(dir_path + g + '/' + a)
            for sub in subjects:

                if sub.startswith('.'):
                    continue

                # print "      subj: ", sub
                images = listdir(dir_path + g + '/' + a + '/' + sub)

                # img = images[0]
                for img in images:

                    if img.startswith('.'):
                        continue

                    try:
                        p = cv2.imread(dir_path + g + '/' + a + '/' + sub + '/' + img)
                        p = cv2.resize(p, im_size)

                    except:
                        print dir_path + g + '/' + a + '/' + sub + '/' + img
                        continue

                    image = p.copy()
                    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                    faces = face_cascade.detectMultiScale(gray, 1.05, 5)

                    if len(faces) is not 1:
                        continue

                    x0, y0, w, h = faces[0]
                    """
                    if len(faces) == 0:
                        x0, y0 = 0, 0
                        x1, y1 = gray.shape[:2]
    
                    else:
                        x0 = np.min(faces, axis=0)[0]
                        y0 = np.min(faces, axis=0)[1]
                        x1 = np.max(faces[:, 0] + faces[:, 2], axis=0)
                        y1 = np.max(faces[:, 1] + faces[:, 3], axis=0)
    
                        # cv2.rectangle(image, (x0, y0), (x1, y1), (255, 0, 0), 2)
    
                        # print "      box: ", x1 - x0, y1 - y0
                    #cv2.rectangle(gray, (0, 20), (50, 100), (255, 0, 0), 2)
                    minsiz = (min(minsiz[0], x1 - x0), min(minsiz[1], y1 - y0))
                    data.append(gray[y0: y1, x0: x1])
                    """
                    minsiz = (min(minsiz[0], h), min(minsiz[1], w))

                    data.append(gray[y0: y0 + h, x0: x0 + w])
                    labels.append(categories[g][a])
                # print "      label: ", curlab, ages_labels_count

                # cv2.imshow('image', image)
                # cv2.waitKey(0)
                # cv2.destroyAllWindows()

    # reshaping data
    x = []

    print minsiz
    for img in data:
        x.append(cv2.resize(img, minsiz))

    # converting data to array
    x = np.array(x)
    y = np.array(labels)

    return x, y


def relabel(y):

    yg = y.copy()
    ya = y.copy()

    # gender labels
    yg[yg < 3] = 0
    yg[yg > 0] = 1

    # age label
    # ya[ya == 3] = 0
    # ya[ya == 4] = 1
    # ya[ya == 5] = 2
    ya[ya > 2] -= 3

    return yg, ya

#   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #  #
#                                                                                                                      #
#                                                FISHER FACE                                                           #
#                                                                                                                      #
#   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #  #

def face_training(x, y):

    fr = fc.FisherFaceRecognizer_create()
    # adding normalization
    fr.train(x, y)
    # adding saving
    return fr


def fr_test(fr, xe, ye, fe=None, n_classes=None):

    # normalize data
    if fe is None:
        fe = np.array([fr.predict_label(xe[i]) for i in range(xe.shape[0])])

    cfm = confusion_matrix(ye, fe)
    n_classes = np.unique(ye).shape[0]
    accs = cfm[range(n_classes), range(n_classes)] / np.sum(cfm, axis=1, dtype=float)

    return cfm, accs, np.mean(accs)


#   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #  #
#                                                                                                                      #
#                                                PCA                                                                   #
#                                                                                                                      #
#   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #  #

def face_pca(xtrain, xtest, n_components=50):

    pca = PCA(n_components=n_components, svd_solver='randomized', whiten=True).fit(xtrain)
    return pca, pca.transform(xtrain), pca.transform(xtest)


def face_svc_train(x, y):

    param_grid = {'C': [1e3, 5e3, 1e4, 5e4, 1e5],
                  'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1], }
    clf = GridSearchCV(SVC(kernel='rbf', class_weight='balanced', probability=True), param_grid)
    clf = clf.fit(x, y)

    return clf


def face_cnn_train(x, y):

    from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
    from keras.models import Sequential
    from keras.callbacks import EarlyStopping

    # options
    f1 = 32
    k1 = 7
    p1 = 2
    """
    f2 = 2
    k2 = 5
    p2 = 2
    f3 = 2
    k3 = 3
    p3 = 2
    """
    hu = 256
    n_classes = np.unique(y).shape[0]

    # initialization
    clf = Sequential()
    # 1st layer
    clf.add(Conv2D(f1, (k1, k1), activation='relu', padding='same', input_shape=(x.shape[1:])))
    clf.add(MaxPooling2D((p1, p1), padding='same'))
    """
    # 2nd layer
    clf.add(Conv2D(f2, (k2, k2), activation='relu', padding='same'))
    clf.add(MaxPooling2D((p2, p2), padding='same'))
    # 1st layer
    clf.add(Conv2D(f3, (k3, k3), activation='relu', padding='same'))
    clf.add(MaxPooling2D((p3, p3), padding='same'))
    """
    # fully connected
    clf.add(Flatten())
    clf.add(Dense(hu, activation='relu'))
    clf.add(Dense(n_classes, activation='softmax'))
    clf.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

    # generating target
    t_train = np.zeros((y.shape[0], n_classes))
    t_train[np.arange(y.shape[0]), y] = 1
    # training
    clf.fit(x, t_train, epochs=1, validation_split=0.1, shuffle=True,
            callbacks=[EarlyStopping(monitor='val_loss', min_delta=1e-1, patience=5, verbose=0, mode='auto')])

    return clf

def face_predict(gray, clf, pca, bbox_size):

    faces = face_cascade.detectMultiScale(gray, 1.05, 5)
    print faces
    if len(faces) > 0:
        f = []
        for (x0, y0, w, h) in faces:
            x = gray[y0: y0 + h, x0: x0 + w].copy()
            x = cv2.resize(x, bbox_size).reshape(1, -1) / 255.0
            x = pca.transform(x)
            f.append(clf.predict_proba(x))
    else:

        if classifier == 'svm':
            x = cv2.resize(gray, bbox_size).reshape(1, -1) / 255.0
            x = pca.transform(x)
            f = clf.predict_proba(x)

        elif classifier == 'cnn':
            x = cv2.resize(gray, bbox_size) / 255.0
            f = clf.predict(x)

    return faces, f


def face_prediction_show(path_to_img, clf, pca, bbox_size):

    p = cv2.imread(path_to_img)
    try:
        p = cv2.resize(p, im_size)
    except:
        return

    gray = cv2.cvtColor(p, cv2.COLOR_BGR2GRAY)
    faces, f = face_predict(gray, clf, pca, bbox_size)
    i = 0
    for (x0, y0, w, h) in faces:
        t = ''
        for j in f[i][0]:
            t += str(j) + '; '
        image = p.copy()
        cv2.rectangle(image, (x0, y0), (x0 + w, y0 + h), (255, 0, 0), 2)
        cv2.imshow(t, image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        i += 1
    return faces, f


def data_augmentation(x, y, n):
    datagen = image.ImageDataGenerator(featurewise_center=False, featurewise_std_normalization=False, rotation_range=20,
                                       width_shift_range=0.2, height_shift_range=0.2, shear_range=0.2,
                                       zoom_range=0.2, horizontal_flip=True)

    # data augmentation
    i = 0
    xa = x.reshape(x.shape + (1,)).copy()
    ya = y.copy()
    for x_batch, y_batch in datagen.flow(xa, ya, batch_size=32):
        if i >= n:
            break
        xa = np.append(xa, x_batch, axis=0)
        ya = np.append(ya, y_batch, axis=0)
        i += 1

    return xa, ya


#   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #  #
#                                                                                                                      #
#                                          START TRAINING                                                              #
#                                                                                                                      #
#   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #  #


# print "Preparing data . . . "
#x, y = load_face(dir_path)  # loading data

print "Loading data . . . "
x = hkl.load(dir_path + 'data')
y = hkl.load(dir_path + 'label')

#

xb = x.copy()
yb = y.copy()

if classifier == 'svm':
    xb = xb.reshape(-1, np.prod(xb.shape[1:])) / 255.0  # reshaping and formatting data
elif classifier == 'cnn':
    xb, yb = balance_set(x, y)  # classes balancing
    xb = xb/255.0

yg, ya = relabel(yb)  # resplitting gender/ages labels

xt_g, xe_g, yt_g, ye_g = train_test_split(xb, yg, test_size=0.5)  # gender train test split
xt_a, xe_a, yt_a, ye_a = train_test_split(xb, ya, test_size=0.5)  # age train test split


if classifier == 'svm':

    xb = xb.reshape(-1, np.prod(xb.shape[1:])) / 255.0  # reshaping and formatting data

    print "PCA projection . . . "
    pca_g, xt_g, xe_g = face_pca(xt_g, xe_g, n_components=n_components)
    pca_a, xt_a, xe_a = face_pca(xt_a, xe_a, n_components=n_components)

    train_function = face_svc_train

elif classifier == 'cnn':

    if data_aug > 0:
        xt_g, yt_g = data_augmentation(xt_g, yt_g, data_aug)
        xt_a, yt_a = data_augmentation(xt_a, yt_a, data_aug)
    else:
        xt_g = xt_g.reshape(xt_g.shape + (1, ))
        xt_a = xt_a.reshape(xt_a.shape + (1, ))

    xe_g = xe_g.reshape(xe_g.shape + (1, ))
    xe_a = xe_a.reshape(xe_a.shape + (1, ))
    train_function = face_cnn_train


# training
print "Gender training . . . "
clf_g = train_function(xt_g, yt_g)

print "Age training . . . "
clf_a = train_function(xt_a, yt_a)

fe_g = clf_g.predict(xe_g)
fe_a = clf_a.predict(xe_a)

if classifier == 'cnn':
    fe_g = np.argmax(fe_g, axis=1)
    fe_a = np.argmax(fe_a, axis=1)

ConfMat_g, Acc_g, acc_g = fr_test([], [], ye_g, fe=fe_g)
print "Confusion:"
print ConfMat_g
print "Accuracy: ", Acc_g, " - ", acc_g

ConfMat_a, Acc_a, acc_a = fr_test([], [], ye_a, fe=fe_a)
print "Confusion:"
print ConfMat_a
print "Accuracy: ", Acc_a, " - ", acc_a

bbox_size = x.shape[1:]

clf_g.save("models/gender_model")
clf_a.save("models/age_model")
# print "train: ", x.shape, y.shape
# print "test: ", xe.shape, ye.shape







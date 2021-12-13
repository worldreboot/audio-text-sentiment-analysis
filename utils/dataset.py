import os
import numpy as np
import pandas as pd
from .audio import convert_audio_to_log_mel_spectrogram
import model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_datasets as tfds
import sys

from mmsdk import mmdatasdk
import numpy as np
import nltk
from nltk.corpus import stopwords
import pickle

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

from tensorflow.keras.utils import to_categorical

import model
import model_2

from sklearn import tree
from sklearn.dummy import DummyClassifier

config = tf.compat.v1.ConfigProto(gpu_options=tf.compat.v1.GPUOptions(allow_growth=True))
sess = tf.compat. v1.Session(config=config)

def fix_gpu():
    config = ConfigProto()
    config.gpu_options.allow_growth = True
    session = InteractiveSession(config=config)

# def gen(input, output):
#     for i in range(input.size[0]):
#         yield input[i], output[i]

def numpy_fillna(data):
    # Get lengths of each row of data
    lens = np.array([len(i) for i in data])

    # Mask of valid places in each row
    mask = np.arange(lens.max()) < lens[:,None]

    # Setup output array and put elements from data into masked positions
    out = np.zeros(mask.shape)
    out[mask] = np.concatenate(data)
    return out

def load_ravdess_dataset(path_to_ravdess):
    # CREATE DIRECTORY OF AUDIO FILES 
    audio = path_to_ravdess
    actor_folders = os.listdir(audio) #list files in audio directory
    actor_folders.sort()

    # # CREATE FUNCTION TO EXTRACT EMOTION NUMBER, ACTOR AND GENDER LABEL
    emotion = []
    gender = []
    actor = []
    file_path = []
    for i in actor_folders:
        filename = os.listdir(audio + i) #iterate over Actor folders
        for f in filename: # go through files in Actor folder
            part = f.split('.')[0].split('-')
            emotion.append(int(part[2]))
            actor.append(int(part[6]))
            bg = int(part[6])
            if bg % 2 == 0:
                bg = "female"
            else:
                bg = "male"
            gender.append(bg)
            file_path.append(audio + i + '/' + f)

    # # # PUT EXTRACTED LABELS WITH FILEPATH INTO DATAFRAME
    audio_df = pd.DataFrame(emotion)
    audio_df = audio_df.replace({1:'neutral', 2:'calm', 3:'happy', 4:'sad', 5:'angry', 6:'fear', 7:'disgust', 8:'surprise'})
    audio_df = pd.concat([pd.DataFrame(gender),audio_df,pd.DataFrame(actor)],axis=1)
    audio_df.columns = ['gender','emotion','actor']
    audio_df = pd.concat([audio_df,pd.DataFrame(file_path, columns = ['path'])],axis=1)

    # print(emotion)

    # EXPORT TO CSV
    # audio_df.to_csv('audio_df.csv')

    df = pd.DataFrame(columns=['mel_spectrogram'])

    counter=0
    # print(audio_df.path)
    # print(audio_df.path[0:10])
    print("Loading dataset...")
    log_mels = []
    for index, path in enumerate(file_path):
        # print(path)
        log_spectrogram = convert_audio_to_log_mel_spectrogram(path)
        log_mels.append(log_spectrogram)
        df.loc[counter] = [log_spectrogram]
        counter = counter + 1
    print("Finished loading dataset")   
    # print(max(len(log_mel) for log_mel in log_mels))
    # print(len(df))
    # df.head()

    # np.set_printoptions(threshold=sys.maxsize)
    # df.to_csv('ravdess.csv')

    # audio_df = pd.read_csv('audio_df.csv')
    # df = pd.read_csv('ravdess.csv')

    # print(df['mel_spectrogram'].values.tolist()[1])

    # TURN ARRAY INTO LIST AND JOIN WITH AUDIO_DF TO GET CORRESPONDING EMOTION LABELS
    # df_combined = pd.concat([audio_df,pd.DataFrame(df['mel_spectrogram'].values.tolist())],axis=1)
    # df_combined = df_combined.fillna(0)
    # for thingy in df['mel_spectrogram'].values.tolist():
        # print(thingy)

    # dataset = tf.data.Dataset.from_generator(lambda: (df['mel_spectrogram'].values.tolist(), 42), output_signature=(tf.TensorSpec(shape=(128, None), dtype=tf.float32, name=None)))
    
    log_mels = numpy_fillna(log_mels)

    # for list in numpy_fillna(log_mels):
    #     print(len(list))
    # print(numpy_fillna(log_mels))

    # test_dataset = tf.data.Dataset.from_tensor_slices((numpy_fillna(log_mels), emotion[0:96]))
    # print(test_dataset)
    # for test in test_dataset:
    #     print(test)

    lb = LabelEncoder()
    emotion_one_hot = to_categorical(lb.fit_transform(emotion))

    def gen():
        for i in range(len(log_mels)):
            yield log_mels[i], emotion_one_hot[i]
    
    # def gen():
    #     for i in range(len(log_mels)):
    #         yield log_mels[i]
    # dataset = tf.data.Dataset.from_generator(gen, output_signature=(tf.TensorSpec(shape=(128, None), dtype=tf.float32, name=None), tf.TensorSpec(shape=(8,), dtype=tf.float32)))
    # dataset = tf.data.Dataset.from_generator(gen, output_signature=(tf.TensorSpec(shape=(None, ), dtype=tf.float32, name=None), tf.TensorSpec(shape=(8,), dtype=tf.float32)))
    # dataset = tf.data.Dataset.from_generator(gen, output_signature=(tf.TensorSpec(shape=(128, None), dtype=tf.float32, name=None)))
    # print(df_combined)

    # for element in dataset:
        # print(element[0].shape())

    # print(dataset)
    # print(dataset.element_spec)
    # print(list(dataset.take(5)))

    # train_dataset = dataset.padded_batch(96)
    # # train_dataset = dataset.take(72)
    # # test_dataset = dataset.skip(72)
    # batches = train_dataset.take(1)
    # test_batches = train_dataset.skip(72)
    # print(list(batches.take(5)))
    # print(batches.padded_shapes)

    # print(train_dataset)

    # input_size = 0
    # for batch in batches:
    #     input_size = batch[0].shape[1]
    #     print(input_size)

    # df_combined = tfds.as_dataframe(dataset)

    # df_combined = pd.concat([audio_df, df_combined],axis=1)
    # df_combined = df_combined.fillna(0)

    # DROP PATH COLUMN FOR MODELING
    # df_combined.drop(columns='path',inplace=True)

    # print(df_combined)

    # # # # TRAIN TEST SPLIT DATA
    # train, test = train_test_split(df_combined, test_size=0.2, random_state=0,
    #                            stratify=df_combined[['emotion','gender','actor']])
    
    # X_train = train.iloc[:, 3:]
    # y_train = train.iloc[:,:2].drop(columns=['gender'])
    # # print(X_train.shape)
    # # print(y_train.shape)

    # X_test = test.iloc[:,3:]
    # y_test = test.iloc[:,:2].drop(columns=['gender'])
    # # print(X_test.shape)
    # # print(y_test.shape)

    # # # # # TURN DATA INTO ARRAYS FOR KERAS
    # X_train = np.array(X_train)
    # y_train = np.array(y_train)
    # X_test = np.array(X_test)
    # y_test = np.array(y_test)

    # print(X_train)

    # X_train = tf.convert_to_tensor(X_train)
    # y_train = tf.convert_to_tensor(y_train)
    # X_test = tf.convert_to_tensor(X_test)
    # y_test = tf.convert_to_tensor(y_test)

    # X_train = np.stack(X_train, axis=0)
    # y_train = np.stack(y_train)
    # X_test = np.stack(X_test, axis=0)
    # y_test = np.stack(y_test)

    # print(tf.convert_to_tensor(X_train  , dtype=tf.float32))

    # ONE HOT ENCODE THE TARGET
    # CNN REQUIRES INPUT AND OUTPUT ARE NUMBERS
    lb = LabelEncoder()
    y_train = to_categorical(lb.fit_transform(emotion))
    # print(y_train[0:5])
    # y_test = to_categorical(lb.fit_transform(y_test))

    # print(y_train)

    # tensor = tf.ragged.constant(dataset)

    # train_dataset = tf.data.Dataset.from_tensor_slices((list(X_train), y_train))

    # print(y_train[0:10])

    # print(lb.classes_)

    # cnn = model.CNN(0, 1)
    # cnn.model.load_weights('best_initial_model_2.hdf5')

    # eval = cnn.model.evaluate(test_batches)
    # print("loss: " + str(eval[0]))
    # print("accuracy: " + str(eval[1]))

    # # RESHAPE DATA TO INCLUDE 3D TENSOR 
    # X_train = X_train[:,:,np.newaxis]
    # X_test = X_test[:,:,np.newaxis]

    # print(tf.convert_to_tensor(X_train))
    # X_train = np.asarray(X_train).astype('object')
    # print(np.array(dataset.take(128)))
    # print(type(dataset))

    # print(cnn.model.predict(batches).shape)

    # log_mels = np.array(log_mels)
    # log_mels = log_mels[:, :, np.newaxis]
    print(log_mels.shape)

    combined = [[log_mels[i], emotion_one_hot[i]] for i in range(len(log_mels))]

    train, test = train_test_split(combined, test_size=0.2, random_state=0)

    print(train[0])

    x_train = [sample[0] for sample in train]
    y_train = [sample[1] for sample in train]

    x_test = [sample[0] for sample in test]
    y_test = [sample[1] for sample in test]

    x_train = np.array(x_train)
    y_train = np.array(y_train)
    x_test = np.array(x_test)
    y_test = np.array(y_test)

    # print(x_train)
    # print(y_train)

    # x_train = x_train[:, :, np.newaxis]
    # y_train = y_train[:, :, np.newaxis]
    # x_test = x_test[:, :, np.newaxis]

    # print(x_train[0])

    dummy_clf = DummyClassifier(strategy="stratified")
    dummy_clf.fit(x_train, y_train)
    DummyClassifier(strategy='stratified')
    dummy_clf.predict(x_test)
    print("dummy: ", str(dummy_clf.score(x_test, y_test)))

    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(x_train, y_train)
    clf.predict(x_test)
    print("tree: ", str(clf.score(x_test, y_test)))

    x_train = x_train[:, :, np.newaxis]
    # y_train = y_train[:, :, np.newaxis]
    x_test = x_test[:, :, np.newaxis]

    cnn = model_2.CNN(x_train.shape[1], 8)

    # FIT MODEL AND USE CHECKPOINT TO SAVE BEST MODEL
    checkpoint = ModelCheckpoint("best_initial_model_ryerson.hdf5", monitor='accuracy', verbose=1,
        save_best_only=True, mode='max', period=1, save_weights_only=True)

    # [print(i.shape, i.dtype) for i in cnn.model.inputs]

    # model_history = cnn.model.fit(train_dataset, epochs=40, callbacks=[checkpoint])
    # model_history = cnn.model.fit(batches, epochs=40, validation_data=test_batches, callbacks=[checkpoint])
    # model_history=cnn.model.fit(log_mels[0:1152], y_train[0:1152],batch_size=32, epochs=40, validation_data=(log_mels[1152:], y_train[1152:]), callbacks=[checkpoint])
    model_history=cnn.model.fit(x_train, y_train, batch_size=32, epochs=40, validation_data=(x_test, y_test), callbacks=[checkpoint])
    # model_history=cnn.model.fit(test_dataset.batch(32), batch_size=32, epochs=40, callbacks=[checkpoint])
    # PLOT MODEL HISTORY OF ACCURACY AND LOSS OVER EPOCHS
    plt.plot(model_history.history['accuracy'])
    plt.plot(model_history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.savefig('Initial_Model_Accuracy_4.png')
    plt.show()
    # # summarize history for loss
    plt.plot(model_history.history['loss'])
    plt.plot(model_history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig('Initial_Model_loss_4.png')
    plt.show()

def load_cmu_dataset(path_to_cmu):
    mydictLabels={'myfeaturesLabels':'cmumosi/CMU_MOSI_Opinion_Labels.csd'}
    mydatasetLabels = mmdatasdk.mmdataset(mydictLabels)
    
    audio = path_to_cmu
    actor_folders = os.listdir(audio) # list files in audio directory
    actor_folders.sort()

    emotion = []
    file_path = []

    filename = os.listdir(audio) # iterate over Actor folders
    for f in filename: # go through files in Actor folder
        part = f.split('.')[0].split('_')
        print(part)
        clip = ""
        for i in range(len(part) - 1):
            if i > 0:
                clip += "_"
            clip += part[i]
        segment = int(part[len(part) - 1]) - 1
        print(clip + " " + str(segment))
        if int(mydatasetLabels.computational_sequences['myfeaturesLabels'].data[clip]['features'][segment][0]) < 0:
            emotion.append(0)
        else:
            emotion.append(1)
        # emotion.append(if int(mydatasetLabels.computational_sequences['myfeaturesLabels'].data[clip]['features'][segment][0]) < 0: 0 else: 1)
        print(mydatasetLabels.computational_sequences['myfeaturesLabels'].data[clip]['features'][segment][0])
        file_path.append(audio + '/' + f)
    
    print(emotion)

    print("Loading CMU dataset...")
    log_mels = []
    for index, path in enumerate(file_path):
        # print(path)
        log_spectrogram = convert_audio_to_log_mel_spectrogram(path)
        log_mels.append(log_spectrogram)
    
    log_mels = numpy_fillna(log_mels)
    
    print("Finished loading CMU dataset")

    lb = LabelEncoder()
    emotion_one_hot = to_categorical(lb.fit_transform(emotion))

    # def gen():
    #     for i in range(len(log_mels)):
    #         yield log_mels[i], emotion_one_hot[i]
    
    # dataset = tf.data.Dataset.from_generator(gen, output_signature=(tf.TensorSpec(shape=(128, None), dtype=tf.float32, name=None), tf.TensorSpec(shape=(7,), dtype=tf.float32)))

    # print(dataset)
    # print(dataset.element_spec)
    
    # train_dataset = dataset.take(1650)
    # test_dataset = dataset.skip(1650)
    # batches = train_dataset.padded_batch(8)
    # test_batches = test_dataset.padded_batch(8)

    # cnn = model.CNN(0, 1)

    # log_mels = np.array(log_mels)
    # log_mels = log_mels[:, :, np.newaxis]

    combined = [[log_mels[i], emotion_one_hot[i]] for i in range(len(log_mels))]

    train, test = train_test_split(combined, test_size=0.2, random_state=0)

    x_train = [sample[0] for sample in train]
    y_train = [sample[1] for sample in train]

    x_test = [sample[0] for sample in test]
    y_test = [sample[1] for sample in test]

    dummy_clf = DummyClassifier(strategy="stratified")
    dummy_clf.fit(x_train, y_train)
    DummyClassifier(strategy='stratified')
    dummy_clf.predict(x_test)
    print("dummy: ", str(dummy_clf.score(x_test, y_test)))

    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(x_train, y_train)
    clf.predict(x_test)
    print("tree: ", str(clf.score(x_test, y_test)))

    x_train = np.array(x_train)
    y_train = np.array(y_train)
    x_test = np.array(x_test)
    y_test = np.array(y_test)

    print(x_train)
    print(y_train)

    x_train = x_train[:, :, np.newaxis]
    # y_train = y_train[:, :, np.newaxis]
    x_test = x_test[:, :, np.newaxis]
    # y_test = y_test[:, :, np.newaxis]

    cnn = model_2.CNN(x_train.shape[1], 2)

    checkpoint = ModelCheckpoint("best_initial_model_3.hdf5", monitor='accuracy', verbose=1,
        save_best_only=True, mode='max', period=1, save_weights_only=True)

    # model_history = cnn.model.fit(batches, epochs=40, validation_data=test_batches, callbacks=[checkpoint])
    model_history=cnn.model.fit(x_train, y_train, batch_size=32, epochs=40, validation_data=(x_test, y_test), callbacks=[checkpoint])

    # plot accuracy
    plt.plot(model_history.history['accuracy'])
    plt.plot(model_history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.savefig('Initial_Model_Accuracy_CMU.png')
    plt.show()

    # plot loss
    plt.plot(model_history.history['loss'])
    plt.plot(model_history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig('Initial_Model_loss_CMU.png')
    plt.show()
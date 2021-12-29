import numpy as np
import pandas as pd
import keras
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from matplotlib import pyplot as plt
from tensorflow.keras import callbacks
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping


# Datan lukeminen

def data_lataa():
    data = pd.read_csv('Metsa/src/cover_data.csv', sep=',')
    print(f'Rivit: {data.shape[0]}')
    print(f'Kolumnit: {data.shape[1]}')
    return data

def data_pienenna(data: pd.DataFrame):
    # Pienennetään dataa
    # Poistetaan luokkia    
    data = data[data['class'] != 4]
    data = data[data['class'] != 5]
    data = data[data['class'] != 6]

    # Pienennetään havaintojen määrää poistamalla 1 ja 2 metsätyyppien havaintoja
    data = data.sort_values(by=['class'])
    data = data.iloc[160000:]
    data = data.drop(data.index[51840:280000])

    # uudelleen labeloidaan selittävä muuuttuja
    data['class'] = data['class'].replace([7], 4)

    print(f'Rivit: {data.shape[0]}')
    print(f'Kolumnit: {data.shape[1]}')
    print(data.head())
    print(data.info())
    print(data.describe())

    selittettava_muuttuja = data.iloc[:,-1]
    selittavat_muuttujat = data.iloc[:,:-1]
    return selittettava_muuttuja, selittavat_muuttujat


def data_muokkaa(selittettava_muuttuja: pd.Series, selittavat_muuttujat: pd.DataFrame):
    
    train_selittavat, test_selittavat, train_selitettava, test_selitettava = train_test_split(
        selittavat_muuttujat,
        selittettava_muuttuja,
        test_size = 0.25,
        random_state = 42)

    # numeeristen muutujien normalisointi
    kolumnien_nimet = list(selittavat_muuttujat.iloc[:,0:10])
    kolumnien_muuttaja = ColumnTransformer(
        [('numeric',
        Normalizer(),
        kolumnien_nimet
        )],
        remainder= 'passthrough')

    train_selittavat_norm = kolumnien_muuttaja.fit_transform(train_selittavat)
    test_selittavat_norm = kolumnien_muuttaja.fit_transform(test_selittavat)

    # one-hot enkoodaus selitettävälle muuttujalle
    label_koodari = LabelEncoder()
    train_selitettava_koo = label_koodari.fit_transform(train_selitettava.astype(str))
    test_selitettava_koo = label_koodari.transform(test_selitettava.astype(str))
    train_selitettava_koo = tf.keras.utils.to_categorical(train_selitettava_koo, dtype='int64')
    test_selitettava_koo = tf.keras.utils.to_categorical(test_selitettava_koo, dtype='int64')

    return train_selittavat_norm, test_selittavat_norm, train_selitettava_koo, test_selitettava_koo


def mallin_koonti(train_selittavat_norm : pd.DataFrame, train_selitettava_koo: pd.Series, epochs : int = 300, batch_size : int = 256, learning_rate : float = 0.01):
    
    malli = tf.keras.models.Sequential()
    # input kerros
    malli.add(tf.keras.layers.InputLayer(input_shape=(train_selittavat_norm.shape[1],)))
    # yksi hidden kerros
    malli.add(tf.keras.layers.Dense(16, activation = 'relu'))
    # output kerros
    malli.add(tf.keras.layers.Dense(4, activation = 'softmax'))
    
    opt = Adam(learning_rate = learning_rate)
    malli.compile(
        loss = 'categorical_crossentropy', 
        metrics = 'accuracy', 
        optimizer = opt)
    
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=100)
    history = malli.fit(
        train_selittavat_norm,
        train_selitettava_koo,
        epochs = epochs,
        batch_size = batch_size,
        validation_split = 0.2,
        verbose = 1,
        callbacks = [es])

    print(type(malli))
    print(type(history))

    return malli, history


def mallin_arviointi(test_selittavat_norm : pd.DataFrame, test_selitettava_koo: pd.Series, malli):
    # Mallin arviointi
    loss, acc = malli.evaluate(test_selittavat_norm, test_selitettava_koo, verbose = 1)
    print('categorical crossentropy', loss)
    print('accuracy', acc)

    yhat_classes = np.argmax(malli.predict(test_selittavat_norm), axis = -1)
    y_true = np.argmax(test_selitettava_koo, axis = 1)
    nimet = [
        'Spruce/Fir',
        'Lodgepole Pine',
        'Ponderosa Pine',
        'Krummholz']

    print(classification_report(y_true, yhat_classes, target_names=nimet))


def piirraloss(history, index = 0, hyperparametri = 0):
    if hyperparametri == 0:
        ax = plt.subplot(2,1, index +1)
    else:
        ax = plt.subplot(2, 3, index + 1)
    ax.plot(history.history['loss'], label = 'training loss')
    ax.plot(history.history['val_loss'], label = 'validation loss')
    ax.set_title('Training and validation loss ' + str(hyperparametri))
    ax.set_xlabel('epochs')
    ax.set_ylabel('loss')
    ax.legend()
    return ax

def piirraacc(history, index = 0, hyperparametri = 0):
    if hyperparametri == 0:
        bx = plt.subplot(2, 1, index + 2)
    else:
        bx = plt.subplot(2, 3, index + 4)
    bx.plot(history.history['accuracy'], label = 'training accuracy')
    bx.plot(history.history['val_accuracy'], label = 'validation accuracy')
    bx.set_title('Training and validation accuracy ' + str(hyperparametri))
    bx.set_xlabel('epochs')
    bx.set_ylabel('accuracy')
    bx.legend()
    return bx


def main():
    data = data_lataa()
    selittettava_muuttuja, selittavat_muuttujat = data_pienenna(data)
    train_selittavat_norm, test_selittavat_norm, train_selitettava_koo, test_selitettava_koo = data_muokkaa(selittettava_muuttuja, selittavat_muuttujat)
    malli, history = mallin_koonti(train_selittavat_norm, train_selitettava_koo)
    mallin_arviointi(test_selittavat_norm, test_selitettava_koo, malli)
    ax = piirraloss(history)
    bx = piirraacc(history)
    (ax, bx) = plt.subplots(2,1)
    plt.close(2)
    plt.show()

def hyperparameter():
    batch_size = [128, 256, 512]
    epochs = [200, 400, 600]
    learning_rate = [0.1, 0.01, 0.001]
    data = data_lataa()
    selittettava_muuttuja, selittavat_muuttujat = data_pienenna(data)
    train_selittavat_norm, test_selittavat_norm, train_selitettava_koo, test_selitettava_koo = data_muokkaa(selittettava_muuttuja, selittavat_muuttujat)
    axeslist = []
    bxeslist = []
    for i in range(3):
        malli, history = mallin_koonti(train_selittavat_norm, train_selitettava_koo, batch_size = batch_size[i])
        mallin_arviointi(test_selittavat_norm, test_selitettava_koo, malli)
        ax = piirraloss(history, index=i, hyperparametri=batch_size[i])
        bx = piirraacc(history, index=i, hyperparametri=batch_size[i])
        axeslist.append(ax)
        bxeslist.append(bx)
    fig, ((axeslist[0], axeslist[1]), (axeslist[2],bxeslist[0]), (bxeslist[1], bxeslist[2])) = plt.subplots(3,2)
    plt.close(2)
    plt.show()
    
main()
#hyperparameter()

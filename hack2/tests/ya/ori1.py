import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, GlobalAveragePooling2D, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.layers import RandomRotation
from keras import backend as K


# You will need to set this path according to where you've stored the data set on your machine!
filename = "/data/wbc/A3net/hack2/" + 'cluster_TNG_data.fits'
hdul = fits.open(filename)

image_size = hdul[1].data.shape[1]

train_ind = np.argwhere(hdul[2].data['train'] == 1)
train_X = hdul[1].data[train_ind].reshape(-1, image_size, image_size, 1)
train_Y = hdul[2].data['log_M500'][train_ind]

val_ind = np.argwhere(hdul[2].data['validate'] == 1)
val_X = hdul[1].data[val_ind].reshape(-1, image_size, image_size, 1)
val_Y = hdul[2].data['log_M500'][val_ind]

test_ind = np.argwhere(hdul[2].data['test'] == 1)
test_X = hdul[1].data[test_ind].reshape(-1, image_size, image_size, 1)
test_Y = hdul[2].data['log_M500'][test_ind]

norm = np.nanmin(hdul[2].data['log_M500'])
train_Y -= norm
val_Y -= norm
test_Y -= norm

input_shape = (image_size, image_size, 1)

model = Sequential()
model.add(Conv2D(16, kernel_size=(3, 3), input_shape=input_shape, activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(GlobalAveragePooling2D())
model.add(Dropout(0.1))
model.add(Dense(200, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(100, activation='relu'))
model.add(Dense(20, activation='relu'))
model.add(Dense(1, activation='linear'))

model.compile(loss='mean_squared_error', optimizer=keras.optimizers.Adam())#lr=0.0002

Ss = 0
epochs = 100
batch_size = 16 #lower this value if you get a memory error

for i in range(10):
    
    hist = model.fit(train_X, train_Y, validation_data = (val_X, val_Y), batch_size=batch_size, verbose=True, epochs=epochs)

    prediction = model.predict(test_X, verbose=0, batch_size=batch_size).flatten()
    # Remember when we subtracted off the min in an earlier cell?  In the next line, we're putting it back in!

    truevalue = test_Y + norm
    predvalue = prediction + norm
    truevalue = truevalue.flatten()

    
    for ts, ps in zip(truevalue, predvalue):
        Ss = Ss + (ts - ps)**2

    plt.scatter(test_Y + norm, prediction + norm, c='C0')
    x = np.linspace(np.min(test_Y+norm), np.max(test_Y+norm), 100)
    plt.plot(x,x,ls='--', c='C1')

    plt.xlabel('True '+r'$\log\left(M_{500c}\right)$')
    plt.ylabel('Predicted '+r'$\log\left(M_{500c}\right)$')

MSE = Ss / len(truevalue) / 10

plt.title('Results (with validation) after {:.0f} epochs, MSE={:.4f}'.format(epochs, MSE)) 
plt.scatter([], [], c='C0', label='model predictions')
plt.plot([], [], label='one-to-one line', c='C1', ls='--')
plt.legend()
plt.savefig('/data/wbc/A3net/hack2/tests/ya/or1.png')
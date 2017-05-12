import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import SGD

x = np.array([[0,0], [0,1], [1,0], [1,1]])
y = np.array([[0], [1], [1], [0]])

model = Sequential()
model.add(Dense(2, input_shape=(2,)))  
model.add(Activation('sigmoid'))
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='mean_squared_error', optimizer=SGD(lr=0.02), metrics=['accuracy'])

model.summary()

model.fit(x,y, epochs=100000, batch_size=4)

model.predict(x, verbose=1)


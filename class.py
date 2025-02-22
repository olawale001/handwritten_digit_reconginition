import keras
import numpy as np
import tensorflow as tf
from tensorflow import Keras
from tensorflow.keras.model import Sequential
from tensorflow.keras.layers import Conv2d, MaxPooling2d, Flatten, Dense, Dropout
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt


(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train / 255.0
x_test = x_test /255.0

x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)

y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)


model = Sequential([
    Conv2d(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2d(pool_size=(2,2)),
    Conv2d(64, (3,3), activation='relu'),
    MaxPooling2d(pool_size=(2,2)),
    Flatten(),
    Dropout(0.5),
    Dense(10, activation='softmax')
])

model.compile(optimizer = 'adam', loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit(x_train, y_train, epoch=15, validation_data=(x_test, y_test), batch_size=64)
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"Test Accuracy: {test_acc:.4f}")

prediction = model.predict(x_test)

plt.figure(fig_size=(10, 5))
for i in range(5):
    plt.subplot(1, 5, i+1)

plt.imshow(x_test[i].reshape(28, 28), cmap='gray')
plt.title(f"pred: {np.argmax(prediction[i])}")
plt.axes('off')
plt.show()

model.save("mnist_cnn_model.h5")

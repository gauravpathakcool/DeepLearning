import tensorflow as tf
from keras_preprocessing.image import ImageDataGenerator
from keras.callbacks import TensorBoard
from time import time
import os
import PreProcessing

tboard_log_dir = os.path.join("logs",format(time()))
custom_callback = TensorBoard(log_dir=tboard_log_dir)

train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)
train_generator = train_datagen.flow_from_directory(
    directory='C:/Gaurav/DeepLearning/images/Train_resize_Grey',
    target_size=(100, 100),
    shuffle=True,
    seed=42,
    color_mode="grayscale",
    class_mode='binary')

validation_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
)
validation_generator = validation_datagen.flow_from_directory(
    directory='C:/Gaurav/DeepLearning/images/Validation_Resize_Grey',
    target_size=(100, 100),
    shuffle=True,
    color_mode="grayscale",
    seed=42,
    class_mode='binary')

# Create the model
init = tf.keras.initializers.glorot_uniform(seed=1)
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3),
                                 strides=(1, 1), padding='same', kernel_initializer=init,
                                 activation='relu', input_shape=(100, 100, 1)))
model.add(tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3),
                                 strides=(1, 1), padding='same', kernel_initializer=init,
                                 activation='relu'))
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
model.add(tf.keras.layers.Dropout(0.2))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(units=100, kernel_initializer=init,
                                activation='relu'))
model.add(tf.keras.layers.Dropout(0.2))
model.add(tf.keras.layers.Dense(units=2, kernel_initializer=init,
                                activation='softmax'))

# Compile model
epochs = 20
lrate = 0.01
decay = lrate / epochs
sgd = tf.keras.optimizers.SGD(lr=lrate, momentum=0.5, decay=decay, nesterov=False)
#model.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam, metrics=['accuracy'])
model.compile(loss=tf.keras.losses.sparse_categorical_crossentropy,
              optimizer=sgd,
              metrics=['accuracy'])
model.summary()

# Fit the model
model.fit_generator(
    generator=train_generator,
    epochs=epochs,
    validation_data=validation_generator,
   # verbose=2,callbacks=[custom_callback]
)

##save model
# serialize model to JSON
model_json = model.to_json()
with open("cat_dog_model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save("cat_dog_model.h5")
print("Saved model to disk")


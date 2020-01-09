# load model
import tensorflow as tf
import cv2
import numpy as np
import time

model = tf.keras.models.load_model('C:/Gaurav/DeepLearning/cat_dog_model.h5')
# model = tf.keras.models.load_model('C:/Gaurav/DeepLearning/converted_keras (1)/keras_model.h5')

# cap = cv2.VideoCapture(1)
Dict = dict([(0, 'cat'), (1, 'dog')])
# while True:
#     # Capture frame-by-frame
#     ret, frame = cap.read()
#
#     # Display the resulting frame
#     if ret:
#         cv2.imshow('frame', frame)
#         img = cv2.resize(frame, (224, 224))
#         img = np.expand_dims(img, axis=0)
#         img = tf.cast(img, tf.float32)
#         img = (img / 255)
#        # print(model.predict(img))
#         #print(np.amax(model.predict(img)))
#         if np.amax(model.predict(img)) > .98:
#             print(Dict[np.argmax(model.predict(img))])
#         else:
#             print("uidentified!!")
#     else:
#         print("not able to capture frame")
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
#         # When everything done, release the capture
#         cap.release()
#         cv2.destroyAllWindows()

#
#
# # prediction
img = cv2.imread("C:/Gaurav/DeepLearning/images/test/000014.jpg", 0)
img = cv2.resize(img, (100, 100))
img = np.array(img)
img = img / 255
print(img.shape)
img = np.transpose(img,(1,0,))
print(img.shape)
img = np.expand_dims(img, axis=0)
img = np.expand_dims(img, axis=3)
print(img.shape)
print(model.predict(img))
print(model.predict_classes(img))
print(Dict[np.argmax(model.predict(img))])

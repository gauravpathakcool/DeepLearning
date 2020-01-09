import os
import cv2
import shutil

from sklearn.model_selection import train_test_split

classes = ["cat", "dog"]

root_dir = "C:/Gaurav/DeepLearning/images"
train_path = "C:/Gaurav/DeepLearning/images/Train"
validation_path = "C:/Gaurav/DeepLearning/images/Validation"


# def split_train_validation():
#     Class1Files = os.listdir(train_path + "/" + classes[0])
#     Class2Files = os.listdir(train_path + "/" + classes[1])
#     Class1_split_ratio = len(Class1Files) * .75
#     Class2_split_ratio = len(Class2Files) * .75
#     counter = 0
#     if not os.exists(train_path):
#         os.makedirs(train_path)
#         if not os.exists(train_path + "/" + classes[0]):
#             os.makedirs(train_path + "/" + classes[0])
#         if not os.exists(train_path + "/" + classes[1]):
#             os.makedirs(train_path + "/" + classes[1])
#             for fi in os.listdir(root_dir):
#                 for f in os.listdir(os.path.join(root_dir, fi)):
#                     shutil.move(f, train_path + "/" + classes[0])
#                     counter = counter + 1
#                     if counter > Class1_split_ratio:
#                         break
#
#     if not os.exists(validation_path):
#         os.makedirs(validation_path)
#         if not os.exists(validation_path + "/" + classes[0]):
#             os.makedirs(validation_path + "/" + classes[0])
#         if not os.exists(validation_path + "/" + classes[1]):
#             os.makedirs(validation_path + "/" + classes[1])


###pre processing of data
FileList = os.listdir(train_path)
for f in FileList:
    for fi in os.listdir(train_path + "/" + f):
        try:
            im = cv2.imread(train_path + "/" + f + "/" + fi)
            img = cv2.resize(im, (100, 100))
            grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            if not os.path.exists(train_path + "_Resize_Grey/" + f):
                os.mkdir(train_path + "_Resize_Grey/" + f)
            cv2.imwrite(train_path + "_Resize_Grey/" + f + "/" + fi, grey)
        except:
            pass

FileListVal = os.listdir(validation_path)
for f in FileListVal:
    for fi in os.listdir(validation_path + "/" + f):
        try:
            im = cv2.imread(validation_path + "/" + f + "/" + fi)
            img = cv2.resize(im, (100, 100))
            grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            if not os.path.exists(validation_path + "_Resize_Grey/" + f):
                os.mkdir(validation_path + "_Resize_Grey/" + f)
            cv2.imwrite(validation_path + "_Resize_Grey/" + f + "/" + fi, grey)
        except:
            pass


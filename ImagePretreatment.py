from PIL import Image, ImageFile
import os, glob
import numpy as np

# IOErrorを防ぐため
ImageFile.LOAD_TRUNCATED_IMAGES = True

# 識別する種類を選択
classes = ["cat", "dog"]  
num_classes = len(classes)  # クラスの数
image_size = 64             # イメージサイズ
num_testdata = 250            # テストデータの数

train_images = []
test_images = []
train_labels = []
test_labels = []

for index, classlabel in enumerate(classes):
    files = glob.glob("./" + classlabel + "/*.jpg")
    for i, file in enumerate(files):
        
        try:
            image = Image.open(file)
            image = image.convert("RGB")    #RGB変換（グレスケに変換もあり？）
            image = image.resize((image_size, image_size))
            data = np.asarray(image)
            if i < num_testdata:
                test_images.append(data)
                test_labels.append(index)
            else:
                train_images.append(data)
                train_labels.append(index)
        except Exception as e:
            print(f"Error processing file {file}: {e}")

train_images = np.array(train_images)
train_labels = np.array(train_labels)
test_images = np.array(test_images)
test_labels = np.array(test_labels)

print(train_images.shape)
print(train_labels.shape)
print(test_images.shape)
print(test_labels.shape)

# データセットを保存
np.savez("animal_dataset.npz", train_images=train_images, test_images=test_images, train_labels=train_labels, test_labels=test_labels)

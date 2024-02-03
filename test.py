import keras
import numpy as np
from PIL import Image
from keras.models import load_model

# 画像サイズの定義
image_size = (224, 224)

# Google Colabでの画像アップロード方法についての説明（コメント）

# テストする画像とモデルのパス
test_image_path = "./dog.jpg"
trained_model_path = "./resnet152_model.h5"

def preprocess_image(image_path):
    """
    画像を読み込み、前処理する関数
    """
    img = Image.open(image_path)
    img = img.convert('RGB')
    img = img.resize(image_size)
    img = np.asarray(img)
    img = img / 255.0
    return img

# 学習済みモデルの読み込み
model = load_model(trained_model_path)

# 画像の前処理とモデルを使った予測
processed_img = preprocess_image(test_image_path)
prediction = model.predict(np.array([processed_img]))
print(prediction) # 予測確率の表示

# 予測されたクラスの表示
predicted_class = np.argmax(prediction, axis=1)
if predicted_class == 0:
    print(">>> 犬")
elif predicted_class == 1:
    print(">>> 猫")

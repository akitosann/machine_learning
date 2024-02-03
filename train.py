from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Activation, Dropout, Flatten, Dense, BatchNormalization
from tensorflow.keras.optimizers import RMSprop, Adam
from keras.utils import np_utils
from keras.callbacks import Callback, EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
import numpy as np

# クラスの定義：犬と猫
class_labels = ["dog", "cat"]
num_classes = len(class_labels)
image_size = 64

class ProgressCallback(Callback):
    def on_epoch_begin(self, epoch, logs=None):
        print(f"開始 エポック {epoch+1}/{self.params['epochs']}")

    def on_epoch_end(self, epoch, logs=None):
        print(f"終了 エポック {epoch+1}/{self.params['epochs']}")
        print(f"損失: {logs['loss']:.4f}, 精度: {logs['accuracy']:.4f}")

def load_dataset():
    """
    データセットを読み込み、前処理を行う関数
    """
    # .npzファイルをロード
    data = np.load("animal_dataset.npz", allow_pickle=True)
    
    # 各配列を正しく抽出
    train_images = data['train_images']
    test_images = data['test_images']
    train_labels = data['train_labels']
    test_labels = data['test_labels']

    # 画像データの画素値を0-1に正規化
    train_images = train_images.astype("float") / 255
    test_images = test_images.astype("float") / 255

    # ラベルデータをone-hotベクトルに変換
    train_labels = np_utils.to_categorical(train_labels, num_classes)
    test_labels = np_utils.to_categorical(test_labels, num_classes)

    return train_images, train_labels, test_images, test_labels

def build_and_train_model(train_images, train_labels, test_images, test_labels):
    """
    CNNモデルを構築し、訓練する関数
    """
    model = Sequential()

    # CNNモデルの構築
    model.add(Conv2D(32, (3, 3), padding='same', input_shape=train_images.shape[1:]))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.1))

    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    # 全結合層
    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))

    # 出力層
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))

    # モデルのコンパイル
    optimizer = Adam(learning_rate=0.001)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    # データ拡張の設定
    datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True)

    # モデルの訓練
    early_stopping = EarlyStopping(monitor='val_loss', patience=5)
    progress_callback = ProgressCallback()
    model.fit(datagen.flow(train_images, train_labels, batch_size=32),
              steps_per_epoch=len(train_images) / 32, epochs=50,
              validation_data=(test_images, test_labels),
              callbacks=[progress_callback, early_stopping])

    # モデルをHDF5ファイルに保存
    model.save('./cnn_model.h5')

    return model


"""
メイン関数：データセットの読み込みとモデルの訓練
"""
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

train_images, train_labels, test_images, test_labels = load_dataset()

model = build_and_train_model(train_images, train_labels, test_images, test_labels)


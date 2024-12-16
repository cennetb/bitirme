import os

import matplotlib.pyplot as plt
import numpy as np
from keras import Model
from keras.src.legacy.preprocessing.image import ImageDataGenerator
from keras.src.utils import load_img, img_to_array
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Input, concatenate, Conv2DTranspose

# Klasör yolları
data_dir = './data'
train_dir = os.path.join(data_dir, 'train')
test_dir = os.path.join(data_dir, 'test')
model_path = './bitirme/cilt_kanseri_tespit_modeli.keras'

# Görsel boyutu
IMG_SIZE = (256, 256, 1)

# U-Net Modeli Tanımı
def unet_model():
    inputs = Input(IMG_SIZE)

    c1 = Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    c1 = Conv2D(64, (3, 3), activation='relu', padding='same')(c1)
    p1 = MaxPooling2D((2, 2))(c1)

    c2 = Conv2D(128, (3, 3), activation='relu', padding='same')(p1)
    c2 = Conv2D(128, (3, 3), activation='relu', padding='same')(c2)
    p2 = MaxPooling2D((2, 2))(c2)

    c3 = Conv2D(256, (3, 3), activation='relu', padding='same')(p2)
    c3 = Conv2D(256, (3, 3), activation='relu', padding='same')(c3)
    p3 = MaxPooling2D((2, 2))(c3)

    u4 = Conv2DTranspose(256, (3, 3), strides=(2, 2), padding='same')(p3)
    u4 = concatenate([u4, c3])
    c4 = Conv2D(256, (3, 3), activation='relu', padding='same')(u4)
    c4 = Conv2D(256, (3, 3), activation='relu', padding='same')(c4)

    u5 = Conv2DTranspose(128, (3, 3), strides=(2, 2), padding='same')(c4)
    u5 = concatenate([u5, c2])
    c5 = Conv2D(128, (3, 3), activation='relu', padding='same')(u5)
    c5 = Conv2D(128, (3, 3), activation='relu', padding='same')(c5)

    u6 = Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same')(c5)
    u6 = concatenate([u6, c1])
    c6 = Conv2D(64, (3, 3), activation='relu', padding='same')(u6)
    c6 = Conv2D(64, (3, 3), activation='relu', padding='same')(c6)

    outputs = Conv2D(1, (1, 1), activation='sigmoid')(c6)

    model = Model(inputs, outputs)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

model = unet_model()

# Veri yükleyici
train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir, target_size=(256, 256), color_mode='grayscale', batch_size=32, class_mode='input')

test_generator = test_datagen.flow_from_directory(
    test_dir, target_size=(256, 256), color_mode='grayscale', batch_size=32, class_mode='input')

# Model eğitimi ve tarihçesi
history = model.fit(train_generator, epochs=5, validation_data=test_generator)

# Modeli kaydet
model.save(model_path)  # Modeli .keras formatında kaydet

# Eğitim ve doğrulama doğruluğu grafiği
plt.figure(figsize=(12, 6))

# Doğruluk
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

# Kayıp
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()

# Test veri setinden bir görüntü al
test_image_path = os.path.join(test_dir, './skin_cancer/skin_cancer_117.jpg')  # Örnek bir test görseli yolu
img = load_img(test_image_path, target_size=(256, 256), color_mode='grayscale')  # Görüntüyü yükle
img_array = img_to_array(img) / 255.0  # Görüntüyü numpy dizisine çevir ve normalize et
img_array = np.expand_dims(img_array, axis=0)  # Model için uygun hale getirmek için boyutunu genişlet

# Segmentasyonu tahmin et
predicted_mask = model.predict(img_array)  # Modelin tahminini al

# Segmentasyon maskesini (beyaz=kanserli alan, siyah=sağlıklı) görselleştir
plt.figure(figsize=(12, 6))

# Orijinal görüntü
plt.subplot(1, 2, 1)
plt.imshow(img, cmap='gray')
plt.title('Original Image')
plt.axis('off')

# Segmentasyon maskesi
plt.subplot(1, 2, 2)
plt.imshow(predicted_mask[0], cmap='gray')  # predicted_mask[0] çünkü modelin çıktısı batch olarak gelir
plt.title('Predicted Segmentation Mask')
plt.axis('off')

plt.tight_layout()
plt.show()
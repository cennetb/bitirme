import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Veri yükleme fonksiyonu
def load_data(image_dir, mask_dir):
    images, masks = [], []
    file_names = sorted(os.listdir(image_dir))
    for file_name in file_names:
        image_path = os.path.join(image_dir, file_name)
        mask_file_name = file_name.replace("X_img", "Y_img")
        mask_path = os.path.join(mask_dir, mask_file_name)
        if os.path.exists(image_path) and os.path.exists(mask_path):
            image = tf.keras.preprocessing.image.load_img(image_path, target_size=(224, 224))
            mask = tf.keras.preprocessing.image.load_img(mask_path, target_size=(224, 224), color_mode='grayscale')
            images.append(tf.keras.preprocessing.image.img_to_array(image) / 255.0)
            masks.append(tf.keras.preprocessing.image.img_to_array(mask) / 255.0)
        else:
            print(f"Eksik dosya: {file_name} veya {mask_file_name}")
    return np.array(images), np.array(masks)

# Veri yolları
image_dir = "ph2/trainx"
mask_dir = "ph2/trainy"
X, Y = load_data(image_dir, mask_dir)

# Eğitim ve test verilerini ayır
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# U-Net benzeri model tanımlama
def unet_model():
    inputs = layers.Input((224, 224, 3))

    c1 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    c1 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(c1)
    p1 = layers.MaxPooling2D((2, 2))(c1)

    c2 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(p1)
    c2 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(c2)
    p2 = layers.MaxPooling2D((2, 2))(c2)

    u2 = layers.UpSampling2D((2, 2))(p2)
    u2 = layers.Concatenate()([u2, c2])
    c3 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(u2)
    c3 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(c3)

    u3 = layers.UpSampling2D((2, 2))(c3)
    u3 = layers.Concatenate()([u3, c1])
    c4 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(u3)
    c4 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(c4)

    outputs = layers.Conv2D(1, (1, 1), activation='sigmoid')(c4)

    model = models.Model(inputs, outputs)
    return model

model = unet_model()
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

# Modeli eğit
history = model.fit(X_train, Y_train, epochs=3, validation_data=(X_test, Y_test), batch_size=8)

# Modeli test et
def display_results(model, X_samples, threshold=0.4):
    predictions = model.predict(X_samples)
    plt.figure(figsize=(15, 5))

    for i in range(3):
        plt.subplot(2, 3, i + 1)
        plt.title("Orijinal")
        plt.imshow(X_samples[i])
        plt.axis('off')

        # Eşik uygulama işlemi
        binary_prediction = (predictions[i] > threshold).astype(np.uint8)

        plt.subplot(2, 3, i + 4)
        plt.title("Tahmin (Binary)")
        plt.imshow(binary_prediction.squeeze(), cmap='gray')
        plt.axis('off')

    plt.show()

# Sonuçları göster
loss, accuracy = model.evaluate(X_test, Y_test)
print(f"Test Kayıp: {loss:.4f}, Test Doğruluk: {accuracy:.4f}")

display_results(model, X_test)
# Sonuçları göster
loss, accuracy = model.evaluate(X_test, Y_test)
print(f"Test Kayıp: {loss:.4f}, Test Doğruluk: {accuracy:.4f}")

display_results(model, X_test)

import cv2
import numpy as np
import os
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Veri seti yolu
data_dir = "C:/Users/ASUS/Desktop/bitirme/train/train/"
categories = ["skin_cancer", "not_skin_cancer"]

# Görselleri ve etiketleri saklamak için listeler
images = []
labels = []

# Örnek bir görseli kontrol etme
image_path = "C:/Users/ASUS/Desktop/bitirme/train/train/skin_cancer/skin_cancer_117.jpg"
image = cv2.imread(image_path)
if image is None:
    print(f"'{image_path}' yolundan görüntü yüklenemedi. Yol veya dosya adı hatalı olabilir.")
else:
    print("Görüntü başarıyla yüklendi.")

# Veri setini yükleme
for category in categories:
    path = os.path.join(data_dir, category)
    label = categories.index(category)  # 0: skin_cancer, 1: not_skin_cancer
    for img in os.listdir(path):
        try:
            img_path = os.path.join(path, img)
            image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # Gri tonlama
            image = cv2.resize(image, (128, 128))  # Görselleri yeniden boyutlandırma
            images.append(image)
            labels.append(label)
        except Exception as e:
            print(f"Görsel yüklenirken hata oluştuu: {e}")




# Görselleri ve etiketleri numpy array'e dönüştürme
images = np.array(images).reshape(-1, 128, 128, 1) / 255.0  # Normalizasyon
labels = np.array(labels)

# Eğitim ve test verisi olarak ayırma
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

# CNN Modeli
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 1)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(2, activation='softmax')  # 2 sınıf: skin_cancer ve not_skin_cancer
])

# Modeli derleme
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Modeli eğitme
history = model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test), batch_size=32)

# Eğitim sonuçlarını görselleştirme
plt.plot(history.history['accuracy'], label='Eğitim Doğruluğu')
plt.plot(history.history['val_accuracy'], label='Doğrulama Doğruluğu')
plt.xlabel('Epochs')
plt.ylabel('Doğruluk')
plt.legend()
plt.title('Eğitim ve Doğrulama Doğruluğu')
plt.show()

plt.plot(history.history['loss'], label='Eğitim Kaybı')
plt.plot(history.history['val_loss'], label='Doğrulama Kaybı')
plt.xlabel('Epochs')
plt.ylabel('Kayıp')
plt.legend()
plt.title('Eğitim ve Doğrulama Kaybı')
plt.show()

# Modeli kaydetme
model.save("cilt_kanseri_tespit_modeli.h5")

# Tahmin fonksiyonu
def tahmin_yap(image_path):
    try:
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            return f"Görüntü yüklenemedi: {image_path}"
        image = cv2.resize(image, (128, 128)) / 255.0
        image = image.reshape(1, 128, 128, 1)
        prediction = model.predict(image)
        class_index = np.argmax(prediction)
        class_name = categories[class_index]
        return class_name
    except Exception as e:
        return f"Hata oluştu: {e}"

# Örnek tahmin
sonuc = tahmin_yap("C:/Users/ASUS/Desktop/bitirme/train/train/skin_cancer/skin_cancer_118.jpg")
print(f"Tahmin edilen sınıf: {sonuc}")


import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf



# Eğitilmiş modeli yükleme fonksiyonu
def load_model(model_path):
    return tf.keras.models.load_model(model_path)


# Modeli yükle
model = load_model('unet_model.keras')


# Görseli yükleme ve işleme fonksiyonu
def preprocess_image(image_path):
    image = tf.keras.preprocessing.image.load_img(image_path,
                                                  target_size=(180, 180))  # Görseli belirtilen boyutta yükle
    image = tf.keras.preprocessing.image.img_to_array(image) / 255.0  # Görseli numpy dizisine çevir ve normalize et
    return np.expand_dims(image, axis=0), image  # Model giriş boyutu için ek boyut ekler


# Tahmin yapma ve sonucu görselleştirme fonksiyonu
def predict_and_display(model, image_path, threshold=0.4):
    # Görseli işle
    input_image, original_image = preprocess_image(image_path)

    # Model ile tahmin yapma
    prediction = model.predict(input_image)

    # Prediction değişkeninin kontrolü ve boyutunu yazdırma
    print("Prediction Shape:", prediction.shape)  # Tahmin çıktısının boyutunu kontrol et

    # Eşikleme işlemi: Prediction'ı threshold ile karşılaştırıp ikili (binary) formata dönüştür
    binary_prediction = (prediction[0] > threshold).astype(np.uint8)

    # Görselleri yan yana göster
    plt.figure(figsize=(10, 5))

    # Orijinal görseli göster
    plt.subplot(1, 2, 1)
    plt.title("Orijinal Görsel")
    plt.imshow(original_image)  # Orijinal görseli göster
    plt.axis('off')

    # Segmentasyon sonucunu göster
    plt.subplot(1, 2, 2)
    plt.title("Segmentasyon Sonucu")
    plt.imshow(binary_prediction.squeeze(), cmap='gray')  # Segmentasyon sonucunu gri tonlamalı göster
    plt.axis('off')

    plt.tight_layout()
    plt.show()


# Test görselinin yolunu belirleyin
image_path = 'ph2/trainx/X_img_8.bmp'

# Fonksiyonu çağırarak tahmin yapın ve görselleştirme işlemi yapın
predict_and_display(model, image_path)

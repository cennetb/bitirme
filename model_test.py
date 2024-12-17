import matplotlib.pyplot as plt
import numpy as np
from keras.src.saving import load_model
from keras.src.utils import load_img, img_to_array

# Model yolu (eğitilmiş modelin kaydedildiği yol)
model_path = './bitirme/cilt_kanseri_tespit_modeli.keras'

# Test görselinin yolu
test_image_path = './data/test/skin_cancer/skin_cancer_117.jpg'  # Test görselinin yolu

# Görsel boyutu
IMG_SIZE = (256, 256)

# Eğitilmiş modeli yükle
model = load_model(model_path)  # Modeli yükle

# Test görselini yükle ve ön işleme
img = load_img(test_image_path, target_size=IMG_SIZE, color_mode='grayscale')  # Görüntüyü yükle
img_array = img_to_array(img) / 255.0  # Görüntüyü normalize et
img_array = np.expand_dims(img_array, axis=0)  # Model için uygun hale getirmek için boyutunu genişlet

# Segmentasyonu tahmin et
predicted_mask = model.predict(img_array)

# Segmentasyonu orijinal görüntü ile birleştir
predicted_mask = predicted_mask[0]  # İlk (ve tek) batch'in tahminini al
#predicted_mask = (predicted_mask > 0.5).astype(np.uint8)  # Maskeyi ikili (binary) hale getir

# Maskeyi orijinal görüntüye uygun şekilde birleştir
segmented_image = img_array[0] * predicted_mask  # Maskeyi orijinal görüntü ile çarp

# Görselleştir
plt.figure(figsize=(12, 6))

# Orijinal görüntü
plt.subplot(1, 3, 1)
plt.imshow(img, cmap='gray')
plt.title('Original Image')
plt.axis('off')

# Segmentasyon maskesi
plt.subplot(1, 3, 2)
plt.imshow(predicted_mask, cmap='gray')
plt.title('Predicted Segmentation Mask')
plt.axis('off')

# Segmentasyon maskesi ile orijinal görüntüyü birleştirilmiş hali
plt.subplot(1, 3, 3)
plt.imshow(segmented_image, cmap='gray')
plt.title('Segmented Region')
plt.axis('off')

plt.tight_layout()
plt.show()

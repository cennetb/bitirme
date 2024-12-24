from PIL import Image
import os

# Girdi ve çıktı klasörleri
input_folder = 'data_cnn/test/not_skin_cancer'  # Orijinal görsellerin bulunduğu klasör
output_folder = 'data_cnn/test/not_skin_cancer_rs'  # Yeniden boyutlandırılmış görsellerin kaydedileceği klasör

# Hedef boyut
target_size = (64, 64)

# Çıktı klasörü yoksa oluştur
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Girdi klasöründeki tüm dosyaları işle
for filename in os.listdir(input_folder):
    if filename.endswith(('.png', '.jpg', '.jpeg')):  # Görsel formatlarını kontrol et
        img_path = os.path.join(input_folder, filename)
        img = Image.open(img_path)
        img_resized = img.resize(target_size)  # Görseli yeniden boyutlandır

        # Yeniden boyutlandırılmış görseli kaydet
        output_path = os.path.join(output_folder, filename)
        img_resized.save(output_path)

print(
    f'Görseller {target_size[0]}x{target_size[1]} boyutuna yeniden boyutlandırıldı ve "{output_folder}" klasörüne kaydedildi.')

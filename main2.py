import numpy as np
import os
import tensorflow.keras.preprocessing.image
import matplotlib.pyplot as plt
from keras.src.legacy.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
import tensorflow.keras.models

# Veri artırma işlemleri
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

test_datagen = ImageDataGenerator(rescale=1. / 255)

# Eğitim ve test verilerini yükleme
train_generator = train_datagen.flow_from_directory(
    'bitirme/dataset/train',  # Kendi dataset yolunu buraya yazın
    target_size=(128, 128),
    batch_size=32,
    class_mode='binary'
)

validation_generator = test_datagen.flow_from_directory(
    'train1/train',  # Kendi dataset yolunu buraya yazın
    target_size=(128, 128),
    batch_size=32,
    class_mode='binary'
)


# U-Net Modeli
def unet_model(input_size=(128, 128, 3)):
    inputs = layers.Input(input_size)

    # Encoder kısmı: Konvolüsyonel katmanlar ve max pooling
    conv1 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    pool1 = layers.MaxPooling2D((2, 2))(conv1)

    conv2 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(pool1)
    pool2 = layers.MaxPooling2D((2, 2))(conv2)

    conv3 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(pool2)
    pool3 = layers.MaxPooling2D((2, 2))(conv3)

    # Decoder kısmı: Up-sampling ve konvolüsyonlar
    up1 = layers.UpSampling2D((2, 2))(pool3)
    conv4 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(up1)

    up2 = layers.UpSampling2D((2, 2))(conv4)
    conv5 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(up2)

    # Çıktı katmanı
    output = layers.Conv2D(1, (1, 1), activation='sigmoid')(conv5)

    model = models.Model(inputs, output)
    return model


# Modeli oluşturma
model = unet_model()

# Modeli derleme
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Modeli eğitme
history = model.fit(
    train_generator,
    steps_per_epoch=100,  # her epoch'ta kaç adımda eğitim yapılacak
    epochs=50,
    validation_data=validation_generator,
    validation_steps=50  # her epoch'ta kaç adımda doğrulama yapılacak
)

# Modelin performansını değerlendirme
test_loss, test_acc = model.evaluate(validation_generator)
print('Test Loss:', test_loss)
print('Test Accuracy:', test_acc)

# Modeli kaydetme
model.save('skin_lesion_model.h5')

# Modeli yükleme
model = load_model('skin_lesion_model.h5')


# Görüntü ve segmentasyon maskesini görselleştirme
def plot_image_and_mask(image, true_mask, predicted_mask):
    fig, ax = plt.subplots(1, 3, figsize=(12, 6))

    ax[0].imshow(image)
    ax[0].set_title("Input Image")
    ax[0].axis('off')

    ax[1].imshow(true_mask, cmap='gray')
    ax[1].set_title("True Mask")
    ax[1].axis('off')

    ax[2].imshow(predicted_mask, cmap='gray')
    ax[2].set_title("Predicted Mask")
    ax[2].axis('off')

    plt.show()


# Test verisi üzerinde segmentasyon yapma
for i in range(5):
    image = validation_generator[i][0]  # test görseli
    true_mask = validation_generator[i][1]  # gerçek maske
    predicted_mask = model.predict(image)  # tahmin edilen maske

    plot_image_and_mask(image[0], true_mask[0], predicted_mask[0])

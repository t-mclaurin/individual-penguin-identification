# utils/augmentation.py

from tensorflow.keras.preprocessing.image import ImageDataGenerator

def get_static_augmentation():
    return ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=30,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        fill_mode='nearest'
    )


def get_progressive_augmentation(epoch, max_epoch):
    """Returns ImageDataGenerator with augmentation that scales up with the epoch."""
    scale = min(epoch / max_epoch, 1.0)  # scale from 0.0 to 1.0

    return ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=scale * 30,
        width_shift_range=scale * 0.1,
        height_shift_range=scale * 0.1,
        shear_range=scale * 0.1,
        zoom_range=scale * 0.1,
        fill_mode='nearest'
    )
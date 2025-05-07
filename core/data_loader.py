from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

def load_fer2013(data_dir, batch_size=64, validation_split=0.1):
    train_dir = os.path.join(data_dir, 'train')
    test_dir = os.path.join(data_dir, 'test')

    if not os.path.exists(train_dir) or not os.path.exists(test_dir):
        raise FileNotFoundError(f"Train or test directory not found in {data_dir}")

    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        validation_split=validation_split
    )
    test_datagen = ImageDataGenerator(rescale=1. / 255)

    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(48, 48),
        color_mode='grayscale',
        batch_size=batch_size,
        class_mode='categorical',
        subset='training'
    )
    validation_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(48, 48),
        color_mode='grayscale',
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation'
    )
    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=(48, 48),
        color_mode='grayscale',
        batch_size=batch_size,
        class_mode='categorical'
    )

    return train_generator, validation_generator, test_generator
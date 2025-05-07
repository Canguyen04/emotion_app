import os
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from emotion_recognition.core.data_loader import load_fer2013
from emotion_recognition.core.model import build_emotion_model

def train(data_dir, output_model='emotion_model.h5', epochs=30, batch_size=64):
    train_generator, validation_generator, test_generator = load_fer2013(data_dir, batch_size=batch_size)
    model = build_emotion_model()

    # Kiểm tra và tạo thư mục đầu ra
    output_dir = os.path.dirname(output_model)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    callbacks = [
        ModelCheckpoint(output_model, save_best_only=True, monitor='val_accuracy'),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3)
    ]
    history = model.fit(
        train_generator,
        validation_data=validation_generator,
        epochs=epochs,
        callbacks=callbacks
    )
    return history

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Train emotion model')
    parser.add_argument('--data', type=str, required=True, help='Path to data directory (containing train/test)')
    parser.add_argument('--out', type=str, default='emotion_model.h5', help='Path to save model')
    parser.add_argument('--epochs', type=int, default=30, help='Number of epochs')
    parser.add_argument('--batch', type=int, default=64, help='Batch size')
    args = parser.parse_args()
    train(args.data, args.out, args.epochs, args.batch)
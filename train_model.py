import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import RMSprop

# Define data generators with data augmentation
train_datagen = ImageDataGenerator(rescale=1./255,
                                   rotation_range=40,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

train_dataset = train_datagen.flow_from_directory(
    'dataset/training/',
    target_size=(200, 200),
    batch_size=32,
    class_mode='categorical')

validation_dataset = test_datagen.flow_from_directory(
    'dataset/validation/',
    target_size=(200, 200),
    batch_size=32,
    class_mode='categorical')

# Define the model architecture
model = Sequential([
    Conv2D(16, (3, 3), activation='relu', input_shape=(200, 200, 3)),
    MaxPooling2D(2, 2),
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(512, activation='relu'),
    Dense(5, activation='softmax')  # 5 output classes
])

# Compile the model
model.compile(loss='categorical_crossentropy',
              optimizer=RMSprop(learning_rate=0.001),
              metrics=['accuracy'])

# Train the model
history = model.fit(train_dataset,
                    steps_per_epoch=32,
                    epochs=50,
                    validation_data=validation_dataset,
                    validation_steps=32)

# Save the model
model.save('plant_disease_model.weights.h5')

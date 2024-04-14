import os
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import RMSprop


# Load the trained model
model = tf.keras.models.load_model('c1plant_disease_model.weights.h5')

# Define data generator for test data
test_datagen = ImageDataGenerator(rescale=1./255)

test_dataset = test_datagen.flow_from_directory(
    'dataset/test/',
    target_size=(200, 200),
    batch_size=32,
    class_mode='categorical')

# Compile the model
model.compile(loss='categorical_crossentropy',
              optimizer=RMSprop(learning_rate=0.001),
              metrics=['accuracy'])

# Evaluate the model on test data
loss, accuracy = model.evaluate(test_dataset)

# Print the evaluation results
print("Test Accuracy: {:.2f}%".format(accuracy * 100))
print("Test Loss: {:.4f}".format(loss))

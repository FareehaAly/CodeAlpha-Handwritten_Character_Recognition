import os
import zipfile
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing import image
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

# Zip Path and Zip extraction
zip_path = '/content/archive (3).zip'
extraction_dir = 'data/train'  # dataset will be extracted in 'data/train'

# Extracting zip file to the specified directory
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extraction_dir)

# Setting random seed for reproducibility
np.random.seed(42)

# Path to the extracted dataset
train_dir = '/content/data/train'  # contains training dat

# Data augmentation and rescaling using ImageDataGenerator
datagen = ImageDataGenerator(
    rescale=1./255,  # Normalize the images by scaling pixel values to[0, 1]
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    horizontal_flip=False,
    validation_split=0.2  # using 20% data for validation
)

# Training data generator
train_generator = datagen.flow_from_directory(
    train_dir,
    target_size=(28, 28),  # Resizing images
    batch_size=32,
    class_mode='categorical',
    color_mode='grayscale',
    subset='training'
)

# Validation of data generator
validation_generator = datagen.flow_from_directory(
    train_dir,
    target_size=(28, 28),
    batch_size=32,
    class_mode='categorical',
    color_mode='grayscale',
    subset='validation',
    shuffle=False
)

# Number of classes (digits and alphabetss)
num_classes = train_generator.num_classes

# Using CNN model
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),  # Max-pooling layer to downsample the feature map
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),  #Using  Dropout layer to prevent overfitting
    layers.Dense(num_classes, activation='softmax')  # Output layer with softmax for multiclass classifi
])

# Compiling model with Adam optimizer
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Training model with training data and validating
history = model.fit(
    train_generator,
    epochs=10,
    validation_data=validation_generator
)

# Saving trained model
model.save('handwritten_character_recognition_model.keras')

# Evaluation of model on the validation set
validation_loss, validation_acc = model.evaluate(validation_generator, verbose=2)
print(f"Validation accuracy: {validation_acc}")

# Generating predictions for entire validation set
validation_generator.reset()
y_pred = model.predict(validation_generator)
y_pred_classes = np.argmax(y_pred, axis=1)

# True labels from validation data
y_true = validation_generator.classes

# Class labels
class_labels = list(validation_generator.class_indices.keys())

# Generating and displaying classification report
print("Classification Report:")
print(classification_report(y_true, y_pred_classes, target_names=class_labels))

# loading and preprocessing image for prediction
img_path = '/content/data/train/Alphabets/img011-051.png'
img = image.load_img(img_path, target_size=(28, 28), color_mode='grayscale')
img_array = image.img_to_array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)

# prediction on image
predictions = model.predict(img_array)
predicted_class = np.argmax(predictions, axis=-1)
print(f"Predicted class: {predicted_class}")

# Displaying predicted image
plt.imshow(image.load_img(img_path, target_size=(28, 28), color_mode='grayscale'), cmap='gray')
plt.title(f"Predicted class: {predicted_class[0]}")
plt.axis('off')
plt.show()

# Classification report
class_labels = list(validation_generator.class_indices.keys())
print("Classification Report:")
print(classification_report(y_true, y_pred_classes, target_names=class_labels))


# Plotting training and validation accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

# Plotting training and validation loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

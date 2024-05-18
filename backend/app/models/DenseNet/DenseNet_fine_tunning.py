import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD
from sklearn.metrics import confusion_matrix, classification_report, precision_score, recall_score, f1_score
from tensorflow.keras.preprocessing import image

# Load configuration file
with open('config.json') as config_file:
    config = json.load(config_file)

data_dir = config["data_dir"]
train_dir = config["train_dir"]
val_dir = config["val_dir"]

# Data Preprocessing
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    rotation_range=10,
    width_shift_range=0.2,
    height_shift_range=0.2,
    brightness_range=[0.8, 1.2],
    fill_mode='nearest'
)

val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    shuffle=True
)

val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    shuffle=False
)

# Plotting the distribution of images in each category
def plot_category_distribution(generator, title):
    class_counts = {k: 0 for k, v in generator.class_indices.items()}
    for _, labels in generator:
        for label in labels:
            class_counts[generator.class_indices_inv[np.argmax(label)]] += 1

    plt.figure(figsize=(10, 5))
    plt.bar(class_counts.keys(), class_counts.values())
    plt.title(title)
    plt.xlabel('Category')
    plt.ylabel('Number of Images')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('category_distribution.png')
    plt.show()

plot_category_distribution(train_generator, "Distribution of Training Images")
plot_category_distribution(val_generator, "Distribution of Validation Images")

# Load Pre-trained DenseNet Model
base_model = DenseNet121(weights='imagenet', include_top=False)

# Add custom layers on top of DenseNet121
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(len(train_generator.class_indices), activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

# Freeze the layers of the base model
for layer in base_model.layers:
    layer.trainable = False

# Compile the model
model.compile(optimizer=SGD(learning_rate=0.001, momentum=0.9),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Training the model
num_epochs = 25
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    validation_data=val_generator,
    validation_steps=val_generator.samples // val_generator.batch_size,
    epochs=num_epochs
)

# Unfreeze some layers of the base model for fine-tuning
for layer in base_model.layers[:313]:  # Adjust the layer index if needed
    layer.trainable = False
for layer in base_model.layers[313:]:
    layer.trainable = True

# Re-compile the model for fine-tuning
model.compile(optimizer=SGD(learning_rate=0.0001, momentum=0.9),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Fine-tuning the model
fine_tune_epochs = 10
total_epochs = num_epochs + fine_tune_epochs

history_fine = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    validation_data=val_generator,
    validation_steps=val_generator.samples // val_generator.batch_size,
    epochs=total_epochs,
    initial_epoch=history.epoch[-1]
)

# Save the model
model.save('disaster_model_densenet121.h5')

print('Training complete')

# Evaluate the model and print confusion matrix
y_true = val_generator.classes
y_pred = model.predict(val_generator)
y_pred_classes = np.argmax(y_pred, axis=1)

conf_matrix = confusion_matrix(y_true, y_pred_classes)
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=val_generator.class_indices.keys(), yticklabels=val_generator.class_indices.keys())
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.tight_layout()
plt.savefig('confusion_matrix.png')
plt.show()

print(classification_report(y_true, y_pred_classes, target_names=val_generator.class_indices.keys()))

# Calculate precision, recall, and F1 score
precision = precision_score(y_true, y_pred_classes, average='weighted')
recall = recall_score(y_true, y_pred_classes, average='weighted')
f1 = f1_score(y_true, y_pred_classes, average='weighted')

print(f'Precision: {precision:.4f}')
print(f'Recall: {recall:.4f}')
print(f'F1 Score: {f1:.4f}')

# Example of model output
def show_model_output(generator, model):
    sample_image, sample_label = next(generator)
    prediction = model.predict(sample_image)
    predicted_class = np.argmax(prediction, axis=1)

    plt.figure(figsize=(10, 10))
    for i in range(9):
        plt.subplot(3, 3, i+1)
        plt.imshow(sample_image[i])
        true_label = list(generator.class_indices.keys())[sample_label[i].argmax()]
        pred_label = list(generator.class_indices.keys())[predicted_class[i]]
        plt.title(f"True: {true_label}\nPred: {pred_label}")
        plt.axis('off')
    plt.tight_layout()
    plt.savefig('model_output_example.png')
    plt.show()

show_model_output(val_generator, model)

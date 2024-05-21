import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import Sequence
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import Dense, Conv2D, Conv2DTranspose, Concatenate, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.callbacks import ProgbarLogger
import certifi

# Set SSL_CERT_FILE environment variable to use certifi's certificate bundle
os.environ['SSL_CERT_FILE'] = certifi.where()

# Load configuration file
with open('config.json') as config_file:
    config = json.load(config_file)

data_dir = config["data_dir"]
train_dir = config["train_dir"]
val_dir = config["val_dir"]

class DataGenerator(Sequence):
    def __init__(self, image_dir, batch_size=32, target_size=(224, 224), shuffle=True, mode='train'):
        self.image_dir = image_dir
        self.batch_size = batch_size
        self.target_size = target_size
        self.shuffle = shuffle
        self.mode = mode
        self.image_pairs, self.labels = self._load_data()
        self.indexes = np.arange(len(self.image_pairs))
        self.on_epoch_end()

        if mode == 'train':
            self.datagen = tf.keras.preprocessing.image.ImageDataGenerator(
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
        else:
            self.datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

    def _load_data(self):
        image_pairs = []
        labels = []
        for img_file in os.listdir(os.path.join(self.image_dir, 'images')):
            if 'pre_disaster' in img_file:
                post_img_file = img_file.replace('pre_disaster', 'post_disaster')
                pre_img_path = os.path.join(self.image_dir, 'images', img_file)
                post_img_path = os.path.join(self.image_dir, 'images', post_img_file)
                json_file = img_file.replace('pre_disaster.png', 'labels.json')
                json_path = os.path.join(self.image_dir, 'labels', json_file)

                if os.path.exists(pre_img_path) and os.path.exists(post_img_path) and os.path.exists(json_path):
                    image_pairs.append((pre_img_path, post_img_path))
                    with open(json_path, 'r') as f:
                        label_data = json.load(f)
                        labels.append(label_data)
                else:
                    print(f"Skipping {img_file} as the label file does not exist.")

        return image_pairs, labels

    def __len__(self):
        return int(np.floor(len(self.image_pairs) / self.batch_size))

    def __getitem__(self, index):
        batch_indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        batch_image_pairs = [self.image_pairs[k] for k in batch_indexes]
        batch_labels = [self.labels[k] for k in batch_indexes]

        X, y = self._generate_batch(batch_image_pairs, batch_labels)
        return X, y

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def _generate_batch(self, image_pairs, labels):
        X_pre = np.empty((self.batch_size, *self.target_size, 3))
        X_post = np.empty((self.batch_size, *self.target_size, 3))
        y = np.empty((self.batch_size, *self.target_size, 5))

        for i, (pre_img_path, post_img_path) in enumerate(image_pairs):
            pre_img = img_to_array(load_img(pre_img_path, target_size=self.target_size))
            post_img = img_to_array(load_img(post_img_path, target_size=self.target_size))
            label_img = self._generate_label_image(labels[i])

            if self.mode == 'train':
                pre_img = self.datagen.random_transform(pre_img)
                post_img = self.datagen.random_transform(post_img)

            X_pre[i,] = pre_img / 255.0
            X_post[i,] = post_img / 255.0
            y[i,] = label_img

        return {"pre_disaster_image": X_pre, "post_disaster_image": X_post}, y

    def _generate_label_image(self, label_data):
        label_img = np.zeros((*self.target_size, 5))
        for feature in label_data['features']['lng_lat']:
            if feature['properties']['feature_type'] == 'building':
                subtype = feature['properties']['subtype']
                if subtype == 'undamaged':
                    label = 1
                elif subtype == 'minor-damage':
                    label = 2
                elif subtype == 'major-damage':
                    label = 3
                elif subtype == 'destroyed':
                    label = 4
                else:
                    label = 0
                label_img[:, :, label] = 1
        return label_img

# Data Generators
train_generator = DataGenerator(train_dir, mode='train')
val_generator = DataGenerator(val_dir, mode='val')

# Load Pre-trained EfficientNetB0 Model
input_pre = Input(shape=(224, 224, 3), name='pre_disaster_image')
input_post = Input(shape=(224, 224, 3), name='post_disaster_image')

# Create a shared EfficientNetB0 model
base_model = EfficientNetB0(weights='imagenet', include_top=False)

# Extract features
features_pre = base_model(input_pre)
features_post = base_model(input_post)

# Concatenate features
concatenated_features = Concatenate(axis=-1)([features_pre, features_post])

# Add custom layers on top
x = Conv2D(512, (3, 3), activation='relu', padding='same')(concatenated_features)
x = Conv2DTranspose(256, (3, 3), strides=(2, 2), padding='same', activation='relu')(x)  # 14x14
x = Conv2DTranspose(128, (3, 3), strides=(2, 2), padding='same', activation='relu')(x)  # 28x28
x = Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same', activation='relu')(x)   # 56x56
x = Conv2DTranspose(32, (3, 3), strides=(2, 2), padding='same', activation='relu')(x)   # 112x112
x = Conv2DTranspose(16, (3, 3), strides=(2, 2), padding='same', activation='relu')(x)   # 224x224
predictions = Conv2D(5, (1, 1), activation='softmax', padding='same')(x)

model = Model(inputs=[input_pre, input_post], outputs=predictions)

# Freeze the layers of the base model
for layer in base_model.layers:
    layer.trainable = False

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.0001),
              loss=CategoricalCrossentropy(),
              metrics=['accuracy'])

# Training the model
num_epochs = 1

# Adding ProgbarLogger to the callbacks (it displays the estimate time of accomplishment)
progbar_logger = ProgbarLogger()

# Verify generator outputs
for data in train_generator:
    print(f"X_pre shape: {data[0]['pre_disaster_image'].shape}, X_post shape: {data[0]['post_disaster_image'].shape}, y shape: {data[1].shape}")
    print(f"Non-zero values in X_pre: {np.count_nonzero(data[0]['pre_disaster_image'])}")
    print(f"Non-zero values in X_post: {np.count_nonzero(data[0]['post_disaster_image'])}")
    print(f"Non-zero values in y: {np.count_nonzero(data[1])}")
    break

history = model.fit(
    train_generator,
    validation_data=val_generator,
    steps_per_epoch=len(train_generator),
    validation_steps=len(val_generator),
    epochs=num_epochs,
    callbacks=[progbar_logger]
)

# Unfreeze some layers of the base model for fine-tuning
for layer in base_model.layers[:230]:
    layer.trainable = False
for layer in base_model.layers[230:]:
    layer.trainable = True

# Re-compile the model for fine-tuning
model.compile(optimizer=Adam(learning_rate=0.00001),
              loss=CategoricalCrossentropy(),
              metrics=['accuracy'])

# Fine-tuning the model
fine_tune_epochs = 1
total_epochs = num_epochs + fine_tune_epochs

history_fine = model.fit(
    train_generator,
    validation_data=val_generator,
    steps_per_epoch=len(train_generator),
    validation_steps=len(val_generator),
    epochs=total_epochs,
    initial_epoch=history.epoch[-1],
    callbacks=[progbar_logger]
)

# Save the model
model.save('disaster_model_efficientnetb0.h5')

print('Training complete')

# Evaluate the model and print confusion matrix
y_true = []
y_pred = []
for [X_pre, X_post], y in val_generator:
    y_true.append(np.argmax(y, axis=-1))
    y_pred.append(np.argmax(model.predict({"pre_disaster_image": X_pre, "post_disaster_image": X_post}), axis=-1))
    if len(y_true) * val_generator.batch_size >= len(val_generator.labels):
        break
y_true = np.concatenate(y_true)
y_pred = np.concatenate(y_pred)

conf_matrix = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['No Building', 'Undamaged', 'Minor Damage', 'Major Damage', 'Destroyed'], yticklabels=['No Building', 'Undamaged', 'Minor Damage', 'Major Damage', 'Destroyed'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.tight_layout()
plt.savefig('confusion_matrix.png')
plt.show()

print(classification_report(y_true, y_pred, target_names=['No Building', 'Undamaged', 'Minor Damage', 'Major Damage', 'Destroyed']))

# Calculate precision, recall, and F1 score
precision = precision_score(y_true, y_pred, average='weighted')
recall = recall_score(y_true, y_pred, average='weighted')
f1 = f1_score(y_true, y_pred, average='weighted')

print(f'Precision: {precision:.4f}')
print(f'Recall: {recall:.4f}')
print(f'F1 Score: {f1:.4f}')

# Example of model output
def show_model_output(generator, model):
    [X_pre, X_post], y = next(generator)
    y_pred = model.predict({"pre_disaster_image": X_pre, "post_disaster_image": X_post})
    plt.figure(figsize=(10, 10))
    for i in range(9):
        plt.subplot(3, 3, i+1)
        plt.imshow(X_pre[i])  # Show pre-disaster image
        true_label = np.argmax(y[i], axis=-1)
        pred_label = np.argmax(y_pred[i], axis=-1)
        plt.title(f"True: {true_label}\nPred: {pred_label}")
        plt.axis('off')
    plt.tight_layout()
    plt.savefig('model_output_example.png')
    plt.show()

show_model_output(val_generator, model)

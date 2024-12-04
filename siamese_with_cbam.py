import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.optimizers import Adam
import numpy as np
import os
import cv2
from mtcnn import MTCNN
from sklearn.model_selection import train_test_split

# Configuration parameters
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 10
LR = 0.0001


def build_resnet50_model(input_shape=(224, 224, 3)):
    model = ResNet50(include_top=False, weights=None, input_shape=input_shape)
    return model


def load_vggface2_weights(model, weights_path):
    model.load_weights(weights_path)
    print("Loaded VGGFace2 weights successfully.")
    return model


# Build and load weights
input_shape = (224, 224, 3)
vggface2_weights_path = 'vggface2_Keras/model/resnet50_softmax_dim512/weights.h5'
resnet_model = build_resnet50_model(input_shape=input_shape)
resnet_model = load_vggface2_weights(resnet_model, vggface2_weights_path)


# Channel Attention Module
class ChannelAttention(layers.Layer):
    def __init__(self, ratio=16, **kwargs):
        super().__init__(**kwargs)
        self.ratio = ratio

    def build(self, input_shape):
        channel = input_shape[-1]
        self.shared_dense_one = layers.Dense(channel // self.ratio,
                                             activation='relu',
                                             kernel_initializer='he_normal',
                                             use_bias=True)
        self.shared_dense_two = layers.Dense(channel,
                                             kernel_initializer='he_normal',
                                             use_bias=True)

    def call(self, inputs):
        avg_pool = layers.GlobalAveragePooling2D()(inputs)
        max_pool = layers.GlobalMaxPooling2D()(inputs)
        avg_pool = layers.Reshape((1, 1, avg_pool.shape[1]))(avg_pool)
        max_pool = layers.Reshape((1, 1, max_pool.shape[1]))(max_pool)
        avg_out = self.shared_dense_two(self.shared_dense_one(avg_pool))
        max_out = self.shared_dense_two(self.shared_dense_one(max_pool))
        out = layers.Add()([avg_out, max_out])
        return layers.Activation('sigmoid')(out)


# Spatial Attention Module
class SpatialAttention(layers.Layer):
    def __init__(self, kernel_size=7, **kwargs):
        super().__init__(**kwargs)
        self.kernel_size = kernel_size

    def build(self, input_shape):
        self.conv = layers.Conv2D(1, kernel_size=self.kernel_size,
                                  strides=1, padding='same',
                                  activation='sigmoid',
                                  kernel_initializer='he_normal',
                                  use_bias=False)

    def call(self, inputs):
        avg_pool = tf.reduce_mean(inputs, axis=-1, keepdims=True)
        max_pool = tf.reduce_max(inputs, axis=-1, keepdims=True)
        concat = layers.Concatenate(axis=-1)([avg_pool, max_pool])
        return self.conv(concat)


# CBAM Block
class CBAMBlock(layers.Layer):
    def __init__(self, ratio=16, kernel_size=7, **kwargs):
        super().__init__(**kwargs)
        self.channel_attention = ChannelAttention(ratio)
        self.spatial_attention = SpatialAttention(kernel_size)

    def call(self, inputs):
        x = inputs
        x = layers.Multiply()([x, self.channel_attention(x)])
        x = layers.Multiply()([x, self.spatial_attention(x)])
        return x


def integrate_cbam_into_resnet(model):
    x = model.output
    x = CBAMBlock()(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation=None)(x)
    model = models.Model(inputs=model.input, outputs=x)
    return model


# Integrate CBAM into ResNet50
# resnet_model = integrate_cbam_into_resnet(resnet_model)

def build_siamese_network(input_shape=(224, 224, 3)):
    input_1 = layers.Input(shape=input_shape)
    input_2 = layers.Input(shape=input_shape)
    base_network = resnet_model  # Shared network
    embedding_1 = base_network(input_1)
    embedding_2 = base_network(input_2)
    diff = layers.Lambda(lambda tensors: tf.abs(tensors[0] - tensors[1]))(
        [embedding_1, embedding_2])
    output = layers.Dense(1, activation='sigmoid')(diff)
    siamese_model = models.Model(inputs=[input_1, input_2], outputs=output)
    return siamese_model


# Build the Siamese network
# siamese_model = build_siamese_network()


def build_siamese_network(input_shape=(224, 224, 3)):
    input_1 = layers.Input(shape=input_shape)
    input_2 = layers.Input(shape=input_shape)
    base_network = resnet_model  # Shared network
    embedding_1 = base_network(input_1)
    embedding_2 = base_network(input_2)
    diff = layers.Lambda(lambda tensors: tf.abs(tensors[0] - tensors[1]))(
        [embedding_1, embedding_2])
    output = layers.Dense(1, activation='sigmoid')(diff)
    siamese_model = models.Model(inputs=[input_1, input_2], outputs=output)
    return siamese_model


# Build the Siamese network
# siamese_model = build_siamese_network()

def preprocess_images(img_dir, img_size=(224, 224)):
    detector = MTCNN()
    processed_images = []
    labels = []
    for subdir, _, files in os.walk(img_dir):
        for file in files:
            img_path = os.path.join(subdir, file)
            img = cv2.imread(img_path)
            if img is None:
                print(f"Warning: Failed to load image {img_path}")
                continue
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            faces = detector.detect_faces(img_rgb)
            if faces:
                x, y, w, h = faces[0]['box']
                face_img = img_rgb[y:y + h, x:x + w]
                face_img = cv2.resize(face_img, img_size)
                processed_images.append(face_img)
                labels.append(subdir.split('/')[-1])
            else:
                print(f"No face detected in {img_path}")
    return np.array(processed_images), np.array(labels)


def create_pairs(images, labels):
    pair_images_1 = []
    pair_images_2 = []
    pair_labels = []
    label_to_indices = {label: np.where(labels == label)[0]
                        for label in np.unique(labels)}
    # Positive pairs
    for label, indices in label_to_indices.items():
        for i in range(len(indices) - 1):
            pair_images_1.append(images[indices[i]])
            pair_images_2.append(images[indices[i + 1]])
            pair_labels.append(1)
    # Negative pairs
    all_labels = list(label_to_indices.keys())
    for label, indices in label_to_indices.items():
        negative_label = np.random.choice(
            [l for l in all_labels if l != label])
        negative_indices = label_to_indices[negative_label]
        for i in range(len(indices)):
            pair_images_1.append(images[indices[i]])
            pair_images_2.append(
                images[np.random.choice(negative_indices)])
            pair_labels.append(0)
    return [np.array(pair_images_1), np.array(pair_images_2)], np.array(pair_labels)


# Load and preprocess images
img_dir = 'datasets/LFWPeople/lfw_funneled'
images, labels = preprocess_images(img_dir)
pairs, pair_labels = create_pairs(images, labels)
train_pairs, val_pairs, train_labels, val_labels = train_test_split(
    pairs, pair_labels, test_size=0.2)

# Compile the model
siamese_model.compile(optimizer=Adam(learning_rate=LR),
                      loss='binary_crossentropy',
                      metrics=['accuracy'])

# Train the model
history = siamese_model.fit(
    [train_pairs[0], train_pairs[1]], train_labels,
    validation_data=([val_pairs[0], val_pairs[1]], val_labels),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE)

# # Evaluate the model
# val_loss, val_acc = siamese_model.evaluate(
#     [val_pairs[0], val_pairs[1]], val_labels)
# print(f'Validation Loss: {val_loss}, Validation Accuracy: {val_acc}')
#
# # Save the model
# siamese_model.save('siamese_cbam_vggface2_lfw.h5')

if __name__ == "__main__":
    # Build and load the model
    resnet_model = build_resnet50_model(input_shape=IMG_SIZE + (3,))
    resnet_model = load_vggface2_weights(resnet_model, vggface2_weights_path)
    resnet_model = integrate_cbam_into_resnet(resnet_model)
    siamese_model = build_siamese_network(input_shape=IMG_SIZE + (3,))

    # Compile and train
    siamese_model.compile(optimizer=Adam(learning_rate=LR),
                          loss='binary_crossentropy',
                          metrics=['accuracy'])
    history = siamese_model.fit(
        [train_pairs[0], train_pairs[1]], train_labels,
        validation_data=([val_pairs[0], val_pairs[1]], val_labels),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE)

    # Evaluate and save
    val_loss, val_acc = siamese_model.evaluate(
        [val_pairs[0], val_pairs[1]], val_labels)
    print(f'Validation Loss: {val_loss}, Validation Accuracy: {val_acc}')
    siamese_model.save('siamese_cbam_vggface2_lfw.h5')

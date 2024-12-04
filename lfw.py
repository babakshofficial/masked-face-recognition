import tensorflow as tf
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Input, Lambda
from tensorflow.keras.models import Model, load_model
# from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.legacy import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
from sklearn.model_selection import train_test_split
import cv2
import os
import random

mlfw_dir = 'datasets/LFWPeople/lfw_funneled'
pairs_file = 'datasets/LFWPeople/pairs.txt'
h5_path = 'vggface2_Keras/model/resnet50_softmax_dim512/weights.h5'

# Configuration
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 1
LR = 0.0001


# Custom VGGFace model (ResNet50 backbone)
def load_base_model(weights_path):
    # Load the entire model with its architecture
    base_model = load_model(weights_path)

    # Get the output of the loaded model
    x = base_model.output

    # Add a dense layer to reduce dimensionality to 128
    x = Dense(128, activation='relu', name='embedding')(x)

    model = Model(inputs=base_model.input, outputs=x)
    print(f"Loaded model from {weights_path}")
    return model


# Create Siamese network
def create_siamese_model(input_shape, weights_path):
    base_network = load_base_model(weights_path)

    input_a = Input(shape=input_shape)
    input_b = Input(shape=input_shape)

    # Generate embeddings for both inputs
    embedding_a = base_network(input_a)
    embedding_b = base_network(input_b)

    # Calculate L1 distance between embeddings
    l1_distance = Lambda(lambda x: tf.abs(x[0] - x[1]))([embedding_a, embedding_b])

    # Final classification layer
    prediction = Dense(1, activation='sigmoid')(l1_distance)

    return Model(inputs=[input_a, input_b], outputs=prediction)


# Create and compile the model
siamese_model = create_siamese_model((*IMG_SIZE, 3), h5_path)
siamese_model.compile(optimizer=Adam(LR), loss='binary_crossentropy', metrics=['accuracy'])


def load_mlfw_pairs(mlfw_dir, pairs_file):
    def load_image(image_path):
        img = cv2.imread(image_path)
        img = cv2.resize(img, IMG_SIZE)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    pairs = []
    with open(pairs_file, 'r') as f:
        for line in f:
            pair = line.strip().split()
            if len(pair) == 3:
                # Same person
                name, img1_id, img2_id = pair
                img1_path = os.path.join(mlfw_dir, name, f"{name}_{img1_id.zfill(4)}.jpg")
                img2_path = os.path.join(mlfw_dir, name, f"{name}_{img2_id.zfill(4)}.jpg")
                pairs.append((img1_path, img2_path, 1))
            elif len(pair) == 4:
                # Different people
                name1, img1_id, name2, img2_id = pair
                img1_path = os.path.join(mlfw_dir, name1, f"{name1}_{img1_id.zfill(4)}.jpg")
                img2_path = os.path.join(mlfw_dir, name2, f"{name2}_{img2_id.zfill(4)}.jpg")
                pairs.append((img1_path, img2_path, 0))

    random.shuffle(pairs)

    X1, X2, y = [], [], []
    for img1_path, img2_path, label in pairs:
        img1 = load_image(img1_path)
        img2 = load_image(img2_path)
        X1.append(img1)
        X2.append(img2)
        y.append(label)
    print('Loaded data from {}'.format(mlfw_dir))
    return np.array(X1), np.array(X2), np.array(y)


# Load and preprocess data
X1, X2, y = load_mlfw_pairs(mlfw_dir, pairs_file)

# Split data
X1_train, X1_test, X2_train, X2_test, y_train, y_test = train_test_split(X1, X2, y, test_size=0.2, random_state=42)

# Data augmentation for training
train_datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    preprocessing_function=tf.keras.applications.resnet50.preprocess_input
)

# Training
history = siamese_model.fit(
    [X1_train, X2_train], y_train,
    validation_data=([X1_test, X2_test], y_test),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE
)

# Evaluation
test_loss, test_accuracy = siamese_model.evaluate([X1_test, X2_test], y_test)
print(f"Test accuracy: {test_accuracy:.4f}")


# Function to verify two face images
def verify_faces(img1, img2):
    img1 = tf.keras.applications.resnet50.preprocess_input(img1)
    img2 = tf.keras.applications.resnet50.preprocess_input(img2)
    prediction = siamese_model.predict([np.expand_dims(img1, axis=0), np.expand_dims(img2, axis=0)])
    return prediction[0][0]

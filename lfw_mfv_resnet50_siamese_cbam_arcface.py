import tensorflow as tf
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Input, Lambda, Multiply, Flatten
from tensorflow.keras.layers import Reshape, Conv2D, MaxPooling2D, ZeroPadding2D, AveragePooling2D
from tensorflow.keras.layers import Activation, BatchNormalization, add, GlobalMaxPooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
from backbones import CBAM_module as CBAM
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import configparser
from sklearn.model_selection import train_test_split
import cv2
import os
import random
import h5py

# GPU availability check and configuration
device_name = tf.test.gpu_device_name()
print("Nvidia GPU", "available!" if tf.config.list_physical_devices("GPU") else "not available :(  ")
if "GPU" not in device_name:
    pass
else:
    print('Found GPU at: {}'.format(device_name))

# Read configuration from config file
config = configparser.ConfigParser()
config.read('config.ini')

# Configuration parameters
IMG_SIZE = (int(config['GENERAL']['WIDTH']), int(config['GENERAL']['HEIGHT']))
BATCH_SIZE = int(config['GENERAL']['BATCH_SIZE'])
EPOCHS = int(config['GENERAL']['EPOCHS'])
LR = float(config['GENERAL']['LR'])

# Dataset configuration
dataset = config['DIRs']['DATASET']
lfw_dir = config['DIRs']['LFW_DIR']
pairs_file = config['DIRs']['PAIRS_FILE']
h5_path = config['DIRs']['H5_PATH']

# Select dataset directory based on configuration
if dataset == 'lfw':
    lfw_dir = 'datasets/LFWPeople/lfw_funneled'
elif dataset == 'cbam_lfw':
    lfw_dir = 'datasets/cbam_lfw/lfw_funneled'
elif dataset == 'mflw':
    lfw_dir = 'datasets/mflw/lfw_funneled'

# CBAM (Convolutional Block Attention Module) implementation
def cbam_block(input_feature):
    # Apply channel and spatial attention
    channel_refined_feature = CBAM.channel_attention(input_feature)
    spatial_refined_feature = CBAM.spatial_attention(channel_refined_feature)
    return spatial_refined_feature

# ResNet50 identity block with CBAM integration
def identity_block_with_cbam(input_tensor, kernel_size, filters, stage, block):
    filters1, filters2, filters3 = filters
    conv_name_base = f'res{stage}{block}_branch'
    bn_name_base = f'bn{stage}{block}_branch'

    # Convolutional layers with batch normalization and ReLU activation
    x = Conv2D(filters1, (1, 1), name=conv_name_base + '2a')(input_tensor)
    x = BatchNormalization(name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters2, kernel_size, padding='same', name=conv_name_base + '2b')(x)
    x = BatchNormalization(name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters3, (1, 1), name=conv_name_base + '2c')(x)
    x = BatchNormalization(name=bn_name_base + '2c')(x)

    # Apply CBAM attention mechanism
    x = cbam_block(x)

    # Residual connection
    x = add([x, input_tensor])
    x = Activation('relu')(x)
    return x


def conv_block_with_cbam(input_tensor, kernel_size, filters, stage, block, strides=(2, 2)):
    filters1, filters2, filters3 = filters
    conv_name_base = f'res{stage}{block}_branch'
    bn_name_base = f'bn{stage}{block}_branch'

    x = Conv2D(filters1, (1, 1), strides=strides, name=conv_name_base + '2a')(input_tensor)
    x = BatchNormalization(name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters2, kernel_size, padding='same', name=conv_name_base + '2b')(x)
    x = BatchNormalization(name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters3, (1, 1), name=conv_name_base + '2c')(x)
    x = BatchNormalization(name=bn_name_base + '2c')(x)

    x = cbam_block(x)

    shortcut = Conv2D(filters3, (1, 1), strides=strides, name=conv_name_base + '1')(input_tensor)
    shortcut = BatchNormalization(name=bn_name_base + '1')(shortcut)

    x = add([x, shortcut])
    x = Activation('relu')(x)
    return x


def ResNet50_CBAM():
    img_input = Input(shape=(224, 224, 3))

    x = ZeroPadding2D((3, 3))(img_input)
    x = Conv2D(64, (7, 7), strides=(2, 2), name='conv1')(x)
    x = BatchNormalization(name='bn_conv1')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)

    x = conv_block_with_cbam(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1))
    x = identity_block_with_cbam(x, 3, [64, 64, 256], stage=2, block='b')
    x = identity_block_with_cbam(x, 3, [64, 64, 256], stage=2, block='c')

    x = conv_block_with_cbam(x, 3, [128, 128, 512], stage=3, block='a')
    x = identity_block_with_cbam(x, 3, [128, 128, 512], stage=3, block='b')
    x = identity_block_with_cbam(x, 3, [128, 128, 512], stage=3, block='c')
    x = identity_block_with_cbam(x, 3, [128, 128, 512], stage=3, block='d')

    x = conv_block_with_cbam(x, 3, [256, 256, 1024], stage=4, block='a')
    x = identity_block_with_cbam(x, 3, [256, 256, 1024], stage=4, block='b')
    x = identity_block_with_cbam(x, 3, [256, 256, 1024], stage=4, block='c')
    x = identity_block_with_cbam(x, 3, [256, 256, 1024], stage=4, block='d')
    x = identity_block_with_cbam(x, 3, [256, 256, 1024], stage=4, block='e')
    x = identity_block_with_cbam(x, 3, [256, 256, 1024], stage=4, block='f')

    x = conv_block_with_cbam(x, 3, [512, 512, 2048], stage=5, block='a')
    x = identity_block_with_cbam(x, 3, [512, 512, 2048], stage=5, block='b')
    x = identity_block_with_cbam(x, 3, [512, 512, 2048], stage=5, block='c')

    x = AveragePooling2D((7, 7), name='avg_pool')(x)
    x = Flatten()(x)
    x = Dense(512, activation='softmax', name='classifier')(x)

    model = Model(img_input, x, name='vggface_resnet50_cbam')
    return model


def sanitize_name(name):
    return name.replace('/', '_').replace(' ', '_')


def identity_block(input_tensor, kernel_size, filters, stage, block):
    filters1, filters2, filters3 = filters
    conv_name_base = f'res{stage}{block}_branch'
    bn_name_base = f'bn{stage}{block}_branch'

    x = Conv2D(filters1, (1, 1), name=conv_name_base + '2a')(input_tensor)
    x = BatchNormalization(name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters2, kernel_size, padding='same', name=conv_name_base + '2b')(x)
    x = BatchNormalization(name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters3, (1, 1), name=conv_name_base + '2c')(x)
    x = BatchNormalization(name=bn_name_base + '2c')(x)

    x = add([x, input_tensor])
    x = Activation('relu')(x)
    return x


def conv_block(input_tensor, kernel_size, filters, stage, block, strides=(2, 2)):
    filters1, filters2, filters3 = filters
    conv_name_base = f'res{stage}{block}_branch'
    bn_name_base = f'bn{stage}{block}_branch'

    x = Conv2D(filters1, (1, 1), strides=strides, name=conv_name_base + '2a')(input_tensor)
    x = BatchNormalization(name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters2, kernel_size, padding='same', name=conv_name_base + '2b')(x)
    x = BatchNormalization(name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters3, (1, 1), name=conv_name_base + '2c')(x)
    x = BatchNormalization(name=bn_name_base + '2c')(x)

    shortcut = Conv2D(filters3, (1, 1), strides=strides, name=conv_name_base + '1')(input_tensor)
    shortcut = BatchNormalization(name=bn_name_base + '1')(shortcut)

    x = add([x, shortcut])
    x = Activation('relu')(x)
    return x


def ResNet50():
    img_input = Input(shape=(224, 224, 3))

    x = ZeroPadding2D((3, 3))(img_input)
    x = Conv2D(64, (7, 7), strides=(2, 2), name='conv1')(x)
    x = BatchNormalization(name='bn_conv1')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)

    x = conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1))
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='b')
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='c')

    x = conv_block(x, 3, [128, 128, 512], stage=3, block='a')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='b')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='c')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='d')

    x = conv_block(x, 3, [256, 256, 1024], stage=4, block='a')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='b')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='c')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='d')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='e')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='f')

    x = conv_block(x, 3, [512, 512, 2048], stage=5, block='a')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='c')

    x = AveragePooling2D((7, 7), name='avg_pool')(x)
    x = Flatten()(x)
    x = Dense(512, activation='softmax', name='classifier')(x)

    model = Model(img_input, x, name='vggface_resnet50')
    return model


def load_weights_from_h5(model, weights_path):
    with h5py.File(weights_path, 'r') as f:
        for layer in model.layers:
            if layer.name in f:
                layer_weights = []
                for w in f[layer.name]:
                    layer_weights.append(f[layer.name][w][:])
                layer.set_weights(layer_weights)
    print(f"Weights loaded from {weights_path}")


def load_base_model(weights_path):
    base_model = ResNet50_CBAM()
    load_weights_from_h5(base_model, weights_path)

    # Get the output of the loaded model (before the classifier layer)
    x = base_model.layers[-2].output
    # Add a dense layer to reduce dimensionality to 128
    x = Dense(128, activation='relu', name='embedding')(x)
    model = Model(inputs=base_model.input, outputs=x)
    return model


def create_siamese_model(input_shape, weights_path):
    base_network = load_base_model(weights_path)

    input_a = Input(shape=input_shape)
    input_b = Input(shape=input_shape)

    # Generate embeddings for both inputs
    embedding_a = base_network(input_a)
    embedding_b = base_network(input_b)

    # Calculate L1 distance between embeddings
    l1_distance = tf.keras.layers.Lambda(lambda x: tf.abs(x[0] - x[1]))([embedding_a, embedding_b])

    # Final classification layer
    prediction = Dense(1, activation='sigmoid')(l1_distance)

    return Model(inputs=[input_a, input_b], outputs=prediction)


def load_mlfw_pairs(modified_lfw_dir, pairs_file):
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
                img1_path = os.path.join(modified_lfw_dir, name, f"{name}_{img1_id.zfill(4)}.jpg")
                img2_path = os.path.join(modified_lfw_dir, name, f"{name}_{img2_id.zfill(4)}.jpg")
                pairs.append((img1_path, img2_path, 1))
            elif len(pair) == 4:
                # Different people
                name1, img1_id, name2, img2_id = pair
                img1_path = os.path.join(modified_lfw_dir, name1, f"{name1}_{img1_id.zfill(4)}.jpg")
                img2_path = os.path.join(modified_lfw_dir, name2, f"{name2}_{img2_id.zfill(4)}.jpg")
                pairs.append((img1_path, img2_path, 0))

    random.shuffle(pairs)

    X1, X2, y = [], [], []
    for img1_path, img2_path, label in pairs:
        img1 = load_image(img1_path)
        img2 = load_image(img2_path)
        X1.append(img1)
        X2.append(img2)
        y.append(label)
    print('Loaded data from {}'.format(modified_lfw_dir))
    return np.array(X1), np.array(X2), np.array(y)


# Function to verify two face images
def verify_faces(siamese_model, img1, img2):
    img1 = tf.keras.applications.resnet50.preprocess_input(img1)
    img2 = tf.keras.applications.resnet50.preprocess_input(img2)
    prediction = siamese_model.predict([np.expand_dims(img1, axis=0), np.expand_dims(img2, axis=0)])
    return prediction[0][0]


def get_batch_size(num_gpus=1):
    batch_size = BATCH_SIZE_PER_REPLICA * num_gpus
    print(f'BATCH_SIZE = {batch_size}')
    return batch_size


# Create and compile the model
def build_final_model():
    siamese_model = create_siamese_model((*IMG_SIZE, 3), h5_path)
    siamese_model.compile(optimizer=Adam(LR), loss='binary_crossentropy', metrics=['accuracy'])
    return siamese_model


# # Load and preprocess data
X1, X2, y = load_mlfw_pairs(lfw_dir, pairs_file)
# # Split data
X1_train, X1_test, X2_train, X2_test, y_train, y_test = train_test_split(X1, X2, y, test_size=0.2, random_state=42)

BUFFER_SIZE = 10000
BATCH_SIZE_PER_REPLICA = 64

# Wrap data in Dataset objects. Modify to handle multiple feature inputs (X1, X2).
train_data = tf.data.Dataset.from_tensor_slices(((X1_train, X2_train), y_train))
test_data = tf.data.Dataset.from_tensor_slices(((X1_test, X2_test), y_test))

# the batch size will be defined later based on the number of used GPUs
options = tf.data.Options()
options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.OFF
train_data = train_data.with_options(options).cache().shuffle(BUFFER_SIZE).prefetch(tf.data.AUTOTUNE)
test_data = test_data.with_options(options)

# Data augmentation for training
train_datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    preprocessing_function=tf.keras.applications.resnet50.preprocess_input
)

strategy = tf.distribute.MirroredStrategy()
print('Number of devices: {}'.format(strategy.num_replicas_in_sync))

with strategy.scope():
    multi_gpu_model = build_final_model()
multi_gpu_model.summary()

d_train_data = train_data.batch(get_batch_size(strategy.num_replicas_in_sync))
d_test_data = test_data.batch(get_batch_size(strategy.num_replicas_in_sync))
history = multi_gpu_model.fit(d_train_data, validation_data=d_test_data, epochs=EPOCHS)

# Now perform face verification on the test set
correct_predictions = 0
total_predictions = len(X1_test)

# Iterate over the test pairs
for i in range(total_predictions):
    img1, img2 = X1_test[i], X2_test[i]
    ground_truth = y_test[i]

    # Use verify_faces function to predict the similarity
    prediction = verify_faces(multi_gpu_model, img1, img2)

    # Convert prediction to binary outcome (you can adjust the threshold if needed)
    predicted_label = 1 if prediction > 0.5 else 0

    # Check if the prediction matches the ground truth
    if predicted_label == ground_truth:
        correct_predictions += 1

# Calculate and print the accuracy
accuracy = correct_predictions / total_predictions
print(f'Face verification accuracy: {accuracy * 100:.2f}%')

# Plot training & validation accuracy values
plt.figure(figsize=(12, 5))

# Accuracy
plt.subplot(1, 2, 1)
print(history.history)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['loss'], label='Train Loss')
plt.title('Model Accuracy: ' + dataset.upper())
plt.ylabel('Accuracy, Loss')
plt.xlabel('Epoch')
plt.legend(loc='upper left')

# Loss
plt.subplot(1, 2, 2)
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss: ' + dataset.upper())
plt.ylabel('Accuracy, Loss')
plt.xlabel('Epoch')
plt.legend(loc='upper left')

plt.tight_layout()
plt.show()

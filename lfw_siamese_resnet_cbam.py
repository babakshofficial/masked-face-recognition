import tensorflow as tf
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Input, Lambda, Multiply,  Flatten
from tensorflow.keras.layers import Reshape, Conv2D, MaxPooling2D, ZeroPadding2D, AveragePooling2D
from tensorflow.keras.layers import Activation, BatchNormalization, add, GlobalMaxPooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers.legacy import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
from sklearn.model_selection import train_test_split
import cv2
import os
import random
import h5py
print(tf.__version__)

device_name = tf.test.gpu_device_name()
if "GPU" not in device_name:
    print("GPU device not found")

print('Found GPU at: {}'.format(device_name))

print("GPU", "available (YESS!!!!)" if tf.config.list_physical_devices("GPU") else "not available :(")

# Configuration
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 1
LR = 0.0001
mlfw_dir = 'datasets/LFWPeople/lfw_funneled'
pairs_file = 'datasets/LFWPeople/pairs.txt'
h5_path = 'vggface2_Keras/model/resnet50_softmax_dim512/weights.h5'


def channel_attention(input_feature, ratio=8):
    channel = input_feature.shape[-1]

    shared_layer_one = Dense(channel // ratio,
                             activation='relu',
                             kernel_initializer='he_normal',
                             use_bias=True,
                             bias_initializer='zeros')
    shared_layer_two = Dense(channel,
                             kernel_initializer='he_normal',
                             use_bias=True,
                             bias_initializer='zeros')

    avg_pool = GlobalAveragePooling2D()(input_feature)
    avg_pool = Reshape((1, 1, channel))(avg_pool)
    assert avg_pool.shape[1:] == (1, 1, channel)
    avg_pool = shared_layer_one(avg_pool)
    assert avg_pool.shape[1:] == (1, 1, channel // ratio)
    avg_pool = shared_layer_two(avg_pool)
    assert avg_pool.shape[1:] == (1, 1, channel)

    max_pool = GlobalMaxPooling2D()(input_feature)
    max_pool = Reshape((1, 1, channel))(max_pool)
    assert max_pool.shape[1:] == (1, 1, channel)
    max_pool = shared_layer_one(max_pool)
    assert max_pool.shape[1:] == (1, 1, channel // ratio)
    max_pool = shared_layer_two(max_pool)
    assert max_pool.shape[1:] == (1, 1, channel)

    cbam_feature = Activation('sigmoid')(add([avg_pool, max_pool]))

    return Multiply()([input_feature, cbam_feature])


def spatial_attention(input_feature):
    kernel_size = 7

    avg_pool = Lambda(lambda x: tf.keras.backend.mean(x, axis=3, keepdims=True))(input_feature)
    max_pool = Lambda(lambda x: tf.keras.backend.max(x, axis=3, keepdims=True))(input_feature)
    concat = tf.keras.layers.Concatenate(axis=3)([avg_pool, max_pool])
    cbam_feature = Conv2D(filters=1,
                          kernel_size=kernel_size,
                          strides=1,
                          padding='same',
                          activation='sigmoid',
                          kernel_initializer='he_normal',
                          use_bias=False)(concat)

    return Multiply()([input_feature, cbam_feature])


def cbam_block(input_feature):
    channel_refined_feature = channel_attention(input_feature)
    spatial_refined_feature = spatial_attention(channel_refined_feature)
    return spatial_refined_feature


def identity_block_with_cbam(input_tensor, kernel_size, filters, stage, block):
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

    x = cbam_block(x)

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


# Function to verify two face images
def verify_faces(img1, img2):
    img1 = tf.keras.applications.resnet50.preprocess_input(img1)
    img2 = tf.keras.applications.resnet50.preprocess_input(img2)
    prediction = siamese_model.predict([np.expand_dims(img1, axis=0), np.expand_dims(img2, axis=0)])
    return prediction[0][0]


# Create and compile the model
siamese_model = create_siamese_model((*IMG_SIZE, 3), h5_path)
siamese_model.compile(optimizer=Adam(LR), loss='binary_crossentropy', metrics=['accuracy'])

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

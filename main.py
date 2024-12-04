import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
import numpy as np
import pickle
from PIL import Image
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import os
from sklearn.preprocessing import LabelEncoder

# Constants
IMG_HEIGHT, IMG_WIDTH = 224, 224
BATCH_SIZE = 32
EPOCHS = 20


def load_rof_dataset(data_dir, img_height=224, img_width=224):
    images = []
    labels = []
    identities = []
    occlusion_types = []

    occlusion_folders = ['neutral', 'sunglasses', 'masked']

    for occlusion in occlusion_folders:
        occlusion_path = os.path.join(data_dir, occlusion)

        if not os.path.exists(occlusion_path):
            print(f"Warning: {occlusion} folder not found in {data_dir}")
            continue

        for identity in os.listdir(occlusion_path):
            identity_path = os.path.join(occlusion_path, identity)

            # if os.path.isdir(identity_path):
            #     for img_name in os.listdir(identity_path):
            # img_path = os.path.join(identity_path, img_name)

            # img = load_img(identity_path, target_size=(img_height, img_width))
            img = pickle.load(identity_path)
            img = Image.fromarray(img)
            resized_image = img.resize((img_height, img_width))
            img_array = resized_image / 255.0
            resized_image.show()

            images.append(img_array)
            labels.append(identity)
            identities.append(identity)
            occlusion_types.append(occlusion)
            print(identity, 'Added.')

    images = np.array(images)
    labels = np.array(labels)
    identities = np.array(identities)
    occlusion_types = np.array(occlusion_types)

    le = LabelEncoder()
    encoded_labels = le.fit_transform(labels)

    print(f"Loaded {len(images)} images")
    print(f"Number of unique identities: {len(np.unique(identities))}")
    print(f"Occlusion type distribution:")
    for occlusion in np.unique(occlusion_types):
        count = np.sum(occlusion_types == occlusion)
        print(f"  {occlusion}: {count}")

    return images, encoded_labels, identities, occlusion_types


def create_base_model():
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(IMG_HEIGHT, IMG_WIDTH, 3))
    for layer in base_model.layers:
        layer.trainable = False
    x = layers.GlobalAveragePooling2D()(base_model.output)
    x = layers.Dense(512, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(128, activation='relu')(x)
    return models.Model(inputs=base_model.input, outputs=x)


def create_siamese_model():
    base_model = create_base_model()
    input_a = layers.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3))
    input_b = layers.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3))
    processed_a = base_model(input_a)
    processed_b = base_model(input_b)
    distance = layers.Lambda(lambda x: tf.math.abs(x[0] - x[1]))([processed_a, processed_b])
    output = layers.Dense(1, activation='sigmoid')(distance)
    siamese_model = models.Model(inputs=[input_a, input_b], outputs=output)
    return siamese_model


def contrastive_loss(y_true, y_pred):
    margin = 1
    return tf.math.reduce_mean(y_true * tf.math.square(y_pred) +
                               (1 - y_true) * tf.math.square(tf.math.maximum(margin - y_pred, 0)))


def create_pairs(images, labels):
    pairs = []
    labels_pairs = []
    n = min([len(np.where(labels == i)[0]) for i in np.unique(labels)])  # min class size
    for c in np.unique(labels):
        c_idx = np.where(labels == c)[0]
        for i in range(n):
            for j in range(i + 1, n):
                pairs.append([images[c_idx[i]], images[c_idx[j]]])
                labels_pairs.append(1)  # Same class

                diff_c = np.random.choice(np.where(labels != c)[0])
                pairs.append([images[c_idx[i]], images[diff_c]])
                labels_pairs.append(0)  # Different class
    return np.array(pairs), np.array(labels_pairs)


def save_data_pickle(data, filename):
    with open(filename, 'wb') as f:
        pickle.dump(data, f)


def load_data_pickle(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)


def save_model_tf(model, filename):
    model.save(filename)


def load_model_tf(filename):
    return tf.keras.models.load_model(filename, custom_objects={'contrastive_loss': contrastive_loss})


if __name__ == "__main__":
    # Load ROF dataset
    images, labels, identities, occlusion_types = load_rof_dataset('ROF')
    print(len(images), len(labels), len(identities), len(occlusion_types))
    # Create pairs
    pairs, pair_labels = create_pairs(images, labels)

    # Split data
    train_pairs, test_pairs, train_labels, test_labels = train_test_split(pairs, pair_labels, test_size=0.2)

    # Save processed data
    save_data_pickle((train_pairs, test_pairs, train_labels, test_labels), 'rof_processed_data.pkl')

    # Create and compile model
    model = create_siamese_model()
    model.compile(loss=contrastive_loss, optimizer=optimizers.Adam(), metrics=['accuracy'])

    # Train model
    history = model.fit(
        [train_pairs[:, 0], train_pairs[:, 1]], train_labels,
        validation_data=([test_pairs[:, 0], test_pairs[:, 1]], test_labels),
        batch_size=BATCH_SIZE,
        epochs=EPOCHS
    )

    # Save model
    save_model_tf(model, 'rof_siamese_model')

    # Evaluate model
    results = model.evaluate([test_pairs[:, 0], test_pairs[:, 1]], test_labels)
    print(f"Test loss: {results[0]}, Test accuracy: {results[1]}")

    # Example of using the model for verification
    img1 = images[0]  # Just using the first image as an example
    img2 = images[1]  # And the second image
    prediction = model.predict([np.array([img1]), np.array([img2])])
    print(f"Similarity score: {prediction[0][0]}")
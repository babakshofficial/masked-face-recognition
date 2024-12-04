import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import matplotlib.pyplot as plt
from mtcnn import MTCNN
import cv2
from tensorflow.keras.models import Model, load_model
import os
from PIL import Image
from tqdm import tqdm

tf.keras.utils.disable_interactive_logging()

# Configuration
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 1
LR = 0.0001
h5_path = 'vggface2_Keras/model/resnet50_softmax_dim512/weights.h5'
ratio = 16
kernel_size = 7
normalization_unit = 255.0


class ChannelAttention(layers.Layer):
    def __init__(self, ratio=16, **kwargs):
        super(ChannelAttention, self).__init__(**kwargs)
        self.shared_layer_one = None
        self.shared_layer_two = None
        self.ratio = ratio

    def build(self, input_shape):
        channel = input_shape[-1]
        self.shared_layer_one = layers.Dense(channel // self.ratio,
                                             activation='relu',
                                             kernel_initializer='he_normal',
                                             use_bias=True,
                                             bias_initializer='zeros')
        self.shared_layer_two = layers.Dense(channel,
                                             kernel_initializer='he_normal',
                                             use_bias=True,
                                             bias_initializer='zeros')

    def call(self, inputs):
        avg_pool = tf.reduce_mean(inputs, axis=[1, 2], keepdims=True)
        avg_pool = self.shared_layer_one(avg_pool)
        avg_pool = self.shared_layer_two(avg_pool)

        max_pool = tf.reduce_max(inputs, axis=[1, 2], keepdims=True)
        max_pool = self.shared_layer_one(max_pool)
        max_pool = self.shared_layer_two(max_pool)

        attention = tf.sigmoid(avg_pool + max_pool)
        return attention


class EnhancedSpatialAttention(layers.Layer):
    def __init__(self, kernel_size=7, **kwargs):
        super(EnhancedSpatialAttention, self).__init__(**kwargs)
        self.kernel_size = kernel_size

    def build(self, input_shape):
        self.conv = layers.Conv2D(filters=1,
                                  kernel_size=self.kernel_size,
                                  padding='same',
                                  kernel_initializer='he_normal',
                                  use_bias=False)

    def call(self, inputs, face_mask):
        avg_pool = tf.reduce_mean(inputs, axis=3, keepdims=True)
        max_pool = tf.reduce_max(inputs, axis=3, keepdims=True)
        concat = tf.concat([avg_pool, max_pool], axis=3)
        attention = tf.sigmoid(self.conv(concat))

        # Enhance attention using face mask
        enhanced_attention = attention * face_mask
        return enhanced_attention


base_model = load_model(h5_path)
face_mask_input = layers.Input(shape=(7, 7, 1))
conv_output = base_model.layers[170].output

channel_attention = ChannelAttention(ratio)(conv_output)
x = layers.Multiply()([conv_output, channel_attention])
spatial_attention = EnhancedSpatialAttention(kernel_size)(x, face_mask_input)
x = layers.Multiply()([x, spatial_attention])

x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(32, activation='softmax')(x)

model = models.Model(inputs=[base_model.input, face_mask_input], outputs=x)
cbam_layer_model = models.Model(inputs=model.input, outputs=model.layers[174].output)
print(f"Loaded model from {h5_path}")


def create_face_filter(image, size=IMG_SIZE):
    detector = MTCNN()
    image_array = np.array(image)
    result = detector.detect_faces(image_array)

    mask = np.zeros(size)

    if result:
        for person in result:
            bounding_box = person['box']
            keypoints = person['keypoints']

            # Create mask for face area
            x, y, width, height = bounding_box
            mask[y:y + height, x:x + width] = 1

            # Enhance attention on eyes
            left_eye = keypoints['left_eye']
            right_eye = keypoints['right_eye']
            eye_radius = int(width * 0.1)  # Adjust as needed
            cv2.circle(mask, tuple(left_eye), eye_radius, 2, -1)
            cv2.circle(mask, tuple(right_eye), eye_radius, 2, -1)

    # Down-sample mask to 7x7
    mask_downsampled = tf.image.resize(mask[np.newaxis, ..., np.newaxis], (7, 7)).numpy()
    return mask_downsampled


def apply_enhanced_cbam_to_image(image):
    # Preprocess the image
    original_image = image.copy()
    original_size = original_image.size

    # Convert to tensor and resize
    image_tensor = tf.convert_to_tensor(original_image, dtype=tf.float32)
    image_tensor = tf.image.resize(image_tensor, IMG_SIZE)

    # Normalize the image
    input_array = tf.keras.preprocessing.image.img_to_array(image_tensor) / normalization_unit
    input_tensor = tf.expand_dims(input_array, 0)

    # Create and convert face mask
    face_mask = create_face_filter(image_tensor)  # Ensure this returns the correct shape
    face_mask = tf.convert_to_tensor(face_mask, dtype=tf.float32)

    # Resize the face mask to the expected shape of (7, 7, 1)
    face_mask_resized = tf.image.resize(face_mask, (7, 7))

    # Expand dimensions to fit the model's expected input shape
    face_mask_resized = tf.expand_dims(face_mask_resized, axis=-1)  # Shape becomes (1, 7, 7, 1)

    # Call the model directly with both inputs
    cbam_output = cbam_layer_model([input_tensor, face_mask_resized])

    # Get the attention map
    attention_map = tf.reduce_mean(cbam_output, axis=-1).numpy().squeeze()

    # Handle potential NaN or inf values
    attention_map = np.nan_to_num(attention_map)

    # Normalize attention map
    attention_map_min = np.min(attention_map)
    attention_map_max = np.max(attention_map)
    if attention_map_min != attention_map_max:
        attention_map = (attention_map - attention_map_min) / (attention_map_max - attention_map_min)
    else:
        attention_map = np.zeros_like(attention_map)

    # Resize attention map to original image size
    attention_map_resized = cv2.resize(attention_map, (original_size[0], original_size[1]))

    # Apply attention map to original image
    original_array = np.array(original_image)
    result_image = (original_array * attention_map_resized[:, :, np.newaxis]).astype(np.uint8)
    result_image = Image.fromarray(result_image)

    return result_image


def walk_dataset_directory(src_dir, dst_dir):
    # Collect all files
    image_files = []
    cbam_image_files = []
    for root, dirs, files in os.walk(src_dir):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff')):
                image_files.append((root, file))

    for root, dirs, files in os.walk(dst_dir):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff')):
                cbam_image_files.append(file)

    # Initialize progress bar
    image_files_filtered = [tup for tup in image_files if tup[1] not in cbam_image_files]

    with tqdm(total=len(image_files_filtered), desc="Processing Images", unit="image") as pbar:
        # Loop through image files
        for root, file in image_files_filtered:
            src_file_path = os.path.join(root, file)
            try:
                # Open the image using Pillow
                with Image.open(src_file_path) as img:
                    image = img.convert('RGB')
                    cbam_picture = apply_enhanced_cbam_to_image(image)

                    # Generate new file name with suffix
                    filename, ext = os.path.splitext(file)
                    new_filename = f"{filename}{ext}"

                    # Generate the target directory structure
                    relative_path = os.path.relpath(root, src_dir)
                    dst_dir_with_structure = os.path.join(dst_dir, relative_path)

                    # Ensure the target directory exists
                    os.makedirs(dst_dir_with_structure, exist_ok=True)

                    # Define the new file path
                    dst_file_path = os.path.join(dst_dir_with_structure, new_filename)

                    # Save the resized image to the new location
                    cbam_picture.save(dst_file_path)
            except:
                print(f" ERROR: {file}")
                pass

            # Update progress bar
            pbar.update(1)


src_directory = "datasets/LFWPeople/lfw_funneled"
dst_directory = "datasets/lfw_cbam"
walk_dataset_directory(src_directory, dst_directory)
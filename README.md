# Masked Face Recognition

This repository implements methods and models for masked and unmasked face recognition using various neural network architectures and techniques such as ResNet, CBAM (Convolutional Block Attention Module), and Siamese Networks. Below is an overview of the Python scripts and their functions.

---

## Python Scripts

### `CBAM_module.py`

Contains the implementation of the Convolutional Block Attention Module (CBAM) using TensorFlow.

- `channel_attention`: Applies channel-wise attention to enhance feature maps.
- `spatial_attention`: Focuses on significant spatial regions using spatial attention.
- `cbam_block`: Combines channel and spatial attention to form the CBAM module.

### `cbam_on_image.py`

This script applies CBAM to images, enhancing their features for better recognition.

- `create_face_filter`: Creates a filter for facial regions.
- `apply_enhanced_cbam_to_image`: Applies CBAM to enhance image features.
- `walk_dataset_directory`: Iterates through a dataset directory to process images.

### `create_bin_dataset.py`

Generates binary datasets for training and testing.

- `image_to_bytes`: Converts images to byte format.
- `normal_pair_file_creator`: Creates pairs of images for training and testing.
- `tuple_pair_file_creator`: Creates tuples of image pairs.

### `dataset_preparation.py`

Prepares datasets by applying transformations and filtering.

- `create_face_filter`: Similar to the function in `cbam_on_image.py`, creates a facial filter.
- `apply_enhanced_cbam_to_image`: Applies CBAM to enhance dataset images.
- `walk_dataset_directory`: Iterates through directories to process datasets.

### `lfw_mfv_resnet50_siamese_cbam_arcface.py`

Implements a Siamese Network using ResNet50 with CBAM and ArcFace for masked face verification.

- `cbam_block`: Defines the CBAM module.
- `identity_block_with_cbam`: An identity block with CBAM.
- `conv_block_with_cbam`: A convolutional block with CBAM.
- `ResNet50_CBAM`: Builds a ResNet50 architecture with CBAM.
- `sanitize_name`: Sanitizes names in datasets.
- `identity_block`: A standard identity block.
- `conv_block`: A standard convolutional block.
- `ResNet50`: Builds a standard ResNet50 architecture.
- `load_weights_from_h5`: Loads pre-trained weights from an H5 file.
- `load_base_model`: Loads the base model for feature extraction.
- `create_siamese_model`: Constructs a Siamese model.
- `load_mlfw_pairs`: Loads face image pairs for training/testing.
- `verify_faces`: Verifies face pairs.
- `get_batch_size`: Returns batch size for training.
- `build_final_model`: Builds the final model for face verification.

### `lfw_siamese_resnet_cbam.py`

Implements a Siamese Network with ResNet50 and CBAM for face verification.

- Similar functions to `lfw_mfv_resnet50_siamese_cbam_arcface.py` with additional low-level attention mechanisms.

### `lfw.py`

Loads the Labeled Faces in the Wild (LFW) dataset and verifies face pairs.

- `load_base_model`: Loads the base model for feature extraction.
- `create_siamese_model`: Creates a Siamese network for verification.
- `load_mlfw_pairs`: Loads face image pairs from the LFW dataset.
- `verify_faces`: Verifies face pairs.

### `main.py`

The main script for dataset preparation, model training, and evaluation.

- `load_rof_dataset`: Loads a custom face dataset.
- `create_base_model`: Creates a base model for feature extraction.
- `create_siamese_model`: Builds a Siamese model.
- `contrastive_loss`: Defines the contrastive loss function.
- `create_pairs`: Creates pairs of images for training/testing.
- `save_data_pickle`: Saves processed data to a pickle file.
- `load_data_pickle`: Loads data from a pickle file.
- `save_model_tf`: Saves a TensorFlow model.
- `load_model_tf`: Loads a TensorFlow model.

### `resnet100_cbam.py`

Defines a ResNet100 architecture with CBAM.

- `convolutional_block`: A convolutional block with BatchNormalization and ReLU activation.
- `identity_block`: An identity block with BatchNormalization and ReLU activation.
- `cbam_block`: Implements the CBAM module.
- `resnet100_cbam`: Builds a ResNet100 architecture with CBAM.

### `siamese_with_cbam.py`

Implements a Siamese Network integrated with CBAM for masked face recognition.

- `build_resnet50_model`: Builds a ResNet50 model.
- `load_vggface2_weights`: Loads pre-trained weights from the VGGFace2 dataset.
- `integrate_cbam_into_resnet`: Integrates CBAM into the ResNet architecture.
- `build_siamese_network`: Constructs a Siamese network.
- `preprocess_images`: Preprocesses images for training/testing.
- `create_pairs`: Creates image pairs for training/testing.

---

## Usage

1. Clone the repository:
   ```bash
   git clone https://github.com/babakshofficial/masked-face-recognition.git
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the main script:
   ```bash
   python main.py
   ```

---

## Contributions

Feel free to contribute by submitting issues or pull requests.

---
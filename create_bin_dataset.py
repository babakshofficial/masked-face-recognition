import pickle
from PIL import Image
import io


# Helper function to resize an image and convert it to BytesIO
def image_to_bytes(image_path):
    with Image.open(image_path) as img:
        # Resize image to 112x112 and ensure it's RGB (3 channels)
        img = img.resize((112, 112)).convert('RGB')
        with io.BytesIO() as output:
            img.save(output, format='JPEG')  # Use the correct format (e.g., JPEG)
            return output.getvalue()  # Return binary image data


def normal_pair_file_creator(pairs_file_path, lfw_image_dir, dataset_name):
    # Initialize lists for images and labels
    images = []
    labels = []

    # Read the pairs.txt file
    with open(pairs_file_path, 'r') as pairs_file:
        lines = pairs_file.readlines()

        # Process each line in pairs.txt
        for line in lines:
            # Parse image paths and label
            parts = line.strip().split()
            image1_path = parts[0]  # Path to the first image
            image2_path = parts[1]  # Path to the second image
            label = bool(int(parts[2]))  # Convert 1/0 to True/False

            # Convert both images to binary format and add to the list
            image1_data = image_to_bytes(f"{lfw_image_dir}/{image1_path}")
            image2_data = image_to_bytes(f"{lfw_image_dir}/{image2_path}")

            # Append the images and label to respective lists
            # images.extend([image1_data, image2_data])  # Each row has two images
            images.append(image1_data)
            images.append(image2_data)
            labels.append(label)

        # The final structure of the .bin file data
        data = [images, labels]

        # Save the data to a .bin file
        with open(dataset_name, 'wb') as bin_file:
            pickle.dump(data, bin_file)

    print("The normal bin dataset has been created and saved as " + dataset_name)
    return


def tuple_pair_file_creator(pairs_file_path, lfw_image_dir, dataset_name):
    # Initialize lists for images and labels
    images = []
    labels = []

    # Read the new pairs file
    with open(pairs_file_path, 'r') as pairs_file:
        lines = pairs_file.readlines()

        # Process the lines two by two
        for i in range(0, len(lines) - 1, 2):
            # Parse image paths and labels
            line1 = lines[i].strip().split()
            line2 = lines[i + 1].strip().split()

            fullname1 = ' '.join(line1[0].split('_')[0:-1])
            fullname2 = ' '.join(line2[0].split('_')[0:-1])

            # Extract image paths and labels
            image1_path = line1[0]  # Path to the first image
            label1 = line1[1]  # Label number for the first image

            image2_path = line2[0]  # Path to the second image
            label2 = line2[1]  # Label number for the second image

            # Check if the two lines have the same label and identity
            is_same_label = (label1 == label2)
            is_same_name = (fullname1 == fullname2)

            # print(is_same_label, is_same_name, is_same_name and is_same_label)

            # Convert both images to binary format and add to the list
            image1_data = image_to_bytes(f"{lfw_image_dir}/{image1_path}")
            image2_data = image_to_bytes(f"{lfw_image_dir}/{image2_path}")

            # Append the images and label to respective lists
            # images.extend([image1_data, image2_data])  # Each pair has two images
            images.append(image1_data)
            images.append(image2_data)
            labels.append(is_same_name and is_same_label)  # Append True if same, False otherwise

        # The final structure of the .bin file data
        data = [images, labels]

        # Save the data to a .bin file
        with open(dataset_name, 'wb') as bin_file:
            pickle.dump(data, bin_file)

    print("The tuple bin dataset has been created and saved as " + dataset_name)
    return


normal_or_tuple = 'tuple'

pairs_file_path = 'datasets/calfw/pairs_CALFW.txt'
lfw_image_dir = 'datasets/calfw/images&landmarks/images&landmarks/images'
dataset_name = 'calfw.bin'

if normal_or_tuple == 'normal':
    normal_pair_file_creator(pairs_file_path, lfw_image_dir, dataset_name)
elif normal_or_tuple == 'tuple':
    tuple_pair_file_creator(pairs_file_path, lfw_image_dir, dataset_name)


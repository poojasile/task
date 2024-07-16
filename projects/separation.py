## separating 3 folders
import cv2
import os
import random
import shutil

# Define the paths for the input and output folders
input_folder = "/home/pooja-sile/Desktop/2_folder"
output_folder_1 = "/home/pooja-sile/Desktop//Training"
output_folder_2 = "/home/pooja-sile/Desktop//Validation"
output_folder_3 = "/home/pooja-sile/Desktop//Testing"

# Create the output folders if they don't exist
for folder in [output_folder_1, output_folder_2, output_folder_3]:
    if not os.path.exists(folder):
        os.makedirs(folder)

# Get the list of image files in the input folder
image_files = os.listdir(input_folder)

# Calculate the number of images for each category
total_images = len(image_files)
category_1_count = int(total_images * 0.8)
category_2_count = int(total_images * 0.1)
category_3_count = total_images - category_1_count - category_2_count

# Shuffle the image files randomly
random.shuffle(image_files)

# Move the images to the respective folders based on the distribution
for i, image_file in enumerate(image_files):
    image_path = os.path.join(input_folder, image_file)
    if i < category_1_count:
        output_path = os.path.join(output_folder_1, image_file)
    elif i < category_1_count + category_2_count:
        output_path = os.path.join(output_folder_2, image_file)
    else:
        output_path = os.path.join(output_folder_3, image_file)
    shutil.move(image_path, output_path)
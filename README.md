## Image Bounding Box Processor

This script processes images by drawing bounding boxes around objects specified in a CSV file containing bounding box coordinates. 
It also crops the images based on these bounding boxes.

## Installation

Make sure you have Python installed on your system. Additionally, install the required Python libraries by running pip:

pip install pillow

## Usage

1.Defining paths for csv file, image directory,and output directory
2.os.makedirs is used to create a directory 
3.Install required packages

   pip install -r requirements.txt

## Code Explaination

The script performs the following steps:

1.Importing Libraries:
  It imports necessary libraries including os, csv, and PIL for file manipulation, CSV handling, and image processing respectively.

2.Defining Paths:
  The script defines paths for the CSV file containing bounding box information (csv_file), the directory containing the original images (image_dir), and the          directory where the output images with bounding boxes will be saved (output_dir).

3.Creating Output Directory:
  It ensures that the output directory exists. If not, it creates it using os.makedirs(output_dir, exist_ok=True).

4.Defining Functions:
  draw_boxes: Draws bounding boxes around objects specified by the bounding box coordinates.
  crop_image: Crops the images based on the bounding box coordinates.

5.Reading CSV File and Processing Images:
  The script iterates through each row of the CSV file.For each row, it extracts the filename and constructs the full path to the corresponding image file.
    It then opens the image, extracts bounding box coordinates from the CSV row, and crops the image accordingly.
    Cropped images and images with bounding boxes drawn on them are saved to the output directory.

6.Saving Output:
   Cropped images are saved with filenames prefixed by an index and the original image name.
   Images with bounding boxes are saved with filenames prefixed by "full_" and the original image name.
   
## Example

 ### Crop Image
 
  ![image](https://github.com/poojasile/task/assets/169047585/42e726f4-868f-4b80-9895-52e3f068e98a)

### Bounding Box

  ![image](https://github.com/poojasile/task/assets/169047585/b06e8e78-7c98-478c-a4cd-7a1e9fdba215)







 

   







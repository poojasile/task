# Image Bounding Box Processor

   This script processes images by drawing bounding boxes around objects specified in a CSV file containing bounding box coordinates. 
   It also crops the images based on these bounding boxes using argument parser.

## Installation

  Make sure you have Python installed on your system. Additionally, install the required Python libraries by running pip:

    pip install pillow
    pip install argparse

## Usage

1.provide the input path , output path and csv path to the terminal as an argument

2.os.makedirs is used to create a directory 

3.Install required packages

     pip install -r requirements.txt

## Code Explaination

The script performs the following steps:

1.Importing Libraries:
  It imports necessary libraries including os, csv, and PIL for file manipulation, CSV handling, image processing and argparse respectively.

2.Defining Paths:
  The script defines paths for the CSV file containing bounding box information (csv_file), the directory containing the original images (--image_dir), and the directory where the output images with bounding boxes will be saved (--output_dir).
  parser = argparse.ArgumentParser()
  parser.add_argument("--image_path", help = "Enter the path of your image")
  parser.add_argument("--csv", help = "your csv file path , which has bounding box values")
  parser.add_argument("--out_dir", help = "name of output directory where you want to save your output")
  args = parser.parse_args()

3.Creating Output Directory:
  It ensures that the output directory exists. If not, it creates it using os.makedirs(output_dir, exist_ok=True).

4.Defining Functions:
  draw_boxes: Draws bounding boxes around objects specified by the bounding box coordinates.
  crop_image: Crops the images based on the bounding box coordinates.

5.Reading CSV File and Processing Images:
  The script iterates through each row of the CSV file.For each row, it extracts the filename and constructs the full path to the corresponding image file.
    It then opens the image, extracts bounding box coordinates from the CSV row, and crops the image accordingly.
    Cropped images and images with bounding boxes drawn on them are saved to the output directory.

6.passing given input paths to terminal:
      If we use optional arguments in input we have to use same in terminal.
      
7.Saving Output:
   Cropped images and bounding box images are saved with given filename in terminal.
## Example

 ### Crop Image:
 
  ![image](https://github.com/poojasile/task/assets/169047585/42e726f4-868f-4b80-9895-52e3f068e98a)

### Bounding Box:

  ![image](https://github.com/poojasile/task/assets/169047585/b06e8e78-7c98-478c-a4cd-7a1e9fdba215)










  # Histogram Plotter

  A histogram is a type of chart that shows the frequency distribution of data points across a continuous range of numerical values.
  This script reads an image file, computes the histogram for each color channel (blue, green, red), and visualizes the histograms using Matplotlib using argparse

  ## Installation
  
    Install Numpy,Opencv,and Matplotlib using pip:
    
     pip install numpy opencv-python matplotlib
     pip install argparse

  ## Usage
  
  1.provide the input path and output path to the terminal as an argument
    
  2.The script will generate a histogram plot for each color channel and display it.

  3.Install required packages

     pip install -r requirements.txt

  ## Code Explaination

  The script performs the following steps:

  1.Importing Libraries:
  
  It imports necessary libraries including numpy, cv2 (OpenCV), and matplotlib for numerical operations, image processing, plotting and argparse respectively.

  2.using argparse:
     parser = argparse.ArgumentParser()
     parser.add_argument("--image_path", help = "Enter the path of your image")
     parser.add_argument("--output_path", help = "Enter the output path of your image")
     args = vars(parser.parse_args())
     If any new person didnt understand code they can use help

  3.Reading and Writing Image:
  
  It reads the input image using OpenCV's cv.imread() function.
  If the image is successfully read, it writes it to a new location using cv.imwrite().
  It checks if the image is not None, otherwise raises an assertion error.
  

  4.Calculating Histogram and Plotting:
  
  It calculates the histogram for each color channel (BGR) using OpenCV's cv.calcHist() function.
  It plots the histograms using Matplotlib's plt.plot() function.

  5.passing given input paths to terminal:

      we have to pass input --image_path and --output_path .If we use optional arguments in input we have to use same in terminal.
      

  6.Displaying the Plot:
     
  It displays the plot showing histograms for each color channel.

  ## Example

  ### Input:
  
   ![image](https://github.com/poojasile/task/assets/169047585/0336e039-7838-475f-aab1-d55a9ca0af8a)

  ### Output:

   ![image](https://github.com/poojasile/task/assets/169047585/011459bc-e119-4620-a4e9-7cf9fc920b66)








# Iteration

 Iterating the first 10 numbers and in each iteration printing the sum of the current and previous number using argparse

## Usage

The script will print the current number, its previous number using argparse

parser = argparse.ArgumentParser(description="process some integers")
parser.add_argument('current_number',type =int)
parser.add_argument('previous_number',type =int)
args = parser.parse_args()

## Code Explaination

The script performs the following steps:

1.Initialization:

It initializes the variable previous_num to 0.

2.Looping through Range:

It iterates through a range from 0 to 9.

3.passing given input values to terminal:

   we have to give current and previous number in terminal . use help if code did not understand it will show clear details given in input.

3.Calculating Sum:

For each iteration, it calculates the sum of the current number with its previous number.

4.Printing Results:

It prints the current number, its previous number, and the sum.

## Example 

 ### Output:
 
Current number 0, Previous Number 0 is 0

Current number 1, Previous Number 0 is 1

Current number 2, Previous Number 1 is 3

Current number 3, Previous Number 2 is 5

Current number 4, Previous Number 3 is 7

Current number 5, Previous Number 4 is 9

Current number 6, Previous Number 5 is 11

Current number 7, Previous Number 6 is 13

Current number 8, Previous Number 7 is 15

Current number 9, Previous Number 8 is 17



# Live Video capture 

This script is to capture Live video from the webcam using argparse.

## Installation

Install OpenCV: If you haven't already, install the OpenCV library using pip

     pip install opencv-python
     pip install argparse

## Usage

define a video capture object 
video = cv2.VideoCapture(0) 

parser = argparse.ArgumentParser

Install required packages

     pip install -r requirements.txt

## Code Explaination

1.The code begins with importing the necessary modules. cv2 is imported for OpenCV functionalities, mimetypes is imported for MIME type handling and we have to import argparse to pass arguments through terminal

2.Creating Video Capture Object:

 The script creates a video capture object using cv2.VideoCapture(0) which captures video from the default camera (index 0).

3.This block checks if the camera is opened successfully. If the camera fails to open, it prints an error message.The code retrieves the frame width and height from the camera and converts them from floating-point values to integers. These values are used to set the resolution of the captured video.

4.cv2.VideoWriter() initializes a VideoWriter object to write the video frames to a file.

5.This loop continuously captures frames from the camera (video.read()), writes them to the output video file (result.write(frame)), and displays them in a window (cv2.imshow('Frame', frame)). It continues until the user presses the 's' key, which breaks the loop and stops the recording.

6.After the recording is stopped, the resources are released (video.release() and result.release()), and all OpenCV windows are closed (cv2.destroyAllWindows()). Finally, a message is printed indicating that the video was successfully saved with filename that are given in terminal.

## Example

### Output

  https://github.com/poojasile/task/assets/169047585/27ed5b71-f7c8-4837-98b6-e396bcec0623


   










  
  
  







 

   







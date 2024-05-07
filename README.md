## Bounding_Boxes

  1.INSTALL FOLLOWING PACKAGES
  
   ```csv,PIL```
   
  2.CODE
```bash
import os
import csv
from PIL import Image, ImageDraw


csv_file = "/home/pooja-sile/Downloads/7622202030987_bounding_box.csv"
image_dir = "/home/pooja-sile/Downloads/7622202030987"
output_dir = "/home/pooja-sile/Downloads/7622202030987_with_bounding_boxes"


os.makedirs(output_dir, exist_ok=True)


def draw_boxes(image, boxes):
    draw = ImageDraw.Draw(image)
    for box in boxes:
        left = int(box['left'])
        top = int(box['top'])
        right = int(box['right'])
        bottom = int(box['bottom'])
        draw.rectangle([left, top, right, bottom], outline="yellow")
    return image


def crop_image(image, boxes):
    cropped_images = []
    for box in boxes:
        left = int(box['left'])
        top = int(box['top'])
        right = int(box['right'])
        bottom = int(box['bottom'])
        cropped_img = image.crop((left, top, right, bottom))
        cropped_images.append(cropped_img)
    return cropped_images


with open(csv_file, 'r') as file:
    csv_reader = csv.DictReader(file)
    for row in csv_reader:
        image_name = row['filename']
        image_path = os.path.join(image_dir, image_name)
        output_path = os.path.join(output_dir, image_name)
        image = Image.open(image_path)
        boxes = [{'left': row['xmin'], 'top': row['ymin'], 'right': row['xmax'], 'bottom': row['ymax']}]
        cropped_images = crop_image(image, boxes)
        for i, cropped_img in enumerate(cropped_images):
            cropped_img.save(os.path.join(output_dir, f"{i}_{image_name}"))  
        full_image_with_boxes = draw_boxes(image, boxes)
        full_image_with_boxes.save(os.path.join(output_dir, f"full_{image_name}"))
```
## Histogram
```
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
 
img = cv.imread('shinchan.jpg')
cv.imwrite("/home/pooja-sile/Desktop/experiments/my.jpg",img)
assert img is not None, "file could not be read, check with os.path.exists()"
color = ('b','g','r')
for i,col in enumerate(color):
 histr = cv.calcHist([img],[i],None,[256],[0,256])
 plt.plot(histr,color = col)
 plt.xlim([0,256])
plt.show()

```
## Iteration
```
previous_num = 0
for i in range(10):
    sum = previous_num + i
    print(f'Current number {i} Previous Number {previous_num} is {sum}')
    previous_num = i
```
## Webcam
```




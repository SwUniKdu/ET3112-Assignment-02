# D/ENG/24/0023/EE - ET3112
# Assignment 02 on Fitting
# Question 01 

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# 1. Load the cropped image from the Images folder
image_path = '../Images/1c.jpg'
img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

if img is None:
    print(f"Error: Could not load image at {image_path}. Please check your folder names.")
    exit()

# 2. Apply the Canny edge detector algorithm
edges = cv2.Canny(img, 550, 690)

# 3. Assign the extracted feature positions to x and y coordinates
indices = np.where(edges != 0)
x = indices[1]
y = indices[0]

print(f"Successfully extracted {len(x)} edge points.")

# 4. Plot both the original image and the edge image
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.imshow(img, cmap='gray')
plt.title('Original Cropped Image (1c)')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(edges, cmap='gray')
plt.title('Extracted Edges (Canny)')
plt.axis('off')

# Save the plot inside the Question 1 folder
output_filename = 'q1_output.png'
plt.savefig(output_filename, bbox_inches='tight')
print(f"Plot saved successfully as '{output_filename}'.")

# Display the plot to the screen
plt.show()
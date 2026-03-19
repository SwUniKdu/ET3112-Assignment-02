# D/ENG/24/0023/EE - ET3112
# Assignment 02 on Fitting
# Question 02

import cv2
import numpy as np
import matplotlib.pyplot as plt

# 1. Load the image and extract edges (bringing in the data from Q1)
image_path = '../Images/1c.jpg'
img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

if img is None:
    print("Error: Could not load image. Check the path.")
    exit()

edges = cv2.Canny(img, 550, 690)
indices = np.where(edges != 0)
x = indices[1]
y = indices[0]

# 2. Plot the x and y in a scatter plot
plt.figure(figsize=(8, 6))

# s=1 makes the scatter points small so it looks like fine lines
plt.scatter(x, y, s=1, color='black') 

# Invert the y-axis so the plot matches the image orientation
plt.gca().invert_yaxis()

plt.title('Scatter Plot of Extracted Edges (Question 2)')
plt.xlabel('X coordinates')
plt.ylabel('Y coordinates')

# Save the plot inside the Question 2 folder
output_filename = 'q2_output.png'
plt.savefig(output_filename, bbox_inches='tight')
print(f"Scatter plot saved successfully as '{output_filename}'.")

# Display the plot
plt.show()
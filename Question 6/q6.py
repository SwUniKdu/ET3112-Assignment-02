# D/ENG/24/0023/EE - ET3112
# Assignment 02 on Fitting
# Question 06

import cv2
import numpy as np
import matplotlib.pyplot as plt

# 1. Load the image and extract edges
image_path = '../Images/1c.jpg'
img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

if img is None:
    print("Error: Could not load image. Check the path.")
    exit()

edges = cv2.Canny(img, 550, 690)
indices = np.where(edges != 0)
x = indices[1]
y = indices[0]

# 2. Total Least Squares (TLS) Fit using SVD
mean_x = np.mean(x)
mean_y = np.mean(y)

# Center all the data points around (0,0)
x_centered = x - mean_x
y_centered = y - mean_y

# Stack the centered coordinates into a matrix
pts = np.vstack((x_centered, y_centered)).T

# Perform Singular Value Decomposition (SVD) to find the main direction
U, S, Vt = np.linalg.svd(pts)

# The direction of the best fit line is the first row of Vt 
dx, dy = Vt[0]

# Calculate the slope (m = rise/run) and y-intercept (c)
m_tls = dy / dx
c_tls = mean_y - m_tls * mean_x

# 3. Plotting
plt.figure(figsize=(8, 6))

# Plot the original scatter points
plt.scatter(x, y, s=1, color='gray', label='Extracted Edges')

# Generate x values for the line and calculate corresponding y values
x_line = np.linspace(min(x), max(x), 100)
y_line = m_tls * x_line + c_tls

# Plot the TLS fit line in green to distinguish it from the OLS line
plt.plot(x_line, y_line, color='green', linewidth=2, label='TLS Fit Line')

# Invert the y-axis so the plot matches the image orientation
plt.gca().invert_yaxis()

plt.title('Total Least-Squares-Fit Line (Question 6)')
plt.xlabel('X coordinates')
plt.ylabel('Y coordinates')
plt.legend()

output_filename = 'q6_output.png'
plt.savefig(output_filename, bbox_inches='tight')
print(f"Plot saved successfully as '{output_filename}'.")
print(f"Calculated TLS Slope (m): {m_tls:.4f}")

plt.show()
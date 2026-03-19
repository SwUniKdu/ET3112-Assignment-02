# D/ENG/24/0023/EE - ET3112
# Assignment 02 on Fitting
# Question 07

import cv2
import numpy as np

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

x_centered = x - mean_x
y_centered = y - mean_y

pts = np.vstack((x_centered, y_centered)).T
U, S, Vt = np.linalg.svd(pts)

# The direction vector
dx, dy = Vt[0]

# Calculate the slope (m = dy/dx)
m_tls = dy / dx

# 3. Calculate the Angle
# angle in radians = arctan(m)
angle_rad = np.arctan(m_tls)

# Convert to degrees
angle_deg = np.degrees(angle_rad)
# Invert the angle so it matches the visual slope in Figure 1b
visual_angle = -angle_deg

print("\n--- Answer for Question 07 ---")
print(f"Calculated TLS Slope (m): {m_tls:.4f}")
print(f"Estimated Crop Field Angle (TLS): {visual_angle:.2f} degrees\n")
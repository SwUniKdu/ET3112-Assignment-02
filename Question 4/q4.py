# D/ENG/24/0023/EE - ET3112
# Assignment 02 on Fitting
# Question 04

import cv2
import numpy as np

# 1. Load the image and extract edges (same data as Q3)
image_path = '../Images/1c.jpg'
img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

if img is None:
    print("Error: Could not load image. Check the path.")
    exit()

edges = cv2.Canny(img, 550, 690)
indices = np.where(edges != 0)
x = indices[1]
y = indices[0]

# 2. Ordinary Least Squares (OLS) Fit
coefficients = np.polyfit(x, y, 1)
m_ols = coefficients[0] # This is the slope 'm'

# 3. Calculate the Angle
angle_rad = np.arctan(m_ols)

# Convert to degrees
angle_deg = np.degrees(angle_rad)

# We invert the angle mathematically so it matches the visual slope in Figure 1b.
visual_angle = -angle_deg

print("\n--- Answer for Question 04 ---")
print(f"Calculated Slope (m): {m_ols:.4f}")
print(f"Raw Calculated Angle: {angle_deg:.2f} degrees")
print(f"Adjusted Visual Crop Field Angle: {visual_angle:.2f} degrees\n")
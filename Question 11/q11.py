# D/ENG/24/0023/EE - ET3112
# Assignment 02 on Fitting
# Question 11

import cv2
import numpy as np
from sklearn.linear_model import RANSACRegressor

# 1. Load the image and extract edges (same data as before)
image_path = '../Images/1c.jpg'
img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

if img is None:
    print("Error: Could not load image. Check the path.")
    exit()

edges = cv2.Canny(img, 550, 690)
indices = np.where(edges != 0)
x = indices[1]
y = indices[0]

# 2. RANSAC Fit
X = x.reshape(-1, 1)
ransac = RANSACRegressor(random_state=42)
ransac.fit(X, y)

# Extract the slope (m) from the RANSAC estimator
m_ransac = ransac.estimator_.coef_[0]

# 3. Calculate the Angle
# angle in radians = arctan(m)
angle_rad = np.arctan(m_ransac)

# Convert to degrees
angle_deg = np.degrees(angle_rad)

# Invert the angle so it matches the visual slope in Figure 1b
visual_angle = -angle_deg

print("\n--- Answer for Question 11 ---")
print(f"Calculated RANSAC Slope (m): {m_ransac:.4f}")
print(f"Raw Calculated Angle: {angle_deg:.2f} degrees")
print(f"Adjusted Visual Crop Field Angle (RANSAC): {visual_angle:.2f} degrees\n")
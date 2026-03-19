# D/ENG/24/0023/EE - ET3112
# Assignment 02 on Fitting
# Question 03

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

# 2. Ordinary Least Squares (OLS) Fit
coefficients = np.polyfit(x, y, 1)
m_ols, c_ols = coefficients

# 3. Plotting
plt.figure(figsize=(8, 6))

# Plot the original scatter points in gray so the red line stands out
plt.scatter(x, y, s=1, color='gray', label='Extracted Edges')

# Generate x values for the line and calculate corresponding y values
x_line = np.linspace(min(x), max(x), 100)
y_line = m_ols * x_line + c_ols

# Plot the fitted line in red
plt.plot(x_line, y_line, color='red', linewidth=2, label='OLS Fit Line')

# Invert the y-axis so the plot matches the image orientation
plt.gca().invert_yaxis()

plt.title('Least-Squares-Fit Line (Question 3)')
plt.xlabel('X coordinates')
plt.ylabel('Y coordinates')
plt.legend()

# Save the plot inside the Question 3 folder
output_filename = 'q3_output.png'
plt.savefig(output_filename, bbox_inches='tight')
print(f"Plot saved successfully as '{output_filename}'.")

# Display the plot
plt.show()
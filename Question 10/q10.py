# D/ENG/24/0023/EE - ET3112
# Assignment 02 on Fitting
# Question 10

import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import RANSACRegressor

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

# 2. RANSAC Fit
# scikit-learn expects X to be a 2D array (a column of values)
X = x.reshape(-1, 1)

# Initialize the RANSAC regressor 
ransac = RANSACRegressor(random_state=42)
ransac.fit(X, y)

# 3. Plotting
plt.figure(figsize=(8, 6))

# Plot the original scatter points in gray
plt.scatter(x, y, s=1, color='gray', label='Extracted Edges (with noise)')

# Generate x values for the line (from min to max x)
x_line = np.linspace(x.min(), x.max(), 100).reshape(-1, 1)
# Predict the corresponding y values using the fitted RANSAC model
y_line = ransac.predict(x_line)

# Plot the RANSAC fit line in blue
plt.plot(x_line, y_line, color='blue', linewidth=2, label='RANSAC Robust Fit Line')

# Invert the y-axis so the plot matches the image orientation
plt.gca().invert_yaxis()

plt.title('RANSAC Line Estimation (Question 10)')
plt.xlabel('X coordinates')
plt.ylabel('Y coordinates')
plt.legend()

output_filename = 'q10_output.png'
plt.savefig(output_filename, bbox_inches='tight')
print(f"Plot saved successfully as '{output_filename}'.")

# Display the plot
plt.show()
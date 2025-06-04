'''
Author: fatemeh habibzadeh
Project: Color Blindness Simulation & Optimization
Description:
    This script simulates how images appear to people with 
    different types of color blindness (Protanopia, Deuteranopia, Tritanopia),
    and generates optimized versions to improve their visual perception.
    Created as part of my image processing and accessibility research.

GitHub: https://github.com/fatemeh-habibzadeh
linkedin: www.linkedin.com/in/fatemeh-habibzadeh-heris
email: fhabibzadehh@mail.com
'''
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# Color Blindness Simulation Functions
def simulate_protanopia(image):
    """Simulate Protanopia (red color blindness)"""
    transform_matrix = np.array([
        [0.567, 0.433, 0],
        [0.558, 0.442, 0],
        [0, 0.242, 0.758]
    ])
    data = np.dot(np.array(image), transform_matrix.T)
    return Image.fromarray(np.clip(data, 0, 255).astype(np.uint8))

def simulate_deuteranopia(image):
    """Simulate Deuteranopia (green color blindness)"""
    transform_matrix = np.array([
        [0.625, 0.375, 0],
        [0.7, 0.3, 0],
        [0, 0.3, 0.7]
    ])
    data = np.dot(np.array(image), transform_matrix.T)
    return Image.fromarray(np.clip(data, 0, 255).astype(np.uint8))

def simulate_tritanopia(image):
    """Simulate Tritanopia (blue color blindness)"""
    transform_matrix = np.array([
        [0.95, 0.05, 0],
        [0, 0.433, 0.567],
        [0, 0.475, 0.525]
    ])
    data = np.dot(np.array(image), transform_matrix.T)
    return Image.fromarray(np.clip(data, 0, 255).astype(np.uint8))

# Optimization Functions
def optimize_for_protanopia(image):
    """Enhance green and blue for better perception by Protanopia"""
    data = np.array(image)
    data[:, :, 1] = np.clip(data[:, :, 1] * 1.5, 0, 255)  # Green
    data[:, :, 2] = np.clip(data[:, :, 2] * 1.2, 0, 255)  # Blue
    return Image.fromarray(data.astype(np.uint8))

def optimize_for_deuteranopia(image):
    """Enhance red and blue for better perception by Deuteranopia"""
    data = np.array(image)
    data[:, :, 0] = np.clip(data[:, :, 0] * 1.5, 0, 255)  # Red
    data[:, :, 2] = np.clip(data[:, :, 2] * 1.2, 0, 255)  # Blue
    return Image.fromarray(data.astype(np.uint8))

def optimize_for_tritanopia(image):
    """Enhance red and green for better perception by Tritanopia"""
    data = np.array(image)
    data[:, :, 0] = np.clip(data[:, :, 0] * 1.5, 0, 255)  # Red
    data[:, :, 1] = np.clip(data[:, :, 1] * 1.2, 0, 255)  # Green
    return Image.fromarray(data.astype(np.uint8))

# Load Image
image = Image.open("image.jpg").convert("RGB")

# Generate Simulations and Optimizations
protanopia_sim = simulate_protanopia(image)
deuteranopia_sim = simulate_deuteranopia(image)
tritanopia_sim = simulate_tritanopia(image)

protanopia_opt = optimize_for_protanopia(image)
deuteranopia_opt = optimize_for_deuteranopia(image)
tritanopia_opt = optimize_for_tritanopia(image)

# Display in 3 Rows x 2 Columns Layout, organized by type
plt.figure(figsize=(10, 15))

images_to_display = [
    (protanopia_sim, "Protanopia Simulation"),
    (protanopia_opt, "Protanopia Optimization"),
    (deuteranopia_sim, "Deuteranopia Simulation"),
    (deuteranopia_opt, "Deuteranopia Optimization"),
    (tritanopia_sim, "Tritanopia Simulation"),
    (tritanopia_opt, "Tritanopia Optimization")
]

for i, (img, title) in enumerate(images_to_display, start=1):
    plt.subplot(3, 2, i)
    plt.imshow(img)
    plt.title(title)
    plt.axis("off")

plt.tight_layout()
plt.show()

# Save Output Images 
protanopia_sim.save("protanopia_simulation.jpg")
protanopia_opt.save("optimized_protanopia.jpg")

deuteranopia_sim.save("deuteranopia_simulation.jpg")
deuteranopia_opt.save("optimized_deuteranopia.jpg")

tritanopia_sim.save("tritanopia_simulation.jpg")
tritanopia_opt.save("optimized_tritanopia.jpg")
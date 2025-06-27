import numpy as np
import cv2
import matplotlib.pyplot as plt
import os

# Ensure output directory exists
os.makedirs("output", exist_ok=True)

def save_plot(image, title, filename):
    plt.figure()
    plt.imshow(image, cmap='gray')
    plt.title(title)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join("output", filename), bbox_inches='tight')
    plt.close()

# Step 1: Create a synthetic image with 2 objects and background
def create_synthetic_image():
    img = np.zeros((100, 100), dtype=np.uint8)
    cv2.rectangle(img, (20, 20), (40, 80), 85, -1)   # Object 1 (gray level 85)
    cv2.circle(img, (70, 50), 15, 170, -1)           # Object 2 (gray level 170)
    return img

# Step 2: Add Gaussian noise
def add_gaussian_noise(image, mean=0, std=15):
    noise = np.random.normal(mean, std, image.shape).astype(np.int16)
    noisy_img = np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return noisy_img

# Step 3: Apply Otsu’s thresholding
def apply_otsu(image):
    _, otsu_thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return otsu_thresh

# Step 4: Region Growing Implementation
def region_growing(image, seed, threshold=10):
    output = np.zeros_like(image)
    visited = np.zeros_like(image, dtype=bool)
    height, width = image.shape
    region_value = image[seed]
    queue = [seed]

    while queue:
        x, y = queue.pop(0)
        if visited[x, y]:
            continue
        visited[x, y] = True
        if abs(int(image[x, y]) - int(region_value)) <= threshold:
            output[x, y] = 255
            for dx, dy in [(-1,0), (1,0), (0,-1), (0,1)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < height and 0 <= ny < width and not visited[nx, ny]:
                    queue.append((nx, ny))
    return output

# Execute Part 1
img = create_synthetic_image()
noisy_img = add_gaussian_noise(img)
otsu_result = apply_otsu(noisy_img)

# Save plots for Part 1
save_plot(img, "Original Image", "original.png")
save_plot(noisy_img, "Noisy Image", "noisy.png")
save_plot(otsu_result, "Otsu Threshold", "otsu_result.png")

# Execute Part 2: Region Growing
seed1 = (30, 30)  # Inside rectangle
seed2 = (70, 50)  # Inside circle

region1 = region_growing(img, seed1, threshold=15)
region2 = region_growing(img, seed2, threshold=15)
combined = cv2.bitwise_or(region1, region2)

# Save plots for Part 2
save_plot(region1, "Region 1", "region1.png")
save_plot(region2, "Region 2", "region2.png")
save_plot(combined, "Combined Regions", "combined_regions.png")

print("All plots saved in the 'output' folder.")

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

# Load grayscale image
def load_image(filepath):
    img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Image not found at path: {filepath}")
    return img

# Add Gaussian noise
def add_gaussian_noise(image, mean=0, std=15):
    noise = np.random.normal(mean, std, image.shape).astype(np.int16)
    noisy_img = np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return noisy_img

# Apply Otsu's threshold
def apply_otsu(image):
    _, otsu_thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return otsu_thresh

# Region growing algorithm
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

# TASK 1: Noise + Otsu
def task1(filepath):
    img = load_image(filepath)
    noisy_img = add_gaussian_noise(img)
    otsu_result = apply_otsu(noisy_img)

    # Save outputs
    save_plot(img, "Task 1 - Original Image", "task1_original.png")
    save_plot(noisy_img, "Task 1 - Noisy Image", "task1_noisy.png")
    save_plot(otsu_result, "Task 1 - Otsu Result", "task1_otsu_result.png")

    return img

# TASK 2: Region Growing
def task2(image):
    seed1 = (30, 30)  # You may update these seed points for your image
    seed2 = (70, 50)

    region1 = region_growing(image, seed1, threshold=15)
    region2 = region_growing(image, seed2, threshold=15)
    combined = cv2.bitwise_or(region1, region2)

    # Save outputs
    save_plot(region1, "Task 2 - Region 1", "task2_region1.png")
    save_plot(region2, "Task 2 - Region 2", "task2_region2.png")
    save_plot(combined, "Task 2 - Combined Regions", "task2_combined.png")

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    image_path = "original_image.jpg"
    img = task1(image_path)
    task2(img)
    print("All task outputs saved in the 'output' folder.")

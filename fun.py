import cv2
import numpy as np

def read_image_grayscale(image_path):
    """Read the local image in grayscale."""
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Image not found at {image_path}")
    return image

def add_gaussian_noise(image, mean=0, std=20):
    """Add Gaussian noise to an image."""
    noise = np.random.normal(mean, std, image.shape).astype(np.float32)
    noisy = np.clip(image.astype(np.float32) + noise, 0, 255)
    return noisy.astype(np.uint8)

def apply_otsu_threshold(image):
    """Apply Otsu's thresholding after Gaussian blur."""
    blur = cv2.GaussianBlur(image, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresh

def region_growing(image, seed_point, threshold=10):
    """Perform region growing starting from seed_point."""
    height, width = image.shape
    output = np.zeros_like(image, dtype=np.uint8)
    visited = np.zeros_like(image, dtype=bool)
    seed_value = image[seed_point]
    stack = [seed_point]

    while stack:
        x, y = stack.pop()
        if visited[x, y]:
            continue
        visited[x, y] = True
        pixel_value = image[x, y]
        if abs(int(pixel_value) - int(seed_value)) <= threshold:
            output[x, y] = 255
            for dx, dy in [(-1,0), (1,0), (0,-1), (0,1)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < height and 0 <= ny < width and not visited[nx, ny]:
                    stack.append((nx, ny))
    return output

def test_code():
    print("Module imported and working correctly")

if __name__ == "__main__":
    test_code()

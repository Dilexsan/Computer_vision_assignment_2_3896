import cv2
import matplotlib.pyplot as plt
from fun import read_image_grayscale, add_gaussian_noise, apply_otsu_threshold, region_growing, test_code

def show(title, image, cmap='gray'):
    plt.figure()
    plt.title(title)
    plt.imshow(image, cmap=cmap)
    plt.axis('off')

def main(image_path):
     
    # Load and process image
    try:
        base_image = read_image_grayscale(image_path)
    except FileNotFoundError as e:
        print(e)
        return

    noisy_image = add_gaussian_noise(base_image)

    # 1. Otsu’s Thresholding
    otsu_result = apply_otsu_threshold(noisy_image)

    # 2. Region Growing from seed (set a valid seed point)
    seed_point = (30, 30)  # You may change this based on your image
    region = region_growing(noisy_image, seed_point, threshold=15)

    # Show Results
    show("Original Image", base_image)
    show("Noisy Image", noisy_image)
    show("Otsu's Thresholding", otsu_result)
    show("Region Growing Result", region)

    plt.show()

if __name__ == '__main__':

    path = f"D:\\d\\7\\EC7205 Computer Vision and Image Processing\\assignment2\\face.jpg"
    test_code()
    main(path)

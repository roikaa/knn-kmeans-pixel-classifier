"""
Mabrouki Ala Eddin
TP Machine Learning - Master 1 IA
Université de Tamanrasset
Comparaison classification supervisee et non-suppervisee

Auteur: M. TAFFAR
"""

import copy
import random
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


class structure_pixel:
    def __init__(self, x, y, r, g, b, label=None):
        self.x = x
        self.y = y
        self.r = r
        self.g = g
        self.b = b
        self.label = label  # red, green, blue

    def __repr__(self):
        return (
            f"Pixel({self.x}, {self.y}, RGB({self.r},{self.g},{self.b}), {self.label})"
        )


def get_label(r, g, b):
    t = r + g + b
    rp = r / t
    gp = g / t
    bp = b / t
    if t == 0:
        label = "blue"  # r,g,b/0 = error, t=0 is black
    elif rp > gp and rp > bp:
        label = "red"
    elif gp > rp and gp > bp:
        label = "green"
    else:
        label = "blue"
    return label


# load image as pixels each with their attributes
def loadimg(IMG_PATH):
    img = Image.open(IMG_PATH)
    img = img.convert("RGB")

    # I = NxM
    # a_ij= I[i,j]
    width, hight = img.size
    pixels = []

    for x in range(width):
        for y in range(hight):
            r, g, b = img.getpixel((x, y))
            label = get_label(r, g, b)
            pixels.append(structure_pixel(x, y, r, g, b, label))
    return pixels, width, hight


def creat_training_set(pixel, size):
    if size > len(pixel):
        size = len(pixel)
    training_set = random.sample(pixel, size)
    return training_set


def distance_manhattan(p1, p2):
    distance = abs(p1.r - p2.r) + abs(p1.g - p2.g) + abs(p1.b - p2.b)
    return distance


class knn:
    def __init__(self, test, train, k):
        self.test = test
        self.train = train
        self.k = k

    def predict(self, test):
        predictions = [self.predict_label(new_pixel) for new_pixel in test]
        for i, pixel in enumerate(test):
            pixel.label = predictions[i]

        return test

    def predict_label(self, new_pixel):
        distances = [distance_manhattan(pixel, new_pixel) for pixel in self.train]

        k_nearest_indicat = np.argsort(distances)[: self.k]
        k_nearest_label = [self.train[i].label for i in k_nearest_indicat]

        vote = Counter(k_nearest_label).most_common(1)[0][0]
        return vote


# TODO: implement K-Means


def pixels_to_image(classified_pixels, width, height):
    img_array = np.zeros((height, width, 3), dtype=np.uint8)

    color_map = {"red": (255, 0, 0), "green": (0, 255, 0), "blue": (0, 0, 255)}

    for pixel in classified_pixels:
        color = color_map.get(pixel.label, (0, 0, 0))
        img_array[pixel.y, pixel.x] = color

    return img_array


def calculate_accuracy(original, predicted):
    correct = sum(
        1 for i in range(len(original)) if original[i].label == predicted[i].label
    )
    return (correct / len(original)) * 100


def show_comparison(uploaded_img_path, original_img, knn_img, knn_accuracy):
    num_images = 3
    fig, axes = plt.subplots(1, num_images, figsize=(20, 5))

    # Load and show uploaded image
    uploaded = Image.open(uploaded_img_path)
    axes[0].imshow(uploaded)
    axes[0].set_title("Uploaded Image", fontsize=14, fontweight="bold")
    axes[0].axis("off")

    # Show original labels
    axes[1].imshow(original_img)
    axes[1].set_title("Original Labels", fontsize=14, fontweight="bold")
    axes[1].axis("off")

    # Show KNN result with accuracy
    axes[2].imshow(knn_img)
    axes[2].set_title(
        f"KNN Classification\nAccuracy: {knn_accuracy:.2f}%",
        fontsize=14,
        fontweight="bold",
    )
    axes[2].axis("off")

    # Add overall title with accuracy
    fig.suptitle(
        f"Image Classification Comparison - KNN Accuracy: {knn_accuracy:.2f}%",
        fontsize=16,
        fontweight="bold",
        y=0.98,
    )

    plt.tight_layout()
    plt.show()


def main():
    TRAIN_SIZE = 15
    IMG_PATH = "Images/cool-pfp-1300.png"
    K_VALUE = 3

    # Load image pixels
    original_pixels, width, height = loadimg(IMG_PATH)

    knn_pixels = copy.deepcopy(original_pixels)

    # KNN Classification
    training_set = creat_training_set(original_pixels, TRAIN_SIZE)
    knn_classifier = knn(knn_pixels, training_set, K_VALUE)
    knn_predictions = knn_classifier.predict(knn_pixels)

    # Calculate accuracy
    knn_accuracy = calculate_accuracy(original_pixels, knn_pixels)
    print(f"KNN Accuracy: {knn_accuracy:.2f}%")

    original_img = pixels_to_image(original_pixels, width, height)
    knn_img = pixels_to_image(knn_predictions, width, height)

    show_comparison(IMG_PATH, original_img, knn_img, knn_accuracy)


main()

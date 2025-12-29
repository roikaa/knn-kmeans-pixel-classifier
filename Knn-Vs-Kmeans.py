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
        self.label = label  # [Red, Green, Blue]/{Black, Gray, White}

    def __repr__(self):
        return (
            f"Pixel({self.x}, {self.y}, RGB({self.r},{self.g},{self.b}), {self.label})"
        )


def get_label(r, g, b):
    t = r + g + b
    if t == 0:
        label = None  # r,g,b/0 = error, t=0 is black
    else: 
        rp = r / t # R%
        gp = g / t # G%
        bp = b / t # B%
        if rp > gp and rp > bp:
            label = "red"
        elif gp > rp and gp > bp:
            label = "green"
        else:
            label = "blue"
    return label


# NOTE: our image is basicly the dataset
# were is the image is a nxm matrix (I=M*M)
# and each pixel a_ij= (x, y)
# <structure_pixel> = (x, y, r, g, b, label)


# load image as pixels each with attributes
def loadimg(IMG_PATH):
    img = Image.open(IMG_PATH)
    img = img.convert("RGB")
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


# NOTE: K-NN algorithem
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


# NOTE: K-Means algorithem
class k_means:
    def __init__(self, k=3):
        self.k = k
        self.centroids = None

    def fit(self, pixels, max_iterations=10):
        # Step 1: select random centroids
        self.centroids = random.sample(pixels, self.k)

        # Step 2: calculate the distance centroids and each point
        for iteration in range(max_iterations):
            clusters = [[] for _ in range(self.k)]
            # clusters = []
            for pixel in pixels:
                distances = [
                    distance_manhattan(pixel, centroid) for centroid in self.centroids
                ]
                cluster_num = np.argmin(distances)
                clusters[cluster_num].append(pixel)

            # Step3 : Update centroids
            new_centroids = []
            for cluster in clusters:
                if len(cluster) > 0:
                    avg_r = sum(pixel.r for pixle in cluster) / len(cluster)
                    avg_g = sum(pixle.g for pixle in cluster) / len(cluster)
                    avg_b = sum(pixle.b for pixle in cluster) / len(cluster)
                    label = get_label(int(avg_r), int(avg_g), int(avg_b))
                    new_centroids.append(
                        structure_pixel(0, 0, int(avg_r), int(avg_g), int(avg_b), label)
                    )
                else:
                    # Keep old centroid if cluster is empty
                    new_centroids.append(self.centroids[len(new_centroids)])

            self.centroids = new_centroids

        # Step 4: repeat 2 and 3 for n iterations
        return self

    # Step 5 (final): get the label of each cluster
    def predict(self, pixels):
        predictions = []
        for pixel in pixels:
            distances = [
                distance_manhattan(pixel, centroid) for centroid in self.centroids
            ]
            cluster_num = np.argmin(distances)
            pixel.label = self.centroids[cluster_num].label
            predictions.append(pixel)

        return predictions


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


def show_comparison(
    uploaded_img_path, original_img, knn_img, kmeans_img, knn_accuracy, kmeans_accuracy
):
    fig, axes = plt.subplots(1, 4, figsize=(24, 6))

    # Uploaded image
    uploaded = Image.open(uploaded_img_path)
    axes[0].imshow(uploaded)
    axes[0].set_title("Uploaded Image", fontsize=14, fontweight="bold")
    axes[0].axis("off")

    # Original labels
    axes[1].imshow(original_img)
    axes[1].set_title("Original Labels", fontsize=14, fontweight="bold")
    axes[1].axis("off")

    # KNN result
    axes[2].imshow(knn_img)
    axes[2].set_title(
        f"KNN Classification\nAccuracy: {knn_accuracy:.2f}%",
        fontsize=14,
        fontweight="bold",
    )
    axes[2].axis("off")

    # K-Means result
    axes[3].imshow(kmeans_img)
    axes[3].set_title(
        f"K-Means Classification\nAccuracy: {kmeans_accuracy:.2f}%",
        fontsize=14,
        fontweight="bold",
    )
    axes[3].axis("off")

    fig.suptitle(
        f"Image Classification Comparison - KNN: {knn_accuracy:.2f}% | K-Means: {kmeans_accuracy:.2f}%",
        fontsize=16,
        fontweight="bold",
        y=0.98,
    )
    plt.tight_layout()
    plt.show()


def main():
    TRAIN_SIZE = 15
    IMG_PATH = "Images/cool-pfp-1300.png"
    K_MEANS = 3
    K_NN = 5

    # Load image pixels
    original_pixels, width, height = loadimg(IMG_PATH)

    knn_pixels = copy.deepcopy(original_pixels)
    kmeans_pixels = copy.deepcopy(original_pixels)

    # KNN Classification
    training_set = creat_training_set(original_pixels, TRAIN_SIZE)
    knn_classifier = knn(knn_pixels, training_set, K_NN)
    knn_predictions = knn_classifier.predict(knn_pixels)

    # K-Means Classification
    kmeans_classifier = k_means(k=K_MEANS)
    kmeans_classifier.fit(original_pixels, max_iterations=10)
    kmeans_predictions = kmeans_classifier.predict(kmeans_pixels)

    # Calculate accuracy
    knn_accuracy = calculate_accuracy(original_pixels, knn_predictions)
    kmeans_accuracy = calculate_accuracy(original_pixels, kmeans_predictions)

    original_img = pixels_to_image(original_pixels, width, height)
    knn_img = pixels_to_image(knn_predictions, width, height)
    kmeans_img = pixels_to_image(kmeans_predictions, width, height)

    show_comparison(
        IMG_PATH, original_img, knn_img, kmeans_img, knn_accuracy, kmeans_accuracy
    )


main()

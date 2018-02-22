import numpy as np
from matplotlib import pyplot as plt

def makesquares(images, nr_images_per_side):
    images_to_plot = np.concatenate(
        [np.concatenate([images[j*nr_images_per_side+i].reshape((28,28)) for i in range(0,nr_images_per_side)],
                        axis=1)
         for j in range(0,nr_images_per_side)],
        axis=0)
    return images_to_plot

def plotsquares(originals, reconstructs, nr_images_per_side):
    originals_square = makesquares(originals, nr_images_per_side)
    reconstructs_square = makesquares(reconstructs, nr_images_per_side)
    combined = np.concatenate([originals_square, reconstructs_square], axis=1)
    plt.imsave('combined.png', combined)

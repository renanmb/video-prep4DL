from matplotlib import pyplot as plt
from matplotlib import gridspec


def show_image(image):
    plt.imshow(image, cmap='gray', vmin=0, vmax=255)
    plt.show()


def show_images(images):
    grid = gridspec.GridSpec(2, 4)

    for i in range(len(images)):
        plt.subplot(grid[i])
        plt.axis('off')
        plt.imshow(images[i])


def show_images_greyscale(images):
    grid = gridspec.GridSpec(1, len(images))

    for i in range(len(images)):
        plt.subplot(grid[i])
        plt.axis('off')
        plt.imshow(images[i], cmap='gray', vmin=0, vmax=255)


from PIL import Image
import numpy as np
from scipy.ndimage import sobel
from scipy.ndimage import binary_erosion


def isolate_white_parts(image, threshold=125):
    grayscale_image = image.convert("L")

    binary_image = grayscale_image.point(lambda p: p > threshold and 255)

    edges = sobel(np.array(grayscale_image))

    refined_edges = edges * binary_image

    eroded_image = (
        binary_erosion(refined_edges, structure=np.ones((3, 3))).astype(np.uint8) * 255
    )

    return eroded_image


input_image = Image.open("Loo 2.png")

output_image = isolate_white_parts(input_image)

output_image.save("output.jpg")

# input_image.show()
# Image.fromarray(output_image).show()

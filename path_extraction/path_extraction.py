from PIL import Image

def isolate_white_parts(image, threshold=125):
    grayscale_image = image.convert("L")
    
    binary_image = grayscale_image.point(lambda p: p > threshold and 150)
    
    return binary_image

input_image = Image.open("Loo 2.png")
output_image = isolate_white_parts(input_image)
output_image.save("output.jpg")
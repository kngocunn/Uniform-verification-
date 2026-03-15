import cv2

def crop_border(image, ratio=0.1):

    h, w = image.shape[:2]

    top = int(h * ratio)
    bottom = int(h * (1 - ratio))

    left = int(w * ratio)
    right = int(w * (1 - ratio))

    cropped = image[top:bottom, left:right]

    return cropped

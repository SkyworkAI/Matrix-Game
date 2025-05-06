from PIL import Image
def black_image(width, height):
    """generate a black image

    Args:
        width (int): image width
        height (int): image height

    Returns:
        _type_: a black image
    """
    black_image = Image.new("RGB", (width, height), (0, 0, 0))
    return black_image

def numpy_to_pil(images):
    if images.ndim == 3:
        images = images[None, ...]
    images = (images * 255).round().astype("uint8")
    if images.shape[-1] == 1:
        # special case for grayscale (single channel) images
        pil_images = [Image.fromarray(image.squeeze(), mode="L") for image in images]
    else:
        pil_images = [Image.fromarray(image) for image in images]

    return pil_images


"""Shared PIL preprocessing for training, evaluation, and converter calibration."""

from PIL import Image


def resize_crop_image(image, image_size):
    """Return a PIL RGB image; size is an int or (height, width).

    Aspect-fill uses bilinear PIL resizing followed by a rounded center crop.
    For squares this matches torchvision Resize(int) then CenterCrop(int).
    """
    height, width = (image_size, image_size) if isinstance(image_size, int) else image_size
    if any(type(value) is not int or value < 1 for value in (height, width)):
        raise ValueError("Image dimensions must be positive integers")
    image = image.convert("RGB")
    source_width, source_height = image.size
    if source_width * height >= source_height * width:
        resized = (max(width, int(height * source_width / source_height)), height)
    else:
        resized = (width, max(height, int(width * source_height / source_width)))
    if image.size != resized:
        image = image.resize(resized, Image.Resampling.BILINEAR)
    left = int(round((resized[0] - width) / 2.0))
    top = int(round((resized[1] - height) / 2.0))
    image = image.crop((left, top, left + width, top + height))
    return image


def preprocess_image(image, image_size):
    """Return RGB float32 CHW in [0, 255] without input normalization."""
    import torch
    from torchvision.transforms.functional import pil_to_tensor

    return pil_to_tensor(resize_crop_image(image, image_size)).to(dtype=torch.float32)

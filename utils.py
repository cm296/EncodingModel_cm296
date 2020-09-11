import os
from torchvision.transforms import functional as tr
from PIL import Image



def listdir(dir, path=True):
    files = os.listdir(dir)
    files = [f for f in files if f != '.DS_Store']
    files = sorted(files)
    if path:
        files = [os.path.join(dir, f) for f in files]
    return files


def image_to_tensor(image, resolution=None, do_imagenet_norm=True):
    if isinstance(image, str):
        image = Image.open(image).convert('RGB')
    if resolution is not None:
        image = tr.resize(image, resolution)
    if image.width != image.height:     # if not square image, crop the long side's edges
        r = min(image.width, image.height)
        image = tr.center_crop(image, (r, r))
    image = tr.to_tensor(image)
    if do_imagenet_norm:
        image = imagenet_norm(image)
    return image



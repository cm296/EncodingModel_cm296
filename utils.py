import os
from torchvision.transforms import functional as tr
from PIL import Image
# import os.path
import torch
from utils import listdir, image_to_tensor
from tqdm import tqdm


def load_features(model, conditionsPath):
    #takes model and loads the features for that image, needs path to of files in directory
    conditions = listdir(conditionsPath)
    condition_features = {}
    for c in tqdm(conditions):
        c_name = c.split('/')[-1]
        stimuli = listdir(c)
        #resize according to resolution and square the image
        stimuli = [image_to_tensor(s, resolution=resolutionval) for s in stimuli]
        stimuli = torch.stack(stimuli)
        if torch.cuda.is_available():
            stimuli = stimuli.cuda()
        with torch.no_grad():
            #average across the same category
            feats = model(stimuli).mean(dim=0).cpu().numpy()
        condition_features[c_name] = feats
    return condition_features


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



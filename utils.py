import os
from torchvision.transforms import functional as tr
from PIL import Image
# import os.path
import torch
from tqdm import tqdm

# ImageNet mean and standard deviation. All images
# passed to a PyTorch pre-trained model (e.g. AlexNet) must be
# normalized by these quantities, because that is how they were trained.
imagenet_mean = (0.485, 0.456, 0.406)
imagenet_std = (0.229, 0.224, 0.225)

def compute_features(model, conditionsPath, resolutionval):
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


def image_to_tensor(image, resolution=None,padding=None, do_imagenet_norm=True):
    if isinstance(image, str):
        image = Image.open(image).convert('RGB')
    if image.width != image.height:     # if not square image, crop the long side's edges to make it square
        r = min(image.width, image.height)
        image = tr.center_crop(image, (r, r))
    if padding:     # if not square image, crop the long side's edges to make it square
        image = tr.pad(input=data, pad=(padding, padding, padding, padding), mode='constant', value=0)
    if resolution is not None:#f size is an int, smaller edge of the image will be matched to this number
        image = tr.resize(image, resolution) 
    image = tr.to_tensor(image)
    if do_imagenet_norm:
        image = imagenet_norm(image)
    return image

def imagenet_norm(image):
    dims = len(image.shape)
    if dims < 4:
        image = [image]
    image = [tr.normalize(img, mean=imagenet_mean, std=imagenet_std) for img in image]
    image = torch.stack(image, dim=0)
    if dims < 4:
        image = image.squeeze(0)
    return image



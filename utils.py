import os
from torchvision.transforms import functional as tr
from PIL import Image
# import os.path
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from random import randrange

# ImageNet mean and standard deviation. All images
# passed to a PyTorch pre-trained model (e.g. AlexNet) must be
# normalized by these quantities, because that is how they were trained.
imagenet_mean = (0.485, 0.456, 0.406)
imagenet_std = (0.229, 0.224, 0.225)

def compute_features(model, conditionsPath, resolutionval,paddingval=0,padding_mode='constant'):
    #takes model and loads the features for that image, needs path to of files in directory
    conditions = listdir(conditionsPath)
    condition_features = {}
    for c in tqdm(conditions):
        c_name = c.split('/')[-1]
        stimuli = listdir(c)
        #resize according to resolution and square the image
        stimuli = [image_to_tensor(s, resolution=resolutionval,paddingval=paddingval,padding_mode = padding_mode) for s in stimuli]
        stimuli = torch.stack(stimuli)
        # print(stimuli)
        if torch.cuda.is_available():
            stimuli = stimuli.cuda()
        with torch.no_grad():
            #average across the same category
            feats = model(stimuli).mean(dim=0).cpu().numpy()
        condition_features[c_name] = feats
    return condition_features

def plot_tensor_example(conditionsPath, resolution,paddingval=0,padding_mode='constant'):
    #takes model and loads the features for that image, needs path to of files in directory
    conditions = listdir(conditionsPath)
    x = randrange(len(conditions))
    c_name = conditions[x].split('/')[-1]
    stimuli = listdir(conditions[x])
    s = stimuli[0]
    #resize according to resolution and square the image
    tensor_image = image_to_tensor(s, resolution=resolution,paddingval = paddingval,padding_mode = padding_mode)
    # #convert image back to Height,Width,Channels
    img = np.transpose(tensor_image.numpy(), (1,2,0))
    # #show the image
    print('Plotting Image ' + stimuli[0].split('/')[-1])
    plt.imshow(img)
    plt.show()  
    # return condition_features

def listdir(dir, path=True):
    files = os.listdir(dir)
    files = [f for f in files if f != '.DS_Store']
    files = sorted(files)
    if path:
        files = [os.path.join(dir, f) for f in files]
    return files


def image_to_tensor(image, resolution=None,paddingval=None,padding_mode = 'constant', do_imagenet_norm=True, do_padding = True):
    if isinstance(image, str):
        image = Image.open(image).convert('RGB')
    if image.width != image.height:     # if not square image, crop the long side's edges to make it square
        r = min(image.width, image.height)
        image = tr.center_crop(image, (r, r))
    if do_padding:     # if not square image, crop the long side's edges to make it square
        image = tr.pad(image, padding=paddingval, padding_mode=padding_mode, fill=0)
        # image = tr.pad(input=data, mode='reflect', value=0)
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



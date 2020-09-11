import os.path
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

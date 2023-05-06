import torch
import torchvision
import torch.nn as nn
import pandas as pd
from tqdm import tqdm
import argparse
import numpy as np

from models.simple_classifier import *
from utils.fathomnet_loader import *
from utils.collate_fn import *

to_np = lambda x: x.data.cpu().numpy()

def enable_dropout(m):
    for each_module in m.modules():
        if each_module.__class__.__name__.startswith('Dropout'):
            each_module.train()
      
def disable_dropout(m):
    for each_module in m.modules():
        if each_module.__class__.__name__.startswith('Dropout'):
            each_module.eval()

parser = argparse.ArgumentParser(description='Hyperparams')
parser.add_argument('--arch', type=str, default='densenet',
                    help='Architecture to use densenet|resnet101')

args = parser.parse_args()

if args.arch == "resnet101":
    model_dict = torch.load('./saved_models/coco/resnet101_best.pth')
    clsfier_dict = torch.load('./saved_models/coco/resnet101clsfier_best.pth')
    orig_resnet = torchvision.models.resnet101(pretrained=True)
    features = list(orig_resnet.children())
    model= nn.Sequential(*features[0:8])
    clsfier = SimpleClassifier(2048, 290)
elif args.arch == "densenet":
    model_dict = torch.load('./saved_models/coco/densenet_best.pth')
    clsfier_dict = torch.load('./saved_models/coco/densenetclsfier_best.pth')
    orig_densenet = torchvision.models.densenet121(pretrained=True)
    features = list(orig_densenet.features)
    model = nn.Sequential(*features, nn.ReLU(inplace=True))
    clsfier = SimpleClassifier(1024, 290)

model.load_state_dict(model_dict)
clsfier.load_state_dict(clsfier_dict)

model.eval()
clsfier.eval()

test_data = FathomNetLoader(root='./datasets/eval', annFile='./datasets/eval.json')

test_loader = data.DataLoader(test_data, batch_size=1, num_workers=4, shuffle=False, pin_memory=False, collate_fn=coco_collate_val)

results = []
n_samples = 10

for i, (images, path) in tqdm(enumerate(test_loader)):
    nnOutputs = clsfier(model(images)).cuda()
    preds = torch.sigmoid(nnOutputs).cuda()
    
    # osd
    nnOutputs = torch.log(1+torch.exp(nnOutputs))
    
    # Loop over the predictions and extract the indices where the value is 1
    for j in range(preds.shape[0]):
        # _, indices = preds[j].topk(k=5)
        indices = (preds[j] >= 0.5).nonzero().flatten()

        categories = [str(index) for index in to_np(indices)]
        if not len(categories):
            categories = ['1', '2']
		
        enable_dropout(clsfier)
        scores = []
        for i in range(n_samples):
            nnOutput = clsfier(model(images)).cuda()
            nnOutput = torch.log(1+torch.exp(nnOutput))
            score = min(1, to_np(torch.sum(nnOutput[j])))
            scores.append(score)
        disable_dropout(clsfier)
        scores = np.mean(scores)
        
        # Add the image and categories to the results list
        results.append({'id': path[j][:-4],
                        'categories': '[' + ' '.join(categories) + ']', 
                        'osd': round(1-scores, 1)})
# Convert the results list to a Pandas DataFrame
df = pd.DataFrame(results)

# Save the DataFrame to a CSV file
df.to_csv('submission.csv', index=False)
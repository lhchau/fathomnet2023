import torch
import torchvision
import torch.nn as nn
import pandas as pd
from tqdm import tqdm
import argparse

from models.simple_classifier import *
from utils.fathomnet_loader import *
from utils.collate_fn import *

to_np = lambda x: x.data.cpu().numpy()

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
    clsfier = SimpleClassifier(2048, args.n_classes)
elif args.arch == "densenet":
    model_dict = torch.load('./saved_models/coco/densenet_best.pth')
    clsfier_dict = torch.load('./saved_models/coco/densenetclsfier_best.pth')
    orig_densenet = torchvision.models.densenet121(pretrained=True)
    features = list(orig_densenet.features)
    model = nn.Sequential(*features, nn.ReLU(inplace=True))
    clsfier = SimpleClassifier(1024, args.n_classes)

model.load_state_dict(model_dict)
clsfier.load_state_dict(clsfier_dict)

model.eval()
clsfier.eval()

test_data = FathomNetLoader(root='./datasets/eval', annFile='./datasets/eval.json')

test_loader = data.DataLoader(test_data, batch_size=1, num_workers=8, shuffle=False, pin_memory=True, collate_fn=coco_collate)

results = []

for i, (images, path) in tqdm(enumerate(test_loader)):
    nnOutputs = clsfier(model(images)).cuda()
    preds = torch.sigmoid(nnOutputs).cuda()
    
    # categories
    labels = torch.ones(preds.shape).cuda() * (preds >= 0.001)
    
    # osd
    nnOutputs = torch.log(1+torch.exp(nnOutputs))
    scores = to_np(torch.sum(nnOutputs))
    
    if scores > 1:
        scores = 1

    # Loop over the predictions and extract the indices where the value is 1
    for j in range(labels.shape[0]):
        indices = torch.where(labels[j] == 1)[0].tolist()
        categories = [str(index) for index in indices]
        if not len(categories):
            categories = ['1', '2']
		
        # Add the image and categories to the results list
        results.append({'id': path[0][:-4],
                        'categories': '[' + ' '.join(categories) + ']', 
                        'osd': round(1-scores, 1)})
        
# Convert the results list to a Pandas DataFrame
df = pd.DataFrame(results)

# Save the DataFrame to a CSV file
df.to_csv('submission.csv', index=False)
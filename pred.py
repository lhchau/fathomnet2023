import torch
import torchvision
import torch.nn as nn
import pandas as pd
from tqdm import tqdm

from models.simple_classifier import *
from utils.fathomnet_loader import *
from utils.collate_fn import *

model_dict = torch.load('./saved_models/coco/densenet_bs80_epoch30.pth')
clsfier_dict = torch.load('./saved_models/coco/densenetclsfier_bs80_epoch30.pth')

orig_densenet = torchvision.models.densenet121(pretrained=True)
features = list(orig_densenet.features)
model = nn.Sequential(*features, nn.ReLU(inplace=True))
clsfier = SimpleClassifier(1024, 290)

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
    labels = torch.ones(preds.shape).cuda() * (preds >= 0.1)
    
    # osd
    E_f = torch.log(1+torch.exp(nnOutputs))
    scores = torch.sum(E_f, dim=1)
    if scores[0] > 1:
        scores[0] = 1

    # Loop over the predictions and extract the indices where the value is 1
    for j in range(labels.shape[0]):
        indices = torch.where(labels[j] == 1)[0].tolist()
        categories = [str(index) for index in indices]
        if not len(categories):
            categories = ['1', '2']
		
        # Add the image and categories to the results list
        results.append({'id': path[0][:-4],
                        'categories': ' '.join(categories),
                        'osd': 1-scores[0]})
        
# Convert the results list to a Pandas DataFrame
df = pd.DataFrame(results)

# Save the DataFrame to a CSV file
df.to_csv('submission_01.csv', index=False)
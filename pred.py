import torch
import torchvision
import torch.nn as nn
import pandas as pd

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

test_loader = data.DataLoader(test_data, batch_size=2, num_workers=8, shuffle=False, pin_memory=True, collate_fn=coco_collate)

results = []

for i, (images, _) in enumerate(test_loader):
    nnOutputs = clsfier(model(images)).cuda()
    preds = torch.sigmoid(nnOutputs).cuda()
    labels = torch.ones(preds.shape).cuda() * (preds >= 0.5)

    # Loop over the predictions and extract the indices where the value is 1
    for j in range(labels.shape[0]):
        indices = torch.where(labels[j] == 1)[0].tolist()
        categories = [str(index) for index in indices]

        # Add the image and categories to the results list
        results.append({'id': test_data.path,
                        'categories': ' '.join(categories),
                        'osd': 0.0})
        
# Convert the results list to a Pandas DataFrame
df = pd.DataFrame(results)

# Save the DataFrame to a CSV file
df.to_csv('submission_01.csv', index=False)

import torch
from torchvision import transforms

def coco_collate(batch):
    # Define the transformations to be applied to each image
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    images = []
    targets = []

    for sample in batch:
        images.append(transform(sample[0]))
        targets.append(sample[1])

    images = torch.stack(images, dim=0)

    return images, targets

import torch
import torchvision 

def coco_collate_train(batch):
    # Define the transformations to be applied to each image
    normalize = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    img_transform = torchvision.transforms.Compose([
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.RandomResizedCrop((256, 256), scale=(0.5, 2.0)),
            torchvision.transforms.ToTensor(),
            normalize,
        ])
    
    images = []
    targets = []

    for sample in batch:
        images.append(img_transform(sample[0]))
        targets.append(sample[1])

    images = torch.stack(images, dim=0)

    return images, targets

def coco_collate_val(batch):
    # Define the transformations to be applied to each image
    normalize = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    val_transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((256, 256)),
        torchvision.transforms.ToTensor(),
        normalize
    ])
    
    images = []
    targets = []

    for sample in batch:
        images.append(val_transform(sample[0]))
        targets.append(sample[1])

    images = torch.stack(images, dim=0)

    return images, targets
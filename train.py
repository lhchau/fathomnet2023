import os
import torch
import argparse
import torchvision
import torch.nn as nn
from torch.autograd import Variable
from torch.utils import data
from tqdm import tqdm

import validate

from utils.fathomnet_loader import *
from utils.collate_fn import *

from models.simple_classifier import *

def train():
    args.save_dir += args.dataset + '/'
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    # normalize = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
    #                                  std=[0.229, 0.224, 0.225])
    # img_transform = torchvision.transforms.Compose([
    #         torchvision.transforms.RandomHorizontalFlip(),
    #         torchvision.transforms.RandomResizedCrop((256, 256), scale=(0.5, 2.0)),
    #         torchvision.transforms.ToTensor(),
    #         normalize,
    #     ])

    # label_transform = torchvision.transforms.Compose([
    #         ToLabel(),
    #     ])
    # val_transform = torchvision.transforms.Compose([
    #     torchvision.transforms.Resize((256, 256)),
    #     torchvision.transforms.ToTensor(),
    #     normalize
    # ])


    if args.dataset == "coco":
        loader = FathomNetLoader()
        val_data = FathomNetLoader(root='./datasets/train', annFile='./datasets/val.json')
    else:
        raise AssertionError

    args.n_classes = loader.n_classes
    trainloader = data.DataLoader(loader, batch_size=args.batch_size, num_workers=8, shuffle=True, pin_memory=True, collate_fn=coco_collate)
    val_loader = data.DataLoader(val_data, batch_size=args.batch_size, num_workers=8, shuffle=True, pin_memory=True, collate_fn=coco_collate)

    print("number of images = ", len(loader))
    print("number of classes = ", args.n_classes, " architecture used = ", args.arch)

    if args.arch == "resnet101":
        orig_resnet = torchvision.models.resnet101(pretrained=True)
        features = list(orig_resnet.children())
        model= nn.Sequential(*features[0:8])
        clsfier = SimpleClassifier(2048, args.n_classes)
    elif args.arch == "densenet":
        orig_densenet = torchvision.models.densenet121(pretrained=True)
        features = list(orig_densenet.features)
        model = nn.Sequential(*features, nn.ReLU(inplace=True))
        clsfier = SimpleClassifier(1024, args.n_classes)

    model = model.cuda()
    clsfier = clsfier.cuda()
    if torch.cuda.device_count() > 1:
        print("Using", torch.cuda.device_count(), "GPUs")
        model = nn.DataParallel(model)
        clsfier = nn.DataParallel(clsfier)

    optimizer = torch.optim.Adam([{'params': model.parameters(),'lr':args.l_rate/10},{'params': clsfier.parameters()}], lr=args.l_rate)

    if args.load:
        model.load_state_dict(torch.load(args.save_dir + args.arch + ".pth"))
        clsfier.load_state_dict(torch.load(args.save_dir + args.arch +'clsfier' + ".pth"))
        print("Model loaded!")

    bceloss = nn.BCEWithLogitsLoss()
    model.train()
    clsfier.train()
    for epoch in range(args.n_epoch):
        for i, (images, labels) in tqdm(enumerate(trainloader)):
            labels = torch.stack(labels, dim=0)
            
            images = Variable(images.cuda())
            labels = Variable(labels.cuda().float())

            optimizer.zero_grad()
         
            outputs = model(images)
            outputs = clsfier(outputs)
            loss = bceloss(outputs, labels)

            loss.backward()
            optimizer.step()
        torch.save(model.module.state_dict(), args.save_dir + args.arch + ".pth")
        torch.save(clsfier.module.state_dict(), args.save_dir + args.arch + 'clsfier' + ".pth")
        mAP = validate.validate(args, model, clsfier, val_loader)

        print("Epoch [%d/%d] Loss: %.4f mAP: %.4f" % (epoch, args.n_epoch, loss.data, mAP))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hyperparams')
    parser.add_argument('--arch', type=str, default='densenet',
                        help='Architecture to use densenet|resnet101')
    parser.add_argument('--dataset', type=str, default='pascal',
                        help='Dataset to use pascal|coco|nus-wide')
    parser.add_argument('--n_epoch', type=int, default=1,
                        help='# of the epochs')
    parser.add_argument('--n_classes', type=int, default=290,
                        help='# of classes')
    parser.add_argument('--batch_size', type=int, default=2,
                        help='Batch Size')
    # batch_size 320 for resenet101
    parser.add_argument('--l_rate', type=float, default=1e-4,
                        help='Learning Rate')

    #save and load
    parser.add_argument('--load', action='store_true', help='Whether to load models')
    parser.add_argument('--save_dir', type=str, default="./saved_models/",
                        help='Path to save models')
    parser.add_argument('--load_dir', type=str, default="./saved_models",
                        help='Path to load models')
    args = parser.parse_args()
    train()

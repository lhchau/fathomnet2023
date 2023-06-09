import os
import torch
import argparse
import torchvision
import torch.nn as nn
from torch.autograd import Variable
from torch.utils import data
from tqdm import tqdm
from torch.optim.lr_scheduler import OneCycleLR

import validate

from utils.fathomnet_loader import *
from utils.collate_fn import *

from models.simple_classifier import *

def train():
    args.save_dir += args.dataset + '/'
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    if args.dataset == "coco":
        loader = FathomNetLoader()
        val_data = FathomNetLoader(root='./datasets/train', annFile='./datasets/val.json')
    else:
        raise AssertionError

    args.n_classes = loader.n_classes
    trainloader = data.DataLoader(loader, batch_size=args.batch_size, num_workers=4, shuffle=True, pin_memory=False, collate_fn=coco_collate_train)
    val_loader = data.DataLoader(val_data, batch_size=args.batch_size, num_workers=4, shuffle=True, pin_memory=False, collate_fn=coco_collate_val)

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

    optimizer = torch.optim.Adam([{'params': model.parameters(),'lr':args.l_rate/10},
                                  {'params': clsfier.parameters()}], lr=args.l_rate)

    # define the scheduler
    # scheduler = OneCycleLR(optimizer, max_lr=0.01, epochs=args.n_epoch, steps_per_epoch=len(trainloader))
    
    if args.load:
        model.load_state_dict(torch.load(args.save_dir + args.arch + "_best" + ".pth"))
        clsfier.load_state_dict(torch.load(args.save_dir + args.arch +'clsfier_best' + ".pth"))
        print("Model loaded!")

    # unfreeze
    bceloss = nn.BCEWithLogitsLoss()
    model.train()
    clsfier.train()
    best_mAP = 0.0
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
            
            # scheduler.step()

        # Validate
        mAP = validate.validate(args, model, clsfier, val_loader)

        if mAP > best_mAP:
            best_mAP = mAP
            torch.save(model.state_dict(), args.save_dir + args.arch + "_best.pth")
            torch.save(clsfier.state_dict(), args.save_dir + args.arch + 'clsfier_best.pth')

        print("Epoch [%d/%d] Loss: %.4f mAP: %.4f" % (epoch, args.n_epoch, loss.data, mAP))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hyperparams')
    parser.add_argument('--arch', type=str, default='densenet',
                        help='Architecture to use densenet|resnet101')
    parser.add_argument('--dataset', type=str, default='pascal',
                        help='Dataset to use pascal|coco|nus-wide')
    parser.add_argument('--n_epoch', type=int, default=200,
                        help='# of the epochs')
    parser.add_argument('--n_classes', type=int, default=290,
                        help='# of classes')
    parser.add_argument('--batch_size', type=int, default=100,
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

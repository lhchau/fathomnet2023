import torch
import argparse
import torchvision
import numpy as np
from sklearn import metrics
import torch.nn as nn
from torch.autograd import Variable
from torch.utils import data
from tqdm import tqdm

from utils.fathomnet_loader import *
from utils.collate_fn import *

from models.simple_classifier import *

print("Using", torch.cuda.device_count(), "GPUs")
def validate(args, model, clsfier, val_loader):
    model.eval()
    clsfier.eval()

    gts = {i:[] for i in range(0, args.n_classes)}
    preds = {i:[] for i in range(0, args.n_classes)}
    with torch.no_grad():
        for images, labels in tqdm(val_loader):
            labels = torch.stack(labels, dim=0)
            
            images = Variable(images.cuda())
            labels = Variable(labels.cuda().float())
            outputs = model(images)
            outputs = F.relu(outputs, inplace=True)
            outputs = clsfier(outputs)
            outputs = torch.sigmoid(outputs)
            pred = outputs.squeeze().data.cpu().numpy()
            gt = labels.squeeze().data.cpu().numpy()

            for label in range(0, args.n_classes):
                gts[label].extend(gt[:,label])
                preds[label].extend(pred[:,label])

    FinalMAPs = []
    for i in range(0, args.n_classes):
        precision, recall, thresholds = metrics.precision_recall_curve(gts[i], preds[i])
        FinalMAPs.append(metrics.auc(recall, precision))
    # print(FinalMAPs)

    return np.nanmean(FinalMAPs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hyperparams')
    parser.add_argument('--arch', nargs='?', type=str, default='resnet101', 
                        help='Architecture to use')
    parser.add_argument('--dataset', nargs='?', type=str, default='pascal',
                        help='Dataset to use [\'pascal, coco, nus-wide\']')
    parser.add_argument('--load_path', nargs='?', type=str, default='./saved_models/',
                        help='Model path')
    parser.add_argument('--batch_size', nargs='?', type=int, default=2,
                        help='Batch Size')
    parser.add_argument('--n_classes', nargs='?', type=int, default=290)
    args = parser.parse_args()

    if args.dataset == 'coco':
        val_data = FathomNetLoader(root='./datasets/train', annFile='./datasets/val.json')
    else:
        raise AssertionError

    args.n_classes = val_data.n_classes
    val_loader = data.DataLoader(val_data, batch_size=args.batch_size, num_workers=8, shuffle=False, collate_fn=coco_collate)

    if args.arch == "resnet101":
        orig_resnet = torchvision.models.resnet101(pretrained=True)
        features = list(orig_resnet.children())
        model = nn.Sequential(*features[0:8])
        clsfier = SimpleClassifier(2048, args.n_classes)

    elif args.arch == "densenet":
        orig_densenet = torchvision.models.densenet121(pretrained=True)
        features = list(orig_densenet.features)
        model = nn.Sequential(*features, nn.ReLU(inplace=True))
        clsfier = SimpleClassifier(1024, args.n_classes)

    model.load_state_dict(torch.load(args.load_path + args.dataset + '/' +
                                     args.arch + ".pth", map_location="cpu"))
    clsfier.load_state_dict(torch.load(args.load_path + args.dataset + '/' +
                                       args.arch + 'clsfier' + ".pth", map_location="cpu"))

    model = model.cuda()
    clsfier = clsfier.cuda()
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
        clsfier = nn.DataParallel(clsfier)

    mAP = validate(args, model, clsfier, val_loader)
    print("mAP on validation set: %.4f" % (mAP * 100))

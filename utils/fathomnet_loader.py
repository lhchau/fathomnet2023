import os
import torch
from PIL import Image


from torch.utils import data

class FathomNetLoader(data.Dataset):
    def __init__(self, root='./datasets/train', annFile="./datasets/train.json", transform=None, target_transform=None):
        super().__init__()
        """
        Args:
            - root (string): Root dir where images are downloaded to
            - annFile (string): Path to json annotation file
            - transform (callable, optional): A function/transform that takes  in an PIL image and return a transformed version. E.g, `transforms.ToTensor()`
            - target_transform (callable, optional): takes in the target and transforms it
        """
        from pycocotools.coco import COCO
        self.root = root 
        self.coco = COCO(annFile)
        self.n_classes = 290
        self.ids = list(self.coco.imgs.keys())
        self.transform = transform
        self.target_transform = target_transform
        self.split = self.root[-4:]
    
    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: Tuple (image, target). target is the object returned by ``coco.loadAnns``.
        """
        coco = self.coco
        img_id = self.ids[index]
        if self.split == 'eval':
            label = []
        else:
            ann_ids = coco.getAnnIds(imgIds=img_id)
            target = coco.loadAnns(ann_ids)[0]['category_id']
            target = int(target)
            
            label = torch.zeros(self.n_classes)
            label[target] = 1

        path = coco.loadImgs(img_id)[0]['file_name']

        img = Image.open(os.path.join(self.root, path)).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            label = self.target_transform(label)

        if self.split == 'eval':
            return img, path
        else:
            return img, label

    def __len__(self):
        return len(self.ids)
    
    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str
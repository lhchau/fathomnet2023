# FathomNet - FGVC10 workshop at CVPR 2023

- Problem: The challenge is both to classify marine animals in a target image and assess if the image is from a different distribution relative to the training data.

## Kaggle - Top 12/69 on private test

- Classification: ResNet101 or Dense as a backbone and 3 fully connected layers for classification head.
- Out of distribution: 2 approaches, first, use entropy to measure ood score. Second, enable dropout during inference time to sample many outputs and compute variance as ood score

## Download Fathomnet2023 dataset

### Train dataset

```
python3 download_images.py object_detection/train.json --outpath './datasets/train/'
```

### Eval dataset

```
python3 download_images.py object_detection/eval.json --outpath './datasets/eval/'
```

## Training

- Requirement: GPU

```
python3 train.py --arch resnet101 --dataset coco --save_dir ./saved_models/
```

## Evaluation

- Models located: `./saved_models/coco/` [Download models](https://drive.google.com/drive/folders/1LPGsys6UvW0g0ir8eas3pVV4bBjVcuoj?usp=sharing)

```
python3 pred.py --arch resnet101
```

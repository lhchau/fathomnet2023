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

Requirement:

- GPU

```
python3 train.py --arch densenet --dataset coco --save_dir ./saved_models/
```

## Evaluation

Models located: `./saved_models/coco/`

[Download models](https://drive.google.com/drive/folders/1LPGsys6UvW0g0ir8eas3pVV4bBjVcuoj?usp=sharing)

```
python3 pred.py --arch resnet101
```

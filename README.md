## Download Fathomnet2023 dataset

### Requirement

```
export LD_LIBRARY_PATH=/usr/lib/wsl/lib:$LD_LIBRARY_PATH
```

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
python3 train.py --arch resnet101 --dataset coco --save_dir ./saved_models/
```

## Evaluation

Models located: `./saved_models/coco/`

[Download models](https://drive.google.com/drive/folders/1LPGsys6UvW0g0ir8eas3pVV4bBjVcuoj?usp=sharing)

### Fast predict

```
python3 fast_pred.py --arch resnet101
```

### Dropout predict

```
python3 pred.py --arch resnet101
```

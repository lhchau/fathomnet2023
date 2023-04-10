from pathlib import Path

class Cfg:
    DATA_EXPLORING_ROOT = Path('./')
    INPUT_ROOT = Path('./')
    # OUTPUT_ROOT = Path('./working/')
    
    IMAGES_ROOT = DATA_EXPLORING_ROOT / 'datasets'
    TRAIN_IMAGES_ROOT = IMAGES_ROOT / 'train'
    EVAL_IMAGES_ROOT = IMAGES_ROOT / 'eval'
    # DATASET_ROOT = OUTPUT_ROOT / 'dataset'

    TRAIN_IMAGE_DATA = IMAGES_ROOT / 'train_image_data.csv'
    EVAL_IMAGE_DATA = IMAGES_ROOT / 'eval_image_data.csv'
    ANNOTATION_FILE = IMAGES_ROOT / 'annotation.csv'
    CATEGORY_KEY_FILE = IMAGES_ROOT / 'category_key.csv'
    SAMPLE_SUBMISSION_FILE = IMAGES_ROOT / 'sample_submission.csv'
    
    # DATASET_CONFIG = OUTPUT_ROOT / 'dataset.yaml'
    # MODEL_NAME = 'FathomNet-YOLOv8'
    
    N_EPOCHS = 50
    N_BATCH = 16
    RANDOM_STATE = 2023
    SAMPLE_SIZE = 1.0
    TEST_SIZE = .2
    INDEX = 'id'

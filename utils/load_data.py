import pandas as pd
from .configs import *

def read_train_data(file=Cfg.TRAIN_IMAGE_DATA, index_col=Cfg.INDEX):
    return pd.read_csv(file).set_index(Cfg.INDEX)

def read_category_keys(file=Cfg.CATEGORY_KEY_FILE, index_col=Cfg.INDEX):
    return pd.read_csv(file).set_index(Cfg.INDEX)

def read_sample_submission(file=Cfg.SAMPLE_SUBMISSION_FILE, index_col=Cfg.INDEX):
    return pd.read_csv(file).set_index(Cfg.INDEX)

def read_train_image_data(file=Cfg.TRAIN_IMAGE_DATA, index_col=Cfg.INDEX):
    return pd.read_csv(file).set_index(Cfg.INDEX)

def read_eval_image_data(file=Cfg.EVAL_IMAGE_DATA, index_col=Cfg.INDEX):
    return pd.read_csv(file).set_index(Cfg.INDEX)

def read_annotation_data(file=Cfg.ANNOTATION_FILE, index_col=Cfg.INDEX):
    return pd.read_csv(file).set_index(Cfg.INDEX)
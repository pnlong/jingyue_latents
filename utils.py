# README
# Phillip Long
# February 22, 2025

# Utilities.

# python /home/pnlong/jingyue_latents/utils.py

# IMPORTS
##################################################

import json
from typing import Union, List, Tuple
import numpy as np
import pickle
import torch

##################################################


# FILEPATH CONSTANTS
##################################################

# base directory
BASE_DIR = "/deepfreeze/user_shares/pnlong/jingyue_latents"

# subdirectory names
DATA_DIR_NAME = "data"
DATA_SUBDIR_NAME = "data"
CHECKPOINTS_DIR_NAME = "checkpoints"
DEFAULT_MODEL_NAME = "model"

##################################################


# FILE HELPER FUNCTIONS
##################################################

def save_json(filepath: str, data: dict):
    """Save data as a JSON file."""
    with open(filepath, "w", encoding = "utf8") as f:
        json.dump(obj = data, fp = f)

def save_args(filepath: str, args):
    """Save the command-line arguments."""
    args_dict = {}
    for key, value in vars(args).items():
        args_dict[key] = value
    save_json(filepath = filepath, data = args_dict)

def load_json(filepath: str):
    """Load data from a JSON file."""
    with open(filepath, encoding = "utf8") as f:
        return json.load(fp = f)
    
def save_csv(filepath: str, data, header: str = ""):
    """Save data as a CSV file."""
    np.savetxt(fname = filepath, X = data, fmt = "%d", delimiter = ",", header = header, comments = "")

def load_csv(filepath: str, skiprows: int = 1):
    """Load data from a CSV file."""
    return np.loadtxt(fname = filepath, dtype = int, delimiter = ",", skiprows = skiprows)

def save_txt(filepath: str, data: list):
    """Save a list to a TXT file."""
    with open(filepath, "w", encoding = "utf8") as f:
        for item in data:
            f.write(f"{item}\n")

def load_txt(filepath: str):
    """Load a TXT file as a list."""
    with open(filepath, encoding = "utf8") as f:
        return [line.strip() for line in f]
    
def load_pickle(filepath: str):
    """Load a pickle file."""
    with open(filepath, "rb") as f:
        return pickle.load(file = f)
    
def count_lines(filepath: str):
    """Count the number of lines in the given file."""
    n = 0
    with open(filepath, "r", encoding = "utf8") as f:
        for _ in f:
            n += 1
    return n

##################################################


# TRAINING CONSTANTS
##################################################

# partitions
TRAIN_PARTITION_NAME = "train"
VALID_PARTITION_NAME = "valid"
TEST_PARTITION_NAME = "test"
RELEVANT_TRAINING_PARTITIONS = [TRAIN_PARTITION_NAME, VALID_PARTITION_NAME]

# data types for custom dataset
DATA_TYPE = torch.float32
LABEL_TYPE = torch.uint8

# training statistics
LOSS_STATISTIC_NAME = "loss"
ACCURACY_STATISTIC_NAME = "accuracy"
RELEVANT_TRAINING_STATISTICS = [LOSS_STATISTIC_NAME, ACCURACY_STATISTIC_NAME]
TRAINING_STATISTICS_OUTPUT_COLUMNS = ["step", "partition", f"is_{LOSS_STATISTIC_NAME}", "value"]

# data loader
BATCH_SIZE = 12

# training defaults
N_STEPS = 100000
N_VALID_STEPS = 2000
EARLY_STOPPING_TOLERANCE = 10
LEARNING_RATE = 0.0005
LEARNING_RATE_WARMUP_STEPS = 5000
LEARNING_RATE_DECAY_STEPS = 100000
LEARNING_RATE_DECAY_MULTIPLIER = 0.1
WEIGHT_DECAY = 0.00001

##################################################


# TRAINING HELPER FUNCTIONS
##################################################

def pad(seqs: List[torch.Tensor], length: int) -> torch.Tensor:
    """End-zero-pad a given list of sequences to the given length."""

    # pad sequences
    for i, seq in enumerate(seqs):
        if len(seq) < length: # sequence is shorter than length
            seq = torch.nn.functional.pad(input = seq, pad = (0, 0, 0, length - len(seq)), mode = "constant", value = 0)
        else: # sequence is longer than length
            seq = seq[:length]
        seqs[i] = seq # update value in sequences

    # stack sequences
    seqs = torch.stack(tensors = seqs, dim = 0)

    # return padded sequences as single matrix
    return seqs

def mask(seqs: List[torch.Tensor], length: int) -> torch.Tensor:
    """Generate a mask for the given list of sequences with the given length."""

    # create empty mask
    mask = torch.zeros(size = (len(seqs), length), dtype = torch.bool)

    # generate masks for each sequence
    for i in range(len(seqs)):
        mask[i, :len(seqs[i])] = True

    # return the mask
    return mask

def get_lr_multiplier(step: int, warmup_steps: int, decay_end_steps: int, decay_end_multiplier: float) -> float:
    """Return the learning rate multiplier with a warmup and decay schedule.

    The learning rate multiplier starts from 0 and linearly increases to 1
    after `warmup_steps`. After that, it linearly decreases to
    `decay_end_multiplier` until `decay_end_steps` is reached.

    """
    if step < warmup_steps:
        return (step + 1) / warmup_steps
    if step > decay_end_steps:
        return decay_end_multiplier
    position = (step - warmup_steps) / (decay_end_steps - warmup_steps)
    return 1 - (1 - decay_end_multiplier) * position

##################################################


# EVALUATION CONSTANTS
##################################################



##################################################


# SCALAR CONSTANTS
##################################################

LATENT_EMBEDDING_DIM = 128

##################################################


# MISCELLANEOUS CONSTANTS
##################################################

# wandb constants
WANDB_PROJECT_NAME = "jingyue-latents"
WANDB_RUN_NAME_FORMAT_STRING = "%m%d%y%H%M%S" # time format string for determining wandb run names

# file writing
NA_STRING = "NA"

# for multiprocessing
CHUNK_SIZE = 1

# separator line
SEPARATOR_LINE_WIDTH = 140
MAJOR_SEPARATOR_LINE = "".join(("=" for _ in range(SEPARATOR_LINE_WIDTH)))
MINOR_SEPARATOR_LINE = "".join(("-" for _ in range(SEPARATOR_LINE_WIDTH)))
DOTTED_SEPARATOR_LINE = "".join(("- " for _ in range(SEPARATOR_LINE_WIDTH // 2)))

##################################################


# MISCELLANEOUS HELPER FUNCTIONS
##################################################

def inverse_dict(d):
    """Return the inverse dictionary."""
    return {v: k for k, v in d.items()}

def rep(x: object, times: int, flatten: bool = False):
    """
    An implementation of R's rep() function.
    This cannot be used to create a list of empty lists 
    (see https://stackoverflow.com/questions/240178/list-of-lists-changes-reflected-across-sublists-unexpectedly)
    ."""
    l = [x] * times
    if flatten:
        l = sum(l, [])
    return l

def unique(l: Union[List, Tuple]) -> list:
    """Returns the unique values from a list while retaining order."""
    return list(dict.fromkeys(list(l)))

def transpose(l: Union[List, Tuple]) -> list:
    """Tranpose a 2-dimension list."""
    return list(map(list, zip(*l)))

##################################################


# EMOTION CONSTANTS
##################################################

# emotion recognition
EMOTION_DIR_NAME = "emotion"
EMOTION_DIR = f"{BASE_DIR}/{EMOTION_DIR_NAME}"

# mappings
EMOTION_TO_EMOTION_ID = {"happy": "Q1", "angry": "Q2", "sad": "Q3", "relax": "Q4"}
EMOTION_TO_INDEX = {emotion: i for i, emotion in enumerate(EMOTION_TO_EMOTION_ID.keys())}
EMOTION_ID_TO_INDEX = {emotion_id: i for i, emotion_id in enumerate(EMOTION_TO_EMOTION_ID.values())}
EMOTION_ID_TO_EMOTION = inverse_dict(d = EMOTION_TO_EMOTION_ID)
INDEX_TO_EMOTION = inverse_dict(d = EMOTION_TO_INDEX)
INDEX_TO_EMOTION_ID = inverse_dict(d = EMOTION_ID_TO_INDEX)

# number of emotion classes
N_EMOTION_CLASSES = len(EMOTION_TO_EMOTION_ID)

# maximum song length (in bars) for emotion data
EMOTION_MAX_SEQ_LEN = 43

# default model name for emotion
EMOTION_MODEL_NAME = DEFAULT_MODEL_NAME

# emotion evaluation output columns
EMOTION_EVALUATION_LOSS_OUTPUT_COLUMNS = ["model", "path", LOSS_STATISTIC_NAME]
EMOTION_EVALUATION_ACCURACY_OUTPUT_COLUMNS = ["model", "path", "expected", "actual", "is_correct"]

##################################################


# CHORD CONSTANTS
##################################################

# chord progression detection
CHORD_DIR_NAME = "chord"
CHORD_DIR = f"{BASE_DIR}/{CHORD_DIR_NAME}"

# default model name for chord
CHORD_MODEL_NAME = DEFAULT_MODEL_NAME

##################################################


# STYLE CONSTANTS
##################################################

# style classifier
STYLE_DIR_NAME = "style"
STYLE_DIR = f"{BASE_DIR}/{STYLE_DIR_NAME}"

# default model name for style
STYLE_MODEL_NAME = DEFAULT_MODEL_NAME

##################################################
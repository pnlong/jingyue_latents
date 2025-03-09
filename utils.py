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
from torch import uint8, float32

from os.path import exists, dirname, realpath
from os import mkdir
import argparse
import logging
from shutil import rmtree

##################################################


# BASIC CONSTANTS
##################################################

# base directory
BASE_DIR = "/deepfreeze/user_shares/pnlong/jingyue_latents"

# jingyue's directory (and symlinks)
JINGYUE_DIR = "/deepfreeze/user_shares/jingyue"

# tasks
EMOTION_DIR_NAME = "emotion"
CHORD_DIR_NAME = "chord"
STYLE_DIR_NAME = "style"
ALL_TASKS = [EMOTION_DIR_NAME, CHORD_DIR_NAME, STYLE_DIR_NAME]

# subdirectory names
DATA_DIR_NAME = "data"
DATA_SUBDIR_NAME = "data"
PREBOTTLENECK_DATA_SUBDIR_NAME = "prebottleneck"
CHECKPOINTS_DIR_NAME = "checkpoints"
DEFAULT_MODEL_NAME = "model"
SYMLINKS_DIR_NAME = "symlinks"

# symlinks dir, use `ln -sf /path/to/directory /path/to/symlink` to create a new symlink
SYMLINKS_DIR = f"{dirname(realpath(__file__))}/{SYMLINKS_DIR_NAME}"
JINGYUE_DATA_SYMLINKS_DIR_BY_TASK = {task: f"{SYMLINKS_DIR}/{task}" for task in ALL_TASKS}
JINGYUE_DATA_SYMLINK_NAME = DATA_SUBDIR_NAME
JINGYUE_PREBOTTLENECK_DATA_SYMLINK_NAME = PREBOTTLENECK_DATA_SUBDIR_NAME
JINGYUE_SPLITS_SYMLINK_NAME = "splits"

# will all be populated later
DIR_BY_TASK = {task: f"{BASE_DIR}/{task}" for task in ALL_TASKS} # task name to BASE_DIR subdirectory
INDEXER_BY_TASK = dict()  # task name to primary indexer dictionary
N_CLASSES_BY_TASK = dict() # task name to number of classes for that task
MAX_SEQ_LEN_BY_TASK = dict() # task name to maximum sequence length for that task
DEFAULT_MODEL_NAME_BY_TASK = {task: DEFAULT_MODEL_NAME for task in ALL_TASKS} # task name to default model name for that task

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
ALL_PARTITIONS = [TRAIN_PARTITION_NAME, VALID_PARTITION_NAME, TEST_PARTITION_NAME]
RELEVANT_TRAINING_PARTITIONS = ALL_PARTITIONS[:2]

# data types for custom dataset
DATA_TYPE = float32
LABEL_TYPE = uint8

# training statistics
LOSS_STATISTIC_NAME = "loss"
ACCURACY_STATISTIC_NAME = "accuracy"
RELEVANT_TRAINING_STATISTICS = [LOSS_STATISTIC_NAME, ACCURACY_STATISTIC_NAME]
TRAINING_STATISTICS_OUTPUT_COLUMNS = ["step", "partition", f"is_{LOSS_STATISTIC_NAME}", "value"]

# data loader
BATCH_SIZE = 16
FRONT_PAD = True

# training defaults
N_EPOCHS = 100
N_STEPS = 50000
EARLY_STOPPING_TOLERANCE = 10
LEARNING_RATE = 0.0005
WEIGHT_DECAY = 0.1 # alternatively, 0.01

# transformer model defaults
TRANSFORMER_LAYERS = 6
TRANSFORMER_HEADS = 8
TRANSFORMER_DROPOUT = 0.2
TRANSFORMER_FEEDFORWARD_LAYERS = 256

##################################################


# EVALUATION CONSTANTS
##################################################

# default column names
EVALUATION_LOSS_OUTPUT_COLUMNS = ["model", "path", LOSS_STATISTIC_NAME]
EVALUATION_ACCURACY_OUTPUT_COLUMNS = ["model", "path", "expected", "actual", "is_correct"]

# evaluation output columns by task
EVALUATION_LOSS_OUTPUT_COLUMNS_BY_TASK = dict()
EVALUATION_ACCURACY_OUTPUT_COLUMNS_BY_TASK = dict()

##################################################


# SCALAR CONSTANTS
##################################################

# dimension of input data
LATENT_EMBEDDING_DIM = 128
PREBOTTLENECK_LATENT_EMBEDDING_DIM = 512

##################################################


# EMOTION CONSTANTS
##################################################

# mappings
EMOTIONS = ["happy", "angry", "sad", "relax"]
EMOTION_TO_EMOTION_ID = {emotion: f"Q{i + 1}" for i, emotion in enumerate(EMOTIONS)}
EMOTION_TO_INDEX = {emotion: i for i, emotion in enumerate(EMOTION_TO_EMOTION_ID.keys())}
EMOTION_ID_TO_INDEX = {emotion_id: i for i, emotion_id in enumerate(EMOTION_TO_EMOTION_ID.values())}
EMOTION_ID_TO_EMOTION = inverse_dict(d = EMOTION_TO_EMOTION_ID)
INDEX_TO_EMOTION = inverse_dict(d = EMOTION_TO_INDEX)
INDEX_TO_EMOTION_ID = inverse_dict(d = EMOTION_ID_TO_INDEX)
INDEXER_BY_TASK[EMOTION_DIR_NAME] = EMOTION_ID_TO_INDEX

# number of emotion classes
N_CLASSES_BY_TASK[EMOTION_DIR_NAME] = len(EMOTIONS)

# maximum song length (in bars) for emotion data
MAX_SEQ_LEN_BY_TASK[EMOTION_DIR_NAME] = 42

# emotion evaluation output columns
EVALUATION_LOSS_OUTPUT_COLUMNS_BY_TASK[EMOTION_DIR_NAME] = EVALUATION_LOSS_OUTPUT_COLUMNS
EVALUATION_ACCURACY_OUTPUT_COLUMNS_BY_TASK[EMOTION_DIR_NAME] = EVALUATION_ACCURACY_OUTPUT_COLUMNS

##################################################


# CHORD CONSTANTS
##################################################

# mappings
CHORDS = []
CHORD_TO_INDEX = {chord: i for i, chord in enumerate(CHORDS)}
INDEX_TO_CHORD = inverse_dict(d = CHORD_TO_INDEX)
INDEXER_BY_TASK[CHORD_DIR_NAME] = CHORD_TO_INDEX

# number of chord classes
N_CLASSES_BY_TASK[CHORD_DIR_NAME] = len(CHORDS)

# maximum song length (in bars) for chord data
MAX_SEQ_LEN_BY_TASK[CHORD_DIR_NAME] = 1 # must be 1, since this is a bar-by-bar classification task

# chord evaluation output columns
EVALUATION_LOSS_OUTPUT_COLUMNS_BY_TASK[CHORD_DIR_NAME] = EVALUATION_LOSS_OUTPUT_COLUMNS
EVALUATION_ACCURACY_OUTPUT_COLUMNS_BY_TASK[CHORD_DIR_NAME] = EVALUATION_ACCURACY_OUTPUT_COLUMNS

##################################################


# STYLE CONSTANTS
##################################################

# mappings
STYLES = ["Bethel", "Clayderman", "Einaudi", "Hancock", "Hillsong", "Hisaishi", "Ryuichi", "Yiruma"]
STYLE_TO_INDEX = {style: i for i, style in enumerate(STYLES)}
INDEX_TO_STYLE = inverse_dict(d = STYLE_TO_INDEX)
INDEXER_BY_TASK[STYLE_DIR_NAME] = STYLE_TO_INDEX

# number of style classes
N_CLASSES_BY_TASK[STYLE_DIR_NAME] = len(STYLES)

# maximum song length (in bars) for style data
MAX_SEQ_LEN_BY_TASK[STYLE_DIR_NAME] = 42

# style evaluation output columns
EVALUATION_LOSS_OUTPUT_COLUMNS_BY_TASK[STYLE_DIR_NAME] = EVALUATION_LOSS_OUTPUT_COLUMNS
EVALUATION_ACCURACY_OUTPUT_COLUMNS_BY_TASK[STYLE_DIR_NAME] = EVALUATION_ACCURACY_OUTPUT_COLUMNS

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

# filetypes
TENSOR_FILETYPE = "pt"
PICKLE_FILETYPE = "pkl"

##################################################


# ARGUMENTS
##################################################

def parse_args(args = None, namespace = None):
    """Parse command-line arguments."""

    # create argument parser
    parser = argparse.ArgumentParser(prog = "Commands", description = "Produce relevant commands for all tasks, or the one specified.")
    parser.add_argument("-t", "--task", default = None, choices = ALL_TASKS, type = str, help = "Name of task")
    parser.add_argument("-r", "--reset", action = "store_true", help = "Whether or not to reset task directory")
    
    # parse arguments
    args = parser.parse_args(args = args, namespace = namespace)
    
    # return parsed arguments
    return args

##################################################


# MAIN METHOD
##################################################

if __name__ == "__main__":

    # SETUP
    ##################################################

    # parse the command-line arguments
    args = parse_args()

    # interpret arguments
    tasks = ALL_TASKS if (args.task is None) else [args.task]
    
    # set up logging
    logging.basicConfig(level = logging.INFO, format = "%(message)s")

    # set global variables
    SOFTWARE_DIR = dirname(realpath(__file__))
    DEFAULT_GPU = 0

    # helper functions
    join_arguments = lambda args: " ".join(filter(lambda string: len(string) > 0, args))

    ##################################################

    
    # LOOP THROUGH TASKS
    ##################################################

    for task in tasks:

        # create base_dir
        base_dir = f"{BASE_DIR}/{task}"
        if exists(base_dir) and args.reset:
            rmtree(base_dir, ignore_errors = True)
        if not exists(base_dir):
            mkdir(base_dir)
        
        # title
        logging.info(f"{task.upper()}")
        logging.info(MAJOR_SEPARATOR_LINE)

        # create dataset
        logging.info("* Dataset:")
        logging.info(join_arguments(args = [
            f"python {SOFTWARE_DIR}/dataset.py", 
            f"--task {task}",
        ]))
        
        # separator line
        logging.info(MINOR_SEPARATOR_LINE)

        # helper function for generating training commands
        def log_train_command_string(
                use_prebottleneck_latents: bool = False,
                prepool: bool = False,
                use_transformer: bool = False,
            ):
            """Helper function to log the train command string."""
            logging.info("* " + ("Transformer" if use_transformer else "MLP") + (", Prepooled" if prepool else "") + (", Prebottlenecked" if use_prebottleneck_latents else "") + ":")
            logging.info(join_arguments(args = [
                f"python {SOFTWARE_DIR}/train.py",
                f"--task {task}",
                "--use_prebottleneck_latents" if use_prebottleneck_latents else "",
                "--prepool" if prepool else "",
                "--use_transformer" if use_transformer else "",
                "--use_wandb",
                f"--gpu {DEFAULT_GPU}",
                "--model_name " + ("transformer" if use_transformer else "mlp") + str((2 * int(prepool)) + int(use_prebottleneck_latents)),
            ]))

        # log commands for different models
        for prepool in (False, True):
            for use_prebottleneck_latents in (False, True):
                log_train_command_string(use_prebottleneck_latents = use_prebottleneck_latents, prepool = prepool, use_transformer = False)
        for use_prebottleneck_latents in (False, True):
            log_train_command_string(use_prebottleneck_latents = use_prebottleneck_latents, prepool = False, use_transformer = True)
        del log_train_command_string # free up memory
        
        # separator line
        logging.info(MINOR_SEPARATOR_LINE)

        # evaluate
        logging.info("* Evaluate:")
        logging.info(join_arguments(args = [
            f"python {SOFTWARE_DIR}/evaluate.py",
            f"--task {task}",
            f"--gpu {DEFAULT_GPU}",
        ]))
                
        # end margin
        logging.info(MAJOR_SEPARATOR_LINE)
        logging.info("")

    ##################################################

##################################################

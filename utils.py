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

from os.path import exists, dirname, realpath
from os import mkdir
import argparse
import logging
from shutil import rmtree

from os.path import dirname, realpath
import sys
sys.path.insert(0, dirname(realpath(__file__)))

##################################################


# FILEPATH CONSTANTS
##################################################

# base directory
BASE_DIR = "/deepfreeze/user_shares/pnlong/jingyue_latents"

# jingyue's directory
JINGYUE_DIR = "/deepfreeze/user_shares/jingyue"

# task name to jingyue data directory name
TASK_NAME_TO_JINGYUE_DATA_DIR = dict() # will be populated later

# subdirectory names
DATA_DIR_NAME = "data"
DATA_SUBDIR_NAME = "data"
PREBOTTLENECK_DATA_SUBDIR_NAME = "prebottleneck"
CHECKPOINTS_DIR_NAME = "checkpoints"
DEFAULT_MODEL_NAME = "model"
MODELS_FILE_NAME = "models"

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
FRONT_PAD = True

# training defaults
N_STEPS = 100000
N_VALID_STEPS = 2000
EARLY_STOPPING_TOLERANCE = 10
LEARNING_RATE = 0.0005
LEARNING_RATE_WARMUP_STEPS = 5000
LEARNING_RATE_DECAY_STEPS = 100000
LEARNING_RATE_DECAY_MULTIPLIER = 0.1
WEIGHT_DECAY = 0.05 # alternatively, 0.01

# transformer model defaults
TRANSFORMER_LAYERS = 6
TRANSFORMER_HEADS = 8
TRANSFORMER_DROPOUT = 0.2
TRANSFORMER_FEEDFORWARD_LAYERS = 256

##################################################


# TRAINING HELPER FUNCTIONS
##################################################

def pad(seqs: List[torch.Tensor], length: int) -> torch.Tensor:
    """Front-zero-pad a given list of sequences to the given length."""

    # pad sequences
    for i, seq in enumerate(seqs):
        if len(seq) < length: # sequence is shorter than length
            pad = length - len(seq)
            pad = (0, 0, pad, 0) if FRONT_PAD else (0, 0, 0, pad)
            seq = torch.nn.functional.pad(input = seq, pad = pad, mode = "constant", value = 0)
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
        if FRONT_PAD: # front pad
            mask[i, -len(seqs[i]):] = True
        else: # end pad
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

EVALUATION_LOSS_OUTPUT_COLUMNS = ["model", "path", LOSS_STATISTIC_NAME]
EVALUATION_ACCURACY_OUTPUT_COLUMNS = ["model", "path", "expected", "actual", "is_correct"]

##################################################


# SCALAR CONSTANTS
##################################################

# dimension of input data
LATENT_EMBEDDING_DIM = 128
PREBOTTLENECK_LATENT_EMBEDDING_DIM = 128

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

# directory with jingyue's data
EMOTION_DATA_DIR = f"{JINGYUE_DIR}/EMOPIA_emotion_recognition"

# emotion recognition
EMOTION_DIR_NAME = "emotion"
TASK_NAME_TO_JINGYUE_DATA_DIR[EMOTION_DIR_NAME] = EMOTION_DATA_DIR
EMOTION_DIR = f"{BASE_DIR}/{EMOTION_DIR_NAME}"

# mappings
EMOTIONS = ["happy", "angry", "sad", "relax"]
EMOTION_TO_EMOTION_ID = {emotion: f"Q{i + 1}" for i, emotion in enumerate(EMOTIONS)}
EMOTION_TO_INDEX = {emotion: i for i, emotion in enumerate(EMOTION_TO_EMOTION_ID.keys())}
EMOTION_ID_TO_INDEX = {emotion_id: i for i, emotion_id in enumerate(EMOTION_TO_EMOTION_ID.values())}
EMOTION_ID_TO_EMOTION = inverse_dict(d = EMOTION_TO_EMOTION_ID)
INDEX_TO_EMOTION = inverse_dict(d = EMOTION_TO_INDEX)
INDEX_TO_EMOTION_ID = inverse_dict(d = EMOTION_ID_TO_INDEX)

# number of emotion classes
EMOTION_N_CLASSES = len(EMOTIONS)

# maximum song length (in bars) for emotion data
EMOTION_MAX_SEQ_LEN = 42

# default model name for emotion
EMOTION_MODEL_NAME = DEFAULT_MODEL_NAME

# emotion evaluation output columns
EMOTION_EVALUATION_LOSS_OUTPUT_COLUMNS = EVALUATION_LOSS_OUTPUT_COLUMNS
EMOTION_EVALUATION_ACCURACY_OUTPUT_COLUMNS = EVALUATION_ACCURACY_OUTPUT_COLUMNS

##################################################


# CHORD CONSTANTS
##################################################

# directory with jingyue's data
CHORD_DATA_DIR = f"{JINGYUE_DIR}/dir"

# chord progression detection
CHORD_DIR_NAME = "chord"
TASK_NAME_TO_JINGYUE_DATA_DIR[CHORD_DIR_NAME] = CHORD_DATA_DIR
CHORD_DIR = f"{BASE_DIR}/{CHORD_DIR_NAME}"

# mappings
CHORDS = []
CHORD_TO_INDEX = {chord: i for i, chord in enumerate(CHORDS)}
INDEX_TO_CHORD = inverse_dict(d = CHORD_TO_INDEX)

# number of chord classes
CHORD_N_CLASSES = len(CHORDS)

# maximum song length (in bars) for chord data
CHORD_MAX_SEQ_LEN = 42

# default model name for chord
CHORD_MODEL_NAME = DEFAULT_MODEL_NAME

# chord evaluation output columns
CHORD_EVALUATION_LOSS_OUTPUT_COLUMNS = EVALUATION_LOSS_OUTPUT_COLUMNS
CHORD_EVALUATION_ACCURACY_OUTPUT_COLUMNS = EVALUATION_ACCURACY_OUTPUT_COLUMNS

##################################################


# STYLE CONSTANTS
##################################################

# directory with jingyue's data
STYLE_DATA_DIR = f"{JINGYUE_DIR}/Pianist8_style_classification"

# style classifier
STYLE_DIR_NAME = "style"
TASK_NAME_TO_JINGYUE_DATA_DIR[STYLE_DIR_NAME] = STYLE_DATA_DIR
STYLE_DIR = f"{BASE_DIR}/{STYLE_DIR_NAME}"

# mappings
STYLES = ["Bethel", "Clayderman", "Einaudi", "Hancock", "Hillsong", "Hisaishi", "Ryuichi", "Yiruma"]
STYLE_TO_INDEX = {style: i for i, style in enumerate(STYLES)}
INDEX_TO_STYLE = inverse_dict(d = STYLE_TO_INDEX)

# number of style classes
STYLE_N_CLASSES = len(STYLES)

# maximum song length (in bars) for style data
STYLE_MAX_SEQ_LEN = 42

# default model name for style
STYLE_MODEL_NAME = DEFAULT_MODEL_NAME

# style evaluation output columns
STYLE_EVALUATION_LOSS_OUTPUT_COLUMNS = EVALUATION_LOSS_OUTPUT_COLUMNS
STYLE_EVALUATION_ACCURACY_OUTPUT_COLUMNS = EVALUATION_ACCURACY_OUTPUT_COLUMNS

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

# list of all tasks
ALL_TASKS = list(TASK_NAME_TO_JINGYUE_DATA_DIR.keys())

##################################################


# ARGUMENTS
##################################################

def parse_args(args = None, namespace = None):
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(prog = "Commands", description = "Produce relevant commands for all tasks, or the one specified.")
    parser.add_argument("-t", "--task", default = None, choices = [EMOTION_DIR_NAME, CHORD_DIR_NAME, STYLE_DIR_NAME], type = str, help = "Name of task")
    parser.add_argument("-r", "--reset", action = "store_true", help = "Whether or not to reset task directory")
    return parser.parse_args(args = args, namespace = namespace)

##################################################


# MAIN METHOD
##################################################

if __name__ == "__main__":

    # SETUP
    ##################################################

    # parse the command-line arguments
    args = parse_args()

    # interpret arguments
    produce_all_tasks = (args.task is None)
    if not produce_all_tasks:
        args.task = args.task.lower()
        if args.task not in ALL_TASKS:
            raise RuntimeError("Invalid --task argument.")
    tasks = ALL_TASKS if produce_all_tasks else [args.task]
    
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

    for base_dir_name in tasks:

        # create base_dir
        base_dir = f"{BASE_DIR}/{base_dir_name}"
        if exists(base_dir) and args.reset:
            rmtree(base_dir, ignore_errors = True)
        if not exists(base_dir):
            mkdir(base_dir)

        # other variables
        software_dir = f"{SOFTWARE_DIR}/{base_dir_name}"
        data_dir = f"{base_dir}/{DATA_DIR_NAME}"
        jingyue_data_dir = TASK_NAME_TO_JINGYUE_DATA_DIR[base_dir_name]
        
        # title
        logging.info(f"{base_dir_name.upper()}")
        logging.info(MAJOR_SEPARATOR_LINE)

        # create dataset
        logging.info("* Dataset:")
        logging.info(join_arguments(args = [
            f"python {software_dir}/dataset.py", 
            f"--data_dir {jingyue_data_dir}/data_rvq_tokens",
            f"--prebottleneck_data_dir {jingyue_data_dir}/data_encoder_latents",
            f"--partitions_dir {jingyue_data_dir}/data_splits{'_new' if base_dir_name == EMOTION_DIR_NAME else ''}",
            f"--output_dir {base_dir}",
        ]))
        
        # separator line
        logging.info(MINOR_SEPARATOR_LINE)

        # helper function for generating training commands
        def log_train_command_string(
                using_prebottleneck_latents: bool = False,
                prepool: bool = False,
                use_transformer: bool = False,
            ):
            """Helper function to log the train command string."""
            logging.info("* " + ("Transformer" if use_transformer else "MLP") + (", Prepooled" if prepool else "") + (", Prebottlenecked" if using_prebottleneck_latents else "") + ":")
            logging.info(join_arguments(args = [
                f"python {software_dir}/train.py",
                f"--data_dir {data_dir}/{PREBOTTLENECK_DATA_SUBDIR_NAME if using_prebottleneck_latents else DATA_SUBDIR_NAME}",
                f"--paths_train {data_dir}/{TRAIN_PARTITION_NAME}.txt",
                f"--paths_valid {data_dir}/{VALID_PARTITION_NAME}.txt",
                f"--output_dir {base_dir}",
                "--prepool" if prepool else "",
                "--use_transformer" if use_transformer else "",
                "--use_wandb",
                f"--weight_decay {WEIGHT_DECAY}",
                f"--gpu {DEFAULT_GPU}",
                "--model_name " + ("transformer" if use_transformer else "mlp") + str((2 * int(prepool)) + int(using_prebottleneck_latents)),
            ]))

        # log commands for different models
        for prepool in (False, True):
            for using_prebottleneck_latents in (False, True):
                log_train_command_string(using_prebottleneck_latents = using_prebottleneck_latents, prepool = prepool, use_transformer = False)
        for using_prebottleneck_latents in (False, True):
            log_train_command_string(using_prebottleneck_latents = using_prebottleneck_latents, prepool = False, use_transformer = True)
        del log_train_command_string # free up memory
        
        # separator line
        logging.info(MINOR_SEPARATOR_LINE)

        # evaluate
        logging.info("* Evaluate:")
        logging.info(join_arguments(args = [
            f"python {software_dir}/evaluate.py",
            f"--paths_test {data_dir}/{TEST_PARTITION_NAME}.txt",
            f"--models_list {base_dir}/{MODELS_FILE_NAME}.txt",
            f"--gpu {DEFAULT_GPU}",
        ]))
                
        # end margin
        logging.info(MAJOR_SEPARATOR_LINE)
        logging.info("")

    ##################################################

##################################################

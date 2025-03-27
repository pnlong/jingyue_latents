# README
# Phillip Long
# February 22, 2025

# Utilities.

# python /home/pnlong/jingyue_latents/utils.py

# IMPORTS
##################################################

import json
from typing import Union, List, Tuple, Any
import numpy as np
import pandas as pd
import pickle
import torch

from os.path import exists, dirname, realpath
from os import mkdir, makedirs
from shutil import rmtree
import argparse
import logging

##################################################


# BASIC CONSTANTS
##################################################

# base directory
BASE_DIR = "/deepfreeze/pnlong/jingyue_latents"

# jingyue's directory (and symlinks)
JINGYUE_DIR = "/deepfreeze/user_shares/jingyue"

# tasks
EMOTION_DIR_NAME = "emotion"
CHORD_DIR_NAME = "chord"
MELODY_DIR_NAME = "melody"
MELODY_TRANSFORMER_DIR_NAME = "melody_transformer"
ALL_TASKS = [EMOTION_DIR_NAME, CHORD_DIR_NAME, MELODY_DIR_NAME, MELODY_TRANSFORMER_DIR_NAME]

# subdirectory names
DATA_DIR_NAME = "data"
SPLITS_SUBDIR_NAME = "splits"
MAPPINGS_SUBDIR_NAME = "mappings" # map each filename to a label
DATA_SUBDIR_NAME = "data"
PREBOTTLENECK_DATA_SUBDIR_NAME = "prebottleneck"
CHECKPOINTS_DIR_NAME = "checkpoints"
DEFAULT_MODEL_NAME = "model"
SYMLINKS_DIR_NAME = "symlinks"
BASELINE_DIR_NAME = "baseline"
TOKEN_SUBDIR_NAME = "tokens"
BAR_POSITION_SUBDIR_NAME = "bar_positions"

# symlinks dir, use `ln -sf /path/to/directory /path/to/symlink` to create a new symlink
SYMLINKS_DIR = f"{dirname(realpath(__file__))}/{SYMLINKS_DIR_NAME}"
JINGYUE_DATA_SYMLINKS_DIR_BY_TASK = {task: f"{SYMLINKS_DIR}/{task}" for task in ALL_TASKS}
JINGYUE_DATA_SYMLINK_NAME = DATA_SUBDIR_NAME
JINGYUE_PREBOTTLENECK_DATA_SYMLINK_NAME = PREBOTTLENECK_DATA_SUBDIR_NAME
JINGYUE_SPLITS_SYMLINK_NAME = SPLITS_SUBDIR_NAME

# will all be populated later
DIR_BY_TASK = {task: f"{BASE_DIR}/{task}" for task in ALL_TASKS} # task name to BASE_DIR subdirectory
MAX_SEQ_LEN_BY_TASK = {task: 1 for task in ALL_TASKS} # task name to maximum sequence length for that task
TOKEN_MAX_SEQ_LEN_BY_TASK = {task: 0 for task in ALL_TASKS}
MAPPING_NAMES_BY_TASK = {task: [task] for task in ALL_TASKS} # mapping name(s) for each task
N_OUTPUTS_PER_INPUT_BAR_BY_TASK = {task: 1 for task in ALL_TASKS} # number of events per bar for that task
JINGYUE_DIR_BY_TASK = dict() # jingyue's data dir for each task
VOCABULARY_SIZE_BY_TASK = {task: 0 for task in ALL_TASKS} # default to no vocabulary for each task

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

def prettify_table(df: pd.DataFrame, column_widths: list = None) -> str:
    """
    Given a DataFrame `df`, return a string that is the prettified version of that data.
    `column_widths` represent the width of each column in characters -- the length of 
    `column_widths` must be equal to the number of columns in `df`.
    """
    truncate_string = lambda string, n: string + "".join([" " for _ in range(n - len(string))]) if len(string) <= n else string[:n - 3] + "..." # helper function to get strings of constant length
    format_row = lambda row: "  ".join(map(truncate_string, map(lambda elem: str(elem) if not pd.isna(elem) else "", row), column_widths)) # row is a list representing a row in the data frame
    column_names_row = format_row(row = map(lambda column_name: column_name.upper(), df.columns))
    output_string = f"{column_names_row}\n" # add column names to output string
    output_string += "".join(("-" for _ in range(len(column_names_row)))) + "\n" # add separator line to output string
    for i in df.index: # iterate through each row in data frame
        output_string += format_row(row = df.iloc[i].tolist())
        if i != df.index[-1]: # if not the last row
            output_string += "\n"
    return output_string

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

def save_pickle(filepath: str, data: Any):
    """Save an object to a pickle file."""
    with open(filepath, "wb") as f:
        pickle.dump(obj = data, file = f)

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

def directory_creator(directory: str, reset: bool = False):
    """Helper function for creating directories."""
    if not exists(directory) or reset:
        if exists(directory):
            rmtree(directory, ignore_errors = True)
        makedirs(directory, exist_ok = True)

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
DATA_TYPE = torch.float32
TOKEN_TYPE = torch.int32
BAR_POSITION_TYPE = torch.int32
LABEL_TYPE = torch.uint8

# training statistics
LOSS_STATISTIC_NAME = "loss"
ACCURACY_STATISTIC_NAME = "accuracy"
RELEVANT_TRAINING_STATISTICS = [LOSS_STATISTIC_NAME, ACCURACY_STATISTIC_NAME]
TRAINING_STATISTICS_OUTPUT_COLUMNS = ["step", "partition", f"is_{LOSS_STATISTIC_NAME}", "value"]

# data loader
BATCH_SIZE = 16
FRONT_PAD = False

# training defaults
N_EPOCHS = 100
MAX_N_STEPS = 100000 # maximum number of steps in training phase
MAX_VALID_FREQUENCY = 5000 # maximum number of steps between each validation step
MAX_N_VALID_STEPS = 2000 # maximum number of steps in validation phase
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
EMOTION_INDEXER = EMOTION_ID_TO_INDEX

# number of emotion classes
N_EMOTION_CLASSES = len(EMOTIONS)

# maximum song length (in bars) for emotion data
MAX_SEQ_LEN_BY_TASK[EMOTION_DIR_NAME] = 42

# mapping name(s) for emotion
MAPPING_NAMES_BY_TASK[EMOTION_DIR_NAME] = [EMOTION_DIR_NAME]
JINGYUE_DIR_BY_TASK[EMOTION_DIR_NAME] = f"{JINGYUE_DIR}/EMOPIA_emotion_recognition"

# emotion evaluation output columns
EVALUATION_LOSS_OUTPUT_COLUMNS_BY_TASK[EMOTION_DIR_NAME] = EVALUATION_LOSS_OUTPUT_COLUMNS
EVALUATION_ACCURACY_OUTPUT_COLUMNS_BY_TASK[EMOTION_DIR_NAME] = EVALUATION_ACCURACY_OUTPUT_COLUMNS

##################################################


# CHORD CONSTANTS
##################################################

# mappings
DEFAULT_SCALE = ["C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"]
DEFAULT_CHORD_QUALITIES = [
    "maj", "min", "dim", "aug", "7", "maj7", "min7", "dim7", "hdim7", "sus2", "sus4",
    "11", "13", "7(#9)", "9", "maj(11)", "maj(9)", "maj13", "maj6", "maj6(9)", "maj9", "maj9(11)", "min(11)", "min(9)", "min11", "min13", "min6", "min6(9)", "min9", "minmaj7", "sus4(b7)", "sus4(b7,9)",
    ]
NO_CHORD_SYMBOL = "N"
CHORDS11 = [
    "A:7", "A:aug", "A:dim", "A:dim7", "A:hdim7", "A:maj", "A:maj7", "A:min", "A:min7", "A:sus2", "A:sus4",
    "Ab:7", "Ab:aug", "Ab:dim", "Ab:dim7", "Ab:hdim7", "Ab:maj", "Ab:maj7", "Ab:min", "Ab:min7", "Ab:sus2", "Ab:sus4",
    "B:7", "B:aug", "B:dim", "B:dim7", "B:hdim7", "B:maj", "B:maj7", "B:min", "B:min7", "B:sus2", "B:sus4",
    "Bb:7", "Bb:aug", "Bb:dim", "Bb:dim7", "Bb:hdim7", "Bb:maj", "Bb:maj7", "Bb:min", "Bb:min7", "Bb:sus2", "Bb:sus4",
    "C#:7", "C#:aug", "C#:dim", "C#:dim7", "C#:hdim7", "C#:maj", "C#:maj7", "C#:min", "C#:min7", "C#:sus2", "C#:sus4",
    "C:7", "C:aug", "C:dim", "C:dim7", "C:hdim7", "C:maj", "C:maj7", "C:min", "C:min7", "C:sus2", "C:sus4",
    "D:7", "D:aug", "D:dim", "D:dim7", "D:hdim7", "D:maj", "D:maj7", "D:min", "D:min7", "D:sus2", "D:sus4",
    "E:7", "E:aug", "E:dim", "E:dim7", "E:hdim7", "E:maj", "E:maj7", "E:min", "E:min7", "E:sus2", "E:sus4",
    "Eb:7", "Eb:aug", "Eb:dim", "Eb:dim7", "Eb:hdim7", "Eb:maj", "Eb:maj7", "Eb:min", "Eb:min7", "Eb:sus2", "Eb:sus4",
    "F#:7", "F#:aug", "F#:dim", "F#:dim7", "F#:hdim7", "F#:maj", "F#:maj7", "F#:min", "F#:min7", "F#:sus2", "F#:sus4",
    "F:7", "F:aug", "F:dim", "F:dim7", "F:hdim7", "F:maj", "F:maj7", "F:min", "F:min7", "F:sus2", "F:sus4",
    "G:7", "G:aug", "G:dim", "G:dim7", "G:hdim7", "G:maj", "G:maj7", "G:min", "G:min7", "G:sus2", "G:sus4",
    NO_CHORD_SYMBOL]
CHORDS32 = [
    "A:11", "A:13", "A:7", "A:7(#9)", "A:9", "A:aug", "A:dim", "A:dim7", "A:hdim7", "A:maj", "A:maj(11)", "A:maj(9)", 
    "A:maj13", "A:maj6", "A:maj6(9)", "A:maj7", "A:maj9", "A:maj9(11)", "A:min", "A:min(11)", "A:min(9)", "A:min11", 
    "A:min13", "A:min6", "A:min6(9)", "A:min7", "A:min9", "A:minmaj7", "A:sus2", "A:sus4", "A:sus4(b7)", "A:sus4(b7,9)",
    "Ab:11", "Ab:13", "Ab:7", "Ab:7(#9)", "Ab:9", "Ab:aug", "Ab:dim", "Ab:dim7", "Ab:hdim7", "Ab:maj", "Ab:maj(11)", "Ab:maj(9)", 
    "Ab:maj13", "Ab:maj6", "Ab:maj6(9)", "Ab:maj7", "Ab:maj9", "Ab:maj9(11)", "Ab:min", "Ab:min(11)", "Ab:min(9)", "Ab:min11", 
    "Ab:min6", "Ab:min6(9)", "Ab:min7", "Ab:min9", "Ab:minmaj7", "Ab:sus2", "Ab:sus4", "Ab:sus4(b7)", "Ab:sus4(b7,9)", 
    "B:11", "B:13", "B:7", "B:7(#9)", "B:9", "B:aug", "B:dim", "B:dim7", "B:hdim7", "B:maj", "B:maj(11)", "B:maj(9)", 
    "B:maj13", "B:maj6", "B:maj6(9)", "B:maj7", "B:maj9", "B:min", "B:min(11)", "B:min(9)", "B:min11", 
    "B:min13", "B:min6", "B:min6(9)", "B:min7", "B:min9", "B:minmaj7", "B:sus2", "B:sus4", "B:sus4(b7)", "B:sus4(b7,9)", 
    "Bb:11", "Bb:13", "Bb:7", "Bb:7(#9)", "Bb:9", "Bb:aug", "Bb:dim", "Bb:dim7", "Bb:hdim7", "Bb:maj", "Bb:maj(11)", "Bb:maj(9)", 
    "Bb:maj13", "Bb:maj6", "Bb:maj6(9)", "Bb:maj7", "Bb:maj9", "Bb:maj9(11)", "Bb:min", "Bb:min(11)", "Bb:min(9)", "Bb:min11", 
    "Bb:min13", "Bb:min6", "Bb:min6(9)", "Bb:min7", "Bb:min9", "Bb:minmaj7", "Bb:sus2", "Bb:sus4", "Bb:sus4(b7)", "Bb:sus4(b7,9)",
    "C#:11", "C#:13", "C#:7", "C#:7(#9)", "C#:9", "C#:aug", "C#:dim", "C#:dim7", "C#:hdim7", "C#:maj", "C#:maj(11)", "C#:maj(9)", 
    "C#:maj6", "C#:maj6(9)", "C#:maj7", "C#:maj9", "C#:maj9(11)", "C#:min", "C#:min(11)", "C#:min(9)", "C#:min11", 
    "C#:min13", "C#:min6", "C#:min6(9)", "C#:min7", "C#:min9", "C#:minmaj7", "C#:sus2", "C#:sus4", "C#:sus4(b7)", "C#:sus4(b7,9)",
    "C:11", "C:13", "C:7", "C:7(#9)", "C:9", "C:aug", "C:dim", "C:dim7", "C:hdim7", "C:maj", "C:maj(11)", "C:maj(9)", 
    "C:maj13", "C:maj6", "C:maj6(9)", "C:maj7", "C:maj9", "C:maj9(11)", "C:min", "C:min(11)", "C:min(9)", "C:min11", 
    "C:min6", "C:min6(9)", "C:min7", "C:min9", "C:minmaj7", "C:sus2", "C:sus4", "C:sus4(b7)", "C:sus4(b7,9)", 
    "D:11", "D:13", "D:7", "D:7(#9)", "D:9", "D:aug", "D:dim", "D:dim7", "D:hdim7", "D:maj", "D:maj(11)", "D:maj(9)", 
    "D:maj13", "D:maj6", "D:maj6(9)", "D:maj7", "D:maj9", "D:maj9(11)", "D:min", "D:min(11)", "D:min(9)", "D:min11", 
    "D:min13", "D:min6", "D:min7", "D:min9", "D:minmaj7", "D:sus2", "D:sus4", "D:sus4(b7)", "D:sus4(b7,9)", 
    "E:11", "E:13", "E:7", "E:7(#9)", "E:9", "E:aug", "E:dim", "E:dim7", "E:hdim7", "E:maj", "E:maj(11)", "E:maj(9)", 
    "E:maj13", "E:maj6", "E:maj6(9)", "E:maj7", "E:maj9", "E:min", "E:min(11)", "E:min(9)", "E:min11", 
    "E:min6", "E:min6(9)", "E:min7", "E:min9", "E:minmaj7", "E:sus2", "E:sus4", "E:sus4(b7)", "E:sus4(b7,9)", 
    "Eb:11", "Eb:13", "Eb:7", "Eb:7(#9)", "Eb:9", "Eb:aug", "Eb:dim", "Eb:dim7", "Eb:hdim7", "Eb:maj", "Eb:maj(11)", "Eb:maj(9)", 
    "Eb:maj6", "Eb:maj6(9)", "Eb:maj7", "Eb:maj9", "Eb:maj9(11)", "Eb:min", "Eb:min(11)", "Eb:min(9)", "Eb:min11", 
    "Eb:min6", "Eb:min6(9)", "Eb:min7", "Eb:min9", "Eb:minmaj7", "Eb:sus2", "Eb:sus4", "Eb:sus4(b7)", "Eb:sus4(b7,9)", 
    "F#:11", "F#:13", "F#:7", "F#:7(#9)", "F#:9", "F#:aug", "F#:dim", "F#:dim7", "F#:hdim7", "F#:maj", "F#:maj(11)", "F#:maj(9)", 
    "F#:maj6", "F#:maj6(9)", "F#:maj7", "F#:maj9", "F#:maj9(11)", "F#:min", "F#:min(11)", "F#:min(9)", "F#:min11", 
    "F#:min13", "F#:min6", "F#:min6(9)", "F#:min7", "F#:min9", "F#:minmaj7", "F#:sus2", "F#:sus4", "F#:sus4(b7)", "F#:sus4(b7,9)", 
    "F:11", "F:13", "F:7", "F:7(#9)", "F:9", "F:aug", "F:dim", "F:dim7", "F:hdim7", "F:maj", "F:maj(11)", "F:maj(9)", 
    "F:maj13", "F:maj6", "F:maj6(9)", "F:maj7", "F:maj9", "F:maj9(11)", "F:min", "F:min(11)", "F:min(9)", "F:min11", 
    "F:min13", "F:min6", "F:min6(9)", "F:min7", "F:min9", "F:minmaj7", "F:sus2", "F:sus4", "F:sus4(b7)", "F:sus4(b7,9)", 
    "G:11", "G:13", "G:7", "G:9", "G:aug", "G:dim", "G:dim7", "G:hdim7", "G:maj", "G:maj(11)", "G:maj(9)", "G:maj6", "G:maj6(9)", "G:maj7", "G:maj9", 
    "G:min", "G:min(11)", "G:min(9)", "G:min11", 
    "G:min13", "G:min6", "G:min6(9)", "G:min7", "G:min9", "G:minmaj7", "G:sus2", "G:sus4", "G:sus4(b7)", "G:sus4(b7,9)", 
    NO_CHORD_SYMBOL]
CHORD11_TO_INDEX = {chord: i for i, chord in enumerate(CHORDS11)}
INDEX_TO_CHORD11 = inverse_dict(d = CHORD11_TO_INDEX)
CHORD32_TO_INDEX = {chord: i for i, chord in enumerate(CHORDS32)}
INDEX_TO_CHORD32 = inverse_dict(d = CHORD32_TO_INDEX)
CHORD_INDEXER = [CHORD11_TO_INDEX, CHORD32_TO_INDEX] # make sure the order lines up with the order of JINGYUE_CHORD_MAPPING_DIR_NAMES

# number of chords per bar
N_OUTPUTS_PER_INPUT_BAR_BY_TASK[CHORD_DIR_NAME] = 4 # 4, because 4 chords are predicted for each input bar

# number of chord classes
N_CHORD11_CLASSES = len(CHORDS11)
N_CHORD32_CLASSES = len(CHORDS32)

# maximum song length (in bars) for chord data
MAX_SEQ_LEN_BY_TASK[CHORD_DIR_NAME] = 1 # must be 1, since this is a bar-by-bar classification task

# mapping name(s) for chord
JINGYUE_CHORD_MAPPING_DIR_NAMES = ["chord_pkl_11", "chord_pkl_32"]
MAPPING_NAMES_BY_TASK[CHORD_DIR_NAME] = [CHORD_DIR_NAME + "_" + mapping_dir_name.split("_")[-1] for mapping_dir_name in JINGYUE_CHORD_MAPPING_DIR_NAMES]
JINGYUE_DIR_BY_TASK[CHORD_DIR_NAME] = f"{JINGYUE_DIR}/POP909_chord_recognition"
JINGYUE_CHORD_POP909_MIDI_DIR = f"{JINGYUE_DIR_BY_TASK[CHORD_DIR_NAME]}/midi_quantized_480"

# chord evaluation output columns
EVALUATION_LOSS_OUTPUT_COLUMNS_BY_TASK[CHORD_DIR_NAME] = EVALUATION_LOSS_OUTPUT_COLUMNS
EVALUATION_ACCURACY_OUTPUT_COLUMNS_BY_TASK[CHORD_DIR_NAME] = EVALUATION_ACCURACY_OUTPUT_COLUMNS

##################################################


# MELODY CONSTANTS
##################################################

# mappings
MELODIES = ["vocal", "accompaniment", "instrument"]
MELODY_TO_MELODY_ID = {melody: i + 1 for i, melody in enumerate(MELODIES)}
MELODY_TO_INDEX = {melody: i for i, melody in enumerate(MELODY_TO_MELODY_ID.keys())}
MELODY_ID_TO_INDEX = {melody_id: i for i, melody_id in enumerate(MELODY_TO_MELODY_ID.values())}
MELODY_ID_TO_MELODY = inverse_dict(d = MELODY_TO_MELODY_ID)
INDEX_TO_MELODY = inverse_dict(d = MELODY_TO_INDEX)
INDEX_TO_MELODY_ID = inverse_dict(d = MELODY_ID_TO_INDEX)
MELODY_INDEXER = MELODY_ID_TO_INDEX

# number of melody classes
N_MELODY_CLASSES = len(MELODIES)

# remi-related stuff
MELODY_REMI_WORD_TYPES = ["Beat", "Note_Pitch", "Note_Duration"]
MELODY_REMI_VALUES_BY_WORD_TYPE = dict(zip(MELODY_REMI_WORD_TYPES, (
    [0, 2, 3, 4, 6, 8, 9, 10, 12, 14, 15, 16, 18, 20, 21, 22, 24, 26, 27, 28, 30, 32, 33, 34, 36, 38, 39, 40, 42, 44, 45, 46], # onset
    [21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108], # pitch
    [80, 120, 160, 240, 320, 360, 480, 640, 720, 960, 1440, 1920], # duration
)))
MELODY_CLASS_WORD_TYPE = "Note_Track"
MELODY_REMI_VOCABULARY = {word: i for i, word in enumerate([f"{word_type}_{value}" for word_type in MELODY_REMI_WORD_TYPES for value in MELODY_REMI_VALUES_BY_WORD_TYPE[word_type]])}
VOCABULARY_SIZE_BY_TASK[MELODY_DIR_NAME] = len(MELODY_REMI_VOCABULARY)

# maximum song length (in bars) for melody data
MAX_SEQ_LEN_BY_TASK[MELODY_DIR_NAME] = 1 # must be 1, since this is a note-by-note classification task
TOKEN_MAX_SEQ_LEN_BY_TASK[MELODY_DIR_NAME] = 3 # 3 events for eatch sample in a batch

# mapping name(s) for melody
JINGYUE_MELODY_MAPPING_DIR_NAME = "data_events"
MAPPING_NAMES_BY_TASK[MELODY_DIR_NAME] = [MELODY_DIR_NAME]
JINGYUE_DIR_BY_TASK[MELODY_DIR_NAME] = f"{JINGYUE_DIR}/POP909_melody_classification"

# melody evaluation output columns
EVALUATION_LOSS_OUTPUT_COLUMNS_BY_TASK[MELODY_DIR_NAME] = EVALUATION_LOSS_OUTPUT_COLUMNS
EVALUATION_ACCURACY_OUTPUT_COLUMNS_BY_TASK[MELODY_DIR_NAME] = EVALUATION_ACCURACY_OUTPUT_COLUMNS

##################################################


# MELODY TRANSFORMER CONSTANTS
##################################################

# mappings
MELODY_TRANSFORMER_INDEXER = MELODY_ID_TO_INDEX

# number of melody classes
N_MELODY_TRANSFORMER_CLASSES = len(MELODIES) + 1 # add 1 for the no-class option

# remi-related stuff
MELODY_TRANSFORMER_BOS_TOKEN = "BOS"
MELODY_TRANSFORMER_EOS_TOKEN = "EOS"
MELODY_TRANSFORMER_EVENTS = [f"{MELODY_TRANSFORMER_BOS_TOKEN}_None", f"{MELODY_TRANSFORMER_EOS_TOKEN}_None", "Bar_None"] + [f"{MELODY_CLASS_WORD_TYPE}_{i + 1}" for i in range(len(MELODIES))] + [f"Time_Signature_{time_signature}" for time_signature in ("2/2", "2/4", "3/4", "3/8", "4/4", "6/8")] + list(MELODY_REMI_VOCABULARY.keys())
MELODY_TRANSFORMER_REMI_VOCABULARY = {event: i for i, event in enumerate(MELODY_TRANSFORMER_EVENTS)}
VOCABULARY_SIZE_BY_TASK[MELODY_TRANSFORMER_DIR_NAME] = len(MELODY_TRANSFORMER_REMI_VOCABULARY)

# maximum song length (in bars) for melody data
MELODY_TRANSFORMER_CLIP_LENGTH = 16 # number of bars in a clip
MAX_SEQ_LEN_BY_TASK[MELODY_TRANSFORMER_DIR_NAME] = MELODY_TRANSFORMER_CLIP_LENGTH
TOKEN_MAX_SEQ_LEN_BY_TASK[MELODY_TRANSFORMER_DIR_NAME] = 3858 # maximum number of events in a clip

# mapping name(s) for melody
MAPPING_NAMES_BY_TASK[MELODY_TRANSFORMER_DIR_NAME] = [MELODY_TRANSFORMER_DIR_NAME]
JINGYUE_DIR_BY_TASK[MELODY_TRANSFORMER_DIR_NAME] = JINGYUE_DIR_BY_TASK[MELODY_DIR_NAME]

# melody evaluation output columns
EVALUATION_LOSS_OUTPUT_COLUMNS_BY_TASK[MELODY_TRANSFORMER_DIR_NAME] = EVALUATION_LOSS_OUTPUT_COLUMNS
EVALUATION_ACCURACY_OUTPUT_COLUMNS_BY_TASK[MELODY_TRANSFORMER_DIR_NAME] = EVALUATION_ACCURACY_OUTPUT_COLUMNS

##################################################


# MISCELLANEOUS CONSTANTS
##################################################

# error message
INVALID_TASK_ERROR = lambda task: f"Invalid --task `{task}`"

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
JSON_FILETYPE = "json"

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
    tasks = list(map(lambda task: MELODY_DIR_NAME if task == MELODY_TRANSFORMER_DIR_NAME else task, tasks)) # replace melody tranformer with just melody
    tasks = unique(l = tasks)
    
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
        if not exists(base_dir):
            mkdir(base_dir)
        
        # title
        logging.info(f"{task.upper()}")
        logging.info(MAJOR_SEPARATOR_LINE)

        # create dataset
        logging.info("* Dataset:")
        def log_dataset_command_string():
            """Helper function to log the dataset command string."""
            logging.info(join_arguments(args = [
                f"python {SOFTWARE_DIR}/dataset.py", 
                f"--task {task}",
            ]))
        log_dataset_command_string()
        
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
        
        # separator line
        logging.info(MINOR_SEPARATOR_LINE)

        # evaluate
        logging.info("* Evaluate:")
        def log_evaluate_command_string():
            """Helper function to log the evaluate command string."""
            logging.info(join_arguments(args = [
                f"python {SOFTWARE_DIR}/evaluate.py",
                f"--task {task}",
                f"--gpu {DEFAULT_GPU}",
            ]))
        log_evaluate_command_string()

        # special cases
        if task == MELODY_DIR_NAME:
            logging.info(MINOR_SEPARATOR_LINE)
            task = MELODY_TRANSFORMER_DIR_NAME # set task to melody transfrmer
            logging.info("* Special Melody Transformer:")
            log_dataset_command_string()
            logging.info(join_arguments(args = [
                f"python {SOFTWARE_DIR}/train.py",
                f"--task {task}",
                "--use_wandb",
                f"--gpu {DEFAULT_GPU}",
                "--model_name melody_transformer",
            ]))
            log_evaluate_command_string()
                
        # end margin
        logging.info(MAJOR_SEPARATOR_LINE)
        logging.info("")

    ##################################################

##################################################

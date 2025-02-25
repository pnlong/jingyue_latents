# README
# Phillip Long
# February 22, 2025

# Prepare data for training an emotion recognition model.

# python /home/pnlong/jingyue_latents/emotion/dataset.py

# IMPORTS
##################################################

from typing import List
from os.path import basename, exists
from os import mkdir
from shutil import rmtree
import argparse
import logging
import random
import multiprocessing
from tqdm import tqdm

import torch
import torch.utils.data

from os.path import dirname, realpath
import sys
sys.path.insert(0, dirname(realpath(__file__)))
sys.path.insert(0, dirname(dirname(realpath(__file__))))

import utils

##################################################


# DATASET CLASS
##################################################


class EmotionDataset(torch.utils.data.Dataset):

    # intializer
    def __init__(
        self,
        paths: str, # path to file with relevant paths
    ):
        super().__init__()
        self.paths = utils.load_txt(filepath = paths)

    # length attribute
    def __len__(self) -> int:
        return len(self.paths)

    # obtain an item
    def __getitem__(self, index: int) -> dict:

        # get the name
        path = self.paths[index]

        # get label from path
        label = utils.EMOTION_ID_TO_INDEX[basename(path)[:2]] # Q3_77z6Ep3aOmg_1.pkl

        # load in sequence as tensor
        seq = torch.load(f = path, weights_only = True).to(utils.DATA_TYPE)

        # return dictionary of sequence, label, and path
        return {
            "seq": seq,
            "label": label,
            "path": path,
        }
    
    # collate method
    @classmethod
    def collate(cls, batch: List[dict]) -> dict:

        # aggregate list of sequences
        seqs, labels, paths = utils.transpose(l = [sample.values() for sample in batch])

        # return dictionary of sequences, labels, masks, and paths
        return {
            "seq": utils.pad(seqs = seqs, length = utils.EMOTION_MAX_SEQ_LEN).to(utils.DATA_TYPE),
            "label": torch.tensor(labels, dtype = utils.LABEL_TYPE),
            "mask": utils.mask(seqs = seqs, length = utils.EMOTION_MAX_SEQ_LEN).to(torch.bool),
            "path": paths,
        }

##################################################


# DATA CONSTANTS
##################################################

# directory with jingyue's data
EMOTION_DATA_DIR = "/deepfreeze/user_shares/jingyue/EMOPIA_emotion_recognition"

# fraction of files in the validation partition that should be moved into the test partition
FRACTION_OF_VALIDATION_TO_TEST = 0.5

##################################################


# ARGUMENTS
##################################################

def parse_args(args = None, namespace = None):
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(prog = "Data", description = "Wrangle data.")
    parser.add_argument("-dd", "--data_dir", default = f"{EMOTION_DATA_DIR}/data_rvq_tokens_test", type = str, help = "Directory containing pickled data files")
    parser.add_argument("-pd", "--partitions_dir", default = f"{EMOTION_DATA_DIR}/data_splits", type = str, help = "Directory containing pickled training and validation filepaths")
    parser.add_argument("-od", "--output_dir", default = utils.EMOTION_DIR, type = str, help = "Output directory")
    parser.add_argument("-r", "--reset", action = "store_true", help = "Whether or not to recreate files")
    return parser.parse_args(args = args, namespace = namespace)

##################################################


# MAIN METHOD
##################################################

if __name__ == "__main__":

    # SETUP
    ##################################################

    # parse the command-line arguments
    args = parse_args()

    # set up logging
    logging.basicConfig(level = logging.INFO, format = "%(message)s")

    # set random seed
    random.seed(0)

    # number of jobs for multiprocessing and data loader
    jobs = int(multiprocessing.cpu_count() / 4)

    # create output dir
    splits_dir = f"{args.output_dir}/{utils.DATA_DIR_NAME}"
    if not exists(splits_dir) or args.reset:
        if exists(splits_dir):
            rmtree(splits_dir, ignore_errors = True)
        mkdir(splits_dir)
    data_dir = f"{splits_dir}/{utils.DATA_SUBDIR_NAME}"
    if not exists(data_dir) or args.reset:
        if exists(data_dir):
            rmtree(data_dir, ignore_errors = True)
        mkdir(data_dir)

    # list of all paths
    all_paths = []

    # convert path to an absolute path
    convert_to_absolute_path = lambda path: f"{args.data_dir}/{path}"
    convert_to_absolute_output_path = lambda path: f"{data_dir}/{'.'.join(basename(path).split('.')[:-1])}.pt"

    ##################################################


    # READ IN TRAINING AND VALIDATION PATHS
    ##################################################

    # read in training file
    train_paths = utils.load_pickle(filepath = f"{args.partitions_dir}/train.pkl")
    train_paths = list(map(convert_to_absolute_path, train_paths)) # convert to absolute paths
    all_paths.extend(train_paths) # add to all paths
    train_paths = list(map(convert_to_absolute_output_path, train_paths)) # convert to final absolute output paths

    # read in validation file
    valid_paths = utils.load_pickle(filepath = f"{args.partitions_dir}/valid.pkl")
    valid_paths = list(map(convert_to_absolute_path, valid_paths)) # convert to absolute paths
    random.shuffle(valid_paths) # shuffle valid paths to ensure randomness
    all_paths.extend(valid_paths) # add to all paths
    valid_paths = list(map(convert_to_absolute_output_path, valid_paths)) # convert to final absolute output paths

    ##################################################


    # OUTPUT LONGEST SEQUENCE LENGTH, WHILE SERIALIZING TORCH TENSORS
    ##################################################

    # helper function to get the length (in number of bars) of the song at the given path
    def get_song_length(path: str, path_output: str) -> int:
        """Return the length (in number of bars) of the song at the given path."""

        # load in pickle file
        seq = utils.load_pickle(filepath = path) # load in pickle file

        # if resetting, save as torch pickle object
        if not exists(path_output) or args.reset:
            seq = torch.from_numpy(seq).to(utils.DATA_TYPE) # convert from numpy array to torch tensor of desired type
            torch.save(obj = seq, f = path_output) # save sequence as torch pickle object

        # return the number of bars in the sequence
        return len(seq) # seq.shape[0]

    # use multiprocessing
    with multiprocessing.Pool(processes = jobs) as pool:
        song_lengths = list(pool.starmap(
            func = get_song_length,
            iterable = tqdm(
                iterable = zip(all_paths, map(convert_to_absolute_output_path, all_paths)),
                desc = "Retrieving song lengths (in bars)",
                total = len(all_paths)),
            chunksize = utils.CHUNK_SIZE,
        ))

    # print longest sequence length
    logging.info(f"Longest song length (in number of bars): {max(song_lengths)}")

    # free up memory
    del all_paths, song_lengths

    ##################################################


    # OUTPUT TRAINING, VALIDATION, AND TESTING PATHS
    ##################################################

    # output paths
    train_output_path = f"{splits_dir}/{utils.TRAIN_PARTITION_NAME}.txt"
    valid_output_path = f"{splits_dir}/{utils.VALID_PARTITION_NAME}.txt"
    test_output_path = f"{splits_dir}/{utils.TEST_PARTITION_NAME}.txt"

    # split validation partition into validation and test partitions
    n_valid = int(FRACTION_OF_VALIDATION_TO_TEST * len(valid_paths)) + 1
    test_paths = valid_paths[n_valid:]
    valid_paths = valid_paths[:n_valid] 

    # write to file
    if not all(map(exists, (train_output_path, valid_output_path, test_output_path))) or args.reset:
        logging.info(utils.MAJOR_SEPARATOR_LINE)
        utils.save_txt(filepath = train_output_path, data = train_paths)
        logging.info(f"Wrote training partition to {train_output_path}.")
        utils.save_txt(filepath = valid_output_path, data = valid_paths)
        logging.info(f"Wrote validation partition to {valid_output_path}.")
        utils.save_txt(filepath = test_output_path, data = test_paths)
        logging.info(f"Wrote test partition to {test_output_path}.")

    # free up memory
    del n_valid, train_paths, valid_paths, test_paths

    ##################################################


    # TEST DATASET
    ##################################################
    
    # print separator line
    logging.info(utils.MAJOR_SEPARATOR_LINE)

    # create dataset
    dataset = EmotionDataset(paths = valid_output_path)
    data_loader = torch.utils.data.DataLoader(
        dataset = dataset,
        batch_size = 8,
        shuffle = True,
        num_workers = jobs,
        collate_fn = dataset.collate,
    )

    # iterate over the data loader
    n_batches = 0
    n_samples = 0
    for i, batch in enumerate(data_loader):

        # update tracker variables
        n_batches += 1
        n_samples += len(batch)

        # print example on first batch
        if i == 0:
            logging.info("Example:")
            inputs, labels, mask, paths = batch["seq"], batch["label"], batch["mask"], batch["path"]
            logging.info(f"Shape of data: {tuple(inputs.shape)}")
            # logging.info(f"Data: {inputs}")
            logging.info(f"Shape of labels: {tuple(labels.shape)}")
            # logging.info(f"Labels: {labels}")
            logging.info(f"Shape of mask: {tuple(mask.shape)}")
            logging.info(f"Shape of paths: {len(paths)}")

    # print how many batches were loaded
    logging.info(f"Successfully loaded {n_batches} batches ({n_samples} samples).")

    ##################################################

##################################################
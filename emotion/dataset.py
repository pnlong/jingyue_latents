# README
# Phillip Long
# February 22, 2025

# Prepare data for training an emotion recognition model.

# python /home/pnlong/jingyue_latents/emotion/dataset.py

# IMPORTS
##################################################

from typing import List
from os.path import exists
from os import mkdir
from shutil import rmtree
import argparse
import logging
import multiprocessing
from tqdm import tqdm
import numpy as np

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
        directory: str, # directory containing relevant paths
        paths: str, # path to file with relevant path basenames
        pool: bool = False, # whether to pool (average) the num_bar dimension
    ):
        super().__init__()
        self.directory = directory
        self.paths = utils.load_txt(filepath = paths)
        self.pool = pool

    # length attribute
    def __len__(self) -> int:
        return len(self.paths)

    # obtain an item
    def __getitem__(self, index: int) -> dict:

        # get the name
        base = self.paths[index]
        path = f"{self.directory}/{base}"

        # get label from path
        label = utils.EMOTION_ID_TO_INDEX[base.split("_")[0]] # Q3_77z6Ep3aOmg_1.pkl

        # load in sequence as tensor
        seq = torch.load(f = path, weights_only = True).to(utils.DATA_TYPE)

        # pool if necessary
        if self.pool:
            seq = torch.mean(input = seq, dim = 0) # mean pool

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

        # if pooling was applied
        if len(seqs[0].shape) == 1:
            seqs = torch.stack(tensors = seqs, dim = 0)
            mask = torch.ones(size = (len(labels),))

        # if pooling was not applied
        else:
            seqs = utils.pad(seqs = seqs, length = utils.EMOTION_MAX_SEQ_LEN)
            mask = utils.mask(seqs = seqs, length = utils.EMOTION_MAX_SEQ_LEN)

        # return dictionary of sequences, labels, masks, and paths
        return {
            "seq": seqs.to(utils.DATA_TYPE),
            "label": torch.tensor(labels, dtype = utils.LABEL_TYPE),
            "mask": mask.to(torch.bool),
            "path": paths,
        }

##################################################


# ARGUMENTS
##################################################

def parse_args(args = None, namespace = None):
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(prog = "Data", description = "Wrangle data.")
    parser.add_argument("-d", "--data_dir", default = f"{utils.EMOTION_DATA_DIR}/data_rvq_tokens", type = str, help = "Directory containing pickled data files")
    parser.add_argument("-p", "--prebottleneck_data_dir", default = f"{utils.EMOTION_DATA_DIR}/data_encoder_latents", type = str, help = "Directory containing pickled prebottleneck data files")
    parser.add_argument("-s", "--partitions_dir", default = f"{utils.EMOTION_DATA_DIR}/data_splits_new", type = str, help = "Directory containing pickled training and validation filepaths")
    parser.add_argument("-o", "--output_dir", default = utils.EMOTION_DIR, type = str, help = "Output directory")
    parser.add_argument("-r", "--reset", action = "store_true", help = "Whether or not to recreate files")
    parser.add_argument("-j", "--jobs", default = int(multiprocessing.cpu_count() / 4), type = int, help = "Number of workers for data loading")
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

    # create output dir
    def directory_creator(directory: str):
        """Helper function for creating relevant directories."""
        if not exists(directory) or args.reset:
            if exists(directory):
                rmtree(directory, ignore_errors = True)
            mkdir(directory)
    splits_dir = f"{args.output_dir}/{utils.DATA_DIR_NAME}"
    directory_creator(directory = splits_dir)
    data_dir = f"{splits_dir}/{utils.DATA_SUBDIR_NAME}"
    directory_creator(directory = data_dir)
    prebottleneck_data_dir = f"{splits_dir}/{utils.PREBOTTLENECK_DATA_SUBDIR_NAME}"
    directory_creator(directory = prebottleneck_data_dir)

    ##################################################


    # READ IN TRAINING, VALIDATION, AND TEST PATHS
    ##################################################

    # read in paths
    load_pickle = lambda filepath: list(map(lambda path: ".".join(path.split(".")[:-1]), utils.load_pickle(filepath = filepath))) # load pickle, removing filetype while at it
    train_stems = load_pickle(filepath = f"{args.partitions_dir}/train.pkl") # read in training file
    valid_stems = load_pickle(filepath = f"{args.partitions_dir}/valid.pkl") # read in validation file
    test_stems = load_pickle(filepath = f"{args.partitions_dir}/test.pkl") # read in testing file
    all_stems = train_stems + valid_stems + test_stems # list of all path stems
    convert_stems_to_absolute_paths = lambda directory: list(map(lambda stem: f"{directory}/{stem}." + (utils.TENSOR_FILETYPE if directory in (data_dir, prebottleneck_data_dir) else utils.PICKLE_FILETYPE), all_stems)) # helper function
    n_stems = len(all_stems) # number of stems
    del load_pickle # free up memory

    ##################################################


    # OUTPUT LONGEST SEQUENCE LENGTH
    ##################################################

    # helper function to get the length (in number of bars) of the song at the given path
    def get_song_length(path: str) -> int:
        """Return the length (in number of bars) of the song at the given path."""
        seq = utils.load_pickle(filepath = path) # load in pickle file
        return len(seq) # seq.shape[0], the number of bars in the sequence

    # use multiprocessing
    with multiprocessing.Pool(processes = args.jobs) as pool:
        song_lengths = list(pool.map(
            func = get_song_length,
            iterable = tqdm(
                iterable = convert_stems_to_absolute_paths(directory = args.data_dir),
                desc = "Retrieving song lengths (in bars)",
                total = n_stems),
            chunksize = utils.CHUNK_SIZE,
        ))

    # print longest sequence length
    logging.info(f"Longest song length (in number of bars): {max(song_lengths)}")
    logging.info(f"{n_stems} songs (Train: {len(train_stems)}, Validation: {len(valid_stems)}, Test: {len(test_stems)}).")

    # free up memory
    del get_song_length, song_lengths

    ##################################################


    # SAVE TORCH TENSORS
    ##################################################

    # separator line
    logging.info(utils.MAJOR_SEPARATOR_LINE)

    # helper function to save a torch object given the input path to the output path
    def save_path(path: str, path_output: str):
        """Save the input path to the """
        if not exists(path_output) or args.reset:
            seq = utils.load_pickle(filepath = path) # load in pickle file
            seq = torch.from_numpy(seq).to(utils.DATA_TYPE) # convert from numpy array to torch tensor of desired type
            torch.save(obj = seq, f = path_output) # save sequence as torch pickle object

    # use multiprocessing
    def save_paths(input_dir: str, output_dir: str, desc: str):
        """Helper function to use multiprocessing to save paths as torch tensors."""
        with multiprocessing.Pool(processes = args.jobs) as pool:
            _ = pool.starmap(
                func = save_path,
                iterable = tqdm(
                    iterable = zip(
                        convert_stems_to_absolute_paths(directory = input_dir),
                        convert_stems_to_absolute_paths(directory = output_dir),
                    ),
                    desc = desc,
                    total = n_stems),
                chunksize = utils.CHUNK_SIZE,
            )

    # save paths
    save_paths(input_dir = args.data_dir, output_dir = data_dir, desc = "Extracting tensors")
    save_paths(input_dir = args.prebottleneck_data_dir, output_dir = prebottleneck_data_dir, desc = "Extracting prebottleneck tensors")
    
    # free up memory
    del all_stems, convert_stems_to_absolute_paths, n_stems, save_path, save_paths

    ##################################################


    # OUTPUT TRAINING, VALIDATION, AND TESTING PATHS
    ##################################################

    # output paths
    train_output_path = f"{splits_dir}/{utils.TRAIN_PARTITION_NAME}.txt"
    valid_output_path = f"{splits_dir}/{utils.VALID_PARTITION_NAME}.txt"
    test_output_path = f"{splits_dir}/{utils.TEST_PARTITION_NAME}.txt"

    # write to file
    if not all(map(exists, (train_output_path, valid_output_path, test_output_path))) or args.reset:
        logging.info(utils.MAJOR_SEPARATOR_LINE)
        save_txt = lambda filepath, data: utils.save_txt(filepath = filepath, data = list(map(lambda stem: f"{stem}.{utils.TENSOR_FILETYPE}", data)))
        save_txt(filepath = train_output_path, data = train_stems)
        logging.info(f"Wrote training partition to {train_output_path}.")
        save_txt(filepath = valid_output_path, data = valid_stems)
        logging.info(f"Wrote validation partition to {valid_output_path}.")
        save_txt(filepath = test_output_path, data = test_stems)
        logging.info(f"Wrote test partition to {test_output_path}.")
        del save_txt # free up memory

    # free up memory
    del train_stems, valid_stems, test_stems

    ##################################################


    # TEST DATASET
    ##################################################

    # test dataset with different pooling values
    for pool in (False, True):

        # print separator line
        logging.info(utils.MAJOR_SEPARATOR_LINE)
        logging.info("Pooling is " + ("ON" if pool else "OFF") + ".")

        # create dataset
        dataset = EmotionDataset(
            directory = data_dir,
            paths = valid_output_path,
            pool = pool,
        )
        data_loader = torch.utils.data.DataLoader(
            dataset = dataset,
            batch_size = 8,
            shuffle = True,
            num_workers = args.jobs,
            collate_fn = dataset.collate,
        )

        # iterate over the data loader
        n_batches = 0
        n_samples = 0
        for i, batch in enumerate(data_loader):

            # update tracker variables
            n_batches += 1
            n_samples += len(batch["seq"])

            # print example on first batch
            if i == 0:
                logging.info("Example on the validation partition:")
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
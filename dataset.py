# README
# Phillip Long
# March 4, 2025

# Prepare data for training a model.

# python /home/pnlong/jingyue_latents/dataset.py

# IMPORTS
##################################################

from typing import List
from os.path import exists, dirname, realpath
from os import mkdir
from shutil import rmtree
import argparse
import logging
import multiprocessing
from tqdm import tqdm
import numpy as np
import random
import pprint

import torch
import torch.utils.data

from os.path import dirname, realpath
import sys
sys.path.insert(0, dirname(realpath(__file__)))

import utils

##################################################


# CUSTOM DATASET CLASS
##################################################

class CustomDataset(torch.utils.data.Dataset):

    # intializer
    def __init__(
        self,
        directory: str, # directory containing relevant paths
        paths: str, # path to file with relevant path basenames
        indexer: dict, # dictionary that maps labels to indicies
        max_seq_len: int = 1, # maximum sequence length (in bars)
        pool: bool = False, # whether to pool (average) the num_bar dimension
    ):
        super().__init__()
        self.directory = directory
        self.paths = utils.load_txt(filepath = paths)
        self.indexer = indexer
        self.max_seq_len = max_seq_len
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
        label = base.split("_")[0] # extract label from base (e.g. Q3_77z6Ep3aOmg_1.pkl)
        label = self.indexer[label] # convert label to index

        # load in sequence as tensor
        seq = torch.load(f = path, weights_only = True).to(utils.DATA_TYPE)

        # pool if necessary
        if self.pool and len(seq.shape) == 2:
            seq = torch.mean(input = seq, dim = 0) # mean pool
        elif not self.pool and len(seq.shape) == 1:
            seq = seq.unsqueeze(dim = 0) # add a num_bar dimension if there isn't one

        # return dictionary of sequence, label, and path
        return {
            "seq": seq,
            "label": label,
            "path": path,
            "max_seq_len": self.max_seq_len,
        }
    
    # collate method
    @classmethod
    def collate(cls, batch: List[dict]) -> dict:

        # aggregate list of sequences
        seqs, labels, paths = utils.transpose(l = [(sample["seq"], sample["label"], sample["path"]) for sample in batch])
        max_seq_len = batch[0]["max_seq_len"] # max_seq_len is constant across all samples, so just grab the first one

        # if pooling was applied
        if len(seqs[0].shape) == 1:
            seqs = torch.stack(tensors = seqs, dim = 0)
            mask = torch.ones(size = [len(labels)])

        # if pooling was not applied
        else:
            seqs = utils.pad(seqs = seqs, length = max_seq_len)
            mask = utils.mask(seqs = seqs, length = max_seq_len)

        # return dictionary of sequences, labels, masks, and paths
        return {
            "seq": seqs.to(utils.DATA_TYPE),
            "label": torch.tensor(labels, dtype = utils.LABEL_TYPE),
            "mask": mask.to(torch.bool),
            "path": paths,
        }

##################################################


# GET THE CORRECT DATASET GIVEN SOME ARGUMENTS
##################################################

def get_dataset(
        task: str, # relevant task
        directory: str, # directory containing relevant paths
        paths: str, # path to file with relevant path basenames
        pool: bool = False, # whether to pool (average) the num_bar dimension
    ) -> torch.utils.data.Dataset:
    """
    Helper function that returns the correct dataset object
    given some arguments.
    """

    # match the task to the correct dataset arguments
    indexer = utils.INDEXER_BY_TASK[task]
    max_seq_len = utils.MAX_SEQ_LEN_BY_TASK[task]
    pool = pool if task != utils.CHORD_DIR_NAME else False # ensuring pooling is off for chord task
    
    # return dataset with relevant arguments
    return CustomDataset(
        directory = directory, paths = paths,
        indexer = indexer, max_seq_len = max_seq_len,
        pool = pool,
    )

##################################################


# ARGUMENTS
##################################################

def parse_args(args = None, namespace = None):
    """Parse command-line arguments."""

    # create argument parser
    parser = argparse.ArgumentParser(prog = "Data", description = "Wrangle data.")
    parser.add_argument("-t", "--task", required = True, choices = utils.ALL_TASKS, type = str, help = "Name of task")
    parser.add_argument("-r", "--reset", action = "store_true", help = "Whether or not to recreate files")
    parser.add_argument("-j", "--jobs", default = int(multiprocessing.cpu_count() / 4), type = int, help = "Number of workers for data loading")
    
    # parse arguments
    args = parser.parse_args(args = args, namespace = namespace)

    # infer other arguments
    jingyue_data_symlinks_dir = utils.JINGYUE_DATA_SYMLINKS_DIR_BY_TASK[args.task]
    args.data_dir = f"{jingyue_data_symlinks_dir}/{utils.JINGYUE_DATA_SYMLINK_NAME}"
    args.prebottleneck_data_dir = f"{jingyue_data_symlinks_dir}/{utils.JINGYUE_PREBOTTLENECK_DATA_SYMLINK_NAME}"
    args.partitions_dir = f"{jingyue_data_symlinks_dir}/{utils.JINGYUE_SPLITS_SYMLINK_NAME}"
    args.output_dir = utils.DIR_BY_TASK[args.task]

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

    # set up logging
    logging.basicConfig(level = logging.INFO, format = "%(message)s")

    # log arguments
    logging.info(f"Using arguments:\n{pprint.pformat(vars(args))}")

    # set random seed
    random.seed(0)

    # create output dir
    def directory_creator(directory: str):
        """Helper function for creating relevant directories."""
        if not exists(directory) or args.reset:
            if exists(directory):
                rmtree(directory, ignore_errors = True)
            mkdir(directory)
    directory_creator(directory = args.output_dir) # create output directory
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
    stems_by_partition = [
        load_pickle(filepath = f"{args.partitions_dir}/train.pkl"), # read in training file
        load_pickle(filepath = f"{args.partitions_dir}/valid.pkl"), # read in validation file
        load_pickle(filepath = f"{args.partitions_dir}/test.pkl"), # read in testing file
    ]
    all_stems = sum(stems_by_partition, []) # list of all path stems
    random.shuffle(all_stems)
    convert_stems_to_absolute_paths = lambda directory: list(map(lambda stem: f"{directory}/{stem}." + (utils.TENSOR_FILETYPE if directory in (data_dir, prebottleneck_data_dir) else utils.PICKLE_FILETYPE), all_stems)) # helper function
    n_stems = len(all_stems) # number of stems
    del load_pickle # free up memory

    # get ranges of each partition in all_stems
    partition_ranges_in_all_stems = np.cumsum(list(map(len, stems_by_partition))).tolist() # cumulative sum
    partition_ranges_in_all_stems = {partition: (start, end) for partition, start, end in zip(utils.ALL_PARTITIONS, [0] + partition_ranges_in_all_stems[:-1], partition_ranges_in_all_stems)}

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
    logging.info(f"{n_stems} songs (" + ", ".join([f"{partition.lower().title()}: {partition_range[-1] - partition_range[0]}" for partition, partition_range in partition_ranges_in_all_stems.items()]) + ").") # number of songs per partition

    # free up memory
    del get_song_length

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
    del convert_stems_to_absolute_paths, n_stems, save_path, save_paths, song_lengths

    ##################################################


    # OUTPUT TRAINING, VALIDATION, AND TESTING PATHS
    ##################################################

    # output paths
    output_path_by_partition = {partition: f"{splits_dir}/{partition}.txt" for partition in partition_ranges_in_all_stems.keys()}

    # write to file
    if not all(map(exists, output_path_by_partition.values())) or args.reset:
        logging.info(utils.MAJOR_SEPARATOR_LINE)
        for partition in output_path_by_partition.keys():
            output_path = output_path_by_partition[partition]
            start, end = partition_ranges_in_all_stems[partition] # get start and end indicies of partition
            data = all_stems[start:end] # extract correct range for the partition
            data = list(map(lambda stem: f"{stem}.{utils.TENSOR_FILETYPE}", data)) # convert from stems to basenames
            utils.save_txt(filepath = output_path, data = data)
            logging.info(f"Wrote {partition} partition to {output_path}.")

    # free up memory
    del all_stems

    ##################################################


    # TEST DATASET
    ##################################################

    # test dataset with different pooling values
    for use_prebottleneck_latents in (False, True):
        for pool in (False, True):

            # print separator line
            logging.info(utils.MAJOR_SEPARATOR_LINE)
            logging.info("Using " + ("prebottleneck " if use_prebottleneck_latents else "") + "latents.")
            logging.info("Pooling is " + ("ON" if pool else "OFF") + ".")

            # create dataset
            dataset = get_dataset(
                task = args.task,
                directory = f"{utils.DIR_BY_TASK[args.task]}/{utils.DATA_DIR_NAME}/{utils.PREBOTTLENECK_DATA_SUBDIR_NAME if use_prebottleneck_latents else utils.DATA_SUBDIR_NAME}",
                paths = output_path_by_partition[utils.VALID_PARTITION_NAME],
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
                    del inputs, labels, mask, paths # free up memory

            # print how many batches were loaded
            logging.info(f"Successfully loaded {n_batches} batches ({n_samples} samples).")

            # free up memory
            del dataset, data_loader, n_batches, n_samples
    
    # free up memory
    del output_path_by_partition

    ##################################################

##################################################
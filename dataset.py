# README
# Phillip Long
# March 4, 2025

# Prepare data for training a model.

# python /home/pnlong/jingyue_latents/dataset.py

# IMPORTS
##################################################

from typing import List
from os.path import exists, dirname, realpath, basename
from os import mkdir, readlink
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


# HELPER FUNCTIONS
##################################################

def pad(seqs: List[torch.Tensor], length: int) -> torch.Tensor:
    """Front-zero-pad a given list of sequences to the given length."""

    # pad sequences
    for i, seq in enumerate(seqs):
        if len(seq) < length: # sequence is shorter than length
            pad = length - len(seq)
            pad = (0, 0, pad, 0) if utils.FRONT_PAD else (0, 0, 0, pad)
            seq = torch.nn.functional.pad(input = seq, pad = pad, mode = "constant", value = 0)
        else: # sequence is longer than length
            seq = seq[:length]
        seqs[i] = seq # update value in sequences

    # stack sequences
    seqs = torch.stack(tensors = seqs, dim = 0)

    # return padded sequences as single matrix
    return seqs

def get_mask(seqs: List[torch.Tensor], length: int) -> torch.Tensor:
    """Generate a mask for the given list of sequences with the given length."""

    # create empty mask
    mask = torch.zeros(size = (len(seqs), length), dtype = torch.bool)

    # generate masks for each sequence
    for i in range(len(seqs)):
        if utils.FRONT_PAD: # front pad
            mask[i, -len(seqs[i]):] = True
        else: # end pad
            mask[i, :len(seqs[i])] = True

    # return the mask
    return mask

##################################################


# CUSTOM DATASET CLASS
##################################################

class CustomDataset(torch.utils.data.Dataset):

    # intializer
    def __init__(
        self,
        directory: str, # directory containing relevant paths
        paths: str, # path to file with relevant path basenames
        mappings_path: str, # path to JSON that maps basenames to labels
        max_seq_len: int = 1, # maximum sequence length (in bars)
        pool: bool = False, # whether to pool (average) the num_bar dimension
        embedding_dim: int = utils.LATENT_EMBEDDING_DIM, # length of last dimension of data
    ):
        super().__init__()
        self.directory = directory
        self.paths = utils.load_txt(filepath = paths)
        self.mappings = utils.load_json(filepath = mappings_path) # load in mappings
        self.max_seq_len = max_seq_len
        self.pool = pool
        self.embedding_dim = embedding_dim

    # length attribute
    def __len__(self) -> int:
        return len(self.paths)

    # obtain an item
    def __getitem__(self, index: int) -> dict:

        # get the name
        base = self.paths[index]
        path = f"{self.directory}/{base}"

        # get label from path
        label = torch.tensor(self.mappings[base]) # extract label from base, and store as tensor

        # load in sequence as tensor, extract tokens if there are any
        seq = torch.load(f = path, weights_only = True)
        if len(seq.shape) == 1:
            seq, tokens = seq[:self.embedding_dim], seq[self.embedding_dim:]
        else: # len(seq.shape) == 2
            seq, tokens = seq[:, :self.embedding_dim], seq[:, self.embedding_dim:]

        # pool if necessary
        if self.pool and len(seq.shape) == 2:
            seq = torch.mean(input = seq, dim = 0) # mean pool
            tokens = tokens[0] # take tokens from first bar as pooling, really only falters when there are multiple bars per song for a beat- or note-level classification task
        elif not self.pool and len(seq.shape) == 1:
            seq = seq.unsqueeze(dim = 0) # add a num_bar dimension if there isn't one
            tokens = tokens.unsqueeze(dim = 0) # add a num_bar dimension if there isnt' one

        # wrangle data types
        seq = seq.to(utils.DATA_TYPE)
        tokens = tokens.to(utils.TOKEN_TYPE) if tokens.shape[-1] > 0 else None

        # return dictionary of sequence, label, and path
        return {
            "seq": seq,
            "label": label,
            "path": path,
            "max_seq_len": self.max_seq_len,
            "token": tokens,
        }
    
    # collate method
    @classmethod
    def collate(cls, batch: List[dict]) -> dict:

        # aggregate list of sequences
        seqs, labels, paths, tokens = utils.transpose(l = [(sample["seq"], sample["label"], sample["path"], sample["token"]) for sample in batch])
        max_seq_len = batch[0]["max_seq_len"] # max_seq_len is constant across all samples, so just grab the first one
        has_tokens = tokens[0] is not None

        # if pooling was applied
        if len(seqs[0].shape) == 1:
            seqs = torch.stack(tensors = seqs, dim = 0)
            mask = torch.ones(size = [len(labels)])
            tokens = torch.stack(tensors = tokens, dim = 0) if has_tokens else None

        # if pooling was not applied
        else:
            seqs = pad(seqs = seqs, length = max_seq_len)
            mask = get_mask(seqs = seqs, length = max_seq_len)
            tokens = pad(seqs = tokens, length = max_seq_len) if has_tokens else None

        # ensure tokens is the correct data type
        if has_tokens:
            tokens = tokens.to(utils.TOKEN_TYPE)
        
        # labels
        labels = torch.stack(tensors = labels, dim = 0)
        if len(labels.shape) > 1: # flatten labels if it is multi-dimensional
            labels = labels.flatten()

        # return dictionary of sequences, labels, masks, and paths
        return {
            "seq": seqs.to(utils.DATA_TYPE),
            "label": labels.to(utils.LABEL_TYPE),
            "mask": mask.to(torch.bool),
            "path": paths,
            "token": tokens,
        }

##################################################


# GET THE CORRECT DATASET GIVEN SOME ARGUMENTS
##################################################

def get_dataset(
        task: str, # relevant task
        directory: str, # directory containing relevant paths
        paths: str, # path to file with relevant path basenames
        mappings_path: str, # path to JSON file that maps basenames to label(s)
        pool: bool = False, # whether to pool (average) the num_bar dimension
        use_prebottleneck_latents: bool = False, # whether or not prebottleneck latents are being used
    ) -> torch.utils.data.Dataset:
    """
    Helper function that returns the correct dataset object
    given some arguments.
    """

    # get the correct label extractor
    max_seq_len = utils.MAX_SEQ_LEN_BY_TASK[task]
    embedding_dim = utils.PREBOTTLENECK_LATENT_EMBEDDING_DIM if use_prebottleneck_latents else utils.LATENT_EMBEDDING_DIM
    
    # return dataset with relevant arguments
    return CustomDataset(
        directory = directory,
        paths = paths,
        mappings_path = mappings_path,
        max_seq_len = max_seq_len,
        pool = pool,
        embedding_dim = embedding_dim,
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
    args.data_dir = readlink(f"{jingyue_data_symlinks_dir}/{utils.JINGYUE_DATA_SYMLINK_NAME}")
    args.prebottleneck_data_dir = f"{jingyue_data_symlinks_dir}/{utils.JINGYUE_PREBOTTLENECK_DATA_SYMLINK_NAME}"
    if exists(args.prebottleneck_data_dir): # if the symlink exists, read it
        args.prebottleneck_data_dir = readlink(args.prebottleneck_data_dir)
    args.partitions_dir = readlink(f"{jingyue_data_symlinks_dir}/{utils.JINGYUE_SPLITS_SYMLINK_NAME}")
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
    if not exists(args.output_dir):
        mkdir(args.output_dir)
    base_data_dir = f"{args.output_dir}/{utils.DATA_DIR_NAME}"
    utils.directory_creator(directory = base_data_dir, reset = args.reset)
    splits_dir = f"{base_data_dir}/{utils.SPLITS_SUBDIR_NAME}"
    utils.directory_creator(directory = splits_dir, reset = args.reset)
    mappings_dir = f"{base_data_dir}/{utils.MAPPINGS_SUBDIR_NAME}"
    utils.directory_creator(directory = mappings_dir, reset = args.reset)
    data_dir = f"{base_data_dir}/{utils.DATA_SUBDIR_NAME}"
    utils.directory_creator(directory = data_dir, reset = args.reset)
    prebottleneck_data_dir = f"{base_data_dir}/{utils.PREBOTTLENECK_DATA_SUBDIR_NAME}"
    generate_prebottleneck_data_dir = exists(args.prebottleneck_data_dir)
    if generate_prebottleneck_data_dir: 
        utils.directory_creator(directory = prebottleneck_data_dir, reset = args.reset)

    ##################################################


    # READ IN TRAINING, VALIDATION, AND TEST PATHS
    ##################################################

    # read in paths
    get_stem = lambda filepath: ".".join(basename(filepath).split(".")[:-1])
    load_pickle = lambda filepath: list(map(get_stem, utils.load_pickle(filepath = filepath))) # load pickle, removing filetype while at it
    stems_by_partition = [
        load_pickle(filepath = f"{args.partitions_dir}/train.pkl"), # read in training file
        load_pickle(filepath = f"{args.partitions_dir}/valid.pkl"), # read in validation file
        load_pickle(filepath = f"{args.partitions_dir}/test.pkl"), # read in testing file
    ]
    all_stems = sum(stems_by_partition, []) # list of all path stems
    random.shuffle(all_stems)
    n_stems = len(all_stems) # number of stems
    del load_pickle # free up memory

    # get ranges of each partition in all_stems
    partition_ranges_in_all_stems = np.cumsum(list(map(len, stems_by_partition))).tolist() # cumulative sum
    partition_ranges_in_all_stems = {partition: (start, end) for partition, start, end in zip(utils.ALL_PARTITIONS, [0] + partition_ranges_in_all_stems[:-1], partition_ranges_in_all_stems)}

    ##################################################


    # OUTPUT LONGEST SEQUENCE LENGTH
    ##################################################

    # helper function to get the length (in number of bars) of the song at the given stem
    def get_song_length(stem: str) -> int:
        """Return the length (in number of bars) of the song at the given stem."""
        path = f"{args.data_dir}/{stem}.{utils.PICKLE_FILETYPE}" # calculate absolute filepath
        seq = utils.load_pickle(filepath = path) # load in pickle file
        return len(seq) # seq.shape[0], the number of bars in the sequence

    # use multiprocessing
    with multiprocessing.Pool(processes = args.jobs) as pool:
        song_lengths = list(pool.map(
            func = get_song_length,
            iterable = tqdm(
                iterable = all_stems,
                desc = "Retrieving song lengths (in bars)",
                total = n_stems),
            chunksize = utils.CHUNK_SIZE,
        ))

    # print longest sequence length
    logging.info(f"Longest song length (in number of bars): {max(song_lengths)}")
    logging.info(f"{n_stems} songs (" + ", ".join([f"{partition.lower().title()}: {partition_range[-1] - partition_range[0]}" for partition, partition_range in partition_ranges_in_all_stems.items()]) + ").") # number of songs per partition

    # free up memory
    del get_song_length, song_lengths

    ##################################################


    # CREATE PATHS
    ##################################################

    # output paths
    output_path_by_partition = {partition: f"{splits_dir}/{partition}.txt" for partition in partition_ranges_in_all_stems.keys()}

    # write to file
    if not all(map(exists, output_path_by_partition.values())) or args.reset:

        # separator line
        logging.info(utils.MAJOR_SEPARATOR_LINE)

        # TASK-SPECIFIC FUNCTIONS
        ##################################################

        # given a stem, do whatever with it
        match args.task:

            # EMOTION
            ##################################################

            case utils.EMOTION_DIR_NAME:
                def save_stem(stem: str, input_dir: str, output_dir: str) -> List[dict]: # helper function to save a torch object given the input path to the output path
                    """Save the stem as a tensor."""

                    # save tensor
                    path_output = f"{output_dir}/{stem}.{utils.TENSOR_FILETYPE}"
                    if not exists(path_output) or args.reset:
                        seq = utils.load_pickle(filepath = f"{input_dir}/{stem}.{utils.PICKLE_FILETYPE}") # load in pickle file
                        seq = torch.from_numpy(seq).to(utils.DATA_TYPE) # convert from numpy array to torch tensor of desired type
                        torch.save(obj = seq, f = path_output) # save sequence as torch pickle object
                    
                    # return list of dictionary(s) mapping bases to labels
                    base = basename(path_output)
                    label = utils.EMOTION_ID_TO_INDEX[base.split("_")[0]]
                    mapping = [{base: label}]
                    return mapping
            
            ##################################################

            # CHORD
            ##################################################

            case utils.CHORD_DIR_NAME:
                def save_stem(stem: str, input_dir: str, output_dir: str) -> List[dict]: # helper function to save a torch object given the input path to the output path
                    """Save the stem as a tensor."""

                    # get paths, setup
                    seq = utils.load_pickle(filepath = f"{input_dir}/{stem}.{utils.PICKLE_FILETYPE}") # load in pickle file
                    seq = torch.from_numpy(seq).to(utils.DATA_TYPE) # convert from numpy array to torch tensor of desired type

                    # save tensors
                    get_base_from_index = lambda i: f"{stem}_{i}.{utils.TENSOR_FILETYPE}"
                    for i in range(len(seq)):
                        path_output = f"{output_dir}/{get_base_from_index(i = i)}"
                        if not exists(path_output) or args.reset:
                            torch.save(obj = seq[i], f = path_output) # save sequence as torch pickle object
                    del seq # free up memory

                    # iterate through different chord mappings
                    mapping = []
                    for mapping_dir_name, indexer in zip(utils.JINGYUE_CHORD_MAPPING_DIR_NAMES, utils.CHORD_INDEXER):
                        chords = utils.load_pickle(filepath = f"{dirname(input_dir)}/{mapping_dir_name}/{stem}.{utils.PICKLE_FILETYPE}") # load in chords
                        chords = list(map(lambda chord: indexer[chord], chords)) # convert chords to indicies
                        chords = [chords[i:(i + utils.N_OUTPUTS_PER_INPUT_BAR_BY_TASK[utils.CHORD_DIR_NAME])] for i in range(0, len(chords), utils.N_OUTPUTS_PER_INPUT_BAR_BY_TASK[utils.CHORD_DIR_NAME])] # reshape chords so there are four chords per bar
                        mapping.append({get_base_from_index(i = i): chords[i] for i in range(len(chords))}) # add mappings to list of mappings                

                    # return list of dictionary(s) mapping bases to labels
                    return mapping

            ##################################################

            
            # MELODY
            ##################################################

            case utils.MELODY_DIR_NAME:
                def save_stem(stem: str, input_dir: str, output_dir: str) -> List[dict]: # helper function to save a torch object given the input path to the output path
                    """Save the stem as a tensor."""

                    # get paths, setup
                    seq = utils.load_pickle(filepath = f"{input_dir}/{stem}.{utils.PICKLE_FILETYPE}") # load in pickle file
                    seq = torch.from_numpy(seq).to(utils.DATA_TYPE) # convert from numpy array to torch tensor of desired type
                    bar_positions, events = utils.load_pickle(filepath = f"{dirname(input_dir)}/{utils.JINGYUE_MELODY_MAPPING_DIR_NAME}/{stem}.{utils.PICKLE_FILETYPE}") # remi events
                    mapping = dict() # output mapping

                    # deal with uneven lengths
                    if len(bar_positions) <= len(seq):
                        seq = seq[:len(bar_positions), :]
                        bar_positions.append(len(events) - 1) # add eos token to bar_positions, since it is implicitly a bar
                    else: # len(bar_positions) > len(seq)
                        bar_positions = bar_positions[:(len(seq) + 1)]

                    # save tensors
                    idx = 0
                    for i in range(len(seq)):
                        notes_at_bar = events[(bar_positions[i] + 1):(bar_positions[i + 1] - 1)] # get events in bar
                        notes_at_bar = [notes_at_bar[j:(j + len(utils.MELODY_REMI_WORD_TYPES) + 1)] for j in range(0, len(notes_at_bar), len(utils.MELODY_REMI_WORD_TYPES) + 1)] # reshape so each row is a note
                        for note in notes_at_bar:
                            note = {event["name"]: event["value"] for event in note} # make note into a dictionary keyed by remi word type
                            base_output = f"{stem}_{idx}.{utils.TENSOR_FILETYPE}"
                            mapping[base_output] = utils.MELODY_ID_TO_INDEX[note[utils.MELODY_CLASS_WORD_TYPE]] # make note of mapping
                            path_output = f"{output_dir}/{base_output}"
                            if not exists(path_output) or args.reset:
                                remi_tokens = torch.tensor(list(map(lambda word_type: utils.MELODY_REMI_VOCABULARY[f"{word_type}_{note[word_type]}"], utils.MELODY_REMI_WORD_TYPES)), dtype = utils.DATA_TYPE) # convert remi tokens to indicies
                                torch.save(obj = torch.cat(tensors = (seq[i], remi_tokens), dim = 0), f = path_output) # save sequence as torch pickle object
                            idx += 1 # increment idx
                    del seq, bar_positions, events, idx # free up memory

                    # return list of dictionary(s) mapping bases to labels
                    mapping = [mapping]
                    return mapping
                
            ##################################################
                
            # invalid task
            case _:
                raise RuntimeError(utils.INVALID_TASK_ERROR(task = args.task))

        ##################################################


        # SAVE TORCH TENSORS
        ##################################################

        # use multiprocessing
        def save_paths(input_dir: str, output_dir: str, desc: str) -> List[List[dict]]:
            """Helper function to use multiprocessing to save paths as torch tensors."""
            with multiprocessing.Pool(processes = args.jobs) as pool:
                results = utils.transpose(l = list(pool.starmap(
                    func = save_stem,
                    iterable = tqdm(
                        iterable = zip(
                            all_stems,
                            utils.rep(x = input_dir, times = n_stems),
                            utils.rep(x = output_dir, times = n_stems),
                        ),
                        desc = desc,
                        total = n_stems),
                    chunksize = utils.CHUNK_SIZE,
                )))
            return results

        # save paths
        results = save_paths(input_dir = args.data_dir, output_dir = data_dir, desc = "Extracting tensors")
        if generate_prebottleneck_data_dir:
            _ = save_paths(input_dir = args.prebottleneck_data_dir, output_dir = prebottleneck_data_dir, desc = "Extracting prebottleneck tensors")
        
        # generating mapping json files
        bases_by_stem = [list(mapping.keys()) for mapping in results[0]] # base(s) for each stem in all_stems
        for mappings, mapping_name in zip(results, utils.MAPPING_NAMES_BY_TASK[args.task]):
            utils.save_json(
                filepath = f"{mappings_dir}/{mapping_name}.{utils.JSON_FILETYPE}",
                data = {stem: label for mapping in mappings for stem, label in mapping.items()},
            )

        # free up memory
        del n_stems, save_stem, save_paths, results

        ##################################################


        # OUTPUT PARTITION PATHS
        ##################################################

        # write partitions to file
        for partition in output_path_by_partition.keys():
            output_path = output_path_by_partition[partition]
            start, end = partition_ranges_in_all_stems[partition] # get start and end indicies of partition
            data = sum(bases_by_stem[start:end], []) # extract correct range for the partition
            utils.save_txt(filepath = output_path, data = data)
            logging.info(f"Wrote {partition} partition to {output_path}.")

        # free up memory
        del bases_by_stem

        ##################################################

    # free up memory
    del all_stems

    ##################################################


    # TEST DATASET
    ##################################################

    # test dataset with different pooling values
    for use_prebottleneck_latents in [False] + ([True] if generate_prebottleneck_data_dir else []):
        for pool in [False, True]:

            # print separator line
            logging.info(utils.MAJOR_SEPARATOR_LINE)

            # create dataset
            dataset = get_dataset(
                task = args.task,
                directory = f"{utils.DIR_BY_TASK[args.task]}/{utils.DATA_DIR_NAME}/{utils.PREBOTTLENECK_DATA_SUBDIR_NAME if use_prebottleneck_latents else utils.DATA_SUBDIR_NAME}",
                paths = output_path_by_partition[utils.VALID_PARTITION_NAME],
                mappings_path = f"{mappings_dir}/{utils.MAPPING_NAMES_BY_TASK[args.task][0]}.{utils.JSON_FILETYPE}",
                pool = pool,
                use_prebottleneck_latents = use_prebottleneck_latents,
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
            desc = "Using " + ("prebottleneck " if use_prebottleneck_latents else "") + "latents. Pooling is " + ("ON" if pool else "OFF") + "."
            example = ""
            for i, batch in tqdm(iterable = enumerate(data_loader), desc = "Iterating through data loader", total = len(data_loader)):

                # update tracker variables
                n_batches += 1
                n_samples += len(batch["seq"])

                # print example on first batch
                if i == 0:
                    example += "Example on the validation partition:\n"
                    inputs, labels, mask, paths, tokens = batch["seq"], batch["label"], batch["mask"], batch["path"], batch["token"]
                    example += f"  - Shape of data: {tuple(inputs.shape)}\n"
                    # example += f"  - Data: {inputs}\n"
                    example += f"  - Shape of labels: {tuple(labels.shape)}\n"
                    # example += f"  - Labels: {labels}\n"
                    example += f"  - Shape of mask: {tuple(mask.shape)}\n"
                    if tokens is not None:
                        example += f"  - Shape of tokens: {tuple(tokens.shape)}\n"
                    example += f"  - Shape of paths: {len(paths)}"
                    del inputs, labels, mask, paths, tokens # free up memory

            # print example and how many batches were loaded
            logging.info(example)
            logging.info(f"Successfully loaded {n_batches} batches ({n_samples} samples).")

            # free up memory
            del dataset, data_loader, n_batches, n_samples
    
    # free up memory
    del output_path_by_partition

    ##################################################

##################################################
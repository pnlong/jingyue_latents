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

def pad(seqs: List[torch.Tensor], length: int, sublength: int = None) -> torch.Tensor:
    """Front-zero-pad a given list of sequences to the given length."""

    # pad sequences
    for i, seq in enumerate(seqs):

        # pad along the first dimension
        if len(seq) < length: # sequence is shorter than length
            pad = length - len(seq)
            pad = (0, 0, pad, 0) if utils.FRONT_PAD else (0, 0, 0, pad)
            if len(seq.shape) == 1: # remove 2nd dimension if seq is 1-dimensional
                pad = pad[2:]
            seq = torch.nn.functional.pad(input = seq, pad = pad, mode = "constant", value = 0)
        else: # sequence is longer than length
            seq = seq[:length]

        # pad along the second dimension if specified
        if sublength is not None and len(seq.shape) > 1:
            if seq.shape[-1] < sublength: # sequence is shorter than sublength
                pad = sublength - seq.shape[-1]
                pad = (pad, 0) if utils.FRONT_PAD else (0, pad)
                seq = torch.nn.functional.pad(input = seq, pad = pad, mode = "constant", value = 0)
            else: # sequence is longer than sublength
                seq = seq[:, :sublength]

        # update value in sequences
        seqs[i] = seq

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
        tokens_max_seq_len: int = 0, # maximum sequence length of tokens, defaults to 0 (no tokens)
        pool: bool = False, # whether to pool (average) the num_bar dimension
        mask_tokens: bool = False, # whether the mask should be for tokens or for the latents
    ):
        super().__init__()
        self.directory = directory
        self.paths = utils.load_txt(filepath = paths)
        self.mappings = utils.load_json(filepath = mappings_path) # load in mappings
        self.max_seq_len = max_seq_len
        self.tokens_max_seq_len = tokens_max_seq_len
        self.pool = pool
        self.mask_tokens = mask_tokens

    # length attribute
    def __len__(self) -> int:
        return len(self.paths)

    # obtain an item
    def __getitem__(self, index: int) -> dict:

        # get the name
        base = self.paths[index]
        path = f"{self.directory}/{base}"
        tokens_path = f"{self.directory}/{utils.TOKEN_SUBDIR_NAME}/{base}"
        bar_positions_path = f"{self.directory}/{utils.BAR_POSITION_SUBDIR_NAME}/{base}"

        # get label from path
        label = torch.tensor(self.mappings[base]) # extract label from base, and store as tensor

        # load in sequence as tensor, extract tokens if there are any
        seq = torch.load(f = path, weights_only = True)
        tokens = torch.load(f = tokens_path, weights_only = True).to(utils.TOKEN_TYPE) if exists(tokens_path) else None
        bar_positions = torch.load(f = bar_positions_path, weights_only = True) if exists(bar_positions_path) else None

        # pool if necessary
        if self.pool and len(seq.shape) == 2:
            seq = torch.mean(input = seq, dim = 0) # mean pool
        elif not self.pool and len(seq.shape) == 1:
            seq = seq.unsqueeze(dim = 0) # add a num_bar dimension if there isn't one
        seq = seq.to(utils.DATA_TYPE) # wrangle data type 

        # return dictionary of sequence, label, and path
        return {
            "seq": seq,
            "label": label,
            "path": path,
            "token": tokens,
            "bar_position": bar_positions,
            "max_seq_len": self.max_seq_len,
            "tokens_max_seq_len": self.tokens_max_seq_len,
            "mask_tokens": self.mask_tokens,
        }
    
    # collate method
    @classmethod
    def collate(cls, batch: List[dict]) -> dict:

        # aggregate list of sequences
        seqs, labels, paths, tokens, bar_positions = utils.transpose(l = [(sample["seq"], sample["label"], sample["path"], sample["token"], sample["bar_position"]) for sample in batch])
        max_seq_len, tokens_max_seq_len, mask_tokens = batch[0]["max_seq_len"], batch[0]["tokens_max_seq_len"], batch[0]["mask_tokens"] # constant across all samples, so just grab the first one
        has_tokens, has_bar_positions = (tokens[0] is not None), (bar_positions[0] is not None)

        # deal with sequences
        if len(seqs[0].shape) == 1: # if pooling was applied
            mask = torch.ones(size = [len(labels)])
            seqs = torch.stack(tensors = seqs, dim = 0)
        else: # if pooling was not applied
            mask = get_mask(seqs = seqs, length = max_seq_len)
            seqs = pad(seqs = seqs, length = max_seq_len)

        # deal with tokens
        if has_tokens:
            if mask_tokens:
                mask = get_mask(seqs = tokens, length = tokens_max_seq_len if (len(tokens[0].shape) == 1) else max_seq_len)
            if len(tokens[0].shape) == 1: # if pooling was applied
                tokens = pad(seqs = tokens, length = tokens_max_seq_len)
            else: # if pooling was not applied
                tokens = pad(seqs = tokens, length = max_seq_len, sublength = tokens_max_seq_len)
            tokens = tokens.to(utils.TOKEN_TYPE) # ensure tokens is the correct data type
        else:
            tokens = None

        # bar positions is constant size regardless of pooling
        bar_positions = torch.stack(tensors = bar_positions, dim = 0).to(utils.BAR_POSITION_TYPE) if has_bar_positions else None
        
        # labels
        if has_tokens: # all labels are the same shape
            labels = pad(seqs = labels, length = tokens_max_seq_len)
        else: # labels are of different shape
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
            "bar_position": bar_positions,
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
    ) -> torch.utils.data.Dataset:
    """
    Helper function that returns the correct dataset object
    given some arguments.
    """

    # get the correct label extractor
    max_seq_len = utils.MAX_SEQ_LEN_BY_TASK[task]
    tokens_max_seq_len = utils.TOKEN_MAX_SEQ_LEN_BY_TASK[task]
    mask_tokens = (task == utils.MELODY_TRANSFORMER_DIR_NAME)
    
    # return dataset with relevant arguments
    return CustomDataset(
        directory = directory,
        paths = paths,
        mappings_path = mappings_path,
        max_seq_len = max_seq_len,
        tokens_max_seq_len = tokens_max_seq_len,
        pool = pool,
        mask_tokens = mask_tokens,
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
    parser.add_argument("-rp", "--randomize_partitions", action = "store_true", help = "Whether or not to randomize files before partitioning them")
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
    create_tokens_dir = (utils.VOCABULARY_SIZE_BY_TASK[args.task] > 0)
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
    if create_tokens_dir:
        utils.directory_creator(directory = f"{data_dir}/{utils.TOKEN_SUBDIR_NAME}", reset = args.reset)
        utils.directory_creator(directory = f"{data_dir}/{utils.BAR_POSITION_SUBDIR_NAME}", reset = args.reset)
    prebottleneck_data_dir = f"{base_data_dir}/{utils.PREBOTTLENECK_DATA_SUBDIR_NAME}"
    generate_prebottleneck_data_dir = exists(args.prebottleneck_data_dir)
    if generate_prebottleneck_data_dir: 
        utils.directory_creator(directory = prebottleneck_data_dir, reset = args.reset)
        if create_tokens_dir:
            utils.directory_creator(directory = f"{prebottleneck_data_dir}/{utils.TOKEN_SUBDIR_NAME}", reset = args.reset)
            utils.directory_creator(directory = f"{prebottleneck_data_dir}/{utils.BAR_POSITION_SUBDIR_NAME}", reset = args.reset)

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
    if args.randomize_partitions:
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
                                torch.save(obj = seq[i], f = path_output) # save sequence as torch pickle object
                            tokens_path_output = f"{output_dir}/{utils.TOKEN_SUBDIR_NAME}/{base_output}"
                            if not exists(tokens_path_output) or args.reset:
                                torch.save(obj = torch.tensor(list(map(lambda word_type: utils.MELODY_REMI_VOCABULARY[f"{word_type}_{note[word_type]}"], utils.MELODY_REMI_WORD_TYPES)), dtype = utils.TOKEN_TYPE), f = tokens_path_output) # save tokens as torch pickle object
                            idx += 1 # increment idx
                            del base_output, path_output, tokens_path_output # free up memory
                    del seq, bar_positions, events, idx # free up memory

                    # return list of dictionary(s) mapping bases to labels
                    mapping = [mapping]
                    return mapping
                
            ##################################################


            # MELODY TRANSFORMER
            ##################################################

            case utils.MELODY_TRANSFORMER_DIR_NAME:
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
                    for i in range(0, len(seq), utils.MELODY_TRANSFORMER_CLIP_LENGTH):
                        base_output = f"{stem}_{idx}.{utils.TENSOR_FILETYPE}"
                        start, end = i, i + utils.MELODY_TRANSFORMER_CLIP_LENGTH # start and end index
                        if end > len(seq): # ignore clips at the end of sequence that are less than 16 bars 
                            break
                        path_output = f"{output_dir}/{base_output}"
                        if not exists(path_output) or args.reset: # save path output if needed
                            torch.save(obj = seq[start:end], f = path_output) # save sequence as torch pickle object
                        events_in_clip = events[bar_positions[start]:bar_positions[end]]# get events in clip
                        events_in_clip = [{"name": utils.MELODY_TRANSFORMER_BOS_TOKEN, "value": None}] + events_in_clip + [{"name": utils.MELODY_TRANSFORMER_EOS_TOKEN, "value": None}] # add beginning and end of song tokens
                        bar_positions_in_clip = list(map(lambda bar_position: bar_position - bar_positions[start] + 1, bar_positions[start:end])) # get the bar positions
                        bar_positions_in_clip[0] = 0 # set first bar position to 0
                        tokens_path_output = f"{output_dir}/{utils.TOKEN_SUBDIR_NAME}/{base_output}"
                        if not exists(tokens_path_output) or args.reset: # save the tokens if needed
                            torch.save(obj = torch.tensor(list(map(lambda event: utils.MELODY_TRANSFORMER_REMI_VOCABULARY[event["name"] + "_" + str(event["value"])], events_in_clip)), dtype = utils.TOKEN_TYPE), f = tokens_path_output) # save tokens as torch pickle object
                        bar_positions_path_output = f"{output_dir}/{utils.BAR_POSITION_SUBDIR_NAME}/{base_output}"
                        if not exists(bar_positions_path_output) or args.reset: # save bar positions if needed
                            torch.save(obj = torch.tensor(bar_positions_in_clip + [len(events_in_clip)], dtype = utils.TOKEN_TYPE), f = bar_positions_path_output) # save bar positions as torch pickle object
                        mapping[base_output] = [event["value"] if event["name"] == utils.MELODY_CLASS_WORD_TYPE else 0 for event in events_in_clip] # make note of mapping
                        idx += 1 # increment idx
                        del base_output, start, end, path_output, events_in_clip, bar_positions_in_clip, tokens_path_output, bar_positions_path_output # free up memory
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
        for pool in [False] + ([True] if args.task != utils.MELODY_TRANSFORMER_DIR_NAME else []):

            # print separator line
            logging.info(utils.MAJOR_SEPARATOR_LINE)

            # create dataset
            dataset = get_dataset(
                task = args.task,
                directory = f"{utils.DIR_BY_TASK[args.task]}/{utils.DATA_DIR_NAME}/{utils.PREBOTTLENECK_DATA_SUBDIR_NAME if use_prebottleneck_latents else utils.DATA_SUBDIR_NAME}",
                paths = output_path_by_partition[utils.VALID_PARTITION_NAME],
                mappings_path = f"{mappings_dir}/{utils.MAPPING_NAMES_BY_TASK[args.task][0]}.{utils.JSON_FILETYPE}",
                pool = pool,
            )
            data_loader = torch.utils.data.DataLoader(
                dataset = dataset,
                batch_size = utils.BATCH_SIZE,
                shuffle = True,
                num_workers = args.jobs,
                collate_fn = dataset.collate,
            )

            # iterate over the data loader
            n_batches = 0
            n_samples = 0
            logging.info("Using " + ("prebottleneck " if use_prebottleneck_latents else "") + "latents. Pooling is " + ("ON" if pool else "OFF") + ".")
            example = ""
            for i, batch in tqdm(iterable = enumerate(data_loader), desc = "Iterating through data loader", total = len(data_loader)):

                # update tracker variables
                n_batches += 1
                n_samples += len(batch["seq"])

                # print example on first batch
                if i == 0:
                    example += "Example on the validation partition:\n"
                    inputs, labels, mask, paths, tokens, bar_positions = batch["seq"], batch["label"], batch["mask"], batch["path"], batch["token"], batch["bar_position"]
                    example += f"  - Shape of data: {tuple(inputs.shape)}\n"
                    # example += f"  - Data: {inputs}\n"
                    example += f"  - Shape of labels: {tuple(labels.shape)}\n"
                    # example += f"  - Labels: {labels}\n"
                    example += f"  - Shape of mask: {tuple(mask.shape)}\n"
                    if tokens is not None:
                        example += f"  - Shape of tokens: {tuple(tokens.shape)}\n"
                    if bar_positions is not None:
                        example += f"  - Shape of bar positions: {tuple(bar_positions.shape)}\n"
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
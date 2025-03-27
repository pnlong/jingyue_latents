# README
# Phillip Long
# March 13, 2025

# Generate any baselines.

# python /home/pnlong/jingyue_latents/baseline.py

# IMPORTS
##################################################

from typing import Tuple
from os.path import exists, dirname, realpath, basename
from os import mkdir, listdir
import argparse
import logging
import multiprocessing
from tqdm import tqdm

from os.path import dirname, realpath
import sys
sys.path.insert(0, dirname(realpath(__file__)))

import utils

##################################################


# ARGUMENTS
##################################################

def parse_args(args = None, namespace = None):
    """Parse command-line arguments."""

    # create argument parser
    parser = argparse.ArgumentParser(prog = "Baseline", description = "Generate baseline if necessary.")
    parser.add_argument("-t", "--task", required = True, choices = utils.ALL_TASKS, type = str, help = "Name of task")
    parser.add_argument("-r", "--reset", action = "store_true", help = "Whether or not to recreate files")
    parser.add_argument("-j", "--jobs", default = int(multiprocessing.cpu_count() / 4), type = int, help = "Number of workers for data loading")
    
    # parse arguments
    args = parser.parse_args(args = args, namespace = namespace)

    # infer other arguments
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

    # create output dir
    if not exists(args.output_dir):
        mkdir(args.output_dir)
    output_dir = f"{args.output_dir}/{utils.BASELINE_DIR_NAME}"
    utils.directory_creator(directory = output_dir, reset = args.reset)

    # print separator line
    logging.info(utils.MAJOR_SEPARATOR_LINE)

    # helper function to truncate a string
    truncate_string = lambda string, n: string + "".join([" " for _ in range(n - len(string))]) if len(string) <= n else string[:n - 3] + "..." # helper function to get strings of constant length

    ##################################################


    # EMOTION
    ##################################################

    if args.task == utils.EMOTION_DIR_NAME:

        # relevant imports
        import pandas as pd

        # copying table from https://arxiv.org/pdf/2107.05223
        data = pd.DataFrame(data = {
            "token": utils.rep(x = "REMI", times = 4) + utils.rep(x = "CP", times = 4) + ["OctupleMIDI"],
            "model": (["CNN (Lee et al., 2020)", "RNN (Z. Lin et al., 2017)", "Our model (score)", "Our model (performance)"] * 2) + ["MusicBERT (Zeng et al., 2021)"],
            utils.ACCURACY_STATISTIC_NAME: [60.00, 53.46, 67.74, 66.18, 60.00, 54.13, 64.22, 70.64, 77.78],
        })

        # log info
        logging.info("Extracted from Table 2 of \"BERT-like Pre-training for Symbolic Piano Music Classification Tasks\" (Chou et al., 2024).")
        logging.info("Paper available at https://arxiv.org/abs/2107.05223.\n")
        logging.info(utils.prettify_table(df = data, column_widths = [15, 35, 10]))
    
    ##################################################


    # CHORD
    ##################################################

    elif args.task == utils.CHORD_DIR_NAME:

        # relevant imports
        from chorder import Dechorder, Chord
        from miditoolkit import MidiFile

        # get list of midi files
        paths = list(map(lambda base: f"{utils.JINGYUE_CHORD_POP909_MIDI_DIR}/{base}", listdir(utils.JINGYUE_CHORD_POP909_MIDI_DIR)))
        baseline_output_dir = f"{output_dir}/{utils.DATA_SUBDIR_NAME}" # output any files created here to this directory
        utils.directory_creator(directory = baseline_output_dir, reset = args.reset)
        get_stem = lambda path: basename(path).split(".")[0]
        get_output_path = lambda stem: f"{baseline_output_dir}/{stem}.{utils.PICKLE_FILETYPE}"

        # helper function to convert chorder Chord to strings
        def convert_chord(chord: Chord) -> str:
            """Given a chorder Chord object, convert it to a string."""
            if chord.root_pc is None:
                return utils.NO_CHORD_SYMBOL
            root = utils.DEFAULT_SCALE[chord.root_pc] # determine root
            quality = chord.quality
            if quality is None:
                quality = utils.DEFAULT_CHORD_QUALITIES[0]
            else:
                quality = utils.DEFAULT_CHORD_QUALITIES[chord.standard_qualities.index(quality)]
            return f"{root}:{quality}"
        
        # function to read each midi file and calculate chords
        def determine_chords(path: str):
            """
            Helper function that, given the path to the MIDI file, 
            outputs a pickle file with the determined chords (4 chords per bar, assuming 4/4 time).
            """
            midi = MidiFile(filename = path, charset = "utf8") # load in file
            chords = Dechorder.dechord(midi_obj = midi) # determine chords
            del midi # free up memory
            chords = list(map(convert_chord, chords)) # wrangle chords so that they align with our vocabulary
            utils.save_pickle(filepath = get_output_path(stem = get_stem(path = path)), data = chords) # save chords as pickle

        # determine chords
        if len(listdir(baseline_output_dir)) < len(paths):
            with multiprocessing.Pool(processes = args.jobs) as pool:
                _ = pool.map(
                    func = determine_chords,
                    iterable = tqdm(
                        iterable = paths,
                        desc = "Determining Chords in POP909",
                        total = len(paths)),
                    chunksize = utils.CHUNK_SIZE,
                )

        # function to read each stem and calculate whether it was correct
        def calculate_accuracy_for_stem(stem: str) -> Tuple[int, int, int]:
            """
            Helper function that, given the stem of a MIDI file in POP909,
            compares the determined chords to the actual annotated chords.
            """
            chords = utils.load_pickle(filepath = get_output_path(stem = stem)) # load in list of determined chords
            output = utils.rep(x = 0, times = len(utils.JINGYUE_CHORD_MAPPING_DIR_NAMES)) + [len(chords)] # output results list
            for i, mapping_dir_name in enumerate(utils.JINGYUE_CHORD_MAPPING_DIR_NAMES):
                actual_chords = utils.load_pickle(filepath = f"{dirname(utils.JINGYUE_CHORD_POP909_MIDI_DIR)}/{mapping_dir_name}/{stem}.{utils.PICKLE_FILETYPE}") # load in actual chords
                output[i] = sum(map(lambda i: (chords[i] if i < len(chords) else utils.NO_CHORD_SYMBOL) == actual_chords[i], range(len(actual_chords))))
            return tuple(output) # return output list
        
        # calculate accuracy
        with multiprocessing.Pool(processes = args.jobs) as pool:
            results = utils.transpose(l = list(pool.map(
                func = calculate_accuracy_for_stem,
                iterable = tqdm(
                    iterable = map(get_stem, paths),
                    desc = "Calculating Baseline Accuracy",
                    total = len(paths)),
                chunksize = utils.CHUNK_SIZE,
            )))
        n_correct_by_mapping_dir_name = list(map(sum, results)) # sum to get total number correct, and total number of chords
        n = n_correct_by_mapping_dir_name.pop(-1) # get total number of chords
        del results # free up memory

        # log info
        for mapping_dir_name, n_correct in zip(utils.JINGYUE_CHORD_MAPPING_DIR_NAMES, n_correct_by_mapping_dir_name):
            logging.info(f"Baseline Accuracy ({mapping_dir_name.split('_')[-1]} chord qualities): {100 * (n_correct / n):.2f}%")

    ##################################################

    
    # MELODY
    ##################################################

    elif args.task == utils.MELODY_DIR_NAME or args.task == utils.MELODY_TRANSFORMER_DIR_NAME:

        # relevant imports
        import pandas as pd

        # copying table from https://arxiv.org/pdf/2107.05223
        data = pd.DataFrame(data = {
            "token": utils.rep(x = "REMI", times = 4) + utils.rep(x = "CP", times = 4) + ["OctupleMIDI"],
            "model": (["CNN (Lee et al., 2020)", "RNN (Z. Lin et al., 2017)", "Our model (score)", "Our model (performance)"] * 2) + ["MusicBERT (Zeng et al., 2021)"],
            utils.ACCURACY_STATISTIC_NAME: [None, 89.96, 90.97, 89.23, None, 88.66, 96.15, 95.83, None],
        })

        # log info
        logging.info("Extracted from Table 2 of \"BERT-like Pre-training for Symbolic Piano Music Classification Tasks\" (Chou et al., 2024).")
        logging.info("Paper available at https://arxiv.org/abs/2107.05223.\n")
        logging.info(utils.prettify_table(df = data, column_widths = [15, 35, 10]))
        
    ##################################################


    # WRAP UP
    ##################################################

    # invalid task
    else:
        raise RuntimeError(utils.INVALID_TASK_ERROR(task = args.task))
    
    # print separator line
    logging.info(utils.MAJOR_SEPARATOR_LINE)
    
    ##################################################

##################################################
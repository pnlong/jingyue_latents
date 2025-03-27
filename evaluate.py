# README
# Phillip Long
# March 4, 2025

# Evaluate a model.

# python /home/pnlong/jingyue_latents/evaluate.py

# IMPORTS
##################################################

import argparse
import logging
import pprint
import sys
from os.path import exists, dirname, basename
from multiprocessing import cpu_count # for calculating num_workers
import pandas as pd
import torch
import torch.utils.data
from tqdm import tqdm
import warnings
warnings.simplefilter(action = "ignore", category = FutureWarning)

from os.path import dirname, realpath
import sys
sys.path.insert(0, dirname(realpath(__file__)))

from dataset import get_dataset
from model import get_model, get_predictions_from_outputs
import utils

##################################################


# ARGUMENTS
##################################################

def parse_args(args = None, namespace = None):
    """Parse command-line arguments."""

    # create argument parser
    parser = argparse.ArgumentParser(prog = "Evaluate", description = "Evaluate all models.")
    parser.add_argument("-t", "--task", required = True, choices = utils.ALL_TASKS, type = str, help = "Name of task")
    # others
    parser.add_argument("-bs", "--batch_size", default = utils.BATCH_SIZE, type = int, help = "Batch size for data loader")
    parser.add_argument("-g", "--gpu", default = -1, type = int, help = "GPU number")
    parser.add_argument("-j", "--jobs", default = int(cpu_count() / 4), type = int, help = "Number of workers for data loading")
    parser.add_argument("-r", "--reset", action = "store_true", help = "Whether or not to re-evaluate")
    
    # parse arguments
    args = parser.parse_args(args = args, namespace = namespace)
    
    # infer other arguments
    input_dir = utils.DIR_BY_TASK[args.task]
    args.paths_test = f"{input_dir}/{utils.DATA_DIR_NAME}/{utils.SPLITS_SUBDIR_NAME}/{utils.TEST_PARTITION_NAME}.txt"
    args.models_list = f"{input_dir}/models.txt"

    # return parsed arguments
    return args

##################################################


# MAIN METHOD
##################################################

if __name__ == "__main__":

    # SET UP
    ##################################################

    # parse the command-line arguments
    args = parse_args()

    # ensure paths test exist
    if not exists(args.paths_test):
        raise ValueError(f"Invalid --paths_test argument: `{args.paths_test}`. File does not exist.")

    # get output directory
    output_dir = dirname(args.models_list)
    task = basename(output_dir)

    # get directories to eval
    models = utils.load_txt(filepath = args.models_list)

    # set up the logger
    logging.basicConfig(level = logging.INFO, format = "%(message)s", handlers = [logging.FileHandler(filename = f"{output_dir}/evaluate.log", mode = "a"), logging.StreamHandler(stream = sys.stdout)])

    # log command called and arguments, save arguments
    logging.info(f"Running command: python {' '.join(sys.argv)}")
    logging.info(f"Using arguments:\n{pprint.pformat(vars(args))}")
    args_output_filepath = f"{output_dir}/evaluate_args.json"
    logging.info(f"Saved arguments to {args_output_filepath}")
    utils.save_args(filepath = args_output_filepath, args = args)
    del args_output_filepath

    # get the specified device
    device = torch.device(f"cuda:{abs(args.gpu)}" if (torch.cuda.is_available() and args.gpu != -1) else "cpu")
    logging.info(f"Using device: {device}")

    # loss function
    loss_fn = torch.nn.CrossEntropyLoss()

    # output files
    evaluation_loss_output_columns = utils.EVALUATION_LOSS_OUTPUT_COLUMNS_BY_TASK[task]
    evaluation_accuracy_output_columns = utils.EVALUATION_ACCURACY_OUTPUT_COLUMNS_BY_TASK[task]
    loss_output_filepath = f"{output_dir}/evaluation.{utils.LOSS_STATISTIC_NAME}.csv"
    if not exists(loss_output_filepath) or args.reset: # if column names need to be written
        previous_loss_output = pd.DataFrame(columns = evaluation_loss_output_columns)
        previous_loss_output.to_csv(path_or_buf = loss_output_filepath, sep = ",", na_rep = utils.NA_STRING, header = True, index = False, mode = "w")
    else:
        previous_loss_output = pd.read_csv(filepath_or_buffer = loss_output_filepath, sep = ",", na_values = utils.NA_STRING, header = 0, index_col = False)
    accuracy_output_filepath = f"{output_dir}/evaluation.{utils.ACCURACY_STATISTIC_NAME}.csv"
    if not exists(accuracy_output_filepath) or args.reset: # if column names need to be written
        pd.DataFrame(columns = evaluation_loss_output_columns).to_csv(path_or_buf = accuracy_output_filepath, sep = ",", na_rep = utils.NA_STRING, header = True, index = False, mode = "w")

    ##################################################


    # EVALUATE IF WE HAVEN'T YET
    ##################################################

    # determine if we need to evaluate
    actual_line_count_in_output_filepath = utils.count_lines(filepath = accuracy_output_filepath) # get number of lines already written
    test_partition_size = utils.count_lines(filepath = args.paths_test) # get number of songs for evaluation
    expected_line_count_in_output_filepath = (len(models) * test_partition_size) + 1 # for each model, evaluate on all songs, add one for the column labels

    # evaluate if necessary
    if actual_line_count_in_output_filepath < expected_line_count_in_output_filepath: # if the output is complete, based on number of lines

        # REPEAT WITH EACH MODEL IN INPUT DIRECTORY
        ##################################################

        for model_name in models:

            # LOAD MODEL
            ##################################################

            # get directory of model
            model_dir = f"{output_dir}/{model_name}"

            # get previous statistic outputs, if any
            model_column_value = model_name # value that will make up the model column
            previous_loss_output_for_model = previous_loss_output[previous_loss_output["model"] == model_column_value]

            # avoid loading in anything if possible
            if len(previous_loss_output_for_model) == test_partition_size and not args.reset:
                del model_dir, model_column_value, previous_loss_output_for_model
                continue
            else:
                previously_completed_paths = set(previous_loss_output_for_model["path"])

            # load training configurations
            train_args_filepath = f"{model_dir}/train_args.json"
            train_args = utils.load_json(filepath = train_args_filepath)
            del train_args_filepath

            # load dataset and data loader
            dataset = get_dataset(task = task, directory = train_args["data_dir"], paths = args.paths_test, mappings_path = train_args["mappings_path"], pool = train_args["prepool"])
            data_loader = torch.utils.data.DataLoader(
                dataset = dataset,
                batch_size = args.batch_size,
                shuffle = False,
                num_workers = args.jobs,
                collate_fn = dataset.collate,
            )

            # create the model
            model = get_model(args = train_args).to(device)

            # load the checkpoint
            checkpoint_filepath = f"{model_dir}/{utils.CHECKPOINTS_DIR_NAME}/best_model.{utils.VALID_PARTITION_NAME}.pth"
            model_state_dict = torch.load(f = checkpoint_filepath, map_location = device, weights_only = True)
            model.load_state_dict(state_dict = model_state_dict)
            del checkpoint_filepath, model_state_dict # free up memory

            ##################################################


            # EVALUATE
            ##################################################

            # put model into evaluation mode
            model.eval()
            with torch.no_grad():

                # iterate over testing partition
                for batch in tqdm(iterable = data_loader, desc = f"Evaluating Model {model_name}"):

                    # PASS THROUGH MODEL
                    ##################################################

                    # get paths associated with batch
                    paths = batch["path"]

                    # avoid calculating this batch if possible
                    incomplete_paths_in_batch = [path not in previously_completed_paths for path in paths]
                    if not any(incomplete_paths_in_batch) and not args.reset:
                        del paths, incomplete_paths_in_batch # free up memory
                        continue

                    # evaluate
                    current_batch_size = len(paths)
                    model_column = utils.rep(x = model_column_value, times = current_batch_size)
                    
                    # get input and output pair
                    inputs, labels, mask = batch["seq"], batch["label"], batch["mask"]
                    inputs, labels, mask = inputs.to(device), labels.to(device), mask.to(device) # move to device

                    # get tokens if any
                    tokens = batch["token"]
                    if tokens is not None:
                        tokens = tokens.to(device) # move to device

                    # get bar positions if any
                    bar_positions = batch["bar_position"]
                    if bar_positions is not None:
                        bar_positions = bar_positions.to(device)

                    # get outputs
                    if args.task == utils.MELODY_TRANSFORMER_DIR_NAME:
                        outputs = model(enc_inp = tokens, inp_bar_pos = bar_positions, rvq_latent = inputs, padding_mask = mask)
                    else:
                        outputs = model(input = inputs, mask = mask, tokens = tokens)

                    # compute the loss and its gradients
                    loss = loss_fn(outputs, labels)
                    loss = float(loss) # float(loss) because it has a gradient attribute

                    # compute number correct in batch
                    predictions = get_predictions_from_outputs(outputs = outputs)
                    correctness = (predictions == labels)

                    ##################################################


                    # OUTPUT STATISTICS
                    ##################################################

                    # loss
                    loss_batch_output = pd.DataFrame(
                        data = dict(zip(
                            evaluation_loss_output_columns,
                            (
                                model_column, # model
                                paths, # path
                                utils.rep(x = loss, times = current_batch_size), # loss
                            ))),
                        columns = evaluation_loss_output_columns,
                    )
                    loss_batch_output = loss_batch_output[incomplete_paths_in_batch].reset_index(drop = True) # only write paths that haven't been written yet
                    loss_batch_output.to_csv(path_or_buf = loss_output_filepath, sep = ",", na_rep = utils.NA_STRING, header = False, index = False, mode = "a")

                    # accuracy
                    accuracy_batch_output = pd.DataFrame(
                        data = dict(zip(
                            evaluation_accuracy_output_columns,
                            (
                                model_column, # model
                                paths, # path
                                labels.cpu().tolist(), # expected
                                predictions.cpu().tolist(), # actual
                                correctness.cpu().tolist(), # is correct
                            ))),
                        columns = evaluation_accuracy_output_columns,
                    )
                    accuracy_batch_output = accuracy_batch_output[incomplete_paths_in_batch].reset_index(drop = True) # only write paths that haven't been written yet
                    accuracy_batch_output.to_csv(path_or_buf = accuracy_output_filepath, sep = ",", na_rep = utils.NA_STRING, header = False, index = False, mode = "a")

                    # free up memory
                    del paths, incomplete_paths_in_batch, current_batch_size, model_column, inputs, labels, mask, tokens, bar_positions, outputs, loss, predictions, correctness, loss_batch_output, accuracy_batch_output

                    ##################################################

            ##################################################

            # free up memory
            del model, dataset, data_loader, model_dir, model_column_value, previous_loss_output_for_model, previously_completed_paths, train_args

        ##################################################

    ##################################################


    # LOG STATISTICS
    ##################################################

    # separator line
    logging.info(utils.MAJOR_SEPARATOR_LINE)
    column_widths = [30, 10]

    # loss
    loss = pd.read_csv(filepath_or_buffer = loss_output_filepath, sep = ",", na_values = utils.NA_STRING, header = 0, index_col = False)
    loss = loss[["model", utils.LOSS_STATISTIC_NAME]].groupby(by = "model").mean().reset_index(drop = False)
    loss[utils.LOSS_STATISTIC_NAME] = list(map(lambda val: f"{val:.4f}", loss[utils.LOSS_STATISTIC_NAME]))
    logging.info(utils.prettify_table(df = loss, column_widths = column_widths))
    del loss

    # separator line
    logging.info(utils.MAJOR_SEPARATOR_LINE)

    # accuracy
    accuracy = pd.read_csv(filepath_or_buffer = accuracy_output_filepath, sep = ",", na_values = utils.NA_STRING, header = 0, index_col = False)
    accuracy = accuracy[["model", "is_correct"]].groupby(by = "model").mean().reset_index(drop = False).rename(columns = {"is_correct": utils.ACCURACY_STATISTIC_NAME})
    accuracy[utils.ACCURACY_STATISTIC_NAME] = list(map(lambda val: f"{100 * val:.2f}%", accuracy[utils.ACCURACY_STATISTIC_NAME]))
    logging.info(utils.prettify_table(df = accuracy, column_widths = column_widths))
    del accuracy

    ##################################################

##################################################
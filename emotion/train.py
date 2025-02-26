# README
# Phillip Long
# February 22, 2025

# Train an emotion recognition model.

# python /home/pnlong/jingyue_latents/emotion/train.py

# IMPORTS
##################################################

import argparse
import logging
import pprint
import sys
from os.path import exists, basename
from os import mkdir
from multiprocessing import cpu_count # for calculating num_workers
import wandb
from datetime import datetime # for creating wandb run names linked to time of run
import pandas as pd
import torch
import torch.utils.data
from tqdm import tqdm
import warnings
warnings.simplefilter(action = "ignore", category = FutureWarning)

from os.path import dirname, realpath
import sys
sys.path.insert(0, dirname(realpath(__file__)))
sys.path.insert(0, dirname(dirname(realpath(__file__))))

from dataset import EmotionDataset
import utils

##################################################


# ARGUMENTS
##################################################

def parse_args(args = None, namespace = None):
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(prog = "Train", description = "Train a Model.")
    parser.add_argument("-pt", "--paths_train", default = f"{utils.EMOTION_DIR}/{utils.DATA_DIR_NAME}/{utils.TRAIN_PARTITION_NAME}.txt", type = str, help = ".txt file with absolute filepaths in training partition")
    parser.add_argument("-pv", "--paths_valid", default = f"{utils.EMOTION_DIR}/{utils.DATA_DIR_NAME}/{utils.VALID_PARTITION_NAME}.txt", type = str, help = ".txt file with absolute filepaths in validation partition")
    parser.add_argument("-o", "--output_dir", default = utils.EMOTION_DIR, type = str, help = "Output directory")
    parser.add_argument("-mn", "--model_name", default = utils.EMOTION_MODEL_NAME, type = str, help = "Name of the model")
    # training
    parser.add_argument("-bs", "--batch_size", default = utils.BATCH_SIZE, type = int, help = "Batch size for data loader")
    parser.add_argument("--steps", default = utils.N_STEPS, type = int, help = "Number of steps")
    parser.add_argument("--valid_steps", default = utils.N_VALID_STEPS, type = int, help = "Validation frequency")
    parser.add_argument("--early_stopping", action = "store_true", help = "Whether to use early stopping")
    parser.add_argument("--early_stopping_tolerance", default = utils.EARLY_STOPPING_TOLERANCE, type = int, help = "Number of extra validation rounds before early stopping")
    parser.add_argument("-lr", "--learning_rate", default = utils.LEARNING_RATE, type = float, help = "Learning rate")
    parser.add_argument("--lr_warmup_steps", default = utils.LEARNING_RATE_WARMUP_STEPS, type = int, help = "Learning rate warmup steps")
    parser.add_argument("--lr_decay_steps", default = utils.LEARNING_RATE_DECAY_STEPS, type = int, help = "Learning rate decay end steps")
    parser.add_argument("--lr_decay_multiplier", default = utils.LEARNING_RATE_DECAY_MULTIPLIER, type = float, help = "Learning rate multiplier at the end")
    parser.add_argument("-wd", "--weight_decay", default = utils.WEIGHT_DECAY, type = float, help = "Weight decay for L2 regularization")
    # others
    parser.add_argument("-g", "--gpu", default = -1, type = int, help = "GPU number")
    parser.add_argument("-j", "--jobs", default = int(cpu_count() / 4), type = int, help = "Number of workers for data loading")
    parser.add_argument("-r", "--resume", default = None, type = str, help = "Provide the WANDB run name/id to resume a run")
    parser.add_argument("-ir", "--infer_run_name", action = "store_true", help = "Whether or not to infer the WANDB run name when resuming")
    parser.add_argument("--use_wandb", action = "store_true", help = "Whether or not to log progress with WANDB")
    return parser.parse_args(args = args, namespace = namespace)

##################################################


# CUSTOM NEURAL NETWORK
##################################################

class EmotionMLP(torch.nn.Module):

    # initializer
    def __init__(self):
        super().__init__()
        layers = [
            torch.nn.Linear(in_features = utils.LATENT_EMBEDDING_DIM, out_features = 10),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features = 10, out_features = utils.N_EMOTION_CLASSES),
        ]
        # layers = [
        #     torch.nn.Linear(in_features = utils.LATENT_EMBEDDING_DIM, out_features = 2 * utils.LATENT_EMBEDDING_DIM),
        #     torch.nn.ReLU(),
        #     torch.nn.Linear(in_features = 2 * utils.LATENT_EMBEDDING_DIM, out_features = utils.LATENT_EMBEDDING_DIM // 2),
        #     torch.nn.ReLU(),
        #     torch.nn.Linear(in_features = utils.LATENT_EMBEDDING_DIM // 2, out_features = utils.N_EMOTION_CLASSES),
        # ]
        self.mlp = torch.nn.Sequential(*layers)

    # forward pass
    def forward(self, input: torch.Tensor, mask: torch.Tensor):

        # reshape input from (batch_size, num_bar, embedding_dim) to (batch_size * num_bar, embedding_dim)
        input = input.flatten(start_dim = 0, end_dim = 1)

        # run input through model, which yields an output of size (batch_size * num_bar, n_classes)
        output = self.mlp(input)

        # reshape output to size (batch_size, num_bar, n_classes)
        output = output.unflatten(dim = 0, sizes = mask.shape)

        # apply mask (size of (batch_size, num_bar)) to output
        output *= mask.unsqueeze(dim = -1)

        # average output to reduce num_bar dimension; output is now of size (batch_size, n_classes)
        logits = output.sum(dim = 1) / mask.sum(dim = -1).unsqueeze(dim = -1)

        # return final logits
        return logits

##################################################


# HELPER FUNCTIONS
##################################################

def get_predictions_from_outputs(outputs: torch.Tensor) -> torch.Tensor:
    """Directly given the outputs from the model, convert the output into label predictions."""

    # get predictions from outputs
    outputs = torch.nn.functional.softmax(input = outputs, dim = -1) # apply softmax function to convert to probability distribution
    predictions = torch.argmax(input = outputs, dim = -1) # determine final class prediction from softmaxed outputs
    
    # return class predictions
    return predictions

##################################################


# MAIN METHOD
##################################################

if __name__ == "__main__":

    # LOAD UP MODEL
    ##################################################

    # parse the command-line arguments
    args = parse_args()

    # check filepath arguments
    if not exists(args.paths_train):
        raise ValueError("Invalid --paths_train argument. File does not exist.")
    if not exists(args.paths_valid):
        raise ValueError("Invalid --paths_valid argument. File does not exist.")
    run_name = args.resume # get runname
    args.resume = (run_name != None) # convert to boolean value
    
    # get the specified device
    device = torch.device(f"cuda:{abs(args.gpu)}" if (torch.cuda.is_available() and args.gpu != -1) else "cpu")
    print(f"Using device: {device}")

    # create the dataset and data loader
    print(f"Creating the data loader...")
    dataset = {
        utils.TRAIN_PARTITION_NAME: EmotionDataset(paths = args.paths_train),
        utils.VALID_PARTITION_NAME: EmotionDataset(paths = args.paths_valid),
        }
    data_loader = {
        utils.TRAIN_PARTITION_NAME: torch.utils.data.DataLoader(dataset = dataset[utils.TRAIN_PARTITION_NAME], batch_size = args.batch_size, shuffle = True, num_workers = args.jobs, collate_fn = dataset[utils.TRAIN_PARTITION_NAME].collate),
        utils.VALID_PARTITION_NAME: torch.utils.data.DataLoader(dataset = dataset[utils.VALID_PARTITION_NAME], batch_size = args.batch_size, shuffle = False, num_workers = args.jobs, collate_fn = dataset[utils.VALID_PARTITION_NAME].collate),
    }

    # create the model
    print(f"Creating model...")
    model = EmotionMLP().to(device)
    n_parameters = sum(p.numel() for p in model.parameters()) # statistics
    n_parameters_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad) # statistics (model size)

    # determine the output directory based on arguments
    output_parent_dir = args.output_dir
    output_dir_name = args.model_name.replace(" ", "_")
    output_dir = f"{output_parent_dir}/{output_dir_name}" # custom output directory based on arguments
    if not exists(output_dir):
        mkdir(output_dir)
    checkpoints_dir = f"{output_dir}/{utils.CHECKPOINTS_DIR_NAME}" # models will be stored in the output directory
    if not exists(checkpoints_dir):
        mkdir(checkpoints_dir)

    # start a new wandb run to track the script
    if args.use_wandb:
        group_name = basename(output_parent_dir)
        if args.infer_run_name:
            run_names_in_group = [run.name for run in wandb.Api().runs(f"philly/{utils.WANDB_PROJECT_NAME}", filters = {"group": group_name})]
            run_names_in_group = list(filter(lambda name: name.startswith(output_dir_name), run_names_in_group))
            get_datetime_value_from_run_name = lambda run_name: datetime.strptime(run_name[len(output_dir_name) + 1:], utils.WANDB_RUN_NAME_FORMAT_STRING).timestamp()
            run_names_in_group = sorted(run_names_in_group, key = get_datetime_value_from_run_name)[::-1]
            run_name = run_names_in_group[0] if len(run_names_in_group) > 0 else None # try to infer the run name
            args.resume = (run_name != None) # redefine args.resume in the event that no run name was supplied, but we can't infer one either
            del run_names_in_group, get_datetime_value_from_run_name # free up memory
        if run_name is None: # in the event we need to create a new run name
            current_datetime = datetime.now().strftime(utils.WANDB_RUN_NAME_FORMAT_STRING)
            run_name = f"{output_dir_name}-{current_datetime}"
        run = wandb.init(config = dict(vars(args), **{"n_parameters": n_parameters, "n_parameters_trainable": n_parameters_trainable}), resume = "allow", project = utils.WANDB_PROJECT_NAME, group = group_name, name = run_name, id = run_name) # set project title, configure with hyperparameters

    # set up the logger
    logging_output_filepath = f"{output_dir}/train.log"
    log_hyperparameters = not (args.resume and exists(logging_output_filepath))
    logging.basicConfig(level = logging.INFO, format = "%(message)s", handlers = [logging.FileHandler(filename = logging_output_filepath, mode = "a" if args.resume else "w"), logging.StreamHandler(stream = sys.stdout)])

    # log command called and arguments, save arguments
    if log_hyperparameters:
        logging.info(f"Running command: python {' '.join(sys.argv)}")
        logging.info(f"Using arguments:\n{pprint.pformat(vars(args))}")
        args_output_filepath = f"{output_dir}/train_args.json"
        logging.info(f"Saved arguments to {args_output_filepath}")
        utils.save_args(filepath = args_output_filepath, args = args)
        del args_output_filepath # clear up memory
    else: # print previous loggings to stdout
        with open(logging_output_filepath, "r") as logging_output:
            print(logging_output.read())

    # load previous model and summarize if needed
    def log_model_size():
        """Log the size of the model."""
        logging.info(f"Number of parameters: {n_parameters:,}")
        logging.info(f"Number of trainable parameters: {n_parameters_trainable:,}")
    best_model_filepath = {partition: f"{checkpoints_dir}/best_model.{partition}.pth" for partition in utils.RELEVANT_TRAINING_PARTITIONS}
    if args.resume and exists(best_model_filepath[utils.VALID_PARTITION_NAME]):
        model.load_state_dict(torch.load(f = best_model_filepath[utils.VALID_PARTITION_NAME], weights_only = True))
        if args.fine_tune and log_hyperparameters:
            log_model_size()
    else:
        log_model_size()

    # loss function
    loss_fn = torch.nn.CrossEntropyLoss()
    
    # create the optimizer
    optimizer = torch.optim.Adam(params = model.parameters(), lr = args.learning_rate, weight_decay = args.weight_decay)
    best_optimizer_filepath = {partition: f"{checkpoints_dir}/best_optimizer.{partition}.pth" for partition in utils.RELEVANT_TRAINING_PARTITIONS}
    if args.resume and exists(best_optimizer_filepath[utils.VALID_PARTITION_NAME]):
        optimizer.load_state_dict(torch.load(f = best_optimizer_filepath[utils.VALID_PARTITION_NAME], weights_only = True))

    # create the scheduler
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer = optimizer, lr_lambda = lambda step: utils.get_lr_multiplier(step = step, warmup_steps = args.lr_warmup_steps, decay_end_steps = args.lr_decay_steps, decay_end_multiplier = args.lr_decay_multiplier))
    best_scheduler_filepath = {partition: f"{checkpoints_dir}/best_scheduler.{partition}.pth" for partition in utils.RELEVANT_TRAINING_PARTITIONS}
    if args.resume and exists(best_scheduler_filepath[utils.VALID_PARTITION_NAME]):
        scheduler.load_state_dict(torch.load(f = best_scheduler_filepath[utils.VALID_PARTITION_NAME], weights_only = True))
    
    ##################################################


    # TRAINING PROCESS
    ##################################################

    # create a file to record loss/accuracy metrics
    output_filepath = f"{output_dir}/statistics.csv"
    statistics_columns_must_be_written = not (exists(output_filepath) and args.resume) # whether or not to write column names
    if statistics_columns_must_be_written: # if column names need to be written
        pd.DataFrame(columns = utils.TRAINING_STATISTICS_OUTPUT_COLUMNS).to_csv(path_or_buf = output_filepath, sep = ",", na_rep = utils.NA_STRING, header = True, index = False, mode = "w")

    # initialize variables
    step = 0
    best_statistic = dict() # intialize best statistic dictionary tracker
    best_statistic[utils.LOSS_STATISTIC_NAME] = {partition: float("inf") for partition in utils.RELEVANT_TRAINING_PARTITIONS} # current best lost is infinity
    best_statistic[utils.ACCURACY_STATISTIC_NAME] = {partition: 0.0 for partition in utils.RELEVANT_TRAINING_PARTITIONS} # current best accuracy is 0.0%
    if not statistics_columns_must_be_written: # load in previous loss info
        previous_statistic = pd.read_csv(filepath_or_buffer = output_filepath, sep = ",", na_values = utils.NA_STRING, header = 0, index_col = False) # read in previous loss/accuracy values
        if len(previous_statistic) > 0:
            for partition in utils.RELEVANT_TRAINING_PARTITIONS:
                previous_statistic_partition = previous_statistic[previous_statistic["partition"] == partition]
                best_statistic[utils.LOSS_STATISTIC_NAME][partition] = float(previous_statistic_partition.loc[previous_statistic_partition[f"is_{utils.LOSS_STATISTIC_NAME}"], "value"].min(axis = 0)) # get minimum loss
                best_statistic[utils.ACCURACY_STATISTIC_NAME][partition] = float(previous_statistic_partition.loc[~previous_statistic_partition[f"is_{utils.LOSS_STATISTIC_NAME}"], "value"].max(axis = 0)) # get maximum accuracy
            step = int(previous_statistic["step"].max(axis = 0)) # update step
        del previous_statistic
    if args.early_stopping: # stop early?
        count_early_stopping = 0

    # print current step
    print(f"Current Step: {step:,}")

    # iterate for the specified number of steps
    train_iterator = iter(data_loader[utils.TRAIN_PARTITION_NAME])
    while step < args.steps:

        # to store loss/accuracy values
        statistics = {statistic: {partition: 0.0 for partition in utils.RELEVANT_TRAINING_PARTITIONS} for statistic in utils.RELEVANT_TRAINING_STATISTICS}

        # TRAIN
        ##################################################

        # log that model is training
        logging.info(f"Training...")

        # put model into training mode
        model.train()
        count = 0 # count number of songs
        for _ in (progress_bar := tqdm(iterable = range(args.valid_steps), desc = "Training")):

            # get next batch
            try:
                batch = next(train_iterator)
            except (StopIteration):
                train_iterator = iter(data_loader[utils.TRAIN_PARTITION_NAME]) # reinitialize dataset iterator
                batch = next(train_iterator)

            # get input and output pair
            inputs, labels, mask = batch["seq"], batch["label"], batch["mask"]
            inputs, labels, mask = inputs.to(device), labels.to(device), mask.to(device) # move to device
            current_batch_size = len(labels)

            # zero gradients
            optimizer.zero_grad()

            # get outputs
            outputs = model(input = inputs, mask = mask)

            # compute the loss and its gradients
            loss = loss_fn(outputs, labels)
            loss.backward()
            loss = float(loss) # float(loss) because it has a gradient attribute

            # adjust learning weights
            optimizer.step() # update parameters
            scheduler.step() # update scheduler

            # compute accuracy
            predictions = get_predictions_from_outputs(outputs = outputs)
            n_correct_in_batch = int(sum(predictions == labels))
            accuracy = 100 * (n_correct_in_batch / current_batch_size)
                        
            # set progress bar
            progress_bar.set_postfix(loss = f"{loss:8.4f}", accuracy = f"{accuracy:3.2f}%")

            # log training loss/accuracy for wandb
            if args.use_wandb:
                wandb.log({
                    f"{utils.TRAIN_PARTITION_NAME}/{utils.LOSS_STATISTIC_NAME}": loss,
                    f"{utils.TRAIN_PARTITION_NAME}/{utils.ACCURACY_STATISTIC_NAME}": accuracy,
                }, step = step)

            # update count
            count += current_batch_size

            # add to total statistics tracker
            statistics[utils.LOSS_STATISTIC_NAME][utils.TRAIN_PARTITION_NAME] += loss * current_batch_size
            statistics[utils.ACCURACY_STATISTIC_NAME][utils.TRAIN_PARTITION_NAME] += n_correct_in_batch

            # increment step
            step += 1

            # release GPU memory right away
            del inputs, labels, mask, outputs, loss, predictions, n_correct_in_batch, accuracy

        # compute average loss/accuracy across batches
        statistics[utils.LOSS_STATISTIC_NAME][utils.TRAIN_PARTITION_NAME] /= count
        statistics[utils.ACCURACY_STATISTIC_NAME][utils.TRAIN_PARTITION_NAME] /= (count / 100)
        
        # log train info for wandb
        if args.use_wandb:
            wandb.log({
                f"{utils.TRAIN_PARTITION_NAME}/{utils.LOSS_STATISTIC_NAME}": statistics[utils.LOSS_STATISTIC_NAME][utils.TRAIN_PARTITION_NAME],
                f"{utils.TRAIN_PARTITION_NAME}/{utils.ACCURACY_STATISTIC_NAME}": statistics[utils.ACCURACY_STATISTIC_NAME][utils.TRAIN_PARTITION_NAME],
            }, step = step)
        
        ##################################################


        # VALIDATE
        ##################################################

        # log that model is validating
        logging.info(f"Validating...")

        # put model into evaluation mode
        model.eval()
        with torch.no_grad():
            
            # iterate through validation data loader
            count = 0 # count number of songs
            for batch in (progress_bar := tqdm(iterable = data_loader[utils.VALID_PARTITION_NAME], desc = "Validating")):

                # get input and output pair
                inputs, labels, mask = batch["seq"], batch["label"], batch["mask"]
                inputs, labels, mask = inputs.to(device), labels.to(device), mask.to(device) # move to device
                current_batch_size = len(labels)

                # get outputs
                outputs = model(input = inputs, mask = mask)

                # compute the loss and its gradients
                loss = loss_fn(outputs, labels)
                loss = float(loss) # float(loss) because it has a gradient attribute

                # compute accuracy
                predictions = get_predictions_from_outputs(outputs = outputs)
                n_correct_in_batch = int(sum(predictions == labels))
                accuracy = 100 * (n_correct_in_batch / current_batch_size)

                # set progress bar
                progress_bar.set_postfix(loss = f"{loss:8.4f}", accuracy = f"{accuracy:3.2f}%")

                # log training loss/accuracy for wandb
                if args.use_wandb:
                    wandb.log({
                        f"{utils.VALID_PARTITION_NAME}/{utils.LOSS_STATISTIC_NAME}": loss,
                        f"{utils.VALID_PARTITION_NAME}/{utils.ACCURACY_STATISTIC_NAME}": accuracy,
                    }, step = step)

                # update count
                count += current_batch_size

                # add to total statistics tracker
                statistics[utils.LOSS_STATISTIC_NAME][utils.VALID_PARTITION_NAME] += loss * current_batch_size
                statistics[utils.ACCURACY_STATISTIC_NAME][utils.VALID_PARTITION_NAME] += n_correct_in_batch

                # release GPU memory right away
                del inputs, labels, mask, outputs, loss, predictions, n_correct_in_batch, accuracy

        # compute average loss/accuracy across batches
        statistics[utils.LOSS_STATISTIC_NAME][utils.VALID_PARTITION_NAME] /= count
        statistics[utils.ACCURACY_STATISTIC_NAME][utils.VALID_PARTITION_NAME] /= (count / 100)

        # output statistics
        logging.info(f"Validation loss: {statistics[utils.LOSS_STATISTIC_NAME][utils.VALID_PARTITION_NAME]:.4f}")

        # log validation info for wandb
        if args.use_wandb:
            wandb.log({
                f"{utils.VALID_PARTITION_NAME}/{utils.LOSS_STATISTIC_NAME}": statistics[utils.LOSS_STATISTIC_NAME][utils.VALID_PARTITION_NAME],
                f"{utils.VALID_PARTITION_NAME}/{utils.ACCURACY_STATISTIC_NAME}": statistics[utils.ACCURACY_STATISTIC_NAME][utils.VALID_PARTITION_NAME],
            }, step = step)

        ##################################################


        # RECORD LOSS, SAVE MODEL
        ##################################################

        # write output to file
        output = pd.DataFrame(
            data = dict(zip(
                utils.TRAINING_STATISTICS_OUTPUT_COLUMNS,
                (
                    utils.rep(x = step, times = len(utils.RELEVANT_TRAINING_PARTITIONS) * 2), # step
                    utils.RELEVANT_TRAINING_PARTITIONS + utils.RELEVANT_TRAINING_PARTITIONS, # partition
                    [True, True, False, False], # is_loss
                    list(statistics[utils.LOSS_STATISTIC_NAME].values()) + list(statistics[utils.ACCURACY_STATISTIC_NAME].values()), # value
                ))),
            columns = utils.TRAINING_STATISTICS_OUTPUT_COLUMNS)
        output.to_csv(path_or_buf = output_filepath, sep = ",", na_rep = utils.NA_STRING, header = False, index = False, mode = "a")

        # see whether or not to save
        is_an_improvement = False # whether or not the loss has improved
        for partition in utils.RELEVANT_TRAINING_PARTITIONS:

            # check if loss has improved
            partition_loss = statistics[utils.LOSS_STATISTIC_NAME][partition]
            if partition_loss < best_statistic[utils.LOSS_STATISTIC_NAME][partition]:
                best_statistic[utils.LOSS_STATISTIC_NAME][partition] = partition_loss
                logging.info(f"Best {partition}_{utils.LOSS_STATISTIC_NAME} so far!")
                torch.save(obj = model.state_dict(), f = best_model_filepath[partition]) # save the model
                torch.save(obj = optimizer.state_dict(), f = best_optimizer_filepath[partition]) # save the optimizer state
                torch.save(obj = scheduler.state_dict(), f = best_scheduler_filepath[partition]) # save the scheduler state
                if args.early_stopping: # reset the early stopping counter if we found a better model
                    count_early_stopping = 0
                    is_an_improvement = True # we only care about the lack of improvement when we are thinking about early stopping, so turn off this boolean flag, since there was an improvement
            
            # check if accuracy has improved
            partition_accuracy = statistics[utils.ACCURACY_STATISTIC_NAME][partition]
            if partition_accuracy > best_statistic[utils.ACCURACY_STATISTIC_NAME][partition]:
                best_statistic[utils.ACCURACY_STATISTIC_NAME][partition] = partition_accuracy
                logging.info(f"Best {partition}_{utils.ACCURACY_STATISTIC_NAME} so far!")

        # increment the early stopping counter if no improvement is found
        if (not is_an_improvement) and args.early_stopping:
            count_early_stopping += 1 # increment

        # early stopping
        if args.early_stopping and (count_early_stopping > args.early_stopping_tolerance):
            logging.info(f"Stopped the training for no improvements in {args.early_stopping_tolerance} rounds.")
            break

        ##################################################

    ##################################################

    
    # STATISTICS AND CONCLUSION
    ##################################################

    # log minimum validation loss
    logging.info(f"Minimum validation loss achieved: {best_statistic[utils.LOSS_STATISTIC_NAME][utils.VALID_PARTITION_NAME]:.4f}")
    
    # log maximum validation accuracy
    logging.info(f"Maximum validation accuracy achieved: {best_statistic[utils.ACCURACY_STATISTIC_NAME][utils.VALID_PARTITION_NAME]:.2f}%")

    # log final statistics to wandb
    if args.use_wandb:
        wandb.log({f"best_{partition}_{statistic}": best_statistic[statistic][partition] for partition in utils.RELEVANT_TRAINING_PARTITIONS for statistic in utils.RELEVANT_TRAINING_STATISTICS})
        wandb.finish() # finish the wandb run

    # output model name to list of models
    models_output_filepath = f"{output_parent_dir}/models.txt"
    if exists(models_output_filepath):
        models = set(utils.load_txt(filepath = models_output_filepath)) # read in list of trained models and use a set because better for `in` operations
    else:
        models = set()
    with open(models_output_filepath, "a") as models_output:
        if output_dir_name not in models: # check if in list of trained models
            models_output.write(f"{output_dir_name}\n") # add model to list of trained models if it isn't already there

    ##################################################

##################################################


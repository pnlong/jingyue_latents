# README
# Phillip Long
# March 4, 2025

# Prepare a model.

# python /home/pnlong/jingyue_latents/model.py

# IMPORTS
##################################################

import torch
from torch import nn

from os.path import dirname, realpath
import sys
sys.path.insert(0, dirname(realpath(__file__)))

import utils

##################################################


# CUSTOM MULTILAYER PERCEPTRON
##################################################

class CustomMLP(nn.Module):

    # initializer
    def __init__(self,
            input_dim: int, # number of input features
            output_dim: int, # number of output features
            prepool: bool = False, # whether inputs are prepooled
            n_outputs_per_input_bar: int = 1, # whether there are multiple events per bar (0 means this is a sequence-level task)
            vocabulary_size: int = 0, # size of token vocabulary
            use_large: bool = False, # whether to use larger model
            use_small: bool = False, # whether to use smaller model
        ):
        super().__init__()

        # save variables
        self.prepool = prepool
        self.n_outputs_per_input_bar = n_outputs_per_input_bar
        output_dim *= n_outputs_per_input_bar # make sure the output dim is correct for the model

        # draft model architecture
        if use_large: # large model
            layers = [
                nn.Linear(in_features = input_dim, out_features = 2 * input_dim),
                nn.ReLU(),
                nn.Linear(in_features = 2 * input_dim, out_features = input_dim // 2),
                nn.ReLU(),
                nn.Linear(in_features = input_dim // 2, out_features = output_dim),
            ]
        elif use_small: # small model
            layers = [
                nn.Linear(in_features = input_dim, out_features = output_dim),
            ]
        else: # default to normal model
            layers = [
                nn.Linear(in_features = input_dim, out_features = input_dim // 2),
                nn.ReLU(),
                nn.Linear(in_features = input_dim // 2, out_features = output_dim),
            ]

        # create model
        self.mlp = nn.Sequential(*layers)

        # token embeddings (learned)
        if vocabulary_size > 0:
            self.token_embeddings = nn.Embedding(
                num_embeddings = vocabulary_size,
                embedding_dim = input_dim,
            )

    # forward pass
    def forward(self,
            input: torch.Tensor,
            mask: torch.Tensor,
            tokens: torch.Tensor = None,
        ):

        # get some dimensions
        batch_size = input.shape[0]
        num_bar = 1 if self.prepool else input.shape[1]
        embedding_dim = input.shape[-1]

        # deal with tokens
        if tokens is not None:
            tokens = self.token_embeddings(tokens) # pass through embeddings
            tokens = tokens.sum(dim = -2) # sum across different token types
            tokens = tokens.reshape(batch_size * num_bar, embedding_dim) # reshape to two dimensions
        else: # if not using tokens, just have a matrix of zeros (so no effect)
            tokens = torch.zeros(size = (batch_size * num_bar, embedding_dim), dtype = utils.TOKEN_TYPE)

        # if num_bar dimension was already reduced
        if self.prepool:
            logits = self.mlp(input + tokens) # simply get logits by feeding the input through the model

        # if num_bar dimension wasn't reduced, and we pool ourselves
        else:
            input = input.reshape(batch_size * num_bar, embedding_dim) # reshape input from (batch_size, num_bar, embedding_dim) to (batch_size * num_bar, embedding_dim)
            output = self.mlp(input + tokens) # feed input through model, which yields an output of size (batch_size * num_bar, n_classes)
            output = output.reshape(batch_size, num_bar, -1) # reshape output to size (batch_size, num_bar, n_classes)
            output = output * mask.unsqueeze(dim = -1) # apply mask (size of (batch_size, num_bar)) to output
            logits = output.sum(dim = 1) / mask.sum(dim = -1).unsqueeze(dim = -1) # average output to reduce num_bar dimension; output is now of size (batch_size, n_classes)
            del output # free up memory

        # reshape logits any further if necessary
        if self.n_outputs_per_input_bar > 1:
            logits = logits.reshape(batch_size * self.n_outputs_per_input_bar, -1) # reshape to (batch_size * number of events per bar, n_classes)

        # return final logits
        return logits
    
##################################################


# CUSTOM TRANSFORMER
##################################################

class CustomTransformer(nn.Module):

    # initializer
    def __init__(self,
            input_dim: int, # number of input features
            output_dim: int, # number of output features
            max_seq_len: int = 1, # maximum sequence length
            n_outputs_per_input_bar: int = 1, # whether there are multiple events per bar (0 means this is a sequence-level task)
            vocabulary_size: int = 0, # size of token vocabulary
            use_large: bool = False, # whether to use larger model
            use_small: bool = False, # whether to use smaller model
        ):
        super().__init__()

        # save variables
        self.n_outputs_per_input_bar = n_outputs_per_input_bar
        output_dim *= n_outputs_per_input_bar # make sure the output dim is correct for the model

        # draft model architecture
        dropout = utils.TRANSFORMER_DROPOUT # dropout rate
        heads = utils.TRANSFORMER_HEADS # number of attention heads
        layers = utils.TRANSFORMER_LAYERS # number of layers
        feedforward_layers = utils.TRANSFORMER_FEEDFORWARD_LAYERS # number of feedforward layers
        if use_large:
            heads *= 2
            layers *= 2
            feedforward_layers *= 2
        elif use_small:
            heads //= 2
            layers //= 2
            feedforward_layers //= 2

        # token embeddings (learned)
        if vocabulary_size > 0:
            self.token_embeddings = nn.Embedding(
                num_embeddings = vocabulary_size,
                embedding_dim = input_dim,
            )

        # positional embeddings (learned)
        self.position_embeddings = nn.Embedding(
            num_embeddings = max_seq_len,
            embedding_dim = input_dim,
        )
        
        # transformer encoder
        self.transformer = nn.Transformer(
            d_model = input_dim, # embedding dimension
            nhead = heads, # number of attention heads
            num_encoder_layers = layers, # number of transformer layers
            num_decoder_layers = layers, # number of transformer layers
            dim_feedforward = feedforward_layers, # number of feedforward layers
            dropout = dropout, # dropout rate
            activation = nn.functional.relu, # activation function
            batch_first = True, # input and output are of size (batch_size, num_bar, embedding_dim)
        )

        # output layer (for classification task)
        self.fc_out = nn.Linear(in_features = input_dim, out_features = output_dim)

    # forward pass
    def forward(self,
            input: torch.Tensor,
            mask: torch.Tensor,
            tokens: torch.Tensor = None,
        ):

        # get some dimensions
        batch_size, num_bar, embedding_dim = input.shape

        # deal with tokens
        if tokens is not None:
            tokens = self.token_embeddings(tokens) # pass through embeddings
            tokens = tokens.sum(dim = -2) # sum across different token types
        else: # if not using tokens, just have a matrix of zeros (so no effect)
            tokens = torch.zeros(size = (batch_size, num_bar, embedding_dim), dtype = utils.TOKEN_TYPE)

        # wrangle mask
        mask = torch.logical_not(input = mask) # padding values must be True

        # calculate positional embedding
        position_indicies = torch.arange(num_bar, dtype = torch.long, device = input.device) # get positions for a single batch
        position_indicies = position_indicies.unsqueeze(dim = 0).repeat(batch_size, 1) # repeat positions across all sequences in batch to size (batch_size, num_bar)
        position_embeddings = self.position_embeddings(position_indicies) # calculate positional embeddings from positions, which yields a result of size (batch_size, num_bar, embedding_dim)

        # wrangle input
        input = input + tokens + position_embeddings # add positional embeddings to input
        
        # passing through transformer
        output = self.transformer(
            src = input, # source
            tgt = input, # target
            src_key_padding_mask = mask, # source mask
            tgt_key_padding_mask = mask, # mask for target as well
        ) # for simplicity, using input as both src and tgt

        # output of transformer is (batch_size, num_bar, embedding_dim)
        output = output[:, -1, :] # choose the output from the final bar, now of size (batch_size, embedding_dim)

        # pass through the output layer
        logits = self.fc_out(output) # shape is now (batch_size, n_classes)

        # reshape logits any further if necessary
        if self.n_outputs_per_input_bar > 1:
            logits = logits.reshape(batch_size * self.n_outputs_per_input_bar, -1) # reshape to (batch_size * number of events per bar, n_classes)

        # return final logits
        return logits

##################################################


# HELPER FUNCTION TO GET THE CORRECT MODEL GIVEN ARGS
##################################################

def get_model(args: dict) -> nn.Module:
    """Helper function to return the correct model given arguments as a dictionary."""

    # scrape variables from arguments
    task = args.get("task")
    use_transformer = args.get("use_transformer", False)
    input_dim = utils.PREBOTTLENECK_LATENT_EMBEDDING_DIM if args.get("use_prebottleneck_latents", False) else utils.LATENT_EMBEDDING_DIM
    prepool = args.get("prepool", False)
    max_seq_len = utils.MAX_SEQ_LEN_BY_TASK[task]
    model_name = args.get("model_name")
    n_outputs_per_input_bar = utils.N_OUTPUTS_PER_INPUT_BAR_BY_TASK[task]
    vocabulary_size = utils.VOCABULARY_SIZE_BY_TASK[task]

    # task specific arguments
    match task:
        case utils.EMOTION_DIR_NAME:
            output_dim = utils.N_EMOTION_CLASSES
        case utils.CHORD_DIR_NAME:
            output_dim = utils.N_CHORD32_CLASSES if args.get("use_precise_labels", False) else utils.N_CHORD11_CLASSES
        case utils.MELODY_DIR_NAME:
            output_dim = utils.N_MELODY_CLASSES
    
    # determine small or large
    use_large = "large" in model_name
    use_small = "small" in model_name
    if use_large and use_small: # don't want them both at once
        use_small = False

    # create transformer model
    if use_transformer:
        model = CustomTransformer(
            input_dim = input_dim, output_dim = output_dim,
            max_seq_len = max_seq_len,
            n_outputs_per_input_bar = n_outputs_per_input_bar,
            vocabulary_size = vocabulary_size,
            use_large = use_large, use_small = use_small,
        )

    # create MLP model
    else:
        model = CustomMLP(
            input_dim = input_dim, output_dim = output_dim,
            prepool = prepool, 
            n_outputs_per_input_bar = n_outputs_per_input_bar,
            vocabulary_size = vocabulary_size,
            use_large = use_large, use_small = use_small,
        )

    # return the correct model
    return model

##################################################


# HELPER FUNCTION TO CONVERT OUTPUTS INTO CLASS PREDICTIONS
##################################################

def get_predictions_from_outputs(outputs: torch.Tensor) -> torch.Tensor:
    """Directly given the outputs from the model, convert the output into label predictions."""

    # get predictions from outputs
    outputs = nn.functional.softmax(input = outputs, dim = -1) # apply softmax function to convert to probability distribution
    predictions = torch.argmax(input = outputs, dim = -1) # determine final class prediction from softmaxed outputs
    
    # return class predictions
    return predictions

##################################################


# MAIN METHOD
##################################################

if __name__ == "__main__":

    pass

##################################################

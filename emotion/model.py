# README
# Phillip Long
# February 22, 2025

# Prepare emotion recognition model.

# python /home/pnlong/jingyue_latents/emotion/model.py

# IMPORTS
##################################################

import torch
from torch import nn

from os.path import dirname, realpath
import sys
sys.path.insert(0, dirname(realpath(__file__)))
sys.path.insert(0, dirname(dirname(realpath(__file__))))

import utils

##################################################


# CUSTOM MULTILAYER PERCEPTRON
##################################################

class EmotionMLP(nn.Module):

    # initializer
    def __init__(self,
            input_dim: int = utils.LATENT_EMBEDDING_DIM, # number of input features
            output_dim: int = utils.EMOTION_N_CLASSES, # number of output features
            prepool: bool = False, # whether inputs are prepooled
            use_large: bool = False, # whether to use larger model
            use_small: bool = False, # whether to use smaller model
        ):
        super().__init__()

        # save variables
        self.prepool = prepool

        # draft model architecture
        if use_large: # large model has 49,732 parameters
            layers = [
                nn.Linear(in_features = input_dim, out_features = 2 * input_dim),
                nn.ReLU(),
                nn.Linear(in_features = 2 * input_dim, out_features = input_dim // 2),
                nn.ReLU(),
                nn.Linear(in_features = input_dim // 2, out_features = output_dim),
            ]
        elif use_small: # small model has 516 parameters
            layers = [
                nn.Linear(in_features = input_dim, out_features = output_dim),
            ]
        else: # default to normal model with 8,516 parameters
            layers = [
                nn.Linear(in_features = input_dim, out_features = input_dim // 2),
                nn.ReLU(),
                nn.Linear(in_features = input_dim // 2, out_features = output_dim),
            ]

        # create model
        self.mlp = nn.Sequential(*layers)

    # forward pass
    def forward(self,
            input: torch.Tensor,
            mask: torch.Tensor,
        ):

        # if num_bar dimension was already reduced
        if self.prepool:
            logits = self.mlp(input) # simply get logits by feeding the input through the model

        # if num_bar dimension wasn't reduced, and we pool ourselves
        else:
            input = input.flatten(start_dim = 0, end_dim = 1)  # reshape input from (batch_size, num_bar, embedding_dim) to (batch_size * num_bar, embedding_dim)
            output = self.mlp(input) # feed input through model, which yields an output of size (batch_size * num_bar, n_classes)
            output = output.unflatten(dim = 0, sizes = mask.shape) # reshape output to size (batch_size, num_bar, n_classes)
            output *= mask.unsqueeze(dim = -1) # apply mask (size of (batch_size, num_bar)) to output
            logits = output.sum(dim = 1) / mask.sum(dim = -1).unsqueeze(dim = -1) # average output to reduce num_bar dimension; output is now of size (batch_size, n_classes)

        # return final logits
        return logits
    
##################################################


# CUSTOM TRANSFORMER
##################################################

class EmotionTransformer(nn.Module):

    # initializer
    def __init__(self,
            input_dim: int = utils.LATENT_EMBEDDING_DIM, # number of input features
            output_dim: int = utils.EMOTION_N_CLASSES, # number of output features
            max_seq_len: int = utils.EMOTION_MAX_SEQ_LEN, # maximum sequence length
            heads: int = utils.TRANSFORMER_HEADS, # number of attention heads
            layers: int = utils.TRANSFORMER_LAYERS, # number of layers
            dropout: float = utils.TRANSFORMER_DROPOUT, # dropout rate
            feedforward_layers: int = utils.TRANSFORMER_FEEDFORWARD_LAYERS, # number of feedforward layers
        ):
        super().__init__()

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
        ):

        # wrangle mask
        mask = torch.logical_not(input = mask) # padding values must be True

        # calculate positional embedding
        position_indicies = torch.arange(input.shape[1], dtype = torch.long, device = input.device) # get positions for a single batch
        position_indicies = position_indicies.unsqueeze(dim = 0).repeat(input.shape[0], 1) # repeat positions across all sequences in batch to size (batch_size, num_bar)
        position_embeddings = self.position_embeddings(position_indicies) # calculate positional embeddings from positions

        # wrangle input
        input += position_embeddings # add positional embeddings to input
        
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
        logits = self.fc_out(output)

        # return final logits
        return logits

##################################################


# HELPER FUNCTION TO GET THE CORRECT MODEL GIVEN ARGS
##################################################

def get_model(args: dict) -> nn.Module:
    """Helper function to return the correct model given arguments as a dictionary."""

    # scrape variables from arguments
    use_transformer = args.get("use_transformer", False)
    input_dim = utils.PREBOTTLENECK_LATENT_EMBEDDING_DIM if args.get("using_prebottleneck_latents", False) else utils.LATENT_EMBEDDING_DIM
    output_dim = utils.EMOTION_N_CLASSES
    prepool = args.get("prepool", False)
    model_name = args.get("model_name")

    # create transformer model
    if use_transformer:
        model = EmotionTransformer(
            input_dim = input_dim, output_dim = output_dim,
            max_seq_len = utils.EMOTION_MAX_SEQ_LEN,
            heads = utils.TRANSFORMER_HEADS,
            layers = utils.TRANSFORMER_LAYERS,
            dropout = utils.TRANSFORMER_DROPOUT,
            feedforward_layers = utils.TRANSFORMER_FEEDFORWARD_LAYERS,
        )

    # create MLP model
    else:
        model = EmotionMLP(
            input_dim = input_dim, output_dim = output_dim,
            prepool = prepool,
            use_large = "large" in model_name, use_small = "small" in model_name,
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

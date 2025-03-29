# README
# Phillip Long
# March 4, 2025

# Prepare a model.

# python /home/pnlong/jingyue_latents/model.py

# IMPORTS
##################################################

import torch
from torch import nn
import math

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
            tokens = torch.zeros(size = (batch_size * num_bar, embedding_dim), dtype = utils.TOKEN_TYPE).to(input.device)

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
            tokens = torch.zeros(size = (batch_size, num_bar, embedding_dim), dtype = utils.TOKEN_TYPE).to(input.device)

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


# MELODY TRANSFORMER
##################################################

def weight_init_normal(weight, normal_std):
  nn.init.normal_(weight, 0.0, normal_std)

def weight_init_orthogonal(weight, gain):
  nn.init.orthogonal_(weight, gain)

def bias_init(bias):
  nn.init.constant_(bias, 0.0)
  
def weights_init(m):
    classname = m.__class__.__name__
    # print ('[{}] initializing ...'.format(classname))
    if classname.find("Linear") != -1:
        if hasattr(m, "weight") and m.weight is not None:
            weight_init_normal(m.weight, 0.01)
        if hasattr(m, "bias") and m.bias is not None:
            bias_init(m.bias)
    elif classname.find("Embedding") != -1:
        if hasattr(m, "weight"):
            weight_init_normal(m.weight, 0.01)
    elif classname.find("LayerNorm") != -1:
        if hasattr(m, "weight"):
            nn.init.normal_(m.weight, 1.0, 0.01)
        if hasattr(m, "bias") and m.bias is not None:
            bias_init(m.bias)
    elif classname.find("GRU") != -1:
        for param in m.parameters():
            if len(param.shape) >= 2:  # weights
                weight_init_orthogonal(param, 0.01)
            else:                      # biases
                bias_init(param)
    # else:
    #   print ('[{}] not initialized !!'.format(classname))

class MelodyPositionalEncoding(nn.Module):
    def __init__(self,
            d_embed: int,
            max_pos: int = 20480,
        ):
        super(MelodyPositionalEncoding, self).__init__()
        self.d_embed = d_embed
        self.max_pos = max_pos

        pe = torch.zeros(max_pos, d_embed)
        position = torch.arange(0, max_pos, dtype = torch.float).unsqueeze(dim = 1)
        div_term = torch.exp(torch.arange(0, d_embed, 2).float() * (-math.log(10000.0) / d_embed))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(dim = 0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, seq_len: int, bsz: int = None):
        pos_encoding = self.pe[:seq_len, :]

        if bsz is not None:
          pos_encoding = pos_encoding.expand(seq_len, bsz, -1)

        return pos_encoding

class MelodyTokenEmbedding(nn.Module):
  def __init__(self, n_token: int, d_embed: int, d_proj: int):
    super(MelodyTokenEmbedding, self).__init__()

    self.n_token = n_token
    self.d_embed = d_embed
    self.d_proj = d_proj
    self.emb_scale = d_proj ** 0.5

    self.emb_lookup = nn.Embedding(n_token, d_embed)
    if d_proj != d_embed:
      self.emb_proj = nn.Linear(d_embed, d_proj, bias = False)
    else:
      self.emb_proj = None

  def forward(self, inp_tokens):
    inp_emb = self.emb_lookup(inp_tokens)
    
    if self.emb_proj is not None:
      inp_emb = self.emb_proj(inp_emb)

    return inp_emb.mul_(self.emb_scale)

class MelodyTransformerEncoder(nn.Module):
    def __init__(self,
            n_layer: int = 12,
            n_head: int = 8,
            d_model: int = 512,
            d_ff: int = 2048,
            d_seg_emb: int = 128,
            dropout: float = 0.1,
            activation: str = "relu",
        ):
        super(MelodyTransformerEncoder, self).__init__()
        self.n_layer = n_layer
        self.n_head = n_head
        self.d_model = d_model
        self.d_ff = d_ff
        self.dropout = dropout
        self.activation = activation

        self.seg_emb_proj = nn.Linear(d_seg_emb, d_model, bias = False)
        self.encoder_layers = nn.ModuleList()
        for i in range(n_layer):
            self.encoder_layers.append(
                nn.TransformerEncoderLayer(d_model, n_head, d_ff, dropout, activation, batch_first = True)
            )

    def forward(self, x, seg_emb, padding_mask = None):
        out = x
        for i in range(self.n_layer):
            out += seg_emb
            out = self.encoder_layers[i](out, src_key_padding_mask = padding_mask)

        return out

class MelodyTransformer(nn.Module):
    def __init__(self,
            enc_n_layer: int,
            enc_n_head: int,
            enc_d_ff: int, 
            d_embed: int,
            d_rvq_latent: int,
            n_token: int,
            n_class: int,
            enc_dropout: float = 0.1,
            enc_activation: str = "relu",
        ):
        super(MelodyTransformer, self).__init__()
        self.enc_n_layer = enc_n_layer
        self.enc_n_head = enc_n_head
        self.enc_d_ff = enc_d_ff
        self.enc_dropout = enc_dropout
        self.enc_activation = enc_activation

        self.d_embed = d_embed
        self.d_rvq_latent = d_rvq_latent
        self.n_token = n_token # input vocabulary, e.g. 140
        self.n_class = n_class # output classes, e.g. 4

        self.token_emb = MelodyTokenEmbedding(n_token = n_token, d_embed = d_embed, d_proj = d_embed)
        self.emb_dropout = nn.Dropout(p = self.enc_dropout)
        self.pe = MelodyPositionalEncoding(d_embed = d_embed)
        
        self.encoder = MelodyTransformerEncoder(
            n_layer = enc_n_layer,
            n_head = enc_n_head,
            d_model = d_embed,
            d_ff = enc_d_ff,
            dropout = enc_dropout, 
            activation = enc_activation,
        )
        self.enc_out_proj = nn.Linear(in_features = d_embed, out_features = n_class)
        
        self.apply(weights_init)
        
    def forward(self, enc_inp, inp_bar_pos, rvq_latent, padding_mask = None):
        # [shape of enc_inp] (bsize, seqlen_per_sample) => (bsize, seqlen_per_sample, d_embed)
        enc_token_emb = self.token_emb(inp_tokens = enc_inp)
        enc_inp = self.emb_dropout(enc_token_emb) + self.pe(seq_len = enc_inp.size(dim = 1), bsz = enc_inp.size(dim = 0)).transpose(0, 1)

        # [shape of rvq_latent] (bsize, n_bars_per_sample, d_rvq_latent)
        # [shape of enc_seg_emb] (bsize, seqlen_per_sample, d_rvq_latent)
        # [shape of inp_bar_pos] (bsize, n_bars_per_sample + 1)
        # -- stores [[start idx of bar #1, sample #1, ..., start idx of bar #K, sample #1, seqlen of sample #1], [same for another sample], ...]
        enc_seg_emb = torch.zeros(enc_inp.size(0), enc_inp.size(1), self.d_rvq_latent).to(enc_inp.device)
        for n in range(enc_inp.size(0)):
            for b, (st, ed) in enumerate(zip(inp_bar_pos[n, :-1], inp_bar_pos[n, 1:])):
                enc_seg_emb[n, st:ed, :] = rvq_latent[n, b, :]

        # [shape of padding_mask] (bsize, seqlen_per_sample)
        # -- should be `True` for padded indices (i.e., those >= seqlen_per_sample), `False` otherwise
        padding_mask = torch.logical_not(input = padding_mask) # negate, since the padding values must be true
        # [shape of enc_out] (bsize, seqlen_per_sample, d_embed)
        enc_out = self.encoder(x = enc_inp, seg_emb = enc_seg_emb, padding_mask = padding_mask)
        
        # [shape of enc_logits] (bsize, seqlen_per_sample)
        enc_logits = self.enc_out_proj(enc_out)
        enc_logits = enc_logits.reshape(-1, enc_logits.shape[-1]) # (bsize * seqlen_per_sample)

        return enc_logits
            
    def compute_loss(self, logits, tgt):
        loss = torch.nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)), tgt.contiguous().view(-1), 
            ignore_index = 0, reduction = "mean",
        ).float()
        return loss

##################################################


# HELPER FUNCTION TO GET THE CORRECT MODEL GIVEN ARGS
##################################################

def get_model(args: dict) -> nn.Module:
    """Helper function to return the correct model given arguments as a dictionary."""

    # get task
    task = args.get("task")

    # scrape variables
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
        case utils.MELODY_TRANSFORMER_DIR_NAME:
            output_dim = utils.N_MELODY_TRANSFORMER_CLASSES
    
    # determine small or large
    use_large = "large" in model_name
    use_small = "small" in model_name
    if use_large and use_small: # don't want them both at once
        use_small = False

    # throw error for illegal scenarios
    if use_transformer and prepool:
        raise RuntimeError("--use_transformer and --prepool cannot both be specified.")

    # special tasks
    if task == utils.MELODY_TRANSFORMER_DIR_NAME:
        size_adjustment_factor = 1.0 # factor by which to adjust the size of the model
        if use_large: # if using large, include more of everything
            size_adjustment_factor = 2.0
        elif use_small: # if using small, include less of everything
            size_adjustment_factor = 0.5
        model = MelodyTransformer(
            enc_n_layer = int(utils.TRANSFORMER_LAYERS * size_adjustment_factor),
            enc_n_head = int(utils.TRANSFORMER_HEADS * size_adjustment_factor),
            enc_d_ff = int(utils.TRANSFORMER_FEEDFORWARD_LAYERS * size_adjustment_factor), 
            d_embed = input_dim,
            d_rvq_latent = input_dim,
            n_token = vocabulary_size,
            n_class = output_dim,
            enc_dropout = utils.TRANSFORMER_DROPOUT,
            enc_activation = "relu",
        )

    # create transformer model
    elif use_transformer:
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

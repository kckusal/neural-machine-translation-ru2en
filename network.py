import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchtext

from data_utils import SRC, TRG

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.hidden_size = hidden_size

    def forward(self, hidden, encoder_outputs, mask):
        # dot score
        attn_scores = torch.sum(hidden * encoder_outputs, dim=2)

        # Transpose max_length and batch_size dimensions
        attn_scores = attn_scores.t()

        # Apply mask so network does not attend <pad> tokens
        attn_scores = attn_scores.masked_fill(mask == 0, -1e5)

        # Return softmax over attention scores
        return F.softmax(attn_scores, dim=1).unsqueeze(1)


class Encoder(nn.Module):
    def __init__(self, hidden_size, embedding_size, num_layers=2, dropout=0.3):

        super(Encoder, self).__init__()

        # Basic network params
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size
        self.num_layers = num_layers
        self.dropout = dropout

        # Embedding layer that will be shared with Decoder
        self.embedding = nn.Embedding(len(SRC.vocab), embedding_size)
        # GRU layer
        self.gru = nn.GRU(embedding_size, hidden_size,
                          num_layers=num_layers,
                          dropout=dropout)

    def forward(self, input_sequence):
        # Convert input_sequence to word embeddings
        embedded = self.embedding(input_sequence)

        outputs, hidden = self.gru(embedded)

        # The ouput of a GRU has shape -> (seq_len, batch, hidden_size)
        return outputs, hidden


class Decoder(nn.Module):
    def __init__(self, embedding_size, hidden_size, output_size, n_layers=2, dropout=0.3):
        super(Decoder, self).__init__()

        # Basic network params
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout = dropout
        self.embedding = nn.Embedding(output_size, embedding_size)

        self.gru = nn.GRU(embedding_size, hidden_size, n_layers,
                          dropout=dropout)

        self.concat = nn.Linear(hidden_size * 2, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.attn = Attention(hidden_size)

    def forward(self, current_token, hidden_state, encoder_outputs, mask):
        # convert current_token to word_embedding
        embedded = self.embedding(current_token)

        # Pass through GRU
        gru_output, hidden_state = self.gru(embedded, hidden_state)

        # Calculate attention weights
        attention_weights = self.attn(gru_output, encoder_outputs, mask)

        # Calculate context vector (weigthed average)
        context = attention_weights.bmm(encoder_outputs.transpose(0, 1))

        # Concatenate  context vector and GRU output
        gru_output = gru_output.squeeze(0)
        context = context.squeeze(1)
        concat_input = torch.cat((gru_output, context), 1)
        concat_output = torch.tanh(self.concat(concat_input))

        # Pass concat_output to final output layer
        output = self.out(concat_output)

        # Return output and final hidden state
        return output, hidden_state


class seq2seq(nn.Module):
    def __init__(self, embedding_size, hidden_size, vocab_size, device, pad_idx, eos_idx, sos_idx):
        super(seq2seq, self).__init__()

        # Embedding layer shared by encoder and decoder
        self.embedding = nn.Embedding(vocab_size, embedding_size)

        # Encoder network
        self.encoder = Encoder(hidden_size, embedding_size,
                               num_layers=2, dropout=0.3)

        # Decoder network
        self.decoder = Decoder(embedding_size, hidden_size,
                               vocab_size, n_layers=2, dropout=0.3)

        # Indices of special tokens and hardware device
        self.pad_idx = pad_idx
        self.eos_idx = eos_idx
        self.sos_idx = sos_idx
        self.device = device

    def create_mask(self, input_sequence):
        return (input_sequence != self.pad_idx).permute(1, 0)

    def forward(self, input_sequence, output_sequence):

        # Unpack input_sequence tuple
        input_tokens = input_sequence[0]

        # Unpack output_tokens, or create an empty tensor for text generation
        if output_sequence is None:
            inference = True
            output_tokens = torch.zeros((100, input_tokens.shape[1])).long().fill_(
                self.sos_idx).to(self.device)
        else:
            inference = False
            output_tokens = output_sequence[0]

        vocab_size = self.decoder.output_size
        batch_size = len(input_sequence[1])
        max_seq_len = len(output_tokens)

        # tensor to store decoder outputs
        outputs = torch.zeros(max_seq_len, batch_size,
                              vocab_size).to(self.device)

        # pass input sequence to the encoder
        encoder_outputs, hidden = self.encoder(input_tokens)

        # first input to the decoder is the <sos> tokens
        output = output_tokens[0, :]

        # create mask
        mask = self.create_mask(input_tokens)

        # Step through the length of the output sequence one token at a time
        for t in range(1, max_seq_len):
            output = output.unsqueeze(0)

            output, hidden = self.decoder(
                output, hidden, encoder_outputs, mask)
            outputs[t] = output

            if inference:
                output = output.max(1)[1]
            else:
                output = output_tokens[t]

            # If we're in inference mode, keep generating until we produce an
            # <eos> token
            if inference and output.item() == self.eos_idx:
                return outputs[:t]

        return outputs

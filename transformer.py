import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

NEG_INFTY = -1e9

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

#####################################################################################################
# Encoder Arch
# #############################################################################################

def scaled_dot_product(q, k, v, mask=None):
    d_k = q.size()[-1]
    scaled = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(d_k)
    if mask is not None:
        scaled = scaled.permute(1, 0, 2, 3) + mask
        scaled = scaled.permute(1, 0, 2, 3)
    attention = F.softmax(scaled, dim=-1)
    values = torch.matmul(attention, v)
    return values, attention

class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, hidden, drop_prob=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, hidden) #512 x 2048
        self.linear2 = nn.Linear(hidden, d_model) #2048 x 512
        self.relu = nn.ReLU()       
        self.dropout = nn.Dropout(p=drop_prob)

    def forward(self, x):   # 30 x 200 x 512
        x = self.linear1(x) # 30 x 200 x 2048
        x = self.relu(x)    # 30 x 200 x 2048
        x = self.dropout(x) # 30 x 200 x 2048
        x = self.linear2(x) # 30 x 200 x 512
        return x

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_sequence_length):
        super().__init__()
        self.max_sequence_length = max_sequence_length
        self.d_model = d_model
    
    def forward(self):
        even_i = torch.arange(0, self.d_model, 2).float()
        denominator = torch.pow(10000, even_i / self.d_model)
        position = torch.arange(self.max_sequence_length).type(torch.float).unsqueeze(1)
        even_PE = torch.sin(position / denominator)
        odd_PE = torch.cos(position / denominator)
        PE = torch.zeros((self.max_sequence_length, self.d_model))
        PE[:, ::2] = even_PE
        PE[:, 1::2] = odd_PE
        return PE
    
class SentenceEmbedding(nn.Module):
    "For a given sentence, create an embedding"
    def __init__(self, max_sequence_length, d_model, language_to_index, START_TOKEN, END_TOKEN, PADDING_TOKEN):
        super().__init__()
        self.vocab_size = len(language_to_index)
        self.max_sequence_length = max_sequence_length
        self.embedding = nn.Embedding(self.vocab_size, d_model)
        self.language_to_index = language_to_index
        self.position_encoder= PositionalEncoding(d_model, max_sequence_length)
        self.dropout = nn.Dropout(p=0.1)
        self.START_TOKEN = START_TOKEN
        self.END_TOKEN = END_TOKEN
        self.PADDING_TOKEN = PADDING_TOKEN
    
    def batch_tokenize(self, batch, start_token = True, end_token = True):

        def tokenize(sentence, start_token=True, end_token=True):
            sentence_word_indicies = [self.language_to_index[token] for token in list(sentence)]
            if start_token:
                sentence_word_indicies.insert(0, self.language_to_index[self.START_TOKEN])
            if end_token:
                sentence_word_indicies.append(self.language_to_index[self.END_TOKEN])
            for _ in range(len(sentence_word_indicies), self.max_sequence_length):
                sentence_word_indicies.append(self.language_to_index[self.PADDING_TOKEN])
            return torch.tensor(sentence_word_indicies)

        tokenized = []
        for sentence_num in range(len(batch)):
            tokenized.append( tokenize(batch[sentence_num], start_token, end_token) )
        tokenized = torch.stack(tokenized)
        return tokenized.to(get_device())

    def forward(self, x, start_token, end_token):
        x = self.batch_tokenize(x, start_token, end_token)
        x = self.embedding(x)
        pos = self.position_encoder().to(get_device())
        x = self.dropout(x + pos)
        return x

class MultiheadAttention(nn.Module):
    def __init__(self,d_model, num_heads):
        super().__init__()
        self.d_model = d_model  ## attention head embedding size
        # assert d_model % num_heads == 0, "d_model should be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.qkv_linear = nn.Linear(d_model, 3 * d_model)
        self.linear_layer = nn.Linear(d_model, d_model)
    
    def forward(self, x, mask = None):
        batch_size, sequence_length, d_model = x.size()
        # print(f"x.size(),{x.size()}")
        qkv = self.qkv_linear(x) # (batch_size, seq_len, 3 * d_model)
        # print(f"qkv.size(),{qkv.size()}")
        qkv = qkv.reshape(batch_size, sequence_length, self.num_heads, 3*self.head_dim)
        # print(f"qkv.size(), {qkv.size()}")
        qkv = qkv.permute(0, 2, 1, 3) # (batch_size, num_heads, seq_len, head_dim)
        # print(f"qkv.size(), {qkv.size()}")
        q, k, v = qkv.chunk(3, dim=-1)
        # print(f"q.size(), {q.size()}", f"k.size(), {k.size()}", f"v.size(), {v.size()}")
        values, attention_weights = scaled_dot_product(q, k, v, mask)
        # print(f"values.size(), {values.size()}", f"attention_weights.size(), {attention_weights.size()}")
        values = values.permute(0, 2, 1, 3) 
        values = values.reshape(batch_size, sequence_length, self.d_model)
        # print(f"values.size(), {values.size()}")
        out = self.linear_layer(values)
        # print(f"out.size(), {out.size()}")
        return out
    
class LayerNormalization(nn.Module):
    def __init__(self, parameters_shape, eps=1e-5):
        super().__init__()
        self.parameters_shape=parameters_shape # [512]
        self.eps=eps
        self.gamma = nn.Parameter(torch.ones(parameters_shape)) #learnable parameters [512] sd
        self.beta =  nn.Parameter(torch.zeros(parameters_shape)) #learnable parameters [512] mean

    def forward(self, inputs): # 30 x 200 x 512
        dims = [-(i + 1) for i in range(len(self.parameters_shape))] # [-1] last dimension
        mean = inputs.mean(dim=dims, keepdim=True) # mean of all 512 values per word 30 x 200 x 1
        # print(f"Mean \n ({mean.size()}): \n {mean}") 
        var = ((inputs - mean) ** 2).mean(dim=dims, keepdim=True)# variance of all 512 values per word 30 x 200 x 1
        std = (var + self.eps).sqrt() #sd
        # print(f"Standard Deviation \n ({std.size()}): \n {std}")
        y = (inputs - mean) / std # 30 x 200 x 512
        # print(f"y \n ({y.size()}) = \n {y}")
        out = self.gamma * y  + self.beta
        # print(f"out \n ({out.size()}) = \n {out}")
        return out   
    
class EncoderLayer(nn.Module):
    def __init__(self, d_model, ffn_hidden, num_heads, drop_prob):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiheadAttention(d_model=d_model, num_heads = num_heads)
        self.norm1 = LayerNormalization(parameters_shape=[d_model])
        self.dropout1 = nn.Dropout(p=drop_prob)
        self.ffn = PositionwiseFeedForward(d_model=d_model, hidden = ffn_hidden, drop_prob = drop_prob)
        self.norm2 = LayerNormalization(parameters_shape=[d_model])
        self.dropout2 = nn.Dropout(p=drop_prob)

    def forward(self, x, self_attention_mask):
        residual_x = x.clone()
        x = self.self_attn(x, mask=self_attention_mask) # 30 x 200 x 512
        x = self.dropout1(x) # 30 x 200 x 512
        x = self.norm1(x + residual_x) # 30 x 200 x 512
        residual_x = x.clone()
        x = self.ffn(x) 
        x = self.dropout2(x)
        x = self.norm2(x + residual_x)
        return x

class SequentialEncoder(nn.Sequential):
    def forward(self, *inputs):
        x, self_attention_mask = inputs
        for module in self._modules.values():
            x = module(x, self_attention_mask)
        return x
    
class Encoder(nn.Module):
    def __init__(self, 
                 d_model, 
                 ffn_hidden, 
                 num_heads, 
                 drop_prob, 
                 num_layers,
                 max_sequence_length,
                 language_to_index,
                 START_TOKEN,
                 END_TOKEN,
                 PADDING_TOKEN):
        super().__init__()
        
        self.sentence_embedding = SentenceEmbedding(max_sequence_length, d_model, language_to_index, START_TOKEN, END_TOKEN, PADDING_TOKEN)
        self.layers = SequentialEncoder(*[EncoderLayer(d_model, ffn_hidden, num_heads, drop_prob) 
                                      for _ in range(num_layers)]) # num_layers EncoderLayer

    def forward(self, x, self_attention_mask, start_token, end_token): #self_attention_mask is passed
                                                                       # here as it changes for every input
        
        x = self.sentence_embedding(x, start_token, end_token)
        x = self.layers(x, self_attention_mask) # the mask is used to eliminate the padded words
        return x
    
#####################################################################################################
# Decoder Arch
# ############################################################################################# 

class MultiHeadCrossAttention(nn.Module):
    def __init__(self,d_model, num_heads):
        super().__init__()
        self.d_model = d_model  ## attention head embedding size
        # assert d_model % num_heads == 0, "d_model should be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.kv_linear = nn.Linear(d_model, 2 * d_model)
        self.q_linear = nn.Linear(d_model, d_model)
        self.linear_layer = nn.Linear(d_model, d_model)
    
    def forward(self, x, y, mask = None):
        batch_size, sequence_length, d_model = x.size() # 30 x 200 x 512
        # print(f"x.size(),{x.size()}")
        
        kv = self.kv_linear(x) # (batch_size, seq_len, 2 * d_model) 30 x 200 x 1024
        # print(f"qkv.size(),{kv.size()}")
        kv = kv.reshape(batch_size, sequence_length, self.num_heads, 2*self.head_dim) # 30 x 200 x 8 x 128
        # print(f"qkv.size(), {kv.size()}")
        kv = kv.permute(0, 2, 1, 3) # (batch_size, num_heads, seq_len, head_dim) # 30 x 8 x 200 x 128
        # print(f"qkv.size(), {kv.size()}")

        q = self.q_linear(y) # (batch_size, seq_len, d_model) 30 x 200 x 512
        # print(f"qkv.size(),{q.size()}")
        q = q.reshape(batch_size, sequence_length, self.num_heads, self.head_dim) # 30 x 200 x 8 x 64
        # print(f"qkv.size(), {kv.size()}")
        q = q.permute(0, 2, 1, 3) # (batch_size, num_heads, seq_len, head_dim) # 30 x 8 x 200 x 64
        # print(f"qkv.size(), {q.size()}")

        k, v = kv.chunk(2, dim=-1) # 30 x 8 x 200 x 64 each
        # print(f"q.size(), {q.size()}", f"k.size(), {k.size()}", f"v.size(), {v.size()}")

        values, attention_weights = scaled_dot_product(q, k, v, mask) # values:  30 x 8 x 200 x 192, mask: 30 x 8 x 200 x 200
        # print(f"values.size(), {values.size()}", f"attention_weights.size(), {attention_weights.size()}")
        values = values.permute(0, 2, 1, 3) 
        values = values.reshape(batch_size, sequence_length, self.d_model) # 30 x 200 x 512
        # print(f"values.size(), {values.size()}")
        out = self.linear_layer(values) # 30 x 200 x 512
        # print(f"out.size(), {out.size()}")
        return out

class DecoderLayer(nn.Module):
    def __init__(self, d_model, ffn_hidden, num_heads, drop_prob):
        super(DecoderLayer, self).__init__()
        self.self_attn = MultiheadAttention(d_model, num_heads)
        self.norm1 = LayerNormalization(parameters_shape = [d_model])
        self.dropout1 = nn.Dropout(p=drop_prob)
        self.enc_dec_attention = MultiHeadCrossAttention(d_model, num_heads)
        self.norm2 = LayerNormalization(parameters_shape = [d_model])
        self.dropout2 = nn.Dropout(p=drop_prob)
        self.ffn = PositionwiseFeedForward(d_model, ffn_hidden, drop_prob)
        self.norm3 = LayerNormalization(parameters_shape = [d_model])
        self.dropout3 = nn.Dropout(p=drop_prob)
    
    def forward(self, x, y, self_attention_mask, cross_attention_mask):
        _y = y.clone()
        y = self.self_attn(y, mask=self_attention_mask)
        y = self.dropout1(y)
        y = self.norm1(y + _y)

        _y = y.clone()
        y = self.enc_dec_attention(x, y, mask=cross_attention_mask)
        y = self.dropout2(y)
        y = self.norm2(y + _y)

        _y = y.clone()
        y = self.ffn(y)
        y = self.dropout3(y)
        y = self.norm3(y + _y)
        return y

class SequentialDecoder(nn.Sequential):
    def forward(self, *inputs):
        x, y, self_attention_mask, cross_attention_mask = inputs
        for module in self._modules.values():
            y = module(x, y, self_attention_mask, cross_attention_mask)
        return y
    
class Decoder(nn.Module):
    def __init__(self, 
                 d_model, 
                 ffn_hidden, 
                 num_heads, 
                 drop_prob, 
                 num_layers,
                 max_sequence_length,
                 language_to_index,
                 START_TOKEN,
                 END_TOKEN, 
                 PADDING_TOKEN):
        super().__init__()
        self.sentence_embedding = SentenceEmbedding(max_sequence_length, d_model, language_to_index, START_TOKEN, END_TOKEN, PADDING_TOKEN)
        self.layers = SequentialDecoder(*[DecoderLayer(d_model, ffn_hidden, num_heads, drop_prob) for _ in range(num_layers)])

    def forward(self, x, y, self_attention_mask, cross_attention_mask, start_token, end_token):
        y = self.sentence_embedding(y, start_token, end_token)
        y = self.layers(x, y, self_attention_mask, cross_attention_mask)
        return y

#####################################################################################################
# Transformer Arch
# #############################################################################################

class Transformer(nn.Module):
    def __init__(self, 
                d_model, 
                ffn_hidden, 
                num_heads, 
                drop_prob, 
                num_layers,
                max_sequence_length, 
                kn_vocab_size,
                english_to_index,
                kannada_to_index,
                START_TOKEN, 
                END_TOKEN, 
                PADDING_TOKEN
                ):
        super().__init__()
        self.encoder = Encoder(d_model, ffn_hidden, num_heads, drop_prob, num_layers, max_sequence_length, english_to_index, START_TOKEN, END_TOKEN, PADDING_TOKEN)
        self.decoder = Decoder(d_model, ffn_hidden, num_heads, drop_prob, num_layers, max_sequence_length, kannada_to_index, START_TOKEN, END_TOKEN, PADDING_TOKEN)
        self.linear = nn.Linear(d_model, kn_vocab_size)
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    def forward(self, 
                x, 
                y, 
                encoder_self_attention_mask=None, 
                decoder_self_attention_mask=None, 
                decoder_cross_attention_mask=None,
                enc_start_token=False,
                enc_end_token=False,
                dec_start_token=False, # We should make this true
                dec_end_token=False): # x, y are batch of sentences
        x = self.encoder(x, encoder_self_attention_mask, start_token=enc_start_token, end_token=enc_end_token)
        out = self.decoder(x, y, decoder_self_attention_mask, decoder_cross_attention_mask, start_token=dec_start_token, end_token=dec_end_token)
        out = self.linear(out)
        return out

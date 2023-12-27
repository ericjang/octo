import torch
import torch.nn as nn
from collections import OrderedDict
from functools import partial
import torch.nn.functional as F


def mha_flax_params_to_state_dict(params):
    # converts flax MHA module to Pytorch state dict
    state_dict = {}
    embed_dim = params['out']['bias'].shape[0]

    # we require the following params:
    # {'in_proj_weight': torch.Size([2304, 768]), 'in_proj_bias': torch.Size([2304]), 
    # 'out_proj.weight': torch.Size([768, 768]), 'out_proj.bias': torch.Size([768])}
    # params[mhdpa] contains
    # dict_keys(['key', 'out', 'query', 'value'])
    # in_proj is k, q, v stacked together
        
    # breakpoint()
    # then stack q, k, v
    # in_proj_weight should be (3*H*C, I)
    # jax kernel param is (I, H, C), convert to (H, C, I) then concat to (3*H, C, I) then reshape to (3*H*C, I)
    
    # (I, 3*H*C)
    in_proj_weight = torch.concat([torch.from_numpy(params[name]['kernel']).reshape((embed_dim, -1)) for name in ['query', 'key', 'value']], dim=1)
    state_dict['in_proj_weight'] = in_proj_weight.T # torch param is (O, I)
    # state_dict['in_proj_weight'] = torch.concat(
    #     [torch.from_numpy(params[name]['kernel']).reshape((embed_dim, -1)) for name in ['query', 'key', 'value']], dim=0)
    state_dict['in_proj_bias'] = torch.concat(
        [torch.from_numpy(params[name]['bias'].flatten()) for name in ['query', 'key', 'value']], dim=0)
    # out kernel is (H, I, O), reshape to (H*I, O)
    # jax param is (H, C, O) where I=(H, C) corresponding to input dimensions
    state_dict['out_proj.weight'] = torch.from_numpy(params['out']['kernel'].reshape((-1, embed_dim))).T # (I, O) -> (O, I)
    state_dict['out_proj.bias'] = torch.from_numpy(params['out']['bias'].flatten())
    return state_dict


class MlpBlock(nn.Module):
    """Transformer MLP / feed-forward block."""
    def __init__(self, in_features, mlp_dim, out_dim=None, dropout_rate=0.1):
        super(MlpBlock, self).__init__()
        self.dense1 = nn.Linear(in_features=in_features, out_features=mlp_dim)
        out_dim = out_dim or in_features  # if out_dim not specified, defaults to in_features
        self.dense2 = nn.Linear(in_features=mlp_dim, out_features=out_dim)
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, inputs):
        x = self.dense1(inputs)
        x = F.gelu(x)
        x = self.dropout(x)
        output = self.dense2(x)
        output = self.dropout(output)
        return output

    @staticmethod
    def flax_params_to_state_dict(params):
        # dict_keys(['Dense_0', 'Dense_1'])
        state_dict = {}
        for prefix, params2_name in zip(['dense1', 'dense2'], ['Dense_0', 'Dense_1']):
            state_dict[f'{prefix}.weight'] = torch.from_numpy(params[params2_name]['kernel']).T # (I, O) -> (O, I)
            state_dict[f'{prefix}.bias'] = torch.from_numpy(params[params2_name]['bias'])
        return state_dict


class Encoder1DBlock(nn.Module):
    def __init__(self, input_channels, mlp_dim, num_heads, dropout_rate=0.1, attention_dropout_rate=0.1):
        super(Encoder1DBlock, self).__init__()
        # Define layers here
        self.layer_norm_attention = nn.LayerNorm(input_channels)
        self.multi_head_attention = nn.MultiheadAttention(
            embed_dim=input_channels, num_heads=num_heads, dropout=attention_dropout_rate,
            batch_first=True
        )
        self.dropout = nn.Dropout(p=dropout_rate)
        self.layer_norm_mlp = nn.LayerNorm(input_channels)
        self.mlp_block = MlpBlock(in_features=input_channels, mlp_dim=mlp_dim, dropout_rate=dropout_rate)

    def forward(self, inputs, attention_mask):
        x = self.layer_norm_attention(inputs)
        # print('norm_input', x.max())
        # attn_mask has to be (L, S) or (batch*num_heads, L, S). L is tgt seqlen
        x, _ = self.multi_head_attention(x, x, x, attn_mask=attention_mask)
        x = self.dropout(x)
        x = x + inputs
        y = self.layer_norm_mlp(x)
        y = self.mlp_block(y)

        return x + y

    @staticmethod
    def flax_params_to_state_dict(params):
        # dict_keys(['LayerNorm_0', 'LayerNorm_1', 'MlpBlock_0', 'MultiHeadDotProductAttention_0'])

        state_dict = {}
        
        state_dict['layer_norm_attention.weight'] = torch.from_numpy(params['LayerNorm_0']['scale'])
        state_dict['layer_norm_attention.bias'] = torch.from_numpy(params['LayerNorm_0']['bias'])

        state_dict['layer_norm_mlp.weight'] = torch.from_numpy(params['LayerNorm_1']['scale'])
        state_dict['layer_norm_mlp.bias'] = torch.from_numpy(params['LayerNorm_1']['bias'])

        sd = MlpBlock.flax_params_to_state_dict(params['MlpBlock_0'])
        for key, value in sd.items():
            state_dict[f'mlp_block.{key}'] = value

        for name, value in mha_flax_params_to_state_dict(params['MultiHeadDotProductAttention_0']).items():
            state_dict[f'multi_head_attention.{name}'] = value
        return state_dict


class Transformer(nn.Module):
    # Does not include position embedding functionality

    def __init__(self, token_embedding_size, num_layers, mlp_dim, num_attention_heads, dropout_rate=0.1, attention_dropout_rate=0.1, add_position_embedding=False):
        del add_position_embedding
        super(Transformer, self).__init__()
        self.encoder_blocks = nn.ModuleDict({
            f'encoderblock_{lyr}': Encoder1DBlock(
                input_channels=token_embedding_size,
                mlp_dim=mlp_dim,
                dropout_rate=dropout_rate,
                attention_dropout_rate=attention_dropout_rate,
                num_heads=num_attention_heads
            ) for lyr in range(num_layers)
        })
        self.encoder_norm = nn.LayerNorm(token_embedding_size)

    def forward(self, x, attention_mask):
        assert x.dim() == 3  # (batch, len, emb)

        # Input Encoder
        for encoder_block in self.encoder_blocks.values():
            x = encoder_block(x, attention_mask)
        encoded = self.encoder_norm(x)

        return encoded

    @staticmethod
    def flax_params_to_state_dict(params):
        """
        dict_keys([
            'encoder_norm', 
            'encoderblock_0', 
            'encoderblock_1', 
            'encoderblock_10', 
            'encoderblock_11', 
            'encoderblock_2', 
            'encoderblock_3', 
            'encoderblock_4', 
            'encoderblock_5', 
            'encoderblock_6', 
            'encoderblock_7', 
            'encoderblock_8', 
            'encoderblock_9'])
        """
        state_dict = {}
        for jax_layer, params2 in params.items():
            if 'encoderblock' in jax_layer:
                sd = Encoder1DBlock.flax_params_to_state_dict(params2)
                prefix = f'encoder_blocks.{jax_layer}.'
            elif jax_layer == 'encoder_norm':
                sd = {
                    'weight': torch.from_numpy(params2['scale']),
                    'bias': torch.from_numpy(params2['bias']),
                }
                prefix = f'{jax_layer}.'
            else:
                raise ValueError(f'Unrecognized jax variable {jax_layer}')
            for key, value in sd.items():
                state_dict[f'{prefix}{key}'] = value
        return state_dict


class BlockTransformer(nn.Module):
    """Passes image through small convnet and then flattens patch embeddings to get a sequence of tokens.
    """
    def __init__(self):
        super(BlockTransformer, self).__init__()
        self.encoder = vit_encoders.SmallStem16(return_intermediate_features=True)

    def forward(self, observations, tasks, train=True):
        # observations['image_primary'] is BTCHW
        # tasks['image_primary'] is BTCHW 
        enc_inputs = torch.cat([observations['image_primary'], tasks['image_primary']], dim=2)
        b, t, c, h, w = enc_inputs.shape
        enc_inputs = enc_inputs.view(b * t, c, h, w)
        image_tokens, intermediate_features = self.encoder(enc_inputs) # (B*T, C, H, W)
        _, c, h, w = image_tokens.shape
        image_tokens = image_tokens.view(b, t, c, -1).permute(0, 1, 3, 2) # (B, T, H*W, C)
        return image_tokens, intermediate_features

    
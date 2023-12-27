import torch
import torch.nn as nn
from collections import OrderedDict
from functools import partial
import torch.nn.functional as F

def normalize_images(img, img_norm_type="default"):
    if img_norm_type == "default":
        # put pixels in [-1, 1]
        return img.float() / 127.5 - 1.0
    raise ValueError()


def weight_standardize(w, dim, eps):
    """Subtracts mean and divides by standard deviation."""
    w = w - torch.mean(w, dim=dim, keepdim=True)
    w = w / (torch.std(w, dim=dim, keepdim=True, correction=0) + eps)
    return w


class StdConv(nn.Conv2d):
    """Convolution with weight standardization."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x):
        # self.weight is OIHW, normalize against IHW channels
        weight = weight_standardize(self.weight, dim=[1, 2, 3], eps=1e-5)
        return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

    @staticmethod
    def flax_params_to_state_dict(params):
        return {
            'weight': torch.from_numpy(params['kernel']).permute(3, 2, 0, 1), # HWIO -> OIHW
            'bias': torch.from_numpy(params['bias']),
        }


class SmallStem(nn.Module):
    # No FiLM
    patch_size: int = 32
    kernel_sizes: tuple = (3, 3, 3, 3)
    strides: tuple = (2, 2, 2, 2)
    features: tuple = (32, 96, 192, 384)
    padding: tuple = (1, 1, 1, 1)
    num_features: int = 512
    img_norm_type: str = "default"
    input_channels: int = 6

    def __init__(self, return_intermediate_features=False):
        super(SmallStem, self).__init__()

        self.return_intermediate_features = return_intermediate_features
        layers = []
        input_features = self.input_channels
        
        for n, (kernel_size, stride, out_channels, padding) in enumerate(zip(self.kernel_sizes, self.strides, self.features, self.padding)):
            layers.extend([
                (f'StdConv_{n}', StdConv(in_channels=input_features, out_channels=out_channels,
                                        kernel_size=kernel_size, stride=stride, padding=padding)),
                (f'GroupNorm_{n}', nn.GroupNorm(num_groups=32, num_channels=out_channels, eps=1e-6)),
                (f'ReLU_{n}', nn.ReLU(inplace=False)),  # if this is True, the jax-pt test fails because we compare ReLU(GroupNorm(x)) to GroupNorm(x)
            ])
            input_features = out_channels
        self.conv_layers = nn.Sequential(OrderedDict(layers))

        if return_intermediate_features:
            for layer_name, layer in self.conv_layers.named_children():
                layer.register_forward_hook(partial(self.forward_hook, module_name=layer_name))

        self.embedding = nn.Conv2d(
            in_channels=input_features, 
            out_channels=self.num_features, 
            kernel_size=(self.patch_size // 16, self.patch_size // 16), 
            stride=(self.patch_size // 16, self.patch_size // 16), 
            padding="valid")
        self.embedding.register_forward_hook(partial(self.forward_hook, module_name='embedding'))

    def forward(self, observations, train=True, cond_var=None):
        self.intermediate_features = OrderedDict()
        x = normalize_images(observations, self.img_norm_type)
        self.intermediate_features['normalized_x'] = x
        for layer in self.conv_layers:
            x = layer(x)
        x = self.embedding(x)
        if self.return_intermediate_features:
            return x, self.intermediate_features
        else:
            return x
    
    def forward_hook(self, module, input, output, module_name):
        self.intermediate_features[module_name] = output

    @staticmethod
    def flax_params_to_state_dict(params):
        # static method to convert nested flax dictionary of params into a state_dict that 
        # can be loaded 
        # params is a dict of flax params
        # convert to pytorch params
        # then load
        """
        pytorch state dict:
        odict_keys([
            'conv_layers.Conv0.weight', 
            'conv_layers.Conv0.bias', 
            'conv_layers.GroupNorm0.weight', 
            'conv_layers.GroupNorm0.bias', 
            'conv_layers.Conv1.weight', 
            'conv_layers.Conv1.bias', 
            'conv_layers.GroupNorm1.weight', 
            'conv_layers.GroupNorm1.bias', 
            'conv_layers.Conv2.weight', 
            'conv_layers.Conv2.bias', 
            'conv_layers.GroupNorm2.weight', 
            'conv_layers.GroupNorm2.bias', 
            'conv_layers.Conv3.weight', 
            'conv_layers.Conv3.bias', 
            'conv_layers.GroupNorm3.weight', 
            'conv_layers.GroupNorm3.bias', 
            'embedding.weight', 
            'embedding.bias'])

        jax nested dict:
        ['GroupNorm_0', 'GroupNorm_1', 'GroupNorm_2', 'GroupNorm_3', 'StdConv_0', 'StdConv_1', 'StdConv_2', 'StdConv_3', 'embedding']
        """
        state_dict = {}

        prefix = 'conv_layers'
        for jax_layer, params2 in params.items():
            if 'StdConv' in jax_layer:
                sd = StdConv.flax_params_to_state_dict(params2)
                prefix = f'conv_layers.{jax_layer}.'
            elif 'GroupNorm' in jax_layer:
                sd = {
                    'weight': torch.from_numpy(params2['scale']),
                    'bias': torch.from_numpy(params2['bias']),
                }
                prefix = f'conv_layers.{jax_layer}.'
            elif jax_layer == 'embedding':
                sd = {
                    'weight': torch.from_numpy(params2['kernel']).permute(3, 2, 0, 1), # HWIO -> OIHW
                    'bias': torch.from_numpy(params2['bias']),
                }
                prefix = f'{jax_layer}.'
            else:
                raise ValueError(f'Unrecognized jax variable {jax_layer}')
            for key, value in sd.items():
                state_dict[f'{prefix}{key}'] = value # conv_layers.StdConv{n}.weight, conv_layers.StdConv{n}.bias
        return state_dict
        # for n in range(len(self.kernel_sizes)):
        #     state_dict[f'conv_layers.StdConv{n}.weight'] = torch.from_numpy(params[f'StdConv_{n}']['kernel']).permute(3, 2, 0, 1) # HWIO -> OIHW
        #     state_dict[f'conv_layers.StdConv{n}.bias'] = torch.from_numpy(params[f'StdConv_{n}']['bias'])
        #     state_dict[f'conv_layers.GroupNorm{n}.weight'] = torch.from_numpy(params[f'GroupNorm_{n}']['scale'])
        #     state_dict[f'conv_layers.GroupNorm{n}.bias'] = torch.from_numpy(params[f'GroupNorm_{n}']['bias'])
        # state_dict['embedding.weight'] = 
        # state_dict['embedding.bias'] = 

        # self.load_state_dict(state_dict)
        
        # pass


class SmallStem16(SmallStem):
    patch_size: int = 16

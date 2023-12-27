import torch
import torch.nn as nn
from octo.model.pt_components import vit_encoders

class ImageTokenizer(nn.Module):
    """Passes image through small convnet and then flattens patch embeddings to get a sequence of tokens.
    """
    def __init__(self):
        super(ImageTokenizer, self).__init__()
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

    @staticmethod
    def flax_params_to_state_dict(params):
        return {f'encoder.{k}': v for k, v in vit_encoders.SmallStem16.flax_params_to_state_dict(params['SmallStem16_0']).items()}
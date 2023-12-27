import logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
import os
import sys
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
from PIL import Image
import requests
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from absl.testing import absltest
from functools import partial
from collections import OrderedDict
# Jax ecosystem
import orbax.checkpoint
import jax
import flax
import tensorflow as tf
from octo.model.octo_model import OctoModel
from octo.model.components import vit_encoders
from octo.model.components import transformer
from flax import linen as jax_nn
import jax.numpy as jnp

# pytorch ecosystem
import torch
from octo.model.pt_components import vit_encoders as pt_vit_encoders
from octo.model.pt_components import tokenizers as pt_tokenizers
from octo.model.pt_components import transformer as pt_transformer
import torch.nn.functional as F

import torch.nn as nn

# Plotting 
import matplotlib.pyplot as plt

class TestPytorchModules(absltest.TestCase):
    """
    Verify that Pytorch port gives same numerical predictions as Jax.

    """
    def setUp(self):
        self.model = OctoModel.load_pretrained("hf://rail-berkeley/octo-base")

    # 
    # IMAGE INPUTS
    # 

    def _get_img(self):
        # download one example BridgeV2 image
        IMAGE_URL = "https://rail.eecs.berkeley.edu/datasets/bridge_release/raw/bridge_data_v2/datacol2_toykitchen7/drawer_pnp/01/2023-04-19_09-18-15/raw/traj_group0/traj0/images0/im_12.jpg"
        src_img_path = Path('im_12.jpg')
        if src_img_path.exists():
            src_img = Image.open(src_img_path)
        else:
            src_img = Image.open(requests.get(IMAGE_URL, stream=True).raw)
            src_img.save(src_img_path)
        return np.array(src_img.resize((256, 256)))

    def _get_conv_outputs(self):
        x = self._get_img()
        x = jnp.concatenate([x, jnp.zeros_like(x)], axis=2)
        x = x[np.newaxis,...]

        jax_conv_class = vit_encoders.StdConv
        pt_conv_class = pt_vit_encoders.StdConv

        jax_conv = jax_conv_class(
                features=32,
                kernel_size=(3, 3),
                strides=(2, 2),
                padding=1,
            )
        params = self.model.params['octo_transformer']['observation_tokenizers_primary']['SmallStem16_0']['StdConv_0']

        jax_y = jax_conv.apply({'params': params}, vit_encoders.normalize_images(x, "default"))

        # run the same in pytorch
        pt_conv = pt_conv_class(
            in_channels=6,
            out_channels=32,
            kernel_size=(3, 3),
            stride=(2, 2),
            padding=1)

        state_dict = {
            'weight': torch.from_numpy(np.array(params['kernel'])).permute(3, 2, 0, 1), # HWIO -> OIHW
            'bias': torch.from_numpy(np.array(params['bias']))
        }
        pt_conv.load_state_dict(state_dict)

        with torch.no_grad():
            pt_x = torch.from_numpy(np.array(x)).permute(0, 3, 1, 2) # BHWC -> BCHW
            pt_x = pt_vit_encoders.normalize_images(pt_x, img_norm_type="default")
            pt_y = pt_conv(pt_x)
            pt_y_bhwc = pt_y.permute(0, 2, 3, 1).numpy()
        return jax_y, pt_y_bhwc

    def test_restore(self):
        # infer params shape without actually doing any computation
        
        # restore params and count them
        checkpoint_path = '/home/eric/.cache/huggingface/hub/models--rail-berkeley--octo-small/snapshots/03d88976c54a58e10480d2043a8c762b35bc2611/'
        checkpointer = orbax.checkpoint.CheckpointManager(
            checkpoint_path, orbax.checkpoint.PyTreeCheckpointer()
        )
        step = checkpointer.latest_step()
        
        with tf.io.gfile.GFile(
            tf.io.gfile.join(checkpoint_path, "example_batch.msgpack"), "rb"
        ) as f:
            example_batch = flax.serialization.msgpack_restore(f.read())

        params_shape = jax.eval_shape(
            partial(self.model.module.init, train=False),
            jax.random.PRNGKey(0),
            example_batch["observation"],
            example_batch["task"],
            example_batch["observation"]["pad_mask"],
        )["params"]
        
        params = checkpointer.restore(step, params_shape)

    def test_pt_image_tokenizer(self):
        img = self._get_img()
        img = img[np.newaxis,np.newaxis,...] # (B, T, H, W, C)
        observation = {"image_primary": img, "pad_mask": np.array([[True]])}
        task = self.model.create_tasks(texts=["pick up the fork"])
        tokenizer = self.model.module.octo_transformer.observation_tokenizers['primary']
        params = self.model.params['octo_transformer']['observation_tokenizers_primary'] # smallstem is contained within 
        # the tasks cause the input to be channel stacked (1, 256, 256, 6)
        
        # fwd pass of jax encoder
        jax_image_token_group, jax_features = tokenizer.apply(
            {'params': params}, observations=observation, tasks=task)
        jax_image_tokens = jax_image_token_group.tokens # (1, 1, 256, 512)

        pt_tokenizer = pt_tokenizers.ImageTokenizer()
        sd = pt_tokenizers.ImageTokenizer.flax_params_to_state_dict(params)
        pt_tokenizer.load_state_dict(sd)

        with torch.no_grad():
            observation = {"image_primary": torch.tensor(img).permute(0, 1, 4, 2, 3)}
            tasks = {"image_primary": torch.zeros_like(observation['image_primary'])}
            pt_img_tokens, pt_features = pt_tokenizer(observation, tasks)
        pt_img_tokens = pt_img_tokens.numpy()
        np.testing.assert_array_equal(pt_img_tokens.shape, jax_image_tokens.shape)

        layer_names = [
            'normalized_x', 'StdConv_0', 'GroupNorm_0', 'StdConv_1', 
            'GroupNorm_1', 'StdConv_2', 'GroupNorm_2', 'StdConv_3', 'GroupNorm_3', 'embedding']
        diffs = OrderedDict()
        for layer in layer_names:
            pt_bhwc = pt_features[layer].permute(0, 2, 3, 1).numpy()
            diffs[layer] = np.abs(jax_features[layer] - pt_bhwc)
            logging.debug(f'{layer} abs diff mean: {diffs[layer].mean()} std: {diffs[layer].std()} min: {diffs[layer].min()} max: {diffs[layer].max()}')
        
        # plt.boxplot([d.flatten() for d in diffs.values()], labels=layer_names)
        # # plt.bar(layer_names, [np.mean(diffs[layer]) for layer in layer_names])
        # plt.show()

        np.testing.assert_allclose(pt_img_tokens, jax_image_tokens, atol=1e-3)

    def test_stdconv(self):
        jax_y, pt_y_bhwc = self._get_conv_outputs()
        diff = jax_y - pt_y_bhwc
        logging.debug(f'stdconv diff mean: {diff.mean()} std: {diff.std()} ')
        np.testing.assert_allclose(pt_y_bhwc, jax_y, atol=1e-4)

    def test_groupnorm(self):
        # parity between pytorch and jax groupnorm implementations
        x, _ = self._get_conv_outputs()

        jax_norm = jax_nn.GroupNorm()
        # Use first layer groupnorm params instead of random init.
        params = self.model.params['octo_transformer']['observation_tokenizers_primary']['SmallStem16_0']['GroupNorm_0']

        jax_y = jax_norm.apply({'params': params}, x)

        pt_norm = nn.GroupNorm(num_groups=32, num_channels=32, eps=1e-6)
        state_dict = {
            'weight': torch.from_numpy(np.array(params['scale'])),
            'bias': torch.from_numpy(np.array(params['bias']))
        }
        pt_norm.load_state_dict(state_dict)

        with torch.no_grad():
            pt_x = torch.from_numpy(np.array(x)).permute(0, 3, 1, 2) # BHWC -> BCHW
            pt_y = pt_norm(pt_x)
            pt_y_bhwc = pt_y.permute(0, 2, 3, 1).numpy()
        diff = np.abs(jax_y - pt_y_bhwc)
        logging.debug(f'groupnorm abs diff mean: {diff.mean()} std: {diff.std()} min: {diff.min()} max: {diff.max()}')
        np.testing.assert_allclose(pt_y_bhwc, jax_y, atol=1e-5)

    # 
    # LANGUAGE INPUTS
    # 

    def test_language_tokenizer(self):
        """
        LanguageTokenizer is a misnomer, not only do we tokenize text into input ids, we also run it thru T5 encoder 
        to embed into continuous space

        'task_language_pos_embedding', 
        'task_language_projection', 
        'task_tokenizers_language'
        """
        instruction = "pick up the fork and put it down again"
        params = self.model.params['octo_transformer']['task_tokenizers_language']
        task = self.model.create_tasks(texts=[instruction]) # run jax tokenizer
        
        # jax_lang_tok_group.mask is all ones?
        # Instantiate HF tokenizer
        # task['language_instruction']['input_ids']

        # TODO, figure out how to get the right input ids and pass them thru the T5Model
        # to get numpy embeddings without dealing with jax
        # from transformers import T5Tokenizer, T5Model
        # # Note, need pip install sentencepiece for T5 tokenizer to work
        # pt_tokenizer = T5Tokenizer.from_pretrained("t5-base")
        # # model = T5Model.from_pretrained("t5-base")
        # input_ids = pt_tokenizer(
        #     instruction, return_tensors="pt"
        # ).input_ids  # Batch size 1
        
        # fwd pass of jax encoder
        # embeds text input IDs into continuous language embeddings
        tokenizer = self.model.module.octo_transformer.task_tokenizers['language']
        jax_lang_tok_group = tokenizer.apply({'params': params}, observations=None, tasks=task)
        # jax_tokens = jax_lang_tok_group.tokens # (1, 16, 768) # embeddings

        jnp.save('language_embeddings.npy', jax_lang_tok_group.tokens)
        jnp.save('language_mask.npy', jax_lang_tok_group.mask)
        
        # breakpoint()
        # pass
        # decoder_input_ids = tokenizer("Studies show that", return_tensors="pt").input_ids  # Batch size 1


        # transformers.models.t5.configuration_t5.T5Config
        # from transformers import AutoConfig, FlaxAutoModel, FlaxT5EncoderModel

        # config = AutoConfig.from_pretrained(tokenizer.encoder)
        # # assert tokenizer.proper_pad_mask
        # hf_encoder = FlaxT5EncoderModel(config).module
        # jax_tokens2 = hf_encoder(**task["language_instruction"]).last_hidden_state
        # # from transformers import T5Tokenizer, T5ForConditionalGeneration

        # # tokenizer = T5Tokenizer.from_pretrained("t5-base")
        # # model = T5ForConditionalGeneration.from_pretrained("t5-base")

        # breakpoint()
        # pass
        # see if pretrained T5 

        # pt_tokenizer = pt_tokenizers.LanguageTokenizer()
        # pt_tokenizer.load_flax_state_dict(params)

    # 
    # TRANSFORMER
    #
    def test_mha(self):
        # Test that MHA layer behaves the same between jax and PyTorch
        # Load saved inputs.
        intermediates = jnp.load('octo_intermediates.npz', allow_pickle=True)['arr_0'].item()
        x, = intermediates['octo_transformer']['BlockTransformer_0']['Transformer_0']['input']
        attention_mask, = intermediates['octo_transformer']['BlockTransformer_0']['Transformer_0']['attention_mask'] # [batch, heads, query len, key/value len], (1, 1, 273, 273)

        # test zero input
        # x = jnp.zeros_like(x)
        # jax attention_mask is [B, H, S, L]
        # PT attention mask is [B*H, L, S]
        # [batch_sizes..., num_heads, query_length, key/value_length]
        params = self.model.params['octo_transformer']['BlockTransformer_0']['Transformer_0']['encoderblock_0']['MultiHeadDotProductAttention_0']
        
        num_heads = 12
        jax_mha = jax_nn.MultiHeadDotProductAttention(
            broadcast_dropout=False,
            dropout_rate=0.,
            num_heads=num_heads,
        )

        # Random init
        # params = jax_mha.init(jax.random.PRNGKey(0), x)['params']
        # params = jax.tree_util.tree_map(lambda x: np.array(x), params)
        
        # this test passes when kernel is set to 0.
        # implying that in_proj_weight is not being set properly
        # for k in ['out']:
        #     # params[k]['bias'] = np.zeros_like(params[k]['bias'])
        #     params[k]['kernel'] = np.ones_like(params[k]['kernel'])
        
        # https://flax.readthedocs.io/en/latest/_modules/flax/linen/attention.html#MultiHeadDotProductAttention.__call__
        jax_y, mod_vars = jax_mha.apply(
            {'params': params}, inputs_q=x, inputs_k=x, inputs_v=x, mask=attention_mask, sow_weights=True, mutable='intermediates')
        # softmax-normalized attention weights from dot_product_attention_weights
        # probably are (H, S, L)
        attention_weights = mod_vars['intermediates']['attention_weights'][0] # (1, 12, 273, 273)
        jax_q, jax_k, jax_v = mod_vars['intermediates']['query_scaled'][0], mod_vars['intermediates']['key2'][0], mod_vars['intermediates']['value2'][0]
        pt_mha = nn.MultiheadAttention(
            embed_dim=768, num_heads=num_heads, dropout=0.,
            batch_first=True
        )
        state_dict = pt_transformer.mha_flax_params_to_state_dict(params)
        pt_mha.load_state_dict(state_dict)
        with torch.no_grad():
            pt_x = torch.from_numpy(np.array(x))
            # important gotcha: in PyTorch, True = masked out (NOT allowed to attend)
            pt_attention_mask = torch.logical_not(torch.from_numpy(np.array(attention_mask)))[0, 0].T # (L, S)
            # pt_attention_mask = torch.logical_not(torch.tril(torch.ones_like(pt_attention_mask)))
            pt_y, attn_output_weights = pt_mha(pt_x, pt_x, pt_x, attn_mask=pt_attention_mask, need_weights=False)
        
            # (seq, batch, features)
            pt_x = pt_x.permute(1, 0, 2)
            # breakpoint()
            # pt_q, k, v are each of size torch.Size([273, 1, 768])
            pt_y2, attn_output_weights2, (pt_q, pt_q_scaled, pt_k, pt_v) = F.multi_head_attention_forward(
                        query=pt_x,
                        key=pt_x,
                        value=pt_x,
                        attn_mask=pt_attention_mask,
                        embed_dim_to_check=768,
                        num_heads=num_heads,
                        in_proj_weight=state_dict['in_proj_weight'],
                        in_proj_bias=state_dict['in_proj_bias'],
                        out_proj_weight=state_dict['out_proj.weight'],
                        out_proj_bias=state_dict['out_proj.bias'],
                        bias_k=None,
                        bias_v=None,
                        add_zero_attn=False,
                        dropout_p=0.0,
                        average_attn_weights=False)
            # pt_q is torch.Size([12, 273, 64])
            pt_q_scaled = pt_q_scaled.permute(1, 0, 2)[None].numpy() # (1, 273, 12, 64)
            np.testing.assert_allclose(pt_q_scaled, jax_q, atol=1e-4, err_msg='q_proj(x) not close')

            pt_k = pt_k.permute(1, 0, 2)[None].numpy() # (1, 273, 12, 64)
            np.testing.assert_allclose(pt_k, jax_k, atol=3e-4, err_msg='k_proj(x) not close')
            
            pt_v = pt_v.permute(1, 0, 2)[None].numpy() # (1, 273, 12, 64)
            np.testing.assert_allclose(pt_v, jax_v, atol=1e-3, err_msg='v_proj(x) not close')
            
            # (heads, L, S)
            # 9-11% of elements are mismatched between attention weights! 
            # pt is (batch, heads, tgt_len, src_len), convert to (batch, heads, src, tgt)
            pt_attn_np = attn_output_weights2.permute(0, 1, 3, 2).numpy()
            # diff = np.abs(attention_weights - pt_attn_np)
            # logging.debug(f'MHA attention abs diff mean: {diff.mean()} std: {diff.std()} min: {diff.min()} max: {diff.max()}')
            np.testing.assert_allclose(pt_attn_np, attention_weights, atol=1e-1, err_msg='attention doesnt match between jax and pt')

            pt_y2 = pt_y2.permute(1, 0, 2) # (T, B, C) -> (B, T, C)
            np.testing.assert_allclose(pt_y.numpy(), pt_y2.numpy(), atol=1e-3, err_msg='pt mha and module dont match')

            np.testing.assert_allclose(jax_y, pt_y2.numpy(), atol=2e-1, err_msg='jax and pytorch dont match mha output')


    def test_transformer(self):
        # this test is not passing yet! 

        # Load saved inputs.
        intermediates = jnp.load('octo_intermediates.npz', allow_pickle=True)['arr_0'].item()
        x, = intermediates['octo_transformer']['BlockTransformer_0']['Transformer_0']['input']
        attention_mask, = intermediates['octo_transformer']['BlockTransformer_0']['Transformer_0']['attention_mask']

        transformer_kwargs = self.model.module.octo_transformer.transformer_kwargs
        # transformer_kwargs = {
        #     'token_embedding_size': 768,
        #     'attention_dropout_rate': 0.0, 
        #     'add_position_embedding': False, 
        #     'num_layers': 12, 
        #     'mlp_dim': 1536, 
        #     'num_attention_heads': 6, 
        #     'dropout_rate': 0.0}
        params = self.model.params['octo_transformer']['BlockTransformer_0']['Transformer_0']
        jax_y = transformer.Transformer(**transformer_kwargs).apply(
            {'params': params}, x=x, attention_mask=attention_mask, train=False) # (1, 273, 768)

        pt_model = pt_transformer.Transformer(
            token_embedding_size=768, **transformer_kwargs)
        pt_model.load_state_dict(
            pt_transformer.Transformer.flax_params_to_state_dict(params))
        with torch.no_grad():
            pt_x = torch.from_numpy(np.array(x))
            pt_attention_mask = torch.logical_not(torch.from_numpy(np.array(attention_mask)))[0, 0].T # (L, S)
            # Test mha layer standalone
            pt_y = pt_model(pt_x, pt_attention_mask)
        np.testing.assert_allclose(jax_y, pt_y.numpy(), atol=1e-3)


    def test_octo_transformer(self):
        pass

    #
    # FULL INTEGRATION TESET
    #

    def test_sample_actions(self):
        img = self._get_img()
        img = img[np.newaxis,np.newaxis,...] # (B, T, H, W, C)
        observation = {"image_primary": img, "pad_mask": np.array([[True]])}
        task = self.model.create_tasks(texts=["pick up the fork"])
        action, intermediates = self.model.sample_actions(
            observation, task, rng=jax.random.PRNGKey(0), return_intermediate_features=True)
        np.savez('octo_intermediates.npz', intermediates)

if __name__ == "__main__":
    absltest.main()
    
"""

Good porting guide
https://flax.readthedocs.io/en/latest/guides/converting_and_upgrading/convert_pytorch_to_flax.html


Parameters we need to port

Pdb) p params.keys()
dict_keys(['heads_action', 'octo_transformer'])
(Pdb) p params['octo_transformer'].keys()
dict_keys([
    'BlockTransformer_0', 
    'obs_primary_pos_embedding', 
    'obs_primary_projection', 
    'obs_wrist_pos_embedding', 
    'obs_wrist_projection', 
    'observation_tokenizers_primary', [done]
    'observation_tokenizers_wrist', 
    'readout_action_pos_embedding', 
    'task_language_pos_embedding', 
    'task_language_projection', 
    'task_tokenizers_language'
])




OctoModule = [OctoTransformer, heads]
OctoModel = [OctoModule, inference code]


"""




"""
octo-small:

{'seed': 42, 'num_steps': 300000, 'save_dir': None, 'model': {'observation_tokenizers': {'primary': {'module': 'octo.model.components.tokenizers', 'name': 'ImageTokenizer', 'args': [], 'kwargs': {'obs_stack_keys': ['image_primary'], 'task_stack_keys': ['image_primary'], 'encoder': {'module': 'octo.model.components.vit_encoders', 'name': 'SmallStem16', 'args': [], 'kwargs': {}}}}, 'wrist': {'module': 'octo.model.components.tokenizers', 'name': 'ImageTokenizer', 'args': [], 'kwargs': {'obs_stack_keys': ['image_wrist'], 'task_stack_keys': ['image_wrist'], 'encoder': {'module': 'octo.model.components.vit_encoders', 'name': 'SmallStem16', 'args': [], 'kwargs': {}}}}}, 'task_tokenizers': {'language': {'module': 'octo.model.components.tokenizers', 'name': 'LanguageTokenizer', 'args': [], 'kwargs': {'encoder': 't5-base', 'finetune_encoder': False}}}, 'heads': {'action': {'module': 'octo.model.components.action_heads', 'name': 'DiffusionActionHead', 'args': [], 'kwargs': {'readout_key': 'readout_action', 'use_map': False, 'pred_horizon': 4, 'action_dim': 7}}}, 'readouts': {'action': 1}, 'token_embedding_size': 384, 'transformer_kwargs': {'attention_dropout_rate': 0.0, 'add_position_embedding': False, 'num_layers': 12, 'mlp_dim': 1536, 'num_attention_heads': 6, 'dropout_rate': 0.0}, 'max_horizon': 10}, 'window_size': 2, 'dataset_kwargs': {'oxe_kwargs': {'data_mix': 'oxe_magic_soup', 'data_dir': 'gs://rail-octo-central2/resize_256_256', 'load_camera_views': ['primary', 'wrist'], 'load_depth': False}, 'traj_transform_kwargs': {'window_size': 2, 'future_action_window_size': 3, 'goal_relabeling_strategy': 'uniform', 'subsample_length': 100, 'task_augment_strategy': 'delete_task_conditioning', 'task_augment_kwargs': {'keep_image_prob': 0.5}}, 'frame_transform_kwargs': {'num_parallel_calls': 200, 'resize_size': {'primary': [256, 256], 'wrist': [128, 128]}, 'image_augment_kwargs': [{'random_resized_crop': {'scale': [0.8, 1.0], 'ratio': [0.9, 1.1]}, 'random_brightness': [0.1], 'random_contrast': [0.9, 1.1], 'random_saturation': [0.9, 1.1], 'random_hue': [0.05], 'augment_order': ['random_resized_crop', 'random_brightness', 'random_contrast', 'random_saturation', 'random_hue']}, {'random_brightness': [0.1], 'random_contrast': [0.9, 1.1], 'random_saturation': [0.9, 1.1], 'random_hue': [0.05], 'augment_order': ['random_brightness', 'random_contrast', 'random_saturation', 'random_hue']}]}, 'traj_transform_threads': 48, 'traj_read_threads': 48, 'shuffle_buffer_size': 500000, 'batch_size': 128, 'balance_weights': True}, 'optimizer': {'learning_rate': {'name': 'rsqrt', 'init_value': 0.0, 'peak_value': 0.0003, 'warmup_steps': 2000, 'timescale': 10000}, 'weight_decay': 0.1, 'clip_gradient': 1.0, 'frozen_keys': ['*hf_model*']}, 'prefetch_num_batches': 0, 'start_step': None, 'log_interval': 100, 'eval_interval': 5000, 'viz_interval': 20000, 'save_interval': 10000, 'val_kwargs': {'val_shuffle_buffer_size': 1000, 'num_val_batches': 16}, 'viz_kwargs': {'eval_batch_size': 128, 'trajs_for_metrics': 100, 'trajs_for_viz': 8, 'samples_per_state': 8}, 'resume_path': None, 'text_processor': {'module': 'octo.data.utils.text_processing', 'name': 'HFTokenizer', 'args': [], 'kwargs': {'encode_with_model': False, 'tokenizer_kwargs': {'max_length': 16, 'padding': 'max_length', 'truncation': True, 'return_tensors': 'np'}, 'tokenizer_name': 't5-base'}}, 'pretrained_loaders': [{'module': 'octo.utils.train_utils', 'name': 'hf_weights_loader', 'args': [], 'kwargs': {'hf_model': 't5-base'}}], 'wandb': {'project': 'octo', 'group': None, 'entity': None}, 'wandb_resume_id': None, 'eval_datasets': ['bridge_dataset']}

octo_transformer = OctoTransformer(
        # attributes
        observation_tokenizers = {'primary': ImageTokenizer(
            # attributes
            encoder = {'module': 'octo.model.components.vit_encoders', 'name': 'SmallStem16', 'args': [], 'kwargs': {}}
            use_token_learner = False
            num_tokens = 8
            conditioning_type = 'none'
            obs_stack_keys = ['image_primary']
            task_stack_keys = ['image_primary']
            task_film_keys = ()
            proper_pad_mask = True
        ), 'wrist': ImageTokenizer(
            # attributes
            encoder = {'module': 'octo.model.components.vit_encoders', 'name': 'SmallStem16', 'args': [], 'kwargs': {}}
            use_token_learner = False
            num_tokens = 8
            conditioning_type = 'none'
            obs_stack_keys = ['image_wrist']
            task_stack_keys = ['image_wrist']
            task_film_keys = ()
            proper_pad_mask = True
        )}
        task_tokenizers = {'language': LanguageTokenizer(
            # attributes
            encoder = 't5-base'
            finetune_encoder = False
            proper_pad_mask = True
        )}
        readouts = {'action': 1}
        transformer_kwargs = {'attention_dropout_rate': 0.0, 'add_position_embedding': False, 'num_layers': 12, 'mlp_dim': 1536, 'num_attention_heads': 6, 'dropout_rate': 0.0}
        token_embedding_size = 384
        max_horizon = 10
    )
    heads = {'action': DiffusionActionHead(
        # attributes
        readout_key = 'readout_action'
        use_map = False
        pred_horizon = 4
        action_dim = 7
        max_action = 5.0
        loss_type = 'mse'
        time_dim = 32
        num_blocks = 3
        dropout_rate = 0.1
        hidden_dim = 256
        use_layer_norm = True
        diffusion_steps = 20
    )}

octo-base:

{'seed': 42, 'num_steps': 300000, 'save_dir': None, 'model': {'observation_tokenizers': {'primary': {'module': 'octo.model.components.tokenizers', 'name': 'ImageTokenizer', 'args': [], 'kwargs': {'obs_stack_keys': ['image_primary'], 'task_stack_keys': ['image_primary'], 'encoder': {'module': 'octo.model.components.vit_encoders', 'name': 'SmallStem16', 'args': [], 'kwargs': {}}}}, 'wrist': {'module': 'octo.model.components.tokenizers', 'name': 'ImageTokenizer', 'args': [], 'kwargs': {'obs_stack_keys': ['image_wrist'], 'task_stack_keys': ['image_wrist'], 'encoder': {'module': 'octo.model.components.vit_encoders', 'name': 'SmallStem16', 'args': [], 'kwargs': {}}}}}, 'task_tokenizers': {'language': {'module': 'octo.model.components.tokenizers', 'name': 'LanguageTokenizer', 'args': [], 'kwargs': {'encoder': 't5-base', 'finetune_encoder': False}}}, 'heads': {'action': {'module': 'octo.model.components.action_heads', 'name': 'DiffusionActionHead', 'args': [], 'kwargs': {'readout_key': 'readout_action', 'use_map': False, 'pred_horizon': 4, 'action_dim': 7}}}, 'readouts': {'action': 1}, 'token_embedding_size': 768, 'transformer_kwargs': {'attention_dropout_rate': 0.0, 'add_position_embedding': False, 'num_layers': 12, 'mlp_dim': 3072, 'num_attention_heads': 12, 'dropout_rate': 0.0}, 'max_horizon': 10}, 'window_size': 2, 'dataset_kwargs': {'oxe_kwargs': {'data_mix': 'oxe_magic_soup', 'data_dir': 'gs://rail-octo-central2/resize_256_256', 'load_camera_views': ['primary', 'wrist'], 'load_depth': False}, 'traj_transform_kwargs': {'window_size': 2, 'future_action_window_size': 3, 'goal_relabeling_strategy': 'uniform', 'subsample_length': 100, 'task_augment_strategy': 'delete_task_conditioning', 'task_augment_kwargs': {'keep_image_prob': 0.5}}, 'frame_transform_kwargs': {'num_parallel_calls': 200, 'resize_size': {'primary': [256, 256], 'wrist': [128, 128]}, 'image_augment_kwargs': [{'random_resized_crop': {'scale': [0.8, 1.0], 'ratio': [0.9, 1.1]}, 'random_brightness': [0.1], 'random_contrast': [0.9, 1.1], 'random_saturation': [0.9, 1.1], 'random_hue': [0.05], 'augment_order': ['random_resized_crop', 'random_brightness', 'random_contrast', 'random_saturation', 'random_hue']}, {'random_brightness': [0.1], 'random_contrast': [0.9, 1.1], 'random_saturation': [0.9, 1.1], 'random_hue': [0.05], 'augment_order': ['random_brightness', 'random_contrast', 'random_saturation', 'random_hue']}]}, 'traj_transform_threads': 48, 'traj_read_threads': 48, 'shuffle_buffer_size': 500000, 'batch_size': 128, 'balance_weights': True}, 'optimizer': {'learning_rate': {'name': 'rsqrt', 'init_value': 0.0, 'peak_value': 0.0003, 'warmup_steps': 2000, 'timescale': 10000}, 'weight_decay': 0.1, 'clip_gradient': 1.0, 'frozen_keys': ['*hf_model*']}, 'prefetch_num_batches': 0, 'start_step': None, 'log_interval': 100, 'eval_interval': 5000, 'viz_interval': 20000, 'save_interval': 10000, 'val_kwargs': {'val_shuffle_buffer_size': 1000, 'num_val_batches': 16}, 'viz_kwargs': {'eval_batch_size': 128, 'trajs_for_metrics': 100, 'trajs_for_viz': 8, 'samples_per_state': 8}, 'resume_path': None, 'text_processor': {'module': 'octo.data.utils.text_processing', 'name': 'HFTokenizer', 'args': [], 'kwargs': {'encode_with_model': False, 'tokenizer_kwargs': {'max_length': 16, 'padding': 'max_length', 'truncation': True, 'return_tensors': 'np'}, 'tokenizer_name': 't5-base'}}, 'pretrained_loaders': [{'module': 'octo.model.components.hf_weight_loaders', 'name': 'hf_weights_loader', 'args': [], 'kwargs': {'hf_model': 't5-base'}}], 'wandb': {'project': 'octo', 'group': None, 'entity': None}, 'wandb_resume_id': None, 'eval_datasets': ['bridge_dataset']}

OctoModule(
    # attributes
    octo_transformer = OctoTransformer(
        # attributes
        observation_tokenizers = {'primary': ImageTokenizer(
            # attributes
            encoder = {'module': 'octo.model.components.vit_encoders', 'name': 'SmallStem16', 'args': [], 'kwargs': {}}
            use_token_learner = False
            num_tokens = 8
            conditioning_type = 'none'
            obs_stack_keys = ['image_primary']
            task_stack_keys = ['image_primary']
            task_film_keys = ()
            proper_pad_mask = True
        ), 'wrist': ImageTokenizer(
            # attributes
            encoder = {'module': 'octo.model.components.vit_encoders', 'name': 'SmallStem16', 'args': [], 'kwargs': {}}
            use_token_learner = False
            num_tokens = 8
            conditioning_type = 'none'
            obs_stack_keys = ['image_wrist']
            task_stack_keys = ['image_wrist']
            task_film_keys = ()
            proper_pad_mask = True
        )}
        task_tokenizers = {'language': LanguageTokenizer(
            # attributes
            encoder = 't5-base'
            finetune_encoder = False
            proper_pad_mask = True
        )}
        readouts = {'action': 1}
        transformer_kwargs = {'attention_dropout_rate': 0.0, 'add_position_embedding': False, 'num_layers': 12, 'mlp_dim': 3072, 'num_attention_heads': 12, 'dropout_rate': 0.0}
        token_embedding_size = 768
        max_horizon = 10
    )
    heads = {'action': DiffusionActionHead(
        # attributes
        readout_key = 'readout_action'
        use_map = False
        pred_horizon = 4
        action_dim = 7
        max_action = 5.0
        loss_type = 'mse'
        time_dim = 32
        num_blocks = 3
        dropout_rate = 0.1
        hidden_dim = 256
        use_layer_norm = True
        diffusion_steps = 20
    )}
)

Things we need to port:
SmallStem16
DiffusionActionHead

 AttributeError: 'dict' object has no attribute 'shape'
(Pdb) p observations.keys()
dict_keys(['image_primary', 'image_wrist', 'pad_mask', 'pad_mask_dict', 'proprio', 'timestep'])
(Pdb) p observations['image_primary'].shape
(1, 2, 256, 256, 3)

dict_keys(['image_primary', 'image_wrist', 'language_instruction', 'pad_mask_dict', 'proprio', 'timestep'])

example batch looks like:
(Pdb) p enc_inputs.shape
(2, 256, 256, 6)

(Pdb) p observations['image_primary'].shape
(1, 2, 256, 256, 3)

(2, 16, 16, 512)

^ I think the observation has 2 images in time

but at inference time its (1, 16, 16, 6)

time and batch dimensions are processed in parallel
"""
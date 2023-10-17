try:
    import hooks
except:
    try:
        from .. import hooks
    except:
        raise Exception('hooks.py not found. please make sure it is in the same folder as utils.py or in its parent folder')


import functools
import pandas as pd
import copy
import json

import transformers
import torch

device = torch.device('cpu') if not torch.cuda.is_available() else torch.device('cuda')

# a safe way to get attribute of an object
def rgetattr(obj, attr, *args):
    def _getattr(obj, attr):
        return getattr(obj, attr, *args)
    return functools.reduce(_getattr, [obj] + attr.split('.'))

# a safe way to set attribute of an object
def rsetattr(obj, attr, val):
    pre, _, post = attr.rpartition('.')
    return setattr(rgetattr(obj, pre) if pre else obj, post, val)


def wrap_model(model,  
               layers_to_check = ['.mlp', '.mlp.c_proj', '.mlp.c_fc', '', '.attn.c_attn', '.attn.c_proj', '.attn'],
               max_len=256, return_hooks_handler=False):
    '''
    a wrapper function for model to collect hidden states
    returns a dictionary that is updated during the forward pass of the model
    and contains the hidden states of the layers specified in layers_to_check for each layer (collcting inputs and outputs of each)
    the dictionary has the following structure:
    {
        layer_idx: {
            layer_type: {
                'input': [list of hidden states (torch.tensor)],
                'output': [list of hidden states (torch.tensor)]
            }
        }
    }
    you can easily access the hidden states of a specific layer by using the following code:
    hs_collector[layer_idx][layer_type]['input'/'outputs'] # list of hidden states of the input of the layer
    to get the hidden state for the last forward pass, you can use:
    hs_collector[layer_idx][layer_type]['input'/'outputs'][-1] # the last hidden state of the input of the layer

    @ model: a AutoModelForCausalLM model (pytorch)
    @ layers_to_check: one of two options: 
        (a) a list of strings that specify the layers to collect hidden states from
        (b) a path to a configuration file in the format of GraphConfigs (from flow_graph_configs/graph_configs.py)
    @ max_len: the maximum length of the list. if the list is longer than max_len, the oldest hs will be removed
    @ return_hooks_handler: whether to return the hooks handler (to remove the hooks later)
    '''
    
    hs_collector = {}

    if type(layers_to_check) == str:  # assume config file in the format of GraphConfigs
        with open(layers_to_check, 'r') as f:
            tmp_data = json.load(f)
        layers_to_check = set()
        for cell in ["layer_format", "layer_mlp_format", "layer_attn_format", "ln1", 
                     "mlp_ff1", "mlp_ff2", "ln2", "attn_q", "attn_k", "attn_v", "attn_o"]:
            layers_to_check.add(tmp_data[cell])
        layers_to_check = list(layers_to_check)

    if hasattr(model.config, 'n_layer'):  # gpt2, gpt-j
        n_layer = model.config.n_layer
    elif hasattr(model.config, 'num_layers'):  # gpt-neo
        n_layer = model.config.num_layers
    else:
        n_layer = model.config.num_hidden_layers  # llama2
    
    for layer_idx in range(n_layer):
        for layer_type in layers_to_check:
            list_inputs = []
            list_outputs = []

            # the layer_key is key to access the layer in the hs_collector dictionary
            if type(layer_type) == list:
                layer_key, layer_type = layer_type
            else:
                layer_key = layer_type

            try:
                layer_with_idx = layer_type.format(layer_idx)
                # print(f'layer_with_idx: {layer_with_idx}, layer_type: {layer_type}')  # used for debugging
                layer_pointer = rgetattr(model, layer_with_idx)
            except:
                layer_with_idx = f'{layer_idx}{"." if len(layer_type) else ""}{layer_type}'
                 # "transformer.h" is very common prefix in huggingface models like gpt2 and gpt-j.assuming passing only the suffix of the layer
                layer_pointer = rgetattr(model, f"transformer.h.{layer_with_idx}")

            hooks_handler = layer_pointer.register_forward_hook(
                hooks.extract_hs_include_prefix(
                    list_inputs=list_inputs, 
                    list_outputs=list_outputs, 
                    info=layer_with_idx,
                    max_len=max_len
                    )
                )

            if layer_idx not in hs_collector:
                hs_collector[layer_idx] = {}
            
            if layer_key not in hs_collector[layer_idx]:
                hs_collector[layer_idx][layer_key] = {}

            hs_collector[layer_idx][layer_key]['input'] = list_inputs
            hs_collector[layer_idx][layer_key]['output'] = list_outputs

            if return_hooks_handler:  # allows to remove the hooks later by calling hooks_handler.remove()
                hs_collector[layer_idx][layer_key]['hooks_handler'] = hooks_handler

    return hs_collector


def remove_collector_hooks(hs_collector):
    '''
    remove all hooks in hs_collector
    '''
    for layer_idx in hs_collector:
        for layer_type in hs_collector[layer_idx]:
            # print(f'{layer_idx}: layer_type: {layer_type}')
            if 'hooks_handler' not in hs_collector[layer_idx][layer_type]:
                print(f'Warning: no hooks handler for layer {layer_idx} {layer_type}')
            else:
                hooks_handler = hs_collector[layer_idx][layer_type]['hooks_handler']
                hooks_handler.remove()


class model_extra:
    '''
    a class that contains extra functions for language models
    @ model: a pytorch model (currently only support gpt2 models from transformers library)
    @ model_name: the name of the model (e.g. 'gpt2'. if None, will be inferred from the model)
    @ tokenizer: the tokenizer of the model (if None, will be inferred from the model/model_name)
    @ device: the device to run the model on (default: gpu if available, else cpu)
    '''
    def __init__(self, model, model_name=None, tokenizer=None, device=device):
        if model_name is None:
            model_name = model.config._name_or_path

        self.model_name = model_name
        
        if tokenizer is None:
            self.tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
        else:
            self.tokenizer = tokenizer
        
        self.device = device

        try:  # get the model's final layer norm and decoding matrix
            self.ln_f = copy.deepcopy(model.transformer.ln_f).to(self.device).requires_grad_(False) # gpt2, gpt-j, gpt-neo
        except:
            try:
                self.ln_f = copy.deepcopy(model.model.norm).to(self.device).requires_grad_(False)  # models like Llama2
            except:
                raise Exception('cannot find the final layer norm of the model (the model specified might not be supported)')
        self.lm_head = copy.deepcopy(model.lm_head).to(self.device).requires_grad_(False)  # in gpt2, same as model.transformer.wte (transpose)
            


    def hs_to_probs(self, hs, use_ln_f=True):
        '''
        return the probability of each token given a hidden state

        @ hs: a hidden state (torch.tensor) or a list/dataframe in the length of the model's hidden state
        @ use_ln_f: whether to use the final layer norm of the model (if True, the hs will be normalized before processing by the decoding matrix)
        '''
        if type(hs) != torch.Tensor:
            word_embed = torch.tensor(hs).to(self.device)
        else:
            word_embed = hs.clone().detach().to(self.device)
        if use_ln_f:
            word_embed = self.ln_f(word_embed)
        logic_lens = self.lm_head(word_embed)
        probs = torch.softmax(logic_lens, dim=0).detach()
        return probs
    

    def hs_to_token_top_k(self, hs, k_top=12, k_bottom=12, k=None, use_ln_f=True, return_probs=False):
        '''
        return the top and bottom k tokens given a hidden state according to logit of its projection by the decoding matrix
        this functoin is an implementation of the Logit Lens algorithm

        @ hs: a hidden state (torch.tensor) or a list/dataframe in the length of the model's hidden state
        @ k_top: the number of top tokens to return
        @ k_bottom: the number of bottom tokens to return
        @ k: if not None, will be used for both k_top and k_bottom (overwrites k_top and k_bottom)
        @ use_ln_f: whether to use the final layer norm of the model (if True, the hs will be normalized before processing by the decoding matrix)
        @ return_probs: whether to return the probability we get when applying softmax on the logits
        '''
        probs = self.hs_to_probs(hs, use_ln_f=use_ln_f)
        if k is not None:
            k_top = k_bottom = k

        top_k = probs.topk(k_top)
        top_k_idx = top_k.indices
        # convert the indices to tokens
        top_k_words = [self.tokenizer.decode(i, skip_special_tokens=True) for i in top_k_idx]
        
        top_k = probs.topk(k_bottom, largest=False)
        top_k_idx = top_k.indices
        bottom_k_words = [self.tokenizer.decode(i, skip_special_tokens=True) for i in top_k_idx]
        
        res = {'top_k': top_k_words, 'bottom_k': bottom_k_words}
        if return_probs:
            res['probs'] = probs
        return res
    

    def get_token_rank_from_probs(self, token, probs):
        '''
        return the rank of a token given a probability distribution
        highest rank is 0 (most probable). lowest rank is len(probs)-1 (meaning the token is the least probable)

        @ token: a string of a token
        @ probs: a probability distribution (torch.tensor)
        '''
        if type(token) == str:
            token = self.tokenizer.encode(token, return_tensors='pt')[0]
        return (probs > probs[token]).sum().item()
    

    def infrence(self, model_, line, max_length='auto'):
        '''
        a wrapper for the model's generate function
        '''
        if type(max_length) == str and 'auto' in max_length:
            add = 1
            if "+" in max_length:
                add = int(max_length.split('+')[1])
            max_length = len(self.tokenizer.encode(line)) + add

        encoded_line = self.tokenizer.encode(
            line.rstrip(), return_tensors='pt').to(model_.device)

        output = model_.generate(
            input_ids=encoded_line,
            max_length=max_length,
            pad_token_id=self.tokenizer.eos_token_id
        )

        answer_ = self.tokenizer.decode(
            output[:, encoded_line.shape[-1]:][0], skip_special_tokens=True)
        return line + answer_


    def infrence_for_grad(self, model_, line):
        '''
        a wrapper for the model's forward function
        '''
        encoded_line = self.tokenizer.encode(
            line.rstrip(), return_tensors='pt').to(self.device)

        return model_(encoded_line, output_hidden_states=True, output_attentions=True, use_cache=True)
    
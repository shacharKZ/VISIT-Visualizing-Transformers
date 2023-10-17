import torch

def extract_hs_include_prefix(list_inputs, list_outputs, info='', max_len=256):
    '''
    return a hook function that extract the hidden states (hs) before and after the layer

    @ list_inputs: a list. it will be appended with the hs before the layer (torch.tensor)
    @ list_outputs: a list. it will be appended with the hs after the layer (torch.tensor)
    @ info: a string to use while debugging
    @ max_len: the maximum length of the list. if the list is longer than max_len, the oldest hs will be removed
    
    implemention note for future developers:
    - note we use the easiest way to save the hs, by just appending a copy of the hs to a list
    - if you are going to save this data later to a pickle file, you might want to first change the information 
        from torch.tensor wrapped with list, to pandas or numpy. from our experience that can save a lot of space
    - the information is saved without gradient. if you want to save the gradient you can try and also save it separately
    - use the info parameter to identify the layer you are extracting the hs from (we left the comment from our debugging. it might be useful for you)
    - you should verify that the model is not implemented in a way that the hs is not saved in the same order as the input or it processes 
        them inplace so this information is not representative
    '''
    def hook(module, input, output):
        # print(f'info: {info}, len(input): {len(input)}, len(output): {len(output)}')  # for debugging

        # NOTE: in transformers<=4.23.1, every layer of AutoModelForCausalLM upholds that len(input)==len(output)
        # in transformers>=4.24.0, there are some layers where len(input)!=len(output), and many times len(input)==0 (mostly in the attention top sublayers)
        # for those reasons, the following "if" hold the logic of "len(input/outpu)>0"
        if list_inputs is not None and len(input) > 0:
            last_tokens = input[0].clone().detach().squeeze().cpu()
            while len(last_tokens.shape) > 2:
                last_tokens = last_tokens[0]

            # print('last_tokens.shape', last_tokens.shape, f'[{info}]')
            if len(last_tokens.shape) == 1:
                last_tokens = [last_tokens]  # TODO a workaround for one token long inputs
            for last_token in last_tokens:
                last_token = last_token.squeeze()
                list_inputs.append(last_token)

                if len(list_inputs) > max_len:
                    list_inputs.pop(0)

        if list_outputs is not None:
            last_tokens = output[0].clone().detach().squeeze().cpu()
            while len(last_tokens.shape) > 2:
                last_tokens = last_tokens[0]

            # print('last_tokens.shape', last_tokens.shape, f'[{info}]')
            if len(last_tokens.shape) == 1:
                last_tokens = [last_tokens]  # TODO a workaround for one token long inputs
            for last_token in last_tokens:
                last_token = last_token.squeeze()
                # print('last_token.shape', last_token.shape, f'[{info}]')
                list_outputs.append(last_token)

                if len(list_inputs) > max_len:
                    list_inputs.pop(0)
                
    return hook


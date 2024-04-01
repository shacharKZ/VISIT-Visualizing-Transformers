from transformers import AutoTokenizer
import torch
from typing import Optional, Union
import numpy as np
import json

import utils  # custom code that wraps many of the transformers library functions
from flow_graph_configs.graph_configs import GraphConfigs

import plotly.graph_objects as go
import matplotlib as mpl
import plotly.io as pio


try:  # as default, plotly will try to show the plots in external browser. if it fails, it will show the plots in the notebook
    if 'google.colab' in str(get_ipython()):
        print('running on colab. plot will be presented in notebook')
    else:
        # change to "browser" if you want to see the plots in your browser, else omit this line
        pio.renderers.default = "browser"
except:
    print('Warning: pio.renderers.default = "browser" failed. going to use default renderer')


class FlowGraph:
    def __init__(self, model, config_path, model_aux=None, tokenizer=None, device='cpu'):
        '''
        a wrapper for creating a flow graph from a model
        also providing wrappers to infrence the model and collect the data for the graph

        @ model: transformers.AutoModelForCausalLM model (HuggingFace model, for example: gpt2, gpt-j)
        @ config_path: the path to the json file with the configurations for the graph in the format of GraphConfigs
        @ model_aux: the model_aux class (more functions for the original model). if None, will be created from utils.model_extra using model.config
        @ tokenizer: the tokenizer (transformer.AutoTokenizer). if None, will be created according to model.config 
        @ device: the device to use (cpu or cuda)
        '''
        self.model = model
        self.model_aux = model_aux
        if self.model_aux is None:
            self.model_aux = utils.model_extra(model=model, device=device)

        self.tokenizer = tokenizer
        if self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(model.config.model_type)
        elif type(self.tokenizer) == str:
            self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer)
        # else: assume self.tokenizer is already a self.tokenizer object
        
        self.config_path = config_path
        self.config = GraphConfigs.from_json(config_path)
        
        # color definitions for continuous colors (the reset are assumed to be string of colors or rgba)
        self.config.cmap_node_rank_target = mpl.colors.LinearSegmentedColormap.from_list('cmap_node_rank_target', self.config.cmap_node_rank_target)
        self.config.cmap_attn_score = mpl.colors.LinearSegmentedColormap.from_list('cmap_attn_score', self.config.cmap_attn_score)
        self.config.cmap_entropy = mpl.colors.LinearSegmentedColormap.from_list('cmap_entropy', self.config.cmap_entropy)

        # additional conig that are different between models
        self.n_embd = self.model.config.n_embd if hasattr(self.model.config, 'n_embd') else self.model.config.hidden_size
        
        if hasattr(self.model.config, 'n_head'):
            self.n_head = self.model.config.n_head  # gpt2, gpt-j
        elif hasattr(self.model.config, 'num_heads'):  
            self.n_head = self.model.config.num_heads  # gpt-neo
        else:
            self.n_head = self.model.config.num_attention_heads  # llama2-7B
    
    def infrence_model_and_collect_data_for_graph(self, line, layer_to_collect=None, max_hidden_states_to_collect=100):
        '''
        Uses a wrapped infrence of the model to collect the data for the graph
        @ line: the line to infrence
        @ layer_to_collect: the layer to collect the data for the graph. if None, will use the layer in self.config_path (assuming it in the format of GraphConfigs)

        returns the model answer (as str) and the hs_collector dictionary (which contains the data for the graph)      
        '''
        layer_to_collect = self.config_path if layer_to_collect is None else layer_to_collect
        # connect hooks that collect all the layers and hidden states data
        hs_collector = utils.wrap_model(self.model, layers_to_check=layer_to_collect, 
                                        return_hooks_handler=True, max_len=max_hidden_states_to_collect)
        encoded_line = self.tokenizer.encode(line.rstrip(), return_tensors='pt').to(self.model.device)
        output_and_cache = self.model(encoded_line, output_hidden_states=True, output_attentions=True, use_cache=True)
        utils.remove_collector_hooks(hs_collector)  # remove all the hooks that collected data
        
        # use the attention memory (kv cache) to get the hidden states involved parts of the attention layers
        hs_collector['past_key_values'] = output_and_cache.past_key_values  # the "attentnion memory"
        hs_collector['attentions'] = output_and_cache.attentions

        # extract the model answer as string
        model_answer = self.tokenizer.decode(output_and_cache.logits[0, -1, :].argmax().item())

        return model_answer, hs_collector
    

    def merge_two_nodes(self, graph_data, index1, index2, prefix1='', prefix2=''):
        '''
        used to merge two nodes into one node for compacting the graph
        for example: merge the nodes of FF1 with FF2 since there is one-to-one mapping between them
        '''
        sources = graph_data['sources']
        targets = graph_data['targets']
        weights = graph_data['weights']
        colors_nodes = graph_data['colors_nodes']
        colors_links = graph_data['colors_links']
        labels = graph_data['labels']
        line_explained = graph_data['line_explained']
        customdata = graph_data['customdata']

        if index1 +1 != index2 or index2 + 1 != len(labels):
            raise Exception(f'The call for self.config.merge_two_nodes is not valid. it should be done only if the last two nodes are the ones to merge, \
                            and no other nodes were added after them. got index1: {index1}, index2: {index2}, len(labels): {len(labels)}')

        print(f'Start merge_two_nodes for index1: {index1}, index2: {index2}')

        merged_label = f'{labels[index1]} > {labels[index2]}'
        merged_color = colors_nodes[index2]
        merged_customdata = f'{prefix1}{customdata[index1]}<br />{prefix2}{customdata[index2]}'

        # pop old nodes and create new one
        for _ in range(2):
            labels.pop()
            colors_nodes.pop()
            customdata.pop()
        
        labels.append(merged_label)
        colors_nodes.append(merged_color)
        customdata.append(merged_customdata)
        new_index = len(labels) - 1

        # find the common links and remove them
        i = 0
        while i < len(sources):
            if (sources[i] == index1 and targets[i] == index2) or (sources[i] == index2 and targets[i] == index1):
                weights.pop(i)
                colors_links.pop(i)
                sources.pop(i)
                targets.pop(i)
                line_explained.pop(i)
                break
            i += 1

        # find all links with one of the old nodes and change them to the new node
        for old_index in [index1, index2]:
            i = 0
            while i < len(sources):  # len of sources and targets are the same
                if sources[i] == old_index:
                    sources[i] = new_index
                if targets[i] == old_index:
                    targets[i] = new_index
                i += 1
        
        # print(f'Finish self.config.merge_two_nodes for index1: {index1}, index2: {index2}')

        return new_index


    def plot_graph_aux(self, graph_data, title=f'Flow-Graph', save_html=False):
        '''
        A wrapper for graph plotting by plotly express

        @ graph_data: the graph data. see the function @ gen_basic_graph for more details
        @ title: the title of the graph
        @ save_html: if True, the graph will be saved as an html file to {title}.html. if @ save_html is a non empty string, the graph will be saved to {save_html}.html
        '''
        sources = graph_data['sources']
        targets = graph_data['targets']
        weights = graph_data['weights']
        colors_nodes = graph_data['colors_nodes']
        colors_links = graph_data['colors_links']
        labels = graph_data['labels']
        line_explained = graph_data['line_explained']
        customdata = graph_data['customdata']

        fig = go.Figure(data=[go.Sankey(
        valueformat = ".0f",
        valuesuffix = "TWh",
        node = dict(
            pad = 15,
            thickness = 15,
            line = dict(color = self.config.backgroud_color, width = 0.5),
            label = labels,
            color = colors_nodes,
            customdata = customdata,
            hovertemplate='Node: %{customdata}. %{value}<extra></extra>',
        ),
        link = dict(
            source =  sources,
            target =  targets,
            value =  weights,
            color = colors_links,
            customdata = line_explained,
            hovertemplate='%{source.customdata}<br />' + ' '*50 + '----[%{customdata},  %{value}]----><br />%{target.customdata}<extra></extra>',
        ))]
        )

        fig.update_layout(
            hovermode = 'x',
            title_text = title,
            font=dict(size=self.config.font_size, color='white'),
            plot_bgcolor=self.config.backgroud_color,
            paper_bgcolor=self.config.backgroud_color
        )

        fig.show()

        if save_html != False:  # save to html
            path_out = f'{title}.html' if (type(save_html) != str or save_html == '') else f'{save_html}.html'
            fig.write_html(path_out)


    def get_norm_layer(self, x, round_digits=None):
        '''
        return the norm of the vector (or vector-like) x
        the reason we used this function is to handel cases were instead of pytorch vectors our
        data is represent with list, pd.Series, np.array, etc.
        this way we can use the same function for all of them
        '''
        if round_digits is None:
            round_digits = self.config.round_digits

        if type(x) == torch.Tensor:
            res = torch.norm(x).item()
        else:
            res = torch.norm(torch.Tensor(x)).item()
            
        if round_digits > 0:
            return round(res, round_digits)
        
        return res


    def entropy(self, probabilities):
        # calculates the entropy of a probability distribution
        if len(probabilities) != self.model.config.vocab_size:
            probabilities = self.model_aux.hs_to_probs(probabilities)
        # convert the probabilities to a numpy array
        probabilities = np.array(probabilities.cpu().detach())
        # filter out 0 probabilities (to avoid issues with log(0))
        non_zero_probs = probabilities[probabilities != 0]
        entropy = -np.sum(non_zero_probs * np.log2(non_zero_probs))
        return entropy


    def get_color_according_to_entropy(self, hs, max_val=30):
        # return the color according to the entropy of the probability distribution
        entropy_score = self.entropy(hs)
        color_idx = entropy_score / max_val
        color_idx = max(min(color_idx, 1), 0)
        color = f'rgba{self.config.cmap_entropy(color_idx)}'
        return color
    

    def get_node_customdata(self, hs, prefix='', top_or_bottom='top_k', layer_idx=None, target_word=None, color_flag=True):
        '''
        return metadata that uses in the graph plot
        text: the text that will be displayed in the node when hovering over it
        color: the color of the node accodring to its probability of the target word 
        (for example, with the default configs, green if very probable, red if very improbable and grey otherwise)
        '''
        color = self.config.default_color 
        hs_meaning = self.model_aux.hs_to_token_top_k(hs, k_top=5, k_bottom=5, return_probs=True)
        res = f'{hs_meaning[top_or_bottom]}'

        if prefix != '':
            res = f'{prefix}: {res}'
        if layer_idx is not None:
            res = f'{layer_idx}) {res}'
        if target_word is not None:
            target_index = target_word
            if type(target_index) == str:
                target_index = self.tokenizer.encode(target_index, add_special_tokens=False)[0]
            elif type(target_index) == int:
                target_word = self.tokenizer.decode(target_word, skip_special_tokens=True)

            probs = hs_meaning['probs']
            prob = round(probs[target_index].item()*100, self.config.round_digits) #  probs [0,1] as percentage [0.0,100.0]
            ranking = self.model_aux.get_token_rank_from_probs(target_index, probs) + 1  # rank 1 -> most probable, rank #vocab_size -> least probable
            
            res = f'{res} [status: "{target_word}": prob:{prob}%, rank: {ranking})]'
            if color_flag:
                color_idx = ranking/self.model.config.vocab_size  # color index (0,1] (according to ranking)
                color = f'rgba{self.config.cmap_node_rank_target(color_idx)}'
        return res, color


    def connect_link(self, graph_data, idx_source, idx_target, weight, color_or_hs_vector, 
                     explained_prefix='', explained_line=''):
        graph_data['sources'].append(idx_source)
        graph_data['targets'].append(idx_target)
        graph_data['weights'].append(abs(weight))
        color = color_or_hs_vector if type(color_or_hs_vector) == str else self.get_color_according_to_entropy(color_or_hs_vector)
        graph_data['colors_links'].append(color)  # since the operation is LN

        if explained_line == '':
            explained_line = f'{explained_prefix}norm: {weight}'
        else:
            explained_line = f'{explained_prefix}{explained_line}'
        graph_data['line_explained'].append(explained_line)


    def layer_attn_to_graph(self, layer_idx, graph_data, hs_collector, row_idx=-1, target_word=None, line=None):
        '''
        create a graph for the attention (attn) layer
        the subgraph is a graph of the neurons in the Q, K, V O matricies, mostly aggregated into heads
        the nodes are single or small groups of neurons (when they are aggregated into heads or subheads)
        the links are the connections between the neurons (summation of the neurons or when one neuron creates the coefficient of another neuron)
        the graph is created using the graph_data dictionary
        if the graph_data dictionary's lists are empty, they will be initialized
        if the graph_data dictionary 's lists are not empty, they will be updated (try to connect the new nodes to the existing nodes)

        @ layer_idx: the index of the layer in the model
        @ graph_data: the graph data dictionary (if called first time, should include empty list for the keys it uses)
        @ hs_collector: the hs_collector dictionary (created from wrapping the model with the hs_collector class)
        @ row_idx: the index of the row in the hs_collector which correspond to the infrence of the #row_idx token. use -1 for the last token (Note: currently do not support any other value than -1)
        @ target_word: the target word for extracting the status of the neurons (ranking and probability)
        @ line: the line that was used to generate the data in hs_collector (the prompt to the model)
        '''
        colors_nodes = graph_data['colors_nodes']
        labels = graph_data['labels']
        customdata = graph_data['customdata']
        
        idx_attn_input = graph_data['idx_attn_input']  # as created in the setup function for each layer
        # attn_input = hs_collector[layer_idx][self.config.layer_attn_format]['input'][row_idx]  # the input to the attention layer (the output of the previous layer)
        # The above line is correct and equivalent in transformers<=4.23.1, but in transformers>=4.24.0, the implementation of \
        # AutoModelForCausalLM changed and the input to the top attention layer is empty (pass directly to its sublayers)
        attn_input = hs_collector[layer_idx][self.config.ln1]['output'][row_idx]  # solution for transformers>=4.24.0 that is the same as the above line for transformers<=4.23.1
        idx_attn_output = graph_data['idx_attn_output']  # as created in the setup function for each layer

        # uses to show what was the token that generated the attention memory (previous keys and layer)
        # the i-th key and i-th value were generated by the i-th token
        # if @line is not given (None) - will not show this information
        parsed_line = None
        if line is not None:
            parsed_line = self.tokenizer.encode(line, return_tensors='pt')
            # save the parsed line for later use
            parsed_line = [self.tokenizer.decode(x.item()) for x in parsed_line[0]]

        # in gpt2 c_attn is the concatenation of Wq, Wk, Wv (Q, K, V)
        if self.config.attn_q == self.config.attn_v:
            c_attn = utils.rgetattr(self.model, f"{self.config.attn_q.format(layer_idx)}.weight").clone().detach().cpu() # W_QKV (the QKV matrix)
            Wq = c_attn[:, :self.n_embd]
            Wk = c_attn[:, self.n_embd:2*self.n_embd]
            # Wv = c_attn[:, 2*self.n_embd:]  # only for clarity. we get its output values from hs_collector

            c_attn_output = hs_collector[layer_idx][self.config.attn_q]['output'][row_idx].cpu()
            # we can get the output of each of the Q, K, V matrices by splitting the output of the c_attn
            q = c_attn_output[ :self.n_embd]  # this layer query
            k = c_attn_output[self.n_embd:2*self.n_embd]  # this layer key. it is added to the attention memory (to "past_key_values" so also the next tokens can use it)
            v = c_attn_output[2*self.n_embd:] # this layer value. like the key, it is added to the attention memory 
        else:
            Wq = utils.rgetattr(self.model, f"{self.config.attn_q.format(layer_idx)}.weight").clone().detach().cpu() # W_Q (the Query matrix)
            Wk = utils.rgetattr(self.model, f"{self.config.attn_k.format(layer_idx)}.weight").clone().detach().cpu() # W_K (the Key matrix)
            # Wv = utils.rgetattr(self.model, f"{self.config.attn_v.format(layer_idx)}.weight").clone().detach().cpu() # W_V (the Value matrix)

            q = hs_collector[layer_idx][self.config.attn_q]['output'][row_idx].cpu()  # this layer query
            k = hs_collector[layer_idx][self.config.attn_k]['output'][row_idx].cpu()  # this layer key. it is added to the attention memory (to "past_key_values" so also the next tokens can use it)
            v = hs_collector[layer_idx][self.config.attn_v]['output'][row_idx].cpu()  # this layer value. like the key, it is added to the attention memory

        c_proj = utils.rgetattr(self.model, f"{self.config.attn_o.format(layer_idx)}.weight").clone().detach().cpu() # W_O (the Output/projection matrix)

        if self.config.transpose_attn_o:
            c_proj = c_proj.t()

        # projection using the QK circuit
        def pre_project_q(hs_q):
            return hs_q @ Wk

        # projection using the QK circuit
        def pre_project_k(hs_k):
            return Wq @ hs_k
        
        # projection using the OV circuit
        def pre_project_v(hs_v):
            return hs_v @ c_proj
        
        q_projected = pre_project_q(q)
        k_projected = pre_project_k(k)
        v_poject = pre_project_v(v)  

        # create a node for q,k,v together
        q_meaning = self.model_aux.hs_to_token_top_k(q_projected, k_top=1, k_bottom=0)
        q_data, curr_color = self.get_node_customdata(q_projected, prefix=f'q (for current calc)', top_or_bottom='top_k', target_word=target_word)
        k_data, _ = self.get_node_customdata(k_projected, prefix=f'k (for next tokens)', top_or_bottom='top_k', target_word=target_word)
        v_data, _ = self.get_node_customdata(v_poject, prefix=f'v (for next tokens)', top_or_bottom='top_k', target_word=target_word)

        curr_metadata = f'q,k,v (before splitting into heads):' + '<br />' + q_data + '<br />' + k_data + '<br />' + v_data

        # create a new node for query (q) and add the key and value (k, v) metadata. we call this node qkv_full
        labels.append(q_meaning['top_k'][0])
        customdata.append(curr_metadata)
        colors_nodes.append(curr_color)
        idx_qkv_full = len(labels) - 1

        self.connect_link(graph_data, idx_source=idx_attn_input, idx_target=idx_qkv_full, 
                          weight=self.get_norm_layer(attn_input), color_or_hs_vector=q,
                          explained_prefix='', explained_line='')

        # the concated results from all the heads but without the OV circuit projection
        concated_heads_wihtout_projection = hs_collector[layer_idx][self.config.attn_o]['input'][row_idx]

        idx_concated_heads = len(labels)  # attn_c_proj input
        concated_heads_without_projection_meaning = self.model_aux.hs_to_token_top_k(concated_heads_wihtout_projection, k_top=1, k_bottom=0)
        labels.append(concated_heads_without_projection_meaning['top_k'][0])
        curr_metadata, curr_color = self.get_node_customdata(concated_heads_wihtout_projection, prefix=f'concated_heads (without projection)', top_or_bottom='top_k', layer_idx=layer_idx, target_word=target_word)
        customdata.append(curr_metadata)
        colors_nodes.append(self.config.color_for_abstract)

        # print the hidden meaning of each of the heads

        dim_head = self.n_embd // self.n_head

        heads_to_add = [head_idx for head_idx in range(self.n_head)]
        if hasattr(self.config, 'n_heads_to_add') and self.config.n_heads_to_add > 0:
            heads_to_add = concated_heads_wihtout_projection.split(dim_head, dim=0)
            heads_to_add_by_norm = [(head_idx, head.norm().item()) for head_idx, head in enumerate(heads_to_add)]
            # print(f'heads_to_add before filtering BP1: {heads_to_add_by_norm}')
            heads_to_add_by_norm = sorted(heads_to_add_by_norm, key=lambda x: x[1], reverse=True)
            # print(f'heads_to_add after filtering BP2: {heads_to_add_by_norm}')
            heads_to_add = [head_idx for head_idx, _ in heads_to_add_by_norm[:self.config.n_heads_to_add]]
        # print(f'n_heads_to_add: {heads_to_add}, heads_to_add: {heads_to_add}')


        # create for each head the following nodes:
        # (1) the qi - this head part in the query q (we also add its information about ki, vi that were generated at this layer and saved to the attention memory)
        # (2) its #self.config.n_values_per_head top biggest ki and vi (the keys and values from the attention memory) accodring to the attention score
        # (3) the head output - the weighted summation of all the vi into it 
        for head_idx in range(self.n_head):
            if head_idx not in heads_to_add:
                continue
            # hs_collector['past_key_values'][layer_idx][0 for key, 1 for value][entry in batch][head_idx] -> list of the keys/values for this head. the i-th entry is the key/value for the i-th token in the input
            keys = hs_collector['past_key_values'][layer_idx][0][0][head_idx]  
            values = hs_collector['past_key_values'][layer_idx][1][0][head_idx]
            attentions = hs_collector['attentions'][layer_idx][0][head_idx][row_idx]

            # qi with the information about ki, vi (1)
            qi = q[dim_head * head_idx: dim_head * (head_idx + 1)]
            ki = k[dim_head * head_idx: dim_head * (head_idx + 1)]
            vi = v[dim_head * head_idx: dim_head * (head_idx + 1)]

            q_i_fill = torch.zeros(self.n_embd)
            q_i_fill[dim_head*head_idx:dim_head*(head_idx+1)] = qi
            q_i_projected = pre_project_q(q_i_fill)

            k_i_fill = torch.zeros(self.n_embd)
            k_i_fill[dim_head*head_idx:dim_head*(head_idx+1)] = ki
            k_i_projected = pre_project_k(k_i_fill)

            v_i_fill = torch.zeros(self.n_embd)
            v_i_fill[dim_head*head_idx:dim_head*(head_idx+1)] = vi
            v_i_projected = pre_project_v(v_i_fill) 

            q_i_projected_meaning = self.model_aux.hs_to_token_top_k(q_i_projected, k_top=1, k_bottom=0)
            q_data, curr_color = self.get_node_customdata(q_i_projected, prefix=f'qi (for current calc)', top_or_bottom='top_k', target_word=target_word)
            k_data, _ = self.get_node_customdata(k_i_projected, prefix=f'ki (for next tokens)', top_or_bottom='top_k', target_word=target_word)
            v_data, _ = self.get_node_customdata(v_i_projected, prefix=f'vi (for next tokens)', top_or_bottom='top_k', target_word=target_word)

            curr_metadata = f'head {head_idx}:' + '<br />' + q_data + '<br />' + k_data + '<br />' + v_data

            # create a new node for query and add the key and value metadata
            labels.append(q_i_projected_meaning['top_k'][0])
            customdata.append(curr_metadata)
            colors_nodes.append(curr_color)
            idx_q_i_projected = len(labels) - 1

            curr_weight = self.get_norm_layer(q_i_projected)*self.config.factor_head_norm
            self.connect_link(graph_data, idx_source=idx_qkv_full, idx_target=idx_q_i_projected,
                                weight=curr_weight, color_or_hs_vector=q_i_projected,
                                explained_prefix='', explained_line='')

            # create a new node for head output (3)
            head_output = torch.zeros(self.n_embd)
            head_output[dim_head*head_idx:dim_head*(head_idx+1)] = concated_heads_wihtout_projection[dim_head*head_idx:dim_head*(head_idx+1)]
            head_output_projected = pre_project_v(head_output)
            head_output_projected_meaning = self.model_aux.hs_to_token_top_k(head_output_projected, k_top=1, k_bottom=0)

            labels.append(head_output_projected_meaning['top_k'][0])
            curr_metadata, curr_color = self.get_node_customdata(head_output_projected, prefix=f'head {head_idx}: output', top_or_bottom='top_k', layer_idx=layer_idx, target_word=target_word)
            customdata.append(curr_metadata)
            colors_nodes.append(curr_color)
            idx_head_output_projected = len(labels) - 1

            curr_weight = self.get_norm_layer(head_output_projected)
            self.connect_link(graph_data, idx_source=idx_head_output_projected, idx_target=idx_concated_heads,
                                weight=curr_weight*self.config.factor_head_output, color_or_hs_vector=head_output_projected,
                                explained_prefix='', explained_line=f'after projection {curr_weight}')
            
            # the nodes representing the #self.config.n_values_per_head top biggest ki and vi (2)
            best_head_vals = attentions.topk(min(self.config.n_values_per_head,len(attentions)), dim=0)
            for attn_val_idx, attn_score in zip(best_head_vals.indices, best_head_vals.values):
                attn_score = round(attn_score.item(), self.config.round_digits)
                keys_from_attn = keys[attn_val_idx]  # should be in the size of the subhead (for example, 64 for gpt2-medium)
                values_from_attn = values[attn_val_idx]  # should be in the size of the subhead

                keys_from_attn_proj = torch.zeros(self.n_embd)
                keys_from_attn_proj[dim_head*head_idx:dim_head*(head_idx+1)] = keys_from_attn
                keys_from_attn_proj = pre_project_k(keys_from_attn_proj)

                values_from_attn_proj = torch.zeros(self.n_embd)
                values_from_attn_proj[dim_head*head_idx:dim_head*(head_idx+1)] = values_from_attn
                values_from_attn_proj = pre_project_v(values_from_attn_proj)

                keys_from_attn_proj_meaning = self.model_aux.hs_to_token_top_k(keys_from_attn_proj, k_top=1, k_bottom=0)
                values_from_attn_proj_meaning = self.model_aux.hs_to_token_top_k(values_from_attn_proj, k_top=1, k_bottom=0)

                # create node for the top keys ki
                labels.append(keys_from_attn_proj_meaning['top_k'][0])
                if parsed_line is not None:
                    curr_metadata, curr_color = self.get_node_customdata(keys_from_attn_proj, prefix=f'head {head_idx}: key {attn_val_idx} [created from "{parsed_line[attn_val_idx]}"]', top_or_bottom='top_k', layer_idx=layer_idx, target_word=target_word)
                else:
                    curr_metadata, curr_color = self.get_node_customdata(keys_from_attn_proj, prefix=f'head {head_idx}: key {attn_val_idx}', top_or_bottom='top_k', layer_idx=layer_idx, target_word=target_word)
                customdata.append(curr_metadata)
                colors_nodes.append(curr_color)
                idx_keys_from_attn_proj = len(labels) - 1

                # create node for the top values vi
                labels.append(values_from_attn_proj_meaning['top_k'][0])
                if parsed_line is not None:
                    curr_metadata, curr_color = self.get_node_customdata(values_from_attn_proj, prefix=f'head {head_idx}: value {attn_val_idx} [created from "{parsed_line[attn_val_idx]}"]', top_or_bottom='top_k', layer_idx=layer_idx, target_word=target_word)
                else:
                    curr_metadata, curr_color = self.get_node_customdata(values_from_attn_proj, prefix=f'head {head_idx}: value {attn_val_idx}', top_or_bottom='top_k', layer_idx=layer_idx, target_word=target_word)
                customdata.append(curr_metadata)
                colors_nodes.append(curr_color)
                idx_values_from_attn_proj = len(labels) - 1

                # create a link between the query qi and the top keys ki
                color_attn = f'rgba{self.config.cmap_attn_score(attn_score)}'
                curr_weight = self.get_norm_layer(q_i_projected)
                curr_attn_weight = max(attn_score*self.config.factor_attn_score, 0.51)
                self.connect_link(graph_data, idx_source=idx_q_i_projected, idx_target=idx_keys_from_attn_proj,
                                weight=curr_attn_weight, color_or_hs_vector=color_attn,
                                explained_prefix='', explained_line=f'attention score: {attn_score} (qi norm: {curr_weight})')
                                  
                if self.config.compact_attn_k_v_nodes:
                    merged_idx = self.config.merge_two_nodes(graph_data, index1=idx_keys_from_attn_proj, index2=idx_values_from_attn_proj, prefix1='ki:', prefix2='vi:')
                    idx_keys_from_attn_proj = merged_idx
                    idx_values_from_attn_proj = merged_idx
                else:
                    # create the link between the top keys and the top values
                    self.connect_link(graph_data, idx_source=idx_keys_from_attn_proj, idx_target=idx_values_from_attn_proj,
                                weight=curr_attn_weight, color_or_hs_vector=color_attn,
                                explained_prefix='', explained_line=f'attention score: {attn_score} (ki norm: {curr_weight})')

                # create a link between the top values and the idx_head_output_projected
                curr_weight = self.get_norm_layer(values_from_attn_proj)
                curr_attn_weight = max(attn_score*curr_weight*self.config.factor_for_memory_to_head_values, 0.51)
                self.connect_link(graph_data, idx_source=idx_values_from_attn_proj, idx_target=idx_head_output_projected,
                                weight=curr_attn_weight, color_or_hs_vector=values_from_attn_proj,
                                explained_prefix='', 
                                explained_line=f'attention score * norm: {round(attn_score*curr_weight, self.config.round_digits)} (vi norm: {curr_weight})')

        # now we want to present single neurons from the concatenated heads and how they are projected (indevideually) to the output by W_O (the output projection matrix)
        # we pick only the top most activated neurons (positive and negative)
        concated_heads_wihtout_projection_mul_norm = concated_heads_wihtout_projection * c_proj.norm(dim=1)
        for case, n_top, is_largest in [('top_k', self.config.number_of_top_neurons, True), ('bottom_k', self.config.number_of_bottom_neurons, False)]:
            tops = torch.topk(concated_heads_wihtout_projection_mul_norm, k=n_top, largest=is_largest)
            for entry_idx, activision_mul_norm in zip(tops.indices, tops.values):
                activision_mul_norm = round(activision_mul_norm.item(), self.config.round_digits)
                entry_idx = entry_idx.item()
                activision_value = round(concated_heads_wihtout_projection[entry_idx].item(), self.config.round_digits)
                    
                # connect between c_proj_input, which is the concatenated heads, and each of this neurons
                idx_value = len(labels)
                curr_c_proj_meaning = self.model_aux.hs_to_token_top_k(c_proj[entry_idx], k_top=1, k_bottom=1)
                labels.append(curr_c_proj_meaning[case][0])
                curr_metadata, curr_color = self.get_node_customdata(c_proj[entry_idx], prefix=f'value index:{entry_idx} (from head {entry_idx//(self.n_embd//self.n_head)}), activision:{activision_value}: ', 
                                top_or_bottom=case, layer_idx=layer_idx, target_word=target_word)  # value is chosen accodring to activation sign. suppouse to reflect the meaning its adding to the output
                customdata.append(curr_metadata)
                colors_nodes.append(curr_color)

                self.connect_link(graph_data, idx_source=idx_concated_heads, idx_target=idx_value,
                                weight=self.config.defualt_weight, color_or_hs_vector=self.config.positive_activation if is_largest > 0 else self.config.negative_activation,
                                explained_prefix='', explained_line=f'activision:{activision_value}')

                self.connect_link(graph_data, idx_source=idx_value, idx_target=idx_attn_output,
                                weight=abs(activision_mul_norm), color_or_hs_vector=c_proj[entry_idx],
                                explained_prefix='', explained_line=f'activision*norm:{activision_mul_norm}')
        
        if self.config.include_attn_bias:
            # we also add neurons representing the W_O matrix bias vectors
            c_proj_bias = utils.rgetattr(self.model, f"{self.config.attn_o.format(layer_idx)}.bias").clone().detach().cpu()
            c_proj_bias_meaning = self.model_aux.hs_to_token_top_k(c_proj_bias, k_top=1, k_bottom=0)
            labels.append(c_proj_bias_meaning['top_k'][0])
            curr_metadata, curr_color = self.get_node_customdata(c_proj_bias, prefix='c_proj_bias', top_or_bottom='top_k', layer_idx=layer_idx, target_word=target_word)
            customdata.append(curr_metadata)
            colors_nodes.append(self.config.color_for_bias_vector)
            idx_c_proj_bias = len(labels) - 1

            # connect the bias vector
            curr_norm = self.get_norm_layer(c_proj_bias)
            self.connect_link(graph_data, idx_source=idx_concated_heads, idx_target=idx_c_proj_bias,
                                weight=self.config.defualt_weight, color_or_hs_vector=self.config.color_for_bias_vector,
                                explained_prefix='', explained_line=f'norm: {curr_norm}')

            self.connect_link(graph_data, idx_source=idx_c_proj_bias, idx_target=idx_attn_output,
                                weight=curr_norm, color_or_hs_vector=self.config.color_for_bias_vector,
                                explained_prefix='', explained_line=f'norm: {curr_norm}')


    def layer_mlp_to_graph(self, layer_idx: int, graph_data, hs_collector, row_idx=-1, target_word=None):
        '''
        create a subgraph of the feed-forward (FF, MLP) part of the model at layer_idx
        the subgraph is a graph of the neurons in the FF part of the model
        the nodes are the most active neurons in the FF (some of positive and some of negative)
        the links are the connections between the neurons (summation of the neurons or when one neuron creates the coefficient of another neuron)
        the graph is created using the graph_data dictionary
        if the graph_data dictionary's lists are empty, they will be initialized
        if the graph_data dictionary's lists are not empty, they will be updated (try to connect the new nodes to the existing nodes)

        @ layer_idx: the index of the layer in the model
        @ graph_data: the graph data dictionary (if called first time, should include empty list for the keys it uses)
        @ model: the model (for example: gpt2)
        @ hs_collector: the hs_collector dictionary (created from wrapping the model with the hs_collector class)
        @ row_idx: the index of the row in the hs_collector which correspond to the infrence of the #row_idx token. use -1 for the last token (Note: currently do not support any other value than -1)
        @ self.model_aux: the self.model_aux class (more functions for the original model)
        @ target_word: the target word for extracting the status of the neurons (ranking and probability)    
        '''
        colors_nodes = graph_data['colors_nodes']
        labels = graph_data['labels']
        customdata = graph_data['customdata']
        
        idx_mlp_input = graph_data['idx_mlp_input']  # as created in the setup function for each layer
        mlp_input = hs_collector[layer_idx][self.config.layer_mlp_format]['input'][row_idx]  # the input to the mlp
        idx_mlp_output = graph_data['idx_mlp_output']  # as created in the setup function for each layer

        # get the first and second matricies of the mlp
        c_fc = utils.rgetattr(self.model, f"{self.config.mlp_ff1.format(layer_idx)}.weight").clone().detach().cpu()
        if c_fc.shape[0] > c_fc.shape[1]:  # since some models have the matricies in transposed 
            c_fc = c_fc.T
        c_proj = utils.rgetattr(self.model, f"{self.config.mlp_ff2.format(layer_idx)}.weight").clone().detach().cpu()
        if c_proj.shape[0] < c_proj.shape[1]:
            c_proj = c_proj.T

        values_norm = c_proj.norm(dim=1)
        hs =  hs_collector[layer_idx][self.config.mlp_ff2]['input'][row_idx]  # value activation. mid results between the key and value matrix
        hs_mul_norm = hs * values_norm  # this is our metric for the importance of the value activation (value*norm of the second matrix)
        
        # pick the top most activate neurons (according to activasion sign)
        for case, n_top, is_largest in [('top_k', self.config.number_of_top_neurons, True), ('bottom_k', self.config.number_of_bottom_neurons, False)]:
            tops = torch.topk(hs_mul_norm, k=n_top, largest=is_largest)
            for entry_idx, activision_value_mul_norm in zip(tops.indices, tops.values):
                activision_value_mul_norm = round(activision_value_mul_norm.item(), self.config.round_digits)
                activision_value = round(hs[entry_idx].item(), self.config.round_digits)
                entry_idx = entry_idx.item()

                # create a node for the "key" neuron (the first matrix)
                idx_key = len(labels)
                idx_value = idx_key+1
                curr_c_fc_meaning = self.model_aux.hs_to_token_top_k(c_fc.T[entry_idx], k_top=1, k_bottom=0)
                labels.append(curr_c_fc_meaning['top_k'][0])
                curr_metadata, curr_color = self.get_node_customdata(c_fc.T[entry_idx], prefix=f'key{entry_idx}', top_or_bottom='top_k', layer_idx=layer_idx, target_word=target_word)
                customdata.append(curr_metadata)
                colors_nodes.append(curr_color)

                # create a node for the "value" neuron (the second matrix)
                curr_c_proj_meaning = self.model_aux.hs_to_token_top_k(c_proj[entry_idx], k_top=1, k_bottom=1)
                labels.append(curr_c_proj_meaning[case][0])  # value is chosen accodring to activation sign. suppouse to reflect the meaning its adding to the output
                curr_metadata, curr_color = self.get_node_customdata(c_proj[entry_idx], prefix=f'value{entry_idx}', top_or_bottom=case, layer_idx=layer_idx, target_word=target_word)
                customdata.append(curr_metadata)
                colors_nodes.append(curr_color)

                if self.config.compact_mlp_nodes:
                    merged_idx = self.config.merge_two_nodes(graph_data, index1=idx_key, index2=idx_value, prefix1='key:', prefix2='value:')
                    idx_key = merged_idx
                    idx_value = merged_idx
                else:
                    # connect the "key" and the "value"
                    self.connect_link(graph_data, idx_source=idx_key, idx_target=idx_value,
                                    weight=abs(activision_value)*self.config.factor_weight_mlp_key_value_link,
                                    color_or_hs_vector=self.config.positive_activation if is_largest else self.config.negative_activation,
                                    explained_prefix='', explained_line=f'activision:{activision_value}')

                # add between the mlp_input and the "key" neuron
                curr_weight = self.get_norm_layer(c_fc.T[entry_idx])
                self.connect_link(graph_data, idx_source=idx_mlp_input, idx_target=idx_key,
                                weight=self.config.defualt_weight,
                                color_or_hs_vector=c_fc.T[entry_idx],
                                explained_prefix='', explained_line=f'key norm: {curr_weight}')

                # add a link between of the "value" neuron to the mlp_output
                curr_weight = self.get_norm_layer(c_proj[entry_idx])
                self.connect_link(graph_data, idx_source=idx_value, idx_target=idx_mlp_output,
                                weight=activision_value_mul_norm,
                                color_or_hs_vector=c_proj[entry_idx],
                                explained_prefix='', 
                                explained_line=f'activision*norm:{abs(activision_value_mul_norm)}. value norm {curr_weight}')
        
        # we also add neurons representing the matricies bias vectors
        if self.config.include_mlp_bias:
            # we create nodes for each matricies bias vector then connect them to the flow
            # Remineding that the bias vector of FF1 (c_fc) is in the size of ths embedding and what actually added into the residual is FF1.bias @ FF2
            c_fc_bias = utils.rgetattr(self.model, f"{self.config.mlp_ff1.format(layer_idx)}.bias").clone().detach().cpu() @ c_proj
            c_fc_bias_meaning = self.model_aux.hs_to_token_top_k(c_fc_bias, k_top=1, k_bottom=0)
            labels.append(c_fc_bias_meaning['top_k'][0])
            curr_metadata, curr_color = self.get_node_customdata(c_fc_bias, prefix='c_fc_bias', top_or_bottom='top_k', layer_idx=layer_idx, target_word=target_word)
            customdata.append(curr_metadata)
            colors_nodes.append(self.config.color_for_bias_vector) # special color for bias
            idx_c_fc_bias = len(labels) - 1

            c_proj_bias = utils.rgetattr(self.model, f"{self.config.mlp_ff2.format(layer_idx)}.bias").clone().detach().cpu()
            c_proj_bias_meaning = self.model_aux.hs_to_token_top_k(c_proj_bias, k_top=1, k_bottom=0)
            labels.append(c_proj_bias_meaning['top_k'][0])
            curr_metadata, curr_color = self.get_node_customdata(c_proj_bias, prefix='c_proj_bias', top_or_bottom='top_k', layer_idx=layer_idx, target_word=target_word)
            customdata.append(curr_metadata)
            colors_nodes.append(self.config.color_for_bias_vector)
            idx_c_proj_bias = len(labels) - 1

            if self.config.compact_mlp_nodes:
                merged_idx = self.config.merge_two_nodes(graph_data, index1=idx_c_fc_bias, index2=idx_c_proj_bias, prefix1='c_fc_bias:', prefix2='c_proj_bias:')
                idx_c_fc_bias = merged_idx
                idx_c_proj_bias = merged_idx
            else:
                # connect c_fc_bias to c_proj_bias, although it is not really a link (this why its color is self.config.invisible_link_color)
                self.connect_link(graph_data, idx_source=idx_c_fc_bias, idx_target=idx_c_proj_bias,
                                weight=0.05,  # to be barely visible (trick to make it unvisible with the background)
                                color_or_hs_vector=self.config.invisible_link_color,
                                explained_prefix='', explained_line='')

            # connect mlp_input to c_fc_bias
            self.connect_link(graph_data, idx_source=idx_mlp_input, idx_target=idx_c_fc_bias,
                                weight=self.config.defualt_weight,
                                color_or_hs_vector=self.config.color_for_bias_vector,
                                explained_prefix='', explained_line=f'norm:{self.get_norm_layer(mlp_input)}')

            # connect c_proj_bias to mlp_output
            self.connect_link(graph_data, idx_source=idx_c_proj_bias, idx_target=idx_mlp_output,
                                weight=self.get_norm_layer(c_proj_bias),
                                color_or_hs_vector=self.config.color_for_bias_vector,
                                explained_prefix='', explained_line=f'')


    def setup_sequential_block(self, layer_idx, graph_data, hs_collector, row_idx=-1, target_word=None):
        '''
        set up the attention and mlp input and output nodes
        each input node has its own layer norm (LN)
        and also connect the residual of each sub-block
        '''
        colors_nodes = graph_data['colors_nodes']
        labels = graph_data['labels']
        customdata = graph_data['customdata']

        if 'idx_previous_layer' in graph_data: # conect to the first layer in the graph
            idx_block_input = graph_data['idx_previous_layer']
            block_input_norm = graph_data['previous_layer_norm']
        else:  # this is the first layer for the current graph
            # block_input = hs_collector[layer_idx][self.config.layer_format]['input'][row_idx]
            # The above line is correct and equivalent in transformers<=4.23.1, but in transformers>=4.24.0, the implementation of \
            # AutoModelForCausalLM changed and the input to the top attention layer is empty (pass directly to its sublayers)
            block_input = hs_collector[layer_idx][self.config.ln1]['input'][row_idx]  # same logit as the line above (access the first input to the decoder block)
            block_input_meaning = self.model_aux.hs_to_token_top_k(block_input, k_top=1, k_bottom=0)
            labels.append(block_input_meaning['top_k'][0])
            curr_metadata, curr_color = self.get_node_customdata(block_input, prefix='block_input', top_or_bottom='top_k', layer_idx=layer_idx, target_word=target_word)
            customdata.append(curr_metadata)
            colors_nodes.append(curr_color)
            idx_block_input = len(labels) - 1
            block_input_norm = self.get_norm_layer(block_input)

        # create the attn (attention) subblocks input/output nodes
        # between the block input and the attention (attn) input, there is a layer norm (LN)
        attn_input = hs_collector[layer_idx][self.config.ln1]['output'][row_idx]  # attn's LN. its input is the same as the residual (block_input/previous_layer_output)
        attn_input_meaning = self.model_aux.hs_to_token_top_k(attn_input, k_top=1, k_bottom=0)
        labels.append(attn_input_meaning['top_k'][0])
        curr_metadata, curr_color = self.get_node_customdata(attn_input, prefix=f'attn_input', top_or_bottom='top_k', layer_idx=layer_idx, target_word=target_word)
        customdata.append(curr_metadata)
        colors_nodes.append(curr_color)
        idx_attn_input = len(labels) - 1

        # attn output
        attn_output = hs_collector[layer_idx][self.config.layer_attn_format]['output'][row_idx]
        attn_output_meaning = self.model_aux.hs_to_token_top_k(attn_output, k_top=1, k_bottom=0)
        labels.append(attn_output_meaning['top_k'][0])
        curr_metadata, curr_color = self.get_node_customdata(attn_output, prefix=f'attn_output', top_or_bottom='top_k', layer_idx=layer_idx, target_word=target_word)
        customdata.append(curr_metadata)
        colors_nodes.append(curr_color)
        idx_attn_output = len(labels) - 1

        # between the attn and the mlp we create a node that represents the hidden state between them (residual after attn update)
        residual_after_attn = hs_collector[layer_idx][self.config.ln2]['input'][row_idx]  # residual after attn update
        residual_after_attn_meaning = self.model_aux.hs_to_token_top_k(residual_after_attn, k_top=1, k_bottom=0)
        labels.append(residual_after_attn_meaning['top_k'][0])
        curr_metadata, curr_color = self.get_node_customdata(residual_after_attn, prefix=f'residual_after_attn', top_or_bottom='top_k', layer_idx=layer_idx, target_word=target_word)
        customdata.append(curr_metadata)
        colors_nodes.append(curr_color)
        idx_residual_after_attn = len(labels) - 1

        # create the mlp (FF) subblocks input/output nodes
        # the start of the block is the mlp layer norm (LN)
        mlp_input = hs_collector[layer_idx][self.config.layer_mlp_format]['input'][row_idx]  # mlp's LN (its input is the same as the taking ln2's output)
        mlp_input_meaning = self.model_aux.hs_to_token_top_k(mlp_input, k_top=1, k_bottom=0)
        labels.append(mlp_input_meaning['top_k'][0])
        curr_metadata, curr_color = self.get_node_customdata(mlp_input, prefix=f'mlp_input', top_or_bottom='top_k', layer_idx=layer_idx, target_word=target_word)
        customdata.append(curr_metadata)
        colors_nodes.append(curr_color)
        idx_mlp_input = len(labels) - 1

        # mlp output
        mlp_output = hs_collector[layer_idx][self.config.layer_mlp_format]['output'][row_idx]
        mlp_output_meaning = self.model_aux.hs_to_token_top_k(mlp_output, k_top=1, k_bottom=0)
        labels.append(mlp_output_meaning['top_k'][0])
        curr_metadata, curr_color = self.get_node_customdata(mlp_output, prefix=f'mlp_output', top_or_bottom='top_k', layer_idx=layer_idx, target_word=target_word)
        customdata.append(curr_metadata)
        colors_nodes.append(curr_color)
        idx_mlp_output = len(labels) - 1

        # the block output (the sum of the mlp with the residual)
        block_out = hs_collector[layer_idx][self.config.layer_format]['output'][row_idx]  # mlp_output + attention_out + attn_output + residual
        block_out_meaning = self.model_aux.hs_to_token_top_k(block_out, k_top=1, k_bottom=0)
        labels.append(block_out_meaning['top_k'][0])
        curr_metadata, curr_color = self.get_node_customdata(block_out, prefix='block_output', top_or_bottom='top_k', layer_idx=layer_idx, target_word=target_word)
        customdata.append(curr_metadata)
        colors_nodes.append(curr_color)
        idx_block_out = len(labels) - 1


        # now we have all the input/output of the subblock of this layer and we will connect them
        # connect the block input to the attn input
        self.connect_link(graph_data, idx_source=idx_block_input, idx_target=idx_attn_input, 
                          weight=block_input_norm, color_or_hs_vector=self.config.link_with_normalizer, 
                            explained_prefix=f'', explained_line='')

        # all the nodes between the attn input and attn output - handled by the function layer_attn_to_graph

        # connect the attn output to the residual after attn
        self.connect_link(graph_data, idx_source=idx_attn_output, idx_target=idx_residual_after_attn,
                            weight=self.get_norm_layer(attn_output), color_or_hs_vector=attn_output,
                            explained_prefix=f'', explained_line='')
        
        # connect the block input with the residual after attn (the residual of the attn subblock)
        self.connect_link(graph_data, idx_source=idx_block_input, idx_target=idx_residual_after_attn,
                            weight=block_input_norm, color_or_hs_vector=self.config.color_attn_residual,
                            explained_prefix=f'', explained_line='')
        
        # connect the residual after attn to the mlp input
        self.connect_link(graph_data, idx_source=idx_residual_after_attn, idx_target=idx_mlp_input,
                            weight=self.get_norm_layer(residual_after_attn), color_or_hs_vector=self.config.link_with_normalizer,
                            explained_prefix=f'', explained_line='')
        
        # all the nodes between the mlp input and mlp output - handled by the function layer_mlp_to_graph

        # connect the mlp output to the block output
        self.connect_link(graph_data, idx_source=idx_mlp_output, idx_target=idx_block_out,
                            weight=self.get_norm_layer(mlp_output), color_or_hs_vector=mlp_output,
                            explained_prefix=f'', explained_line='')
        
        # connect the residual after attn to the block output
        self.connect_link(graph_data, idx_source=idx_residual_after_attn, idx_target=idx_block_out,
                            weight=self.get_norm_layer(residual_after_attn), color_or_hs_vector=self.config.color_mlp_residual,
                            explained_prefix=f'', explained_line='')
        

        # now, so the functions that create the subblocks will be able to access the input and output nodes, we save them to the graph_data
        graph_data['idx_attn_input'] = idx_attn_input
        graph_data['idx_attn_output'] = idx_attn_output
        
        graph_data['idx_mlp_input'] = idx_mlp_input
        graph_data['idx_mlp_output'] = idx_mlp_output

        # so the call for the next block graph will be able to connect to this block output
        graph_data['idx_previous_layer'] = idx_block_out
        graph_data['previous_layer_norm'] = block_input_norm

        

    def setup_parallel_block(self, layer_idx, graph_data, hs_collector, row_idx=-1, target_word=None):
        '''
        set up the attention and mlp input and output nodes
        each input node has its own layer norm (LN)
        and also connect the residual of each sub-block
        '''

        colors_nodes = graph_data['colors_nodes']
        labels = graph_data['labels']
        customdata = graph_data['customdata']

        if 'idx_previous_layer' in graph_data: # conect to the first layer in the graph
            idx_block_input = graph_data['idx_previous_layer']
            block_input_norm = graph_data['previous_layer_norm']
        else:  # this is the first layer for the current graph
            # block_input = hs_collector[layer_idx][self.config.layer_format]['input'][row_idx]
            # The above line is correct and equivalent in transformers<=4.23.1, but in transformers>=4.24.0, the implementation of \
            # AutoModelForCausalLM changed and the input to the top attention layer is empty (pass directly to its sublayers)
            block_input = hs_collector[layer_idx][self.config.ln1]['input'][row_idx]  # same logit as the line above (access the first input to the decoder block)
            block_input_meaning = self.model_aux.hs_to_token_top_k(block_input, k_top=1, k_bottom=0)
            labels.append(block_input_meaning['top_k'][0])
            curr_metadata, curr_color = self.get_node_customdata(block_input, prefix='block_input', top_or_bottom='top_k', layer_idx=layer_idx, target_word=target_word)
            customdata.append(curr_metadata)
            colors_nodes.append(curr_color)
            idx_block_input = len(labels) - 1
            block_input_norm = self.get_norm_layer(block_input)

        # create the attn (attention) subblocks input/output nodes
        # in the parallel architecture, the input for both subblocks is the block input after layer norm (LN)
        block_input_after_ln = hs_collector[layer_idx][self.config.ln1]['output'][row_idx]  # block only LN. its input is the same as the residual (block_input/previous_layer_output)
        block_input_after_ln_meaning = self.model_aux.hs_to_token_top_k(block_input_after_ln, k_top=1, k_bottom=0)
        labels.append(block_input_after_ln_meaning['top_k'][0])
        curr_metadata, curr_color = self.get_node_customdata(block_input_after_ln, prefix=f'block_input_after_LN', top_or_bottom='top_k', layer_idx=layer_idx, target_word=target_word)
        customdata.append(curr_metadata)
        colors_nodes.append(curr_color)
        idx_block_input_after_ln = len(labels) - 1

        # atten input
        # attn_input = hs_collector[layer_idx][self.config.layer_attn_format]['input'][row_idx]  # the input to the attention layer (the output of the previous layer)
        # The above line is correct and equivalent in transformers<=4.23.1, but in transformers>=4.24.0, the implementation of \
        # AutoModelForCausalLM changed and the input to the top attention layer is empty (pass directly to its sublayers)
        attn_input = hs_collector[layer_idx][self.config.ln1]['output'][row_idx]  # solution for transformers>=4.24.0 that is the same as the above line for transformers<=4.23.1

        attn_input_meaning = self.model_aux.hs_to_token_top_k(attn_input, k_top=1, k_bottom=0)
        labels.append(attn_input_meaning['top_k'][0])
        curr_metadata, curr_color = self.get_node_customdata(attn_input, prefix=f'attn_input', top_or_bottom='top_k', layer_idx=layer_idx, target_word=target_word)
        customdata.append(curr_metadata)
        colors_nodes.append(curr_color)
        idx_attn_input = len(labels) - 1

        # attn output
        attn_output = hs_collector[layer_idx][self.config.layer_attn_format]['output'][row_idx]
        attn_output_meaning = self.model_aux.hs_to_token_top_k(attn_output, k_top=1, k_bottom=0)
        labels.append(attn_output_meaning['top_k'][0])
        curr_metadata, curr_color = self.get_node_customdata(attn_output, prefix=f'attn_output', top_or_bottom='top_k', layer_idx=layer_idx, target_word=target_word)
        customdata.append(curr_metadata)
        colors_nodes.append(curr_color)
        idx_attn_output = len(labels) - 1

        # create the mlp (FF) subblocks input/output nodes
        # mlp input
        mlp_input = hs_collector[layer_idx][self.config.layer_mlp_format]['input'][row_idx]  # mlp's input is the same as the block input after LN
        mlp_input_meaning = self.model_aux.hs_to_token_top_k(mlp_input, k_top=1, k_bottom=0)
        labels.append(mlp_input_meaning['top_k'][0])
        curr_metadata, curr_color = self.get_node_customdata(mlp_input, prefix=f'mlp_input', top_or_bottom='top_k', layer_idx=layer_idx, target_word=target_word)
        customdata.append(curr_metadata)
        colors_nodes.append(curr_color)
        idx_mlp_input = len(labels) - 1

        # mlp output
        mlp_output = hs_collector[layer_idx][self.config.layer_mlp_format]['output'][row_idx]
        mlp_output_meaning = self.model_aux.hs_to_token_top_k(mlp_output, k_top=1, k_bottom=0)
        labels.append(mlp_output_meaning['top_k'][0])
        curr_metadata, curr_color = self.get_node_customdata(mlp_output, prefix=f'mlp_output', top_or_bottom='top_k', layer_idx=layer_idx, target_word=target_word)
        customdata.append(curr_metadata)
        colors_nodes.append(curr_color)
        idx_mlp_output = len(labels) - 1

        # the block output (the sum of the attn, the mlp and the residual)
        block_out = hs_collector[layer_idx][self.config.layer_format]['output'][row_idx]  # mlp_output + attention_out + attn_output + residual
        block_out_meaning = self.model_aux.hs_to_token_top_k(block_out, k_top=1, k_bottom=0)
        labels.append(block_out_meaning['top_k'][0])
        curr_metadata, curr_color = self.get_node_customdata(block_out, prefix='block_output', top_or_bottom='top_k', layer_idx=layer_idx, target_word=target_word)
        customdata.append(curr_metadata)
        colors_nodes.append(curr_color)
        idx_block_out = len(labels) - 1


        # now we have all the input/output of the subblock of this layer and we will connect them
        # connect the block input to LN
        self.connect_link(graph_data, idx_source=idx_block_input, idx_target=idx_block_input_after_ln,
                            weight=block_input_norm, color_or_hs_vector=self.config.link_with_normalizer,
                            explained_prefix=f'', explained_line='')
        
        # connect the block input after LN to the attn input
        self.connect_link(graph_data, idx_source=idx_block_input_after_ln, idx_target=idx_attn_input,
                            weight=self.get_norm_layer(block_input_after_ln), color_or_hs_vector=self.config.link_with_normalizer,
                            explained_prefix=f'', explained_line='')
        
        # all the nodes between the attn input and attn output - handled by the function layer_attn_to_graph

        # connect the attn output to the block output
        self.connect_link(graph_data, idx_source=idx_attn_output, idx_target=idx_block_out,
                            weight=self.get_norm_layer(attn_output), color_or_hs_vector=attn_output,
                            explained_prefix=f'', explained_line='')
        
        # connect the block input after LN to the mlp input
        self.connect_link(graph_data, idx_source=idx_block_input_after_ln, idx_target=idx_mlp_input,
                            weight=self.get_norm_layer(block_input_after_ln), color_or_hs_vector=self.config.link_with_normalizer,
                            explained_prefix=f'', explained_line='')
        
        # all the nodes between the mlp input and mlp output - handled by the function layer_mlp_to_graph

        # connect the mlp output to the block output
        self.connect_link(graph_data, idx_source=idx_mlp_output, idx_target=idx_block_out,
                            weight=self.get_norm_layer(mlp_output), color_or_hs_vector=mlp_output,
                            explained_prefix=f'', explained_line='')
        
        # connect the residual
        self.connect_link(graph_data, idx_source=idx_block_input, idx_target=idx_block_out,
                            weight=block_input_norm, color_or_hs_vector=self.config.color_attn_residual,
                            explained_prefix=f'', explained_line='')
        

        # now, so the functions that create the subblocks will be able to access the input and output nodes, we save them to the graph_data
        graph_data['idx_attn_input'] = idx_attn_input
        graph_data['idx_attn_output'] = idx_attn_output
        
        graph_data['idx_mlp_input'] = idx_mlp_input
        graph_data['idx_mlp_output'] = idx_mlp_output

        # so the call for the next block graph will be able to connect to this block output
        graph_data['idx_previous_layer'] = idx_block_out
        graph_data['previous_layer_norm'] = self.get_norm_layer(block_out)


    def gen_basic_graph(self, 
                        layers: Optional[Union[str, list]], 
                        hs_collector: dict, 
                        target_word: str = None, 
                        line: str = None, 
                        save_html: Optional[Union[str, bool]] = False, 
                        row_idx: int = -1
                        ):
        '''
        The entry point for generating a graph for a given layer or a list of layers
        A wrapper function to generate a graph for given layers

        @ layers: a list of layers to generate the graph for (correctness is guaranteed only if layers are in order)
        @ hs_collector: the hs_collector dictionary (created from wrapping the model with the hs_collector class)
        @ target_word: the target word for extracting the status of the neurons (ranking and probability)
        @ line: the line that was used to generate the data in hs_collector (the prompt to the model)
        @ save_html: if True, the graph will be saved as an html file to {title}.html. if @ save_html is a non empty string, the graph will be saved as to {save_html}.html
        @ row_idx: the row index in the hs_collector to use for the graph generation. Each row is corresponding 
                to a token in the input sequence, so "0" means the first token and "-1" means the last token (which we recommand to be also the the target word)
                (Note: currently do not support any other value than -1)
        '''

        model_old_device = self.model.device
        self.model = self.model.cpu()  # to prevent memory issues

        # init the graph data for this run (this is what we provide to plot_graph_aux which used plotly.express.go.Sankey)
        graph_data = {
            'sources': [],
            'targets': [],
            'weights': [],
            'colors_nodes': [],
            'colors_links': [],
            'labels': [],
            'line_explained': [],
            'customdata': []
        }

        if type(layers) != list:
            layers = [layers]
        
        # generate the graph for each layer
        for layer_idx in layers:
            # create the input and output nodes for the layer and its sub-blocks (the following functions will create the sub-blocks themselves)
            if self.config.parallel_attn_mlp_architecture:  # like gpt-j
                self.setup_parallel_block(layer_idx, graph_data, hs_collector=hs_collector, target_word=target_word, row_idx=row_idx)
            else:  # like gpt2
                self.setup_sequential_block(layer_idx, graph_data, hs_collector=hs_collector, target_word=target_word, row_idx=row_idx)

            # each layer is a block of two sub-blocks: the attention block and the mlp block
            self.layer_attn_to_graph(layer_idx, graph_data, hs_collector=hs_collector, target_word=target_word, line=line, row_idx=row_idx)
            self.layer_mlp_to_graph(layer_idx, graph_data, hs_collector=hs_collector, target_word=target_word, row_idx=row_idx)
        
        self.plot_graph_aux(graph_data, title=f'Flow-Grpah of layers {layers}--> prompt: "{line}". target: "{target_word}"', save_html=save_html)
        
        self.model = self.model.to(model_old_device)

        return graph_data
import json
from dataclasses import dataclass, field


@dataclass
class GraphConfigs:
    """
    Simple wrapper to store feature configs Flow Graphs
    """
    layer_format: str  # how to access (list) each layer given HuggingFace model, for example: in gpt2: model.transformer.h.{}
    layer_mlp_format: str  # how to access each mlp layer, for example: in gpt2: model.transformer.h.{}.model
    layer_attn_format: str  # how to access each attention layer, for example: in gpt2: model.transformer.h.{}.attn

    ln1: str  # layer norm of attention sublayer, for example: in gpt2: model.transformer.h.{}.attn.ln_1
    # in case of only one layer norm for both sublayers, use the same string for ln1 and ln2
    attn_q: str  # Q, query, for example: in gpt2: model.transformer.h.{}.attn.c_attn (in gpt2, attn_q,k,v are in the same matrix, and this case: write the same string for all of them)
    attn_k: str  # K, key
    attn_v: str  # V, value
    attn_o: str  # O, projection

    ln2: str  # layer norm of MLP sublayer, for example: in gpt2: model.transformer.h.{}.attn.ln_2
    mlp_ff1: str  # FF1, first layer of MLP, for example: in gpt2: model.transformer.h.{}.mlp.c_fc
    mlp_ff2: str  # FF2, second layer of MLP, for example: in gpt2: model.transformer.h.{}.mlp.c_proj

    include_mlp_bias: bool = True
    include_attn_bias: bool = True

    transpose_attn_o: bool = False  # if to transpose the attn_o matrix, for example: in gpt-neo and llama2

    config_name: str = "default_v1"

    round_digits: int = 3 
    # the numbers of top and bottom neurons to show at the mlp matricies (FF1, FF2) and the attention projection matrix (W_O / attn.c_proj)
    number_of_top_neurons: int = 20
    number_of_bottom_neurons: int = 10
    defualt_weight: int = 7
    factor_weight_mlp_key_value_link: float = 1.5
    # number of ki, vi to show for each head. great to examine with gpt2-small/medium but for gpt2-large/xl you might want to reduce this number
    n_values_per_head: int = 2
    factor_attn_score: int = 10
    factor_head_output: float = 1.8
    factor_head_norm: float = 0.2
    factor_for_memory_to_head_values: float = 1.3

    # if to merge those types of nodes into one node to save space, since they are having only input and output ranks of 1 and used as direct mapping between each other
    compact_mlp_nodes: bool = False
    compact_attn_k_v_nodes: bool = False
    parallel_attn_mlp_architecture: bool = False

    cmap_node_rank_target: list = field(default_factory=lambda: ['lime', 'greenyellow', 'yellow' , 'yellow', 'yellow'] + ['dimgrey']*53 + ['orangered']*3 + ['red']*4 + ['darkred']*5)
    cmap_attn_score: list = field(default_factory=lambda: ['khaki', 'yellow', 'green'])
    cmap_entropy: list = field(default_factory=lambda: ['darkslategrey', 'lightgrey', 'lightgrey'])
    backgroud_color: str = 'black'
    invisible_link_color: str = 'black'  # better to be the same as backgroud_color
    color_for_abstract: str = 'white'
    default_color: str = 'white'
    link_with_normalizer: str = 'darkviolet'
    color_for_bias_vector: str = 'pink'
    positive_activation: str = 'blue'
    negative_activation: str = 'red'
    color_attn_residual: str = 'rgba(102,0,204,0.3)'  # unique color for attn residual
    color_mlp_residual: str = 'rgba(51,102,153,0.3)'  # unique color for mlp residual


    @classmethod
    def from_json(cls, fpath, shell_port=None):
        with open(fpath, "r") as f:
            data = json.load(f)
            print(data)

        return cls(**data)

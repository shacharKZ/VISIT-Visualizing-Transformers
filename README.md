# VISIT: Visualizing and Interpreting the Semantic Information Flow of Transformers

This repository contains the code for the paper [VISIT: Visualizing and Interpreting the Semantic Information Flow of Transformers](https://arxiv.org/abs/2305.13417).

Try our demo: [![Colab VISIT Demo](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1_iOJvbri_7jzmqTVBb_T6zl_08_hl3hY?usp=sharing)

<!-- And another version without the markdowns: [![Colab VISIT Demo](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1c2Erx-JC3cRLtYKqGJBUZ4QeiKutCZ_T?usp=sharing) -->

By using this notebook you can create dynamic plots that reflect the forward passes of GPTs from a semantic perspective. These plots illustrate the information flow within the models and provide insights into the impact of each component on the semantic information flow.

![Picture230904](https://github.com/shacharKZ/VISIT-Visualizing-Transformers/assets/57663126/2af753ec-c252-4d3d-8021-3bc37d36e8be)


Our implementation currently works with multi-head attention decoders like OpenAI's GPT-2 and EleutherAI's GPT-j (both from HuggingFace).

Feel free to open an issue if you find any problems or contact us to discuss any related topics.

# Requirements:
Tested with Python 3.9.7 . After cloning our code, run the following command in your Python enviroment to install the required packages:
```pip install -r requirements.txt ```

# Generate Flow Graphs:
To generate the flow graphs you can use the interactive notebooks we provided (including our demo on Colab [![Colab VISIT Demo](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1c2Erx-JC3cRLtYKqGJBUZ4QeiKutCZ_T?usp=sharing), this version is with explicit code unlike the version at the top of the page with form-like UI) or to convert them into ``.py`` files using ```jupyter nbconvert <The relevent notebook>.ipynb --to python``` and run them from the command line.

To run the code from the command line, use the following template:
```
python generate_flow_graphs.py --model_name <The model name or path> \
    --line <The input line> \
    --graph_config_path <The path to the graph config file> \
    --layers_to_check <The layers to check>
```

For example:
```
python generate_flow_graphs.py --model_name "gpt2-medium" \
    --line "The capital of Japan is the city of" 
    --graph_config_path "flow_graph_configs/flow_graph_config_basic.json" \
    --layers_to_check "[10,14]"
```
or (uses a color blind friendliness configuration):

```
python generate_flow_graphs.py --model_name "gpt2-xl" \
    --line "Lionel Messi plays for" \
    --graph_config_path "flow_graph_configs/flow_graph_config_basic_color_palette2.json" \
    --layers_to_check "[15,16]"
```


To generate GPT-j model: (use ```--model_revision "float16"```)

```
python generate_flow_graphs.py --model_name "EleutherAI/gpt-j-6B" \
    --model_revision "float16" \
    --line "The capital of Japan is the city of" \
    --graph_config_path "flow_graph_configs/flow_graph_config_gptj.json" \
    --layers_to_check "[10,14]"
```

Note the graph can be plotted via the IDE or available browser, as well as saved to an HTML file.

# GREAT NEWS! We support GPT-Neo and Llama2-7B !
Please make sure to use the correct graph config file for each model and to get the right access to the model if you are using the llama2 model from HuggingFace.

```
python generate_flow_graphs.py --model_name "EleutherAI/gpt-neo-1.3B" \
    --line "The capital of Japan is the city of" \
    --graph_config_path "flow_graph_configs/flow_graph_config_gpt_neo.json" \
    --layers_to_check "[10,14,18]"
```

```
python generate_flow_graphs.py --model_name "meta-llama/Llama-2-7b-chat-hf" \
    --line "The capital of Japan is the city of" \
    --graph_config_path "flow_graph_configs/flow_graph_config_llama2_7B.json" \
    --layers_to_check "[10,14,18]"
```

# More models:
Our tool should be able to handle any GPT-like model (autoregressive decoder with multi-head self-attention). Please check out [flow_graph_configs](https://github.com/shacharKZ/VISIT-Visualizing-Transformers/tree/main/flow_graph_configs) folder for examples of how to configure the tool for other models. If you have any questions, please contact us.


## How to Cite

```bibtex
@article{katz2023visit,
      title={VISIT: Visualizing and Interpreting the Semantic Information Flow of Transformers}, 
      author={Shahar Katz and Yonatan Belinkov},
      year={2023},
      eprint={2305.13417},
      archivePrefix={arXiv},
}
```

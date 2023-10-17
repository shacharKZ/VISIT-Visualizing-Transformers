from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import os
import flow_graph

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(f'Using device: {device} [cuda available? => {torch.cuda.is_available()}]')

# TODO pass the model and input text as arguments or rewrite the default values
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--model_name', type=str, default='gpt2-medium')  # for models such as GPT-j and others we recommend looking at our GitHub repo
parser.add_argument('--model_revision', type=str, default='main')
parser.add_argument('--line', type=str, default='The capital of Japan is the city of')
parser.add_argument('--target_token', type=str, default='AUTO')
parser.add_argument('--save_html', action='store_true', default=True)
parser.add_argument('--graph_config_path', type=str, default='flow_graph_configs/flow_graph_config_basic.json')
parser.add_argument('--layers_to_check', type=str, default='[14]')

args, unknown = parser.parse_known_args()
print('unknown args:', unknown)
print('args:', args)

model_name = args.model_name
line = args.line
target_token = args.target_token

model = AutoModelForCausalLM.from_pretrained(model_name, revision=args.model_revision).to(device).requires_grad_(False).eval()
tokenizer = AutoTokenizer.from_pretrained(model_name)
try:
    os.environ["TOKENIZERS_PARALLELISM"] = "true"  # not blocking, just to prevent warnings messages and faster tokenization
except:
    pass

print(model)
print(model.config)

flow_graph_obj = flow_graph.FlowGraph(model=model,
                                      tokenizer=tokenizer,
                                      config_path=args.graph_config_path,
                                      device=device)

model_answer, hs_collector = flow_graph_obj.infrence_model_and_collect_data_for_graph(line)
print(f'model_answer: "{model_answer}"')

if target_token == 'AUTO':
    target_token = model_answer
    print(f'target_token: "{target_token}" (set by AUTO option)')

layers_to_check = args.layers_to_check.replace('[', '').replace(']', '')
layers_to_check = [int(layer) for layer in layers_to_check.split(',')]
for layer_idx in layers_to_check:
    graph_data = flow_graph_obj.gen_basic_graph(layers=[layer_idx],
                                hs_collector=hs_collector,
                                target_word=target_token,
                                line=line,
                                save_html=args.save_html)

# graph_data = flow_graph_obj.gen_basic_graph(layers=[10, 11],
#                             hs_collector=hs_collector,
#                             target_word=target_token,
#                             line=line,
#                             save_html=args.save_html)

# graph_data = flow_graph_obj.gen_basic_graph(layers=[20, 21],
#                             hs_collector=hs_collector,
#                             target_word=target_token,
#                             line=line,
#                             save_html=args.save_html)

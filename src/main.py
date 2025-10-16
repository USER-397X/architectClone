from arcloader import ArcDataset
from utils import load_unsloth_4bit



base_model = 'da-fr/Llama-3.2-3B-ARChitects-ReArc-bnb-4bit'  # auto-downloaded from huggingface.co
arc_data_path = '../data/re_arc'

# model, tokenizer = load_unsloth_4bit(base_model)

train_data = ArcDataset.load_from_rearc(arc_data_path, n =50, sizes=[3,4,6], seed=42)

print(train_data[1])

# create lora model
lora_layers = ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj', 'embed_tokens', 'lm_head']

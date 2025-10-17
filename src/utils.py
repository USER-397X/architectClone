from transformers import DataCollatorForCasaulLM
from pathlib import Path

def load_unsloth_4bit(model_path):
    from unsloth import FastLanguageModel
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_path,
        dtype=None,
        load_in_4bit=True,
    )
    if model.max_seq_length == 2048 < model.generation_config.max_length:
        print(f'CHANGING MAX_SEQ_LENGTH {model.max_seq_length} -> {model.generation_config.max_length} (unsloth bug?)')
        to_fix = model
        while to_fix is not None:
            to_fix.max_seq_length = model.generation_config.max_length
            to_fix = getattr(to_fix, 'model', None)
    return model, tokenizer


def save_model_and_tokenizer(store_path, model, tokenizer):
    model.save_pretrained(store_path)
    tokenizer.save_pretrained(store_path)
    
    # Convert the string path to a Path object
    store_path_obj = Path(store_path)
    
    # Use the / operator to join paths and create the file path
    to_delete = store_path_obj / 'tokenizer.model'
    
    # Check if the file exists and delete it if it does
    if to_delete.is_file():
        to_delete.unlink()

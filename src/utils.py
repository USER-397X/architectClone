from transformers import DataCollatorForCausalLM
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

# class InputMaskingDataCollator1(DataCollatorForLanguageModeling):
#     def __init__(self, mask_first_n_examples, instruction_template, response_template, **kwargs):
#         self.instruction_template = instruction_template
#         self.response_template = response_template
#         self.mask_first_n_examples = mask_first_n_examples
#         super().__init__(**kwargs)

#     def torch_call(self, examples):
#         batch = super().torch_call(examples)  # call super, masking all inputs
#         for i in range(len(batch['labels'])):
#             for _ in range(self.mask_first_n_examples):
#                 # mask first still unmasked output block
#                 beg_pos = ((batch['labels'][i] != -100).nonzero().min()).item()
#                 mid_pos = ((batch['labels'][i][beg_pos:] == -100).nonzero().min()).item() + beg_pos
#                 end_pos = ((batch['labels'][i] != -100).nonzero().max()).item() + 1
#                 if mid_pos < end_pos:
#                     batch['labels'][i][beg_pos:mid_pos] = -100
#         return batch
    


class InputMaskingDataCollator(DataCollatorForCausalLM):
    def __init__(self, mask_first_n_examples, **kwargs):
        # We don't need instruction/response templates here, they are part of fmt_opts
        self.mask_first_n_examples = mask_first_n_examples
        # The parent class now handles the tokenizer directly
        super().__init__(**kwargs)

    def torch_call(self, examples):
        # 1. Call the parent class (DataCollatorForCausalLM). 
        #    This correctly masks the entire prompt and prepares the labels.
        batch = super().torch_call(examples)
        
        # 2. Apply your custom logic on top to mask the first n examples.
        #    This part of the logic does not need to change.
        for i in range(len(batch['labels'])):
            for _ in range(self.mask_first_n_examples):
                # Find the first unmasked token (start of the entire answer block)
                unmasked_indices = (batch['labels'][i] != -100).nonzero(as_tuple=True)[0]
                if len(unmasked_indices) == 0:
                    continue # Skip if everything is already masked

                beg_pos = unmasked_indices.min().item()
                
                # Find the first separator token AFTER the beginning of the answer block
                masked_after_beg = (batch['labels'][i][beg_pos:] == -100).nonzero(as_tuple=True)[0]
                if len(masked_after_beg) == 0:
                    continue # No separators found, nothing more to mask

                mid_pos = masked_after_beg.min().item() + beg_pos
                
                # Mask the first example's output
                batch['labels'][i][beg_pos:mid_pos] = -100
                
        return batch
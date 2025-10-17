from arcloader import ArcDataset

from utils import load_unsloth_4bit
from utils import save_model_and_tokenizer

from unsloth import FastLanguageModel
from unsloth import UnslothTrainer as Trainer, unsloth_train, is_bfloat16_supported
from unsloth import UnslothTrainingArguments as TrainingArguments

from datasets import Dataset

base_model = "da-fr/Llama-3.2-3B-ARChitects-ReArc-bnb-4bit"  # auto-downloaded from huggingface.co
arc_data_path = "../data/re_arc"
save_model_path = '../output/model'

model = tokenizer = None  # free memory
model, tokenizer = load_unsloth_4bit(base_model)
# keep_tok = list('ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789!?.:,;*+/-=')+tokenizer.tokenize('\n')
# keep_single_char_tokens(model, tokenizer, keep=keep_tok, remove_unk=True)

train_data = ArcDataset.load_from_rearc(arc_data_path, n=50, sizes=[3, 4, 6], seed=42)

# set formatting options
fmt_opts = dict(
    preprompt='ABCDEFGHJKLMNPQRSTUVWXYZabcdefghjklmnpqrstuvwxyz',
    query_beg='I',
    reply_beg='\n+/-=O',
    reply_end='\n' + tokenizer.eos_token,
    lines_sep='\n',
    max_tokens=128000,
)

# create lora model
lora_layers = ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj', 'embed_tokens', 'lm_head']

model = FastLanguageModel.get_peft_model(
    model=model,
    target_modules=lora_layers,
    r=256,
    lora_alpha=24,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing=True,
    random_state=42,
    use_rslora=True,
    loftq_config=None,
)

# run training
FastLanguageModel.for_training(model)
tokenizer.padding_side = "right"
trainer = Trainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=Dataset.from_list(train_data),
    dataset_text_field="text",
    max_seq_length=fmt_opts["max_tokens"],
    packing=False,
    data_collator= InputMaskingDataCollator(
        instruction_template=fmt_opts["query_beg"],
        response_template=fmt_opts["reply_beg"],
        mlm=False,
        tokenizer=tokenizer,
        mask_first_n_examples=1,
    ),
    args=TrainingArguments(
        per_device_train_batch_size=4,
        gradient_accumulation_steps=2,
        warmup_ratio=0.25,
        num_train_epochs=1,
        learning_rate=1e-4,
        embedding_learning_rate=1e-5,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        logging_steps=10,
        optim="adamw_8bit",
        weight_decay=0.00,
        lr_scheduler_type="cosine",
        seed=42,
        output_dir="tmp_output",
        save_strategy="no",
        report_to="none",
    ),
)
trainer_stats = unsloth_train(trainer)
save_model_and_tokenizer(f"{save_model_path}-lora", model, tokenizer)

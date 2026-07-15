from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from peft import LoraConfig, TaskType, get_peft_model

# load base model for finetuning
model = AutoModelForCausalLM.from_pretrained("gpt2", dtype="auto", device_map="auto")
tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

# stores all important parameters for building a PeftModel
lora_config = LoraConfig(
    # task to train (causal language modeling)
    task_type=TaskType.CAUSAL_LM, 

    # names of modules to apply the adapter to
    target_modules=["c_attn"],

    # the dimension of the low-rank matrices
    r=8, 

    # the scaling factors for the low-rank matrices
    lora_alpha=32, 

    # the dropout probability of the LoRA layers
    lora_dropout=0.1 
)

# configures a base model for training with LoRA
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# downloads the first 1% of the WikiText-2 raw training dataset
dataset = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train[:1%]")

# tokenize the text and convert it into PyTorch tensors
def tokenize_dataset(dataset):
    return tokenizer(dataset["text"], truncation=True, max_length=256)
dataset = dataset.map(tokenize_dataset, batched=True, remove_columns=dataset.column_names) # remove a selection of columns while doing the mapping
dataset.set_format(type="torch", columns=["input_ids", "attention_mask"]) # objects should be of type torch

# create batches of data and pass the tokenizer to it
from transformers import DataCollatorForLanguageModeling

# inputs are dynamically padded to the maximum length of a batch if they are not all of the same length
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# set up training features and hyperparameters
from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir="lora-output",
    per_device_train_batch_size=8,

    # number of update steps to accumulate gradients before performing a backward/update pass
    gradient_accumulation_steps=4, 

    num_train_epochs=1,
    
    # number of update steps between two logs
    logging_steps=10,

    save_strategy="no",

    # enable float16 (FP16) mixed precision training
    fp16=True
)

# pass all training components to Trainer and call train() to start
from transformers import Trainer

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    # eval_dataset=dataset["test"],
    # processing_class=tokenizer,
    data_collator=data_collator,
)

trainer.train()

# save model and tokenizer to lora-output directory
model.save_pretrained("lora-output")
tokenizer.save_pretrained("lora-output")

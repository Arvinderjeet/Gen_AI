from transformers import (
    LlamaTokenizerFast,
    LlamaConfig,
    LlamaForCausalLM,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    pipeline
)
from datasets import load_dataset
import os
import sys

# Set model name and load tokenizer FIRST (before any function definitions)
model_name = "Llama-2-7b-hf"
tokenizer = LlamaTokenizerFast.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

def tokenize_function(examples):
    return tokenizer(examples["text"])

def main():
    dataset_file = "data_set.txt"

    if not os.path.exists(dataset_file):
        print(f"Error: Dataset file '{dataset_file}' not found.")
        print("Please create this file in the same directory as the script.")
        sys.exit(1)

    print(f"Loading dataset from '{dataset_file}'...")
    dataset = load_dataset("text", data_files=dataset_file)
    print("Dataset loaded:")
    print(dataset)

    print("Tokenizing dataset...")
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        num_proc=2,      # Safe due to global tokenizer
        remove_columns=["text"]
    )
    print("Dataset tokenized:")
    print(tokenized_dataset)

    print("Configuring a new LLaMA model from scratch...")
    config = LlamaConfig(
        vocab_size=tokenizer.vocab_size,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        num_hidden_layers=6,
        num_attention_heads=6,
        hidden_size=384,
        intermediate_size=1536,
    )

    model = LlamaForCausalLM(config)
    print("Model configured.")
    print(f"Model parameter count: {model.num_parameters():,}")

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    output_model_dir = './llama2-custom-model'

    training_args = TrainingArguments(
        output_dir=output_model_dir,
        eval_steps=50,
        logging_steps=10,
        save_steps=100,
        save_total_limit=2,
        num_train_epochs=3,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        fp16=True,   # set False if on CPU!
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset['train'],
        eval_dataset=tokenized_dataset['train'],
        data_collator=data_collator,
        tokenizer=tokenizer
    )

    print("Starting model training...")
    trainer.train()
    print("Training complete.")

    print(f"Saving final model and tokenizer to '{output_model_dir}'...")
    model.save_pretrained(output_model_dir)
    tokenizer.save_pretrained(output_model_dir)
    print("Model saving complete.")

    print("Loading trained model for text generation...")
    # If using GPU: device=0, for CPU use device=-1
    generator = pipeline('text-generation', model=output_model_dir, tokenizer=output_model_dir, device=-1)

    prompts = ["what is cricket", "the heart is"]
    for prompt in prompts:
        print(f"Generating text for prompt: '{prompt}'")
        output = generator(prompt, max_length=100, num_return_sequences=1)
        print("\n--- Generated Text ---")
        print(output[0]['generated_text'])
        print("------------------------")

if __name__ == "__main__":
    main()
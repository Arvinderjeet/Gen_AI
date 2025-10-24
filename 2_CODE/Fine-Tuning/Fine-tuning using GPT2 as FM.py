from transformers import (
    GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments,
    DataCollatorForLanguageModeling, pipeline
)
from datasets import load_dataset

model_name = 'gpt2'
dataset = load_dataset("text", data_files="shoolini_finetune.txt")
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

tokenizer.pad_token = tokenizer.eos_token
model.resize_token_embeddings(len(tokenizer))

def tokenize_function(examples):
    tokenized_inputs = tokenizer(
        examples["text"],
        padding="max_length",
        truncation=True,
        max_length=128
    )
    tokenized_inputs["labels"] = tokenized_inputs["input_ids"].copy()
    return tokenized_inputs

if __name__ == "__main__":
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        num_proc=2,
        remove_columns=["text"]
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir='./gpt2_shoolini',
        overwrite_output_dir=True,
        num_train_epochs=3,
        per_device_train_batch_size=4,
        save_steps=500,
        save_total_limit=2
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        tokenizer=tokenizer,
        data_collator=data_collator,
        train_dataset=tokenized_dataset['train']
    )

    trainer.train()

    trainer.save_model('./gpt2_shoolini')

    text_gen = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        pad_token_id=tokenizer.eos_token_id
    )

    input_text = "Shoolini University is"

    output = text_gen(input_text, max_length=50, num_return_sequences=1)
    print(output[0]['generated_text'])

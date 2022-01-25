from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)
from huggingface_hub import HfFolder

import nltk

nltk.download("punkt")

import numpy as np
from nltk.tokenize import sent_tokenize
from datasets import load_metric


# !pip install transformers datasets sentencepiece pandas matplotlib rouge_score nltk torch
# !pip install git+https://github.com/huggingface/transformers.git@master "tokenizers==0.11.2"


model_id = "philschmid/pt-test"
dataset_id = "philschmid/prompted-germanquad"


ds = load_dataset(dataset_id)

tokenizer = AutoTokenizer.from_pretrained(model_id)


max_input_length = 512
max_target_length = 128


def preprocess_function(examples):
    model_inputs = tokenizer(examples["inputs"], max_length=max_input_length, truncation=True)
    # Set up the tokenizer for targets
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(examples["targets"], max_length=max_target_length, truncation=True)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


tokenized_datasets = ds["train"].map(preprocess_function, batched=True)

# test size will be 15% of train dataset
test_size = 0.15
seed = 33

processed_dataset = tokenized_datasets.shuffle(seed=seed).train_test_split(test_size=test_size)
# processed_dataset = tokenized_datasets.shuffle(seed=seed).select(range(2000)).train_test_split(test_size=test_size)


model = AutoModelForSeq2SeqLM.from_pretrained(model_id)


data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)


per_device_train_batch_size = 4
per_device_eval_batch_size = 2
num_train_epochs = 3
learning_rate = 5.6e-5
# Show the training loss with every epoch
logging_steps = len(processed_dataset["train"]) // per_device_train_batch_size
model_name = model_id.split("/")[-1]
output_dir = f"{model_name}-prompted-germanquad-2"
hub_token = HfFolder.get_token()  # or your token directly "hf_xxx"
hub_model_id = f'{model_id.split("/")[1]}-{dataset_id}-v1'

args = Seq2SeqTrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=per_device_train_batch_size,
    per_device_eval_batch_size=per_device_eval_batch_size,
    num_train_epochs=num_train_epochs,
    learning_rate=learning_rate,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=3,
    predict_with_generate=True,
    logging_steps=logging_steps,
    load_best_model_at_end=True,
    metric_for_best_model="rougeLsum",
    # push_to_hub=True,
    # hub_model_id=hub_model_id,
    # hub_token=hub_token,
)


rouge_score = load_metric("rouge")


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    # Decode generated summaries into text
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    # Replace -100 in the labels as we can't decode them
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    # Decode reference summaries into text
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    # ROUGE expects a newline after each sentence
    decoded_preds = ["\n".join(sent_tokenize(pred.strip())) for pred in decoded_preds]
    decoded_labels = ["\n".join(sent_tokenize(label.strip())) for label in decoded_labels]
    # Compute ROUGE scores
    result = rouge_score.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
    # Extract the median scores
    result = {key: value.mid.fmeasure * 100 for key, value in result.items()}
    return {k: round(v, 4) for k, v in result.items()}


trainer = Seq2SeqTrainer(
    model,
    args,
    train_dataset=processed_dataset["train"],
    eval_dataset=processed_dataset["test"],
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

trainer.train()

trainer.evaluate()


# trainer.push_to_hub(commit_message="Training complete", tags="text2text")

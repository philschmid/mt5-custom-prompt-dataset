import argparse
import logging
import os
import sys
import nltk

nltk.download("punkt")
import numpy as np
from nltk.tokenize import sent_tokenize
from datasets import load_dataset, load_metric
from transformers import (
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    AutoTokenizer,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)
from huggingface_hub import HfFolder

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # hyperparameters sent by the client are passed as command-line arguments to the script.
    parser.add_argument("--model_id", type=str)
    parser.add_argument("--dataset_id", type=str)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--train_batch_size", type=int, default=8)
    parser.add_argument("--eval_batch_size", type=int, default=16)
    parser.add_argument("--warmup_steps", type=int, default=500)
    parser.add_argument("--learning_rate", type=str, default=5.6e-5)
    # parser.add_argument("--fp16", type=bool, default=True) # not supported https://github.com/huggingface/transformers/pull/10956
    parser.add_argument("--max_input_length", type=int, default=512)
    parser.add_argument("--max_target_length", type=int, default=128)

    # Push to Hub Parameters
    parser.add_argument("--push_to_hub", type=bool, default=True)
    parser.add_argument("--hub_model_id", type=str, default=None)
    # parser.add_argument("--hub_strategy", type=str, default="all_checkpoints")
    parser.add_argument("--hub_strategy", type=str, default="every_save")
    parser.add_argument("--hub_token", type=str, default=None)

    # Data, model, and output directories
    # SageMaker variables
    # parser.add_argument("--output_data_dir", type=str, default=os.environ["SM_OUTPUT_DATA_DIR"])
    # parser.add_argument("--output_dir", type=str, default=os.environ["SM_MODEL_DIR"])
    # parser.add_argument("--n_gpus", type=str, default=os.environ["SM_NUM_GPUS"])
    # parser.add_argument("--training_dir", type=str, default=os.environ["SM_CHANNEL_TRAIN"])
    # parser.add_argument("--test_dir", type=str, default=os.environ["SM_CHANNEL_TEST"])

    args, _ = parser.parse_known_args()

    # sets hub id if not provided
    if args.hub_model_id is None:
        args.hub_model_id = args.model_id.replace("/", "--")

    # Set up logging
    logger = logging.getLogger(__name__)

    logging.basicConfig(
        level=logging.getLevelName("INFO"),
        handlers=[logging.StreamHandler(sys.stdout)],
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # load datasets
    ds = load_dataset(args.dataset_id)

    # load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)

    def preprocess_function(examples):
        model_inputs = tokenizer(examples["inputs"], max_length=args.max_input_length, truncation=True)
        # Set up the tokenizer for targets
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(examples["targets"], max_length=args.max_target_length, truncation=True)

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    tokenized_datasets = ds["train"].map(preprocess_function, batched=True)
    # test size will be 15% of train dataset
    test_size = 0.15
    seed = 33

    processed_dataset = tokenized_datasets.shuffle(seed=seed).train_test_split(test_size=test_size)
    # processed_dataset = tokenized_datasets.shuffle(seed=seed).select(range(2000)).train_test_split(test_size=test_size)

    logger.info(f" loaded train_dataset length is: {len(processed_dataset['train'])}")
    logger.info(f" loaded test_dataset length is: {len(processed_dataset['test'])}")

    # define metrics and metrics function
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

    # model & data collator
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_id)
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    model_name = args.model_id.split("/")[-1]
    logging_steps = len(processed_dataset["train"]) // args.train_batch_size

    args = Seq2SeqTrainingArguments(
        output_dir=f"{model_name}-prompted-germanquad",
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=0.01,
        warmup_steps=args.warmup_steps,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=3,
        predict_with_generate=True,
        logging_steps=logging_steps,
        load_best_model_at_end=True,
        metric_for_best_model="rougeLsum",
        push_to_hub=True,
        hub_model_id=args.hub_model_id,
        hub_token=HfFolder.get_token(),
        hub_strategy=args.hub_strategy,
    )

    # create Trainer instance
    trainer = Seq2SeqTrainer(
        model,
        args,
        train_dataset=processed_dataset["train"],
        eval_dataset=processed_dataset["test"],
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    # train model
    trainer.train()

    # evaluate model
    eval_result = trainer.evaluate()

    # save best model, metrics and create model card
    trainer.create_model_card(model_name=args.hub_model_id)
    trainer.push_to_hub()

    # Saves the model to s3 uses os.environ["SM_MODEL_DIR"] to make sure checkpointing works
    # trainer.save_model(os.environ["SM_MODEL_DIR"])

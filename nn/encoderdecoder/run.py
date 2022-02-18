if "__file__" in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))

from utils import EncoderDecoderAccuracyMetrics
import numpy
# import wandb
import torch
from transformers import (
    EncoderDecoderModel,
    BertTokenizer,
    AutoTokenizer,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    Trainer,
    DataCollatorForSeq2Seq,
)
from datasets import load_dataset
import pandas as pd
import random

homedir = input("Home dir: ")



tokenizer = AutoTokenizer.from_pretrained("kykim/bert-kor-base")
model = EncoderDecoderModel.from_encoder_decoder_pretrained("kykim/bert-kor-base", "kykim/bert-kor-base")  # initialize Bert2Bert from pre-trained checkpoints


model.config.decoder_start_token_id = 2
model.config.pad_token_id = tokenizer.pad_token_id
model.config.vocab_size = model.config.decoder.vocab_size

dataset = load_dataset('csv', data_files='KMWP/data/train.csv', split='train')

dictdataset = dataset.train_test_split(0.02)

max_input_length = 118
max_target_length = 169

def prepare_train_features(examples):
    tokenized_examples = tokenizer(
        text=examples['problem'],
        # text_pair=examples['code'],
        padding='max_length',
        max_length=216
    )
    tokenized_examples["labels"] = tokenizer(
        text=examples['code'],
        # text_pair=examples['code'],
        padding='max_length',
        max_length=216
    )['input_ids']
    return tokenized_examples


tokenized_datasets = dictdataset.map(prepare_train_features, batched=True, remove_columns=dictdataset["train"].column_names).remove_columns('token_type_ids')

args = Seq2SeqTrainingArguments(
    output_dir='enc-to-dec-finetune',
    # overwrite_output_dir = True,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=1,
    num_train_epochs = 5,
    logging_strategy='epoch',
    save_strategy = 'epoch',
    evaluation_strategy = 'steps',
    eval_steps=30,
    # load_best_model_at_end = True,
)

data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

compute_metrics = EncoderDecoderAccuracyMetrics(tokenizer, f"{homedir}", classi_class=False)


trainer = Seq2SeqTrainer(
    model,
    args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

trainer.train()
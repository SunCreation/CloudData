if "__file__" in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))


import os
import torch
import wandb
from transformers import (
    AutoTokenizer, 
    GPT2LMHeadModel,
    TrainingArguments,
    Trainer,
)
from datasets import load_dataset
import pandas as pd
from datasets import load_metric
import numpy as np
from utils import GPTAccuracyMetrics, get_answer
import sys
import time
import random
import argparse as ap
from torch import nn
import torch.nn.functional as F
import json

torch.manual_seed(42)
np.random.seed(42)
random.seed(42) 
torch.cuda.manual_seed_all(42)

tokenizer = AutoTokenizer.from_pretrained('skt/kogpt2-base-v2', bos_token='</s>', sep_token='<sep>', eos_token='</s>', pad_token='<pad>')

model = GPT2LMHeadModel.from_pretrained('skt/kogpt2-base-v2', output_hidden_states=True)

with open('CloudData/math/data/inputdata.json','r') as f:
    testdata = json.load(f)
problem = testdata['1']['1']['problem'] + '<sys> 1<sys>'
code = testdata['1']['1']['code']
data = tokenizer(
    text=problem,
    text_pair=code,
    return_tensors='pt',
    padding='max_length',
    max_length=260
    )
data['attention_mask'] = testdata['1']['1']['attention_mask']
data['labels'] = testdata['1']['1']['lables']
# print(data)
# data['labels'] = data['input_ids']
# print(data)
# output = model(**data)
# print(output[2][-1][-1].shape)

class Verifier(nn.Module):
    def __init__(self):
        super(Verifier, self).__init__()
        self.kogpt = GPT2LMHeadModel.from_pretrained('skt/kogpt2-base-v2', output_hidden_states=True)
        self.linear1 = nn.Linear(768,64)
        self.linear2 = nn.Linear(64,1)
        self.sigmoid = torch.sigmoid

    def forward(self, data):
        self.kogpt.train()
        output = self.kogpt(**data)
        batch, n_head, senlen, emb = output[2][-1][-1].shape
        output = output[2][-1][-1].transpose(1,2).view(batch,-1,768)
        output = self.linear1(output)
        output = self.linear2(output)
        output = self.sigmoid(output)
        return output

verifier = Verifier()



def train():
    global verifier
    verifier.train()
    output = verifier(data)
    mse_loss = nn.MSELoss()
    output = mse_loss(output, data['labels'])
    output = output * data['attention_mask']
    loss = torch.sum(output)
    loss.backward()
    
train()


# homedir = os.getcwd()

# parser = ap.ArgumentParser(description='hyper&input')
# parser.add_argument('-i','--input_data', type=str, default='clean_all_correct', help='you can use csv file. without file extention')
# parser.add_argument('-d','--input_path', type=str, default=None, help='you can use csv filepath.')
# parser.add_argument('-o','--output_dir', type=str, default=homedir, help='std output & model save dir')
# parser.add_argument('-v','--validation_data', type=str, default=None, help='Optional')
# parser.add_argument('-b','--batch_size', type=int, default=16, help='default16')
# parser.add_argument('-s','--valbatch_size_perdevice', type=int, default=8, help='default8')
# parser.add_argument('-n','--modelname', type=str, default='PowerfulMyModel', help='Enter model name')
# parser.add_argument('-p','--projectname', type=str, default='kogpt2', help='Enter model name')

# args = parser.parse_args()

# val = args.validation_data

# filepath = args.input_path

# modelname = args.modelname

# project = args.projectname

# batch_size = args.batch_size

# valbatch_size_perdevice = args.valbatch_size_perdevice

# device_num = torch.cuda.device_count()


# def prepare_train_features(examples):
#     for i, j in enumerate(examples['problem']):
#         examples['problem'][i] = j + '<sys>' + str(examples["class"][i]) + '<sys>'

#     tokenized_examples = tokenizer(
#         text=examples['problem'],
#         text_pair=examples['code'],
#         padding='max_length',
#         max_length=260
#     )
#     tokenized_examples["labels"] = tokenized_examples["input_ids"].copy()
#     return tokenized_examples

# if filepath:
#     filelist = os.listdir(f'{homedir}/CloudData/math/data/{filepath}')
#     for filename in filelist:
#         graph = filename.split('.')[0]
#         mname = modelname + '_' + graph
#         wandb.init(project=project, entity="math-solver", name=mname)

#         tokenizer = AutoTokenizer.from_pretrained('skt/kogpt2-base-v2', bos_token='</s>', sep_token='<sep>', eos_token='</s>', pad_token='<pad>')

#         model = GPT2LMHeadModel.from_pretrained('skt/kogpt2-base-v2')

#         dataset = load_dataset('csv', data_files=f'{homedir}/CloudData/math/data/{filepath}/{filename}', split='train')
#         if val : valdataset = load_dataset('csv', data_files=f'{homedir}/CloudData/math/data/{val}.csv', split='train')
#         else: 
#             dictdataset = dataset.train_test_split(0.06)
#             tokenized_datasets = dictdataset.map(prepare_train_features, batched=True, remove_columns=dataset.column_names)

#         # tokenized_datasets = dataset.map(prepare_train_features, batched=True, remove_columns=dataset.column_names)
#         if val: 
#             tokenized_datasets = dataset.map(prepare_train_features, batched=True, remove_columns=dataset.column_names)
#             valtokenized_datasets = valdataset.map(prepare_train_features, batched=True, remove_columns=valdataset.column_names)

#         compute_metrics = GPTAccuracyMetrics(tokenizer, f"{homedir}", classi_class=True)

#         # print(tokenizer.decode(tokenized_datasets['train'][0]["input_ids"]))

#         args = TrainingArguments(
#             output_dir=mname,
#             overwrite_output_dir = True,
#             per_device_train_batch_size=batch_size//device_num,
#             per_device_eval_batch_size=valbatch_size_perdevice,
#             warmup_steps=400,
#             weight_decay=0.1,
#             max_steps=10000,
#             logging_strategy='steps',
#             logging_steps=100,
#             save_strategy = 'steps',
#             save_steps=100,
#             evaluation_strategy = 'steps',
#             eval_steps=100,
#             load_best_model_at_end = True,
#             report_to="wandb"
#         )
#         if val:
#             trainer = Trainer(
#                 model,
#                 args,
#                 train_dataset=tokenized_datasets,
#                 eval_dataset=valtokenized_datasets,
#                 compute_metrics=compute_metrics,
#             )
#         else:
#             trainer = Trainer(
#                 model,
#                 args,
#                 train_dataset=tokenized_datasets['train'],
#                 eval_dataset=tokenized_datasets['test'],
#                 compute_metrics=compute_metrics,
#             )

#         trainer.train()
#         wandb.finish()

# else:
#     tokenizer = AutoTokenizer.from_pretrained('skt/kogpt2-base-v2', bos_token='</s>', sep_token='<sep>', eos_token='</s>', pad_token='<pad>')

#     model = GPT2LMHeadModel.from_pretrained('skt/kogpt2-base-v2')

#     input_data = args.input_data
#     filename = input_data.split('/')[-1]
#     mname = modelname+'_'+filename.split('.')[0]
#     wandb.init(project=args.projectname, entity="math-solver", name=mname)

#     dataset = load_dataset('csv', data_files=f'{homedir}/CloudData/math/data/{input_data}.csv', split='train')
    
#     if val : valdataset = load_dataset('csv', data_files=f'{homedir}/CloudData/math/data/{val}.csv', split='train')
#     else: 
#         dictdataset = dataset.train_test_split(0.06)
#         tokenized_datasets = dictdataset.map(prepare_train_features, batched=True, remove_columns=dataset.column_names)

#     # tokenized_datasets = dataset.map(prepare_train_features, batched=True, remove_columns=dataset.column_names)
#     if val: 
#         tokenized_datasets = dataset.map(prepare_train_features, batched=True, remove_columns=dataset.column_names)
#         valtokenized_datasets = valdataset.map(prepare_train_features, batched=True, remove_columns=valdataset.column_names)

#     compute_metrics = GPTAccuracyMetrics(tokenizer, f"{homedir}", classi_class=True)

#     # print(tokenizer.decode(tokenized_datasets['train'][0]["input_ids"]))

#     args = TrainingArguments(
#         output_dir=mname,
#         overwrite_output_dir = True,
#         per_device_train_batch_size=batch_size//device_num,
#         per_device_eval_batch_size=valbatch_size_perdevice,
#         # num_train_epochs = 25,
#         warmup_steps=400,
#         weight_decay=0.1,
#         max_steps=10000,
#         logging_strategy='steps',
#         logging_steps=100,
#         save_strategy = 'steps',
#         save_steps=100,
#         evaluation_strategy = 'steps',
#         eval_steps=100,
#         load_best_model_at_end = True,
#         report_to="wandb"
#     )
#     if val:
#         trainer = Trainer(
#             model,
#             args,
#             train_dataset=tokenized_datasets,
#             # eval_dataset=valtokenized_datasets,
#             eval_dataset=valtokenized_datasets,
#             compute_metrics=compute_metrics,
#             # data_collator=data_collator,
#         )
#     else:
#         trainer = Trainer(
#             model,
#             args,
#             train_dataset=tokenized_datasets['train'],
#             # eval_dataset=valtokenized_datasets,
#             eval_dataset=tokenized_datasets['test'],
#             compute_metrics=compute_metrics,
#             # data_collator=data_collator,
#         )

#     trainer.train()

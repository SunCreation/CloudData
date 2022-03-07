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
from torch import nn, optim
import torch.nn.functional as F
import json
from tqdm import tqdm


torch.manual_seed(42)
np.random.seed(42)
random.seed(42) 
torch.cuda.manual_seed_all(42)

homedir = os.getcwd()

parser = ap.ArgumentParser(description='hyper&input')
parser.add_argument('-i','--input_data', type=str, default='val', help='you can use csv file. without file extention')
parser.add_argument('-d','--input_path', type=str, default=None, help='you can use csv filepath.')
parser.add_argument('-o','--output_dir', type=str, default=homedir, help='std output & model save dir')
parser.add_argument('-v','--verifier_model', type=str, default=None, help='verifier_name')
parser.add_argument('--val_dir', type=str, default=None, help='Optional')
parser.add_argument('-b','--batch_size', type=int, default=16, help='default16')
parser.add_argument('-s','--valbatch_size_perdevice', type=int, default=8, help='default8')
parser.add_argument('-n','--modelname', type=str, default='PowerfulMyModel', help='Enter model name')
parser.add_argument('-p','--projectname', type=str, default='kogpt2', help='Enter model name')

args = parser.parse_args()

veri = args.verifier_model

val_dir = args.val_dir

filepath = args.input_path

filename = args.input_data

modelname = args.modelname

project = args.projectname

batch_size = args.batch_size

valbatch_size_perdevice = args.valbatch_size_perdevice

device_num = torch.cuda.device_count()

# def prepare_train_features(examples):
#     for i, j in enumerate(examples['problem']):
#         examples['problem'][i] = j + '<sys>' + str(examples["class"][i]) + '<sys>'


#     tokenized_examples = tokenizer(
#         text=examples['problem'],
#         text_pair=examples['code'],
#         padding='max_length',
#         max_length=260
#     )

#     # for i, j in enumerate(examples['problem']):
#     #     A = tokenizer.encode(examples['problem'][i])
#     #     B = tokenizer.encode(examples['code'][i])
        
#     #     masks = torch.zeros(len(A))
#     #     codes = torch.ones(len(B))
#     #     attention_mask = torch.concat([masks,codes], axis=0)
#     #     supple = torch.zeros(260-attention_mask.shape[0])
#     #     attention_mask = torch.concat([attention_mask,supple], axis=0)

#     #     tokenized_examples['attention_mask'][i] = attention_mask
   
#     tokenized_examples["labels"] = tokenized_examples["input_ids"].copy()

#     return tokenized_examples

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
#     for i in range(len(examples['attention_mask'])):
#         tokenized_examples["attention_mask"][i] = torch.tensor(list(map(int,examples['attention_mask'][i].split()))+[0]*(260-len(examples['attention_mask'][i].split())))
#         tokenized_examples["labels"][i] = torch.tensor(list(map(int,examples['labels'][i].split()))+[0]*(260-len(examples['labels'][i].split())))
    
     
    # return tokenized_examples

tokenizer = AutoTokenizer.from_pretrained('skt/kogpt2-base-v2', bos_token='</s>', sep_token='<sep>', eos_token='</s>', pad_token='<pad>')


dataset = load_dataset('csv', data_files=f'{homedir}/CloudData/math/data/{filename}.csv', split='train')
# tokenized_datasets = dataset.map(prepare_train_features, batched=True, remove_columns=dataset.column_names)


class Verifier(nn.Module):
    def __init__(self, model_path='skt/kogpt2-base-v2'):
        super(Verifier, self).__init__()
        self.kogpt = GPT2LMHeadModel.from_pretrained(model_path, output_hidden_states=True)
        self.linear1 = nn.Linear(768,64)
        self.linear2 = nn.Linear(64,1)
        self.sigmoid = torch.sigmoid

    def forward(self, **data):
        self.kogpt.train()
        output = self.kogpt(**data)
        batch, n_head, senlen, emb = output[2][-1][-1].shape
        output = output[2][-1][-1].transpose(1,2).view(batch,-1,768)
        output = self.linear1(output)
        output = self.linear2(output)
        output = self.sigmoid(output)
        return output

    def joint(self, model):
        self.gen_model=model
        self.gen_model.to('cuda')

    def generate(self, tokenized_data, EVAL_BATCH=8, per_sentence=True, tokenizer=tokenizer):
        input_ = tokenized_data['input_ids']
        self.eval()
        outputs = self.gen_model.generate(input_, max_length = 260, do_sample=True, top_k=50, top_p=0.95, num_return_sequences=100)
        output_shape = outputs.shape
        # outputs_ = outputs.to('cpu')
        # outputs_ = tokenizer.decode(outputs_)
        # print(output_shape)
        attention_mask = tokenized_data['attention_mask']
        # print(len(attention_mask),output_shape)
        attention_mask = F.pad(attention_mask, (0,260-attention_mask.shape[1]),"constant", 0.0)
        # print(attention_mask, attention_mask.shape)
        attention_mask = attention_mask.repeat(EVAL_BATCH,1).to('cuda')
        scores = []
        self.to("cuda")
        for i in range(0,output_shape[0],EVAL_BATCH):
            output = outputs[i:i+EVAL_BATCH]
            batch = output.shape[0]
            score = self(input_ids=output, attention_mask=attention_mask[:batch], labels=attention_mask[:batch])
            # print(score)
            score = torch.sum(score, axis=1)
            # print(score)
            scores.extend(score.detach().to('cpu').numpy().tolist())
            print('WOW!!')
        maxnum = np.argmax(np.array(scores))
        maxoutput = outputs[maxnum].tolist()


        return maxoutput


verifier = Verifier()

verifier.load_state_dict(torch.load(f'{veri}.pt'))

model = GPT2LMHeadModel.from_pretrained(modelname).to('cuda')

verifier.joint(model)

verifier.to('cuda')

learning_rate = 0.00001

optimizer = optim.Adam(verifier.parameters(), lr=learning_rate)

scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.001, 
                steps_per_epoch=1000, epochs=10,anneal_strategy='linear')

BATCH_SIZE = 32
# data = pd.read_csv('CloudData/math/data/val.csv')

# print(data['problem'][2])
# print(tokenizer(data['problem'][0], return_tensors='pt')['input_ids'])
for i in dataset['problem'][:10]:

    maxoutput = verifier.generate(tokenizer(i, return_tensors='pt').to('cuda'))

    print(tokenizer.decode(maxoutput))

# train()

# torch.save(verifier.state_dict(), 'veri/first.pt')

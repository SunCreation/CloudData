from textwrap import indent
import yaml
import pandas as pd
import json
from postprocess import postprocess
import numpy as np
import torch as th
from transformers import (
    AutoTokenizer, 
    GPT2LMHeadModel,
)


with open('CloudData/math/data/inputdata.yaml', "r") as f:
    data = yaml.load(f, Loader=yaml.FullLoader)
# data = pd.DataFrame(data)


answer = pd.read_csv('CloudData/math/data/randint/val.csv')

# print(data)

tokenizer = AutoTokenizer.from_pretrained('skt/kogpt2-base-v2', bos_token='</s>', eos_token='</s>', pad_token='<pad>')


for i, j in data.items():
    for k in range(10):
        if postprocess(answer['answer'][i-1])==postprocess(j[k]['answer']):
            A = tokenizer.encode(answer['problem'][i-1]+'<sys> 1<sys>', return_tensors='np')
            B = tokenizer.encode(j[k]['code'], return_tensors='np')
            
            masks = np.zeros(A[0].shape, dtype=np.int32).tolist()
            codes = np.ones(B[0].shape, dtype=np.int32).tolist()
            attention_mask = masks+codes

            j[k]['attention_mask'] = attention_mask
            j[k]['lables'] = attention_mask
        else:
            A = tokenizer.encode(answer['problem'][i-1]+'<sys> 1<sys>', return_tensors='np')
            B = tokenizer.encode(j[k]['code'], return_tensors='np')
            masks = np.zeros(A[0].shape, dtype=np.int32).tolist()
            codes = np.ones(B[0].shape, dtype=np.int32).tolist()
            labels = np.zeros(B[0].shape, dtype=np.int32).tolist()
            attention_mask = masks+codes
            labels = masks + labels

            j[k]['attention_mask'] = attention_mask
            j[k]['lables'] = labels

data = json.dumps(data, indent=4)
with open('CloudData/math/data/inputdata.json', "w") as f:
    f.write(data)

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

    



filename = input('Enter input filename: ')
with open(f'CloudData/math/data/verifier_data/{filename}.yaml', "r") as f:
    data = yaml.load(f, Loader=yaml.FullLoader)
# data = pd.DataFrame(data)


answer = pd.read_csv('CloudData/math/data/clean_all_correct.csv')

# print(data)

tokenizer = AutoTokenizer.from_pretrained('skt/kogpt2-base-v2', bos_token='</s>', eos_token='</s>', pad_token='<pad>')


bins = pd.DataFrame()



for i, j in data.items():
    for k in range(100):
        if postprocess(answer['answer'][i])==postprocess(j[k]['answer']):
            A = tokenizer.encode(answer['problem'][i]+'<sys> 1<sys>', return_tensors='np')
            B = tokenizer.encode(j[k]['code'], return_tensors='np')
            
            masks = np.zeros(A[0].shape, dtype=np.int32).tolist()
            codes = np.ones(B[0].shape, dtype=np.int32).tolist()
            attention_mask = masks+codes

            j[k]['attention_mask'] = attention_mask
            j[k]['labels'] = attention_mask
        else:
            A = tokenizer.encode(answer['problem'][i]+'<sys> 1<sys>', return_tensors='np')
            B = tokenizer.encode(j[k]['code'], return_tensors='np')
            masks = np.zeros(A[0].shape, dtype=np.int32).tolist()
            codes = np.ones(B[0].shape, dtype=np.int32).tolist()
            labels = np.zeros(B[0].shape, dtype=np.int32).tolist()
            attention_mask = masks+codes
            labels = masks + labels

            j[k]['attention_mask'] = attention_mask
            j[k]['labels'] = labels
    bins = pd.concat([bins,j], axis=1)

bins.to_csv('testsample.csv',encoding='utf-8-sig')

# data = json.dumps(data, indent=4)

# with open('CloudData/math/data/inputdata.json', "w") as f:
#     f.write(data)

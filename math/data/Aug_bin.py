from koeda import EDA
import pandas as pd
import sys
import time
from sklearn.model_selection import train_test_split
from numpy import random
import re

data = pd.read_csv("train.csv")

eda = EDA(
    morpheme_analyzer="Okt", alpha_sr=0.0, alpha_ri=0.1, alpha_rs=0.0, prob_rd=0.0
)

def randomhangle(x):
    num = len(x)
    x = ''.join(list(map(chr,map(int, (random.randn(num)+1) * (48000-44132)/2 + 47432))))
    x = re.sub(r'[^가-힣]', '', x)
    char = ['가','이','는','은','일','로','히','게']
    x = x + '까?'
    for i in range(num//5):
        x = x.replace(x[random.randint(num-10)], char[random.randint(7)]+' ')
    x = x.replace('다', '다. ')
    return x


data_train = pd.DataFrame()
data_test = pd.DataFrame()

data_train["class"], data_test["class"], data_train["problem"], data_test["problem"], data_train["code"], data_test["code"], data_train["answer"], data_test["answer"] = \
    train_test_split(data["class"],data["problem"], data["code"], data["answer"], test_size=0.035, stratify=data['class'])

new1data = data_train
new2data = pd.DataFrame() # data_test[:]




new1data['problem'] = new1data['problem'].apply(lambda x: eda(x))
new2data['problem'] = data_test['problem'].apply(lambda x: randomhangle(x))
new2data['code'] = data_test['code'].apply(lambda x: "y = 'ab'\nprint(y)")
new2data['class'] = [9]*len(new2data)

data = pd.concat([data_train, new1data, new2data], axis=0)

data['problem'] = data['problem'].apply(lambda x: x + '<sys>') + data['class'].apply(lambda x: str(x) + '<sys>')
data_test['problem'] = data_test['problem'].apply(lambda x: x + '<sys>') + data_test['class'].apply(lambda x: str(x) + '<sys>')


data.to_csv("Agu_bin_train.csv")
data_test.to_csv("Val_bin_train.csv")
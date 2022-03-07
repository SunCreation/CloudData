import pandas as pd

data = pd.read_csv('data/verifier_data/verifier_data.csv')
valdata = pd.read_csv('data/val.csv')
# print(data.loc[51600]['problem'])
# print(valdata[valdata['Unnamed: 0']==516]['problem'])
# print(517 in valdata['Unnamed: 0'].to_list())

# traindata = data[data['']]
data['index'] = data['Unnamed: 0'].apply(lambda x:  not x//100 in valdata['Unnamed: 0'].to_list())
del data['Unnamed: 0']
print(data[data['index']])
data[data['index']].to_csv('data/verifier_data/verifier_train.csv')

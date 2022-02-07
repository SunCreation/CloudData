import sys
import pandas as pd
import time
data = pd.read_csv("agutrain.csv")

data['problem'] = data['problem'].apply(lambda x: x + '<sys>') + data['class'].apply(lambda x: str(x) + '<sys>')

print(data)
data.to_csv("Agutrain.csv")

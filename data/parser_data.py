import pandas as pd
import json

data = pd.read_csv('lvyou.csv', sep=',')
results=[]
for index,row in data.iterrows():
    d={"context":"Instruction: {}\nAnswer: ".format(row[0]),"target":row[1]} 
    results.append(json.dumps(d,ensure_ascii=False))

with open("wenlv_data.jsonl","w",encoding="utf-8") as f:
    f.writelines("\n".join(results))
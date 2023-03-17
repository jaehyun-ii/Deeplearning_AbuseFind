import pandas as pd
from soynlp.normalizer import *
import csv

data = pd.read_csv('data.csv')
f = open('data_nomalize.csv','w', newline='')
wr = csv.writer(f)
print(f'정상 글의 비율 = {round(data["isAbuse"].value_counts()[0]/len(data) * 100,3)}%')
print(f'비정상 글의 비율 = {round(data["isAbuse"].value_counts()[1]/len(data) * 100,3)}%')
X_data = data['text']
y_data = data['isAbuse']
wr.writerow(['text','isAbuse'])

for i in range(1,len(X_data)):
    j = only_hangle_number(X_data[i])
    j = emoticon_normalize(j)
    if j != "":
        wr.writerow([j,y_data[i]])



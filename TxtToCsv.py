import pandas as pd
import csv
import numpy as np
from sklearn.model_selection import train_test_split
ft = open('test.txt', 'r')
lines = ft.readlines()
fc = open("data.csv", "w")
writer = csv.writer(fc)
for i in lines:
    i=i.split(",")
    i[1]=i[1].replace("\n", "")
    writer.writerow(i)
ft.close()
fc.close()


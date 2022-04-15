import pandas as pd
import numpy as np
import sys
import os
import pickle
import json
import sklearn.metrics as metrics
from sklearn.ensemble import AdaBoostRegressor

os.getcwd('..')
dir = './data/'

model_file = sys.argv[1]
data_path = os.path.join(sys.argv[2]+"val.csv")
scores_file = sys.argv[3]

print(sys.argv[0])

print(data_path)
val = pd.read_csv(data_path + 'val.csv')

X_val = val.iloc[:,:-1]
y_val = val.iloc[:,-1]

sys.stderr.write("Input matrix size {}\n".format(val.shape))
sys.stderr.write("X matrix size {}\n".format(X_val.shape))
sys.stderr.write("Y matrix size {}\n".format(y_val.shape))

with open(model_file,'rb') as fd:
    model = pickle.load(fd)

res = model.predict(X_val)
MAPE = np.mean(np.abs(y_val - res) / y_val) * 100

res.to_csv(dir+'results.csv')

with open(scores_file, "w") as fd:
    json.dump({"MAPE": MAPE},
              fd,
              indent=4)

sys.stderr.write("Mape = {}%\n".format(round(MAPE,2)))

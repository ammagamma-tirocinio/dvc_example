import pandas as pd
import pickle
import sys
import os
from sklearn.ensemble import AdaBoostRegressor

train_path = os.path.join(sys.argv[1]+'train.csv')
output = sys.argv[2]

print(train_path)
train = pd.read_csv(train_path)

X_train = train.iloc[:,:-1]
y_train = train.iloc[:,-1]

sys.stderr.write("Input matrix size {}\n".format(train.shape))
sys.stderr.write("X matrix size {}\n".format(X_train.shape))
sys.stderr.write("Y matrix size {}\n".format(y_train.shape))

model = AdaBoostRegressor()
model.fit(X_train,y_train)

with open(output, "wb") as fd:
    pickle.dump(model, fd)
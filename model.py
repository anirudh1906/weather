# -*- coding: utf-8 -*-
"""
Created on Thu Mar  9 16:47:46 2023

@author: aniru
"""
import numpy as np
import pandas as pd
import pickle
df= pd.read_csv("3247640.csv", index_col="DATE")
print(df.head())

core= df[["PRCP","SNOW","SNWD","TMAX","TMIN"]].copy(deep=True)
print(core.head())

del core["SNOW"]
del core["SNWD"]

core["PRCP"]= core["PRCP"].fillna(0)

core= core.fillna(method="ffill")

core.index= pd.to_datetime(core.index)

print( core[["TMAX","TMIN"]].plot())

core["TARGET"]= core.shift(-1)["TMAX"]
print(core.head())

core= core.iloc[:-1,:]
core.head()

from sklearn.linear_model import Ridge
reg= Ridge(alpha=0.1)

predictors= ["PRCP","TMAX", "TMIN"]
train= core.loc[:"2020-12-31"]
test= core.loc["2021-1-1":]

reg.fit(train[predictors], train["TARGET"])

predictions= reg.predict(test[predictors])

from sklearn.metrics import mean_absolute_error
print("\n", mean_absolute_error( predictions, test["TARGET"]))

combined= pd.concat( [test["TARGET"],pd.Series(predictions, index=test.index)], axis=1)
combined.columns= ["ACTUAl","PREDICTED"]
print("\n", combined.head())

#data= {"PRCP":0.0, "TMAX":54.0, "TMIN":35.0}
#df2= pd.DataFrame(data, index=[0])
#print("\n", df2)
arr= np.array([[0, 54, 35]])
predict= reg.predict(arr)
print("\n", predict)      
      
pickle.dump( reg, open("model.pkl", "wb"))
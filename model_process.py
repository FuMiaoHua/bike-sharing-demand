import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
pd.options.mode.chained_assignment = None
import warnings
warnings.filterwarnings("ignore")


#提取数据
data = pd.read_csv("train.csv")


#将datetime列数据进行转换
data["date"] = data.datetime.apply(lambda x : x.split()[0])
data["hour"] = data.datetime.apply(lambda x : x.split()[1].split(":")[0]).astype("int")
data["year"] = data.datetime.apply(lambda x : x.split()[0].split("-")[0])
data["weekday"] = data.date.apply(lambda dateString : datetime.strptime(dateString,"%Y-%m-%d").weekday())
data["month"] = data.date.apply(lambda dateString : datetime.strptime(dateString,"%Y-%m-%d").month)

#用随机森林模型预测风速为零（缺失）样本的风速

dataWind0 = data[data["windspeed"]==0]
dataWindNot0 = data[data["windspeed"]!=0]
rfModel_wind = RandomForestRegressor()
windColumns = ["season","weather","humidity","month","temp","year","atemp"]
rfModel_wind.fit(dataWindNot0[windColumns], dataWindNot0["windspeed"])

wind0Values = rfModel_wind.predict(X= dataWind0[windColumns])
dataWind0["windspeed"] = wind0Values
data = dataWindNot0.append(dataWind0)
data.reset_index(inplace=True)
data.drop('index',inplace=True,axis=1)

#转换特征数据类型
categoricalFeatureNames = ["season","holiday","workingday","weather","weekday","month","year","hour"]
numericalFeatureNames = ["temp","humidity","windspeed","atemp"]
dropFeatures = ['casual',"count","datetime","date","registered"]

for var in categoricalFeatureNames:
    data[var] = data[var].astype("category")

#正如visual_process所受，异常值太多，将其对数化
yLabels = np.log1p(data["count"])

#去掉无用特征
dataTrain  = data.drop(dropFeatures,axis=1)

#对随机森林模型进行参数调试，预测模型效果
rf_params = {"n_estimators":[10,65,100,130]}
grid = GridSearchCV(RandomForestRegressor(),param_grid=rf_params, cv=5)

grid.fit(dataTrain,yLabels)
print ("best params:",grid.best_params_)

preds = grid.predict(X= dataTrain)
Labels = np.exp(yLabels)
Label_preds = np.exp(preds)
print ("RMSE of model:",pow(mean_squared_error(Labels,Label_preds),0.5))

plt.hist(Labels,rwidth=0.8,label="actual")
plt.hist(Label_preds,rwidth=0.8,label="pred")
plt.title("actual count and pred count")
plt.legend()
plt.show()
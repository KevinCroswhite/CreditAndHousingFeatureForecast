###################################################
###################################################
####                                          #####
####      Kevin Croswhite                     #####
####      Economics 835                       #####
####      Econometric Methods II              #####
###       Term Paper                          #####
####      University of Wisconsin - Milwaukee #####
####                                          #####
###################################################
###################################################




###################################################
###################################################
##
##           Table of Contents
##
## Libraries .............................
##
## Inputs ................................ 
##
## Data Import.............................
##
## Data Organization ......................
##
## Exploritory Data Analysis ..............
##
## VAR ....................................
##
###################################################
################################################### 







#
##
###
####
##### Libraries
####
###
##
#


import pandas as pd
import pandas_datareader.data as web
import datetime as dt
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import grangercausalitytests
from statsmodels.tsa.vector_ar.vecm import coint_johansen
import statsmodels.tsa.api as smt
from statsmodels.tsa.api import VAR
from statsmodels.stats.stattools import durbin_watson
import seaborn as sn
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor





#
##
###
####
##### Inputs
####
###
##
#




quandl_key=""
data_start = dt.datetime(1985,1,1)
data_end = dt.datetime(2016,8,1)
export_data_set = False




#
##
###
####
##### Data Import
####
###
##
#




# FRED
DRHPI = web.DataReader('QUSR368BIS','fred',data_start,data_end)
SLO = web.DataReader('DRIWCIL','fred',data_start,data_end)
DEMP = web.DataReader('PAYEMS','fred',data_start,data_end)
U = web.DataReader('UNRATE','fred',data_start,data_end)
DRGDP = web.DataReader('A191RL1Q225SBEA','fred',data_start,data_end)
DIIP = web.DataReader('INDPRO','fred',data_start,data_end)
DSALES = web.DataReader('A190RL1Q225SBEA','fred',data_start,data_end) 

# CSV of Gilchrist-Zakrajöekí data from BU website => http://people.bu.edu/sgilchri/Data/data.htm
# CSV has been reformatted to more easily import date data
GZ_import = pd.read_csv("C:/Users/kevin/Documents/Datasets/GZ_EBP.csv")




#
##
###
####
##### Data Organization
####
###
##
#

DEMP = DEMP.loc[(DEMP.index.month == 1) | (DEMP.index.month == 4) | (DEMP.index.month == 7) | (DEMP.index.month == 10)]
DIIP = DIIP.loc[(DIIP.index.month == 1) | (DIIP.index.month == 4) | (DIIP.index.month == 7) | (DIIP.index.month == 10)]
U = U.loc[(U.index.month == 1) | (U.index.month == 4) | (U.index.month == 7) | (U.index.month == 10)] 


DEMP = DEMP.pct_change()
DIIP = DIIP.pct_change()
DRHPI = DRHPI['QUSR368BIS'] / 100 
SLO = SLO['DRIWCIL'] / 100
U = U['UNRATE'] / 100
DRGDP = DRGDP['A191RL1Q225SBEA'] / 100
DSALES = DSALES['A190RL1Q225SBEA'] / 100

dates = []

for i,row in GZ_import.iterrows():
    dates.append(dt.datetime.strptime(GZ_import.iloc[i,2], '%m/%d/%Y'))

GZ_import["Date2"] = dates

GZ_import = GZ_import.set_index("Date2")

EBP = GZ_import.iloc[:,3]
GZ = GZ_import.iloc[:,4]

EBP = EBP.to_frame(name='EBP')
GZ = GZ.to_frame(name='GZ')

GZ = GZ['GZ'] / 100
EBP = EBP['EBP'] / 100

GZ = GZ.loc[(GZ.index.month == 1) | (GZ.index.month == 4) | (GZ.index.month == 7) | (GZ.index.month == 10)]
EBP = EBP.loc[(EBP.index.month == 1) | (EBP.index.month == 4) | (EBP.index.month == 7) | (EBP.index.month == 10)]




data = pd.merge(DRHPI, SLO, left_index=True, right_index=True)
data = pd.merge(data, DEMP, left_index=True, right_index=True)
data = pd.merge(data, U, left_index=True, right_index=True)
data = pd.merge(data, DRGDP, left_index=True, right_index=True)
data = pd.merge(data, DIIP, left_index=True, right_index=True)
data = pd.merge(data, DSALES, left_index=True, right_index=True)
data = pd.merge(data, GZ, left_index=True, right_index=True)
data = pd.merge(data, EBP, left_index=True, right_index=True)

data.columns = ["Real House Price",
                "Survey of Loan Officers",
                "Non Farm Payroll",
                "Unemployment Rate",
                "Real GDP",
                "Industrial Production",
                "Real Domestic Final Sales", 
                "GZ Spread",
                "Excess Bond Premium"]


data_GDP = data[["GZ Spread","Excess Bond Premium","Survey of Loan Officers","Real House Price","Real GDP"]]


data = data.drop_duplicates()

if export_data_set:
    data.to_csv("C:/Users/kevin/Documents/Datasets/835_termpaper_data.csv", index=True)



#
##
###
####
##### Visual Data Inspection
####
###
##
#


for column in data.columns:

    plt.plot(data[column], color = 'tomato', label = column)
    plt.legend(loc='upper right')
    plt.show()
    plt.clf()



corrMatrix = data.corr()

heat = sn.heatmap(corrMatrix, annot=True,linewidths=0,)
x,y = heat.get_ylim()
heat.set_ylim(x + 0.5,y - 0.5)

_ = sn.pairplot(data_GDP,kind="reg",diag_kind="kde",markers=".")


for column in data.columns:
    adf = adfuller(data[column].dropna())
    print("---- ADF p-value ----")
    print(str(column) + ": " +  str(adf[1]))


# ML


training = data_GDP[0:30]
test = data_GDP[30:]

features_train = training.drop('Real GDP', axis=1)
target_train = training['Real GDP']

features_test = test.drop('Real GDP', axis=1)
target_test = test['Real GDP']

# Decision Tree
    
decision_tree = DecisionTreeRegressor(max_depth=5)
decision_tree.fit(features_train,target_train)

print("Training: " + str(decision_tree.score(features_train,target_train)))
print("Test: " + str(decision_tree.score(features_test,target_test)))




# Random Forest

random_forest = RandomForestRegressor(n_estimators=200,max_depth=5,max_features=4,random_state=42)
random_forest.fit(features_train,target_train)
print(random_forest.score(features_train, target_train))
print(random_forest.score(features_test, target_test))


# Feature Performance from Random Forest

importances = random_forest.feature_importances_

feature_names = features_train.columns
sorted_index = np.argsort(importances)[::-1]

x = range(len(importances))
labels = np.array(feature_names)[sorted_index]

plt.bar(x,importances[sorted_index],tick_label=labels)
plt.xticks(rotation=90)
plt.show

# Gradient Boosting

gbr = GradientBoostingRegressor(max_features=4,learning_rate=0.01,n_estimators=200,subsample=0.6,random_state=42)
gbr.fit(features_train,target_train)

print(gbr.score(features_train, target_train))
print(gbr.score(features_test, target_test))


# KNN

knn = KNeighborsRegressor(n_neighbors=10)
knn.fit(features_train, target_train)

print(knn.score(features_train, target_train))
print(knn.score(features_test, target_test))



# NN

model = Sequential()





#Scaling ########################################################### PICK UP HERE


sc = StandardScaler()
scaled_train_features = sc.fit_transform(features_train)
scaled_test_features = sc.fit(features_test)


gbr = GradientBoostingRegressor(max_features=4,learning_rate=0.01,n_estimators=200,subsample=0.6,random_state=42)
gbr.fit(scaled_train_features,target_train)

print(gbr.score(scaled_train_features, target_train))
print(gbr.score(scaled_test_features, target_test))










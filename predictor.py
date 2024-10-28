import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GridSearchCV


data = pd.read_csv('gym_members_exercise_tracking.csv')
data = pd.get_dummies(data, columns=['Workout_Type','Gender'], drop_first=True)

x = data[['Session_Duration (hours)','Weight (kg)','Max_BPM',
          'Avg_BPM', 'Workout_Type_HIIT', 'Workout_Type_Strength',      
       'Workout_Type_Yoga','BMI','Gender_Male']]

y=data['Calories_Burned']

#data not used in training for final testing
testData = pd.read_csv('testData.csv')
testData = pd.get_dummies(testData, columns=['Workout_Type','Gender'], drop_first=True)

xTest = testData[['Session_Duration (hours)','Weight (kg)','Max_BPM',
          'Avg_BPM', 'Workout_Type_HIIT', 'Workout_Type_Strength',      
       'Workout_Type_Yoga','BMI','Gender_Male']]
yTest = testData['Calories_Burned']

#train for linear regression model
x_trainLR, x_testLR, y_trainLR, y_testLR = train_test_split(x,y,test_size=0.3,random_state=42)

#train for random forest model
x_trainRF,x_testRF, y_trainRF, y_testRF = train_test_split(x,y, test_size=0.3, random_state=42)

#linear regression
lrModel = LinearRegression()
lrModel.fit(x_trainLR,y_trainLR)
y_predLR = lrModel.predict(x_testLR)

#Random Forest 
rfModel = RandomForestRegressor(n_estimators=120, min_samples_leaf=3,min_samples_split=2,max_depth=30)
rfModel.fit(x_trainRF,y_trainRF)
y_predRF = rfModel.predict(x_testRF)


#hyperparameter tuning

param_grid = {
    'n_estimators': [100,110,120,130],
    'max_depth': [None, 30,40,50,60,70],
    'min_samples_split': [1.5,2,2.5],
    'min_samples_leaf': [2.5,3,3.5]
}
grid_search = GridSearchCV(estimator=rfModel, param_grid=param_grid, cv=5, scoring='neg_mean_absolute_error',n_jobs=-1)
grid_search.fit(x_trainRF,y_trainRF)
print("Best parameters found: ", grid_search.best_params_)
print("Best mean absolute error: ", -grid_search.best_score_)

#Best parameters found:  {'max_depth': 30, 'min_samples_leaf': 3, 'min_samples_split': 2, 'n_estimators': 120}


#linear regression model evaluation
maeLR = mean_absolute_error(y_testLR,y_predLR)
mapeLR = np.mean(np.abs((y_testLR-y_predLR)/y_testLR))*100
r2LR = r2_score(y_testLR,y_predLR)
print("Linear Regression - Mean Absolute Error: ",maeLR)
print("Linear Regression - Mean Absolute Error percentage: ",mapeLR,"%")
print("Linear Regression - R-Squared: ",r2LR)
print()

#Random Forest model training data prediction evaluation
maeRF = mean_absolute_error(y_testRF,y_predRF)
mapeRF = np.mean(np.abs((y_testRF-y_predRF)/y_testRF))*100
r2RF = r2_score(y_testRF,y_predRF)
print("Random Forest - Mean Absolute Error: ",maeRF)
print("Random Forest - Mean Absolute Error percentage: ",mapeRF,"%")
print("Random Forest - R-Squared: ",r2RF)
print()

#RF test data prediction evaluation
yTPred = rfModel.predict(xTest)
maeRFT = mean_absolute_error(yTest,yTPred)
mapeRFT = np.mean(np.abs((yTest-yTPred)/yTest))*100
r2RFT = r2_score(yTest,yTPred)
print("Random Forest - Mean Absolute Error: ",maeRFT)
print("Random Forest - Mean Absolute Error percentage: ",mapeRFT,"%")
print("Random Forest - R-Squared: ",r2RFT)

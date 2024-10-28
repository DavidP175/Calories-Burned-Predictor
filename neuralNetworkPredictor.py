import pandas as pd 
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Dense # type: ignore
from sklearn.metrics import mean_absolute_error, r2_score

data = pd.read_csv('gym_members_exercise_tracking.csv')
data = pd.get_dummies(data, columns=['Workout_Type','Gender'], drop_first=True)
columns = ['Session_Duration (hours)','Weight (kg)','Max_BPM',
          'Avg_BPM', 'Workout_Type_HIIT', 'Workout_Type_Strength',      
       'Workout_Type_Yoga','BMI','Gender_Male']
x = data[columns]
y = data['Calories_Burned']

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.3)
y_train = y_train.values
#scaling the dataset
scaler  = StandardScaler()

x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

model = Sequential()

#add input layer and first hidden layer with 64 neurons
model.add(Dense(64, input_dim=x_train_scaled.shape[1], activation='relu'))

#add second hidden layer with 32 neurons
model.add(Dense(32, activation='relu'))

#add output layer with 1 neuron
model.add(Dense(1))

model.compile(optimizer='adam', loss = 'mean_squared_error', metrics=['mae'])

history = model.fit(x_train_scaled,y_train,epochs=100,batch_size=10,validation_split=0.2)

y_pred = model.predict(x_test_scaled)
y_pred=y_pred.flatten()

mae = mean_absolute_error(y_test,y_pred)
mape = np.mean(abs((y_test-y_pred)/y_test))
r2 = r2_score(y_test,y_pred)

print(f'Mean Absolute Error: {mae}')
print(f'Mean Absolute Error Percentage: {mape}')
print(f'R-Squared: {r2}')
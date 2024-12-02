#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from prophet import Prophet
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.metrics import precision_score, recall_score, f1_score


url = r"C:\Users\urvi9\Provisional_COVID-19_Deaths_by_Sex_and_Age.csv"
df = pd.read_csv(url)
df


# In[2]:


print(df.info)


# In[3]:


#data cleaning
print("Duplicates found:",df.duplicated().sum())
df.drop(columns=['Month', 'Data As Of', 'Footnote'], inplace=True)
df['Start Date'] = pd.to_datetime(df['Start Date'])
df['End Date'] = pd.to_datetime(df['End Date'])

df = df.fillna(0)
df


# In[4]:


male_data = df[df['Sex'] == 'Male']
female_data = df[df['Sex'] == 'Female']


# In[10]:


# 1
#To find relatioship between Covid 19 deaths and Males
X = male_data[['Sex']]
y = male_data['COVID-19 Deaths']
X = pd.get_dummies(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#Linear Regression Male
lin_reg = LinearRegression()
lin_reg.fit(X_train_scaled, y_train)
lin_reg_pred = lin_reg.predict(X_test_scaled)
lin_reg_mse = mean_squared_error(y_test, lin_reg_pred)
print("Linear Regression Male MSE:", lin_reg_mse)

#Neural Network Male
nn_reg = MLPRegressor(hidden_layer_sizes=(100, 50), activation='relu', solver='adam', max_iter=1000)
nn_reg.fit(X_train_scaled, y_train)
nn_reg_pred = nn_reg.predict(X_test_scaled)
nn_reg_mse = mean_squared_error(y_test, nn_reg_pred)
print("Neural Network Male MSE:", nn_reg_mse)


# In[40]:


#To find relatioship between Covid 19 deaths and Females
X = female_data[['Sex']]
y = female_data['COVID-19 Deaths']
X = pd.get_dummies(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#Linear Regression Female
lin_reg = LinearRegression()
lin_reg.fit(X_train_scaled, y_train)
lin_reg_pred = lin_reg.predict(X_test_scaled)
lin_reg_mse = mean_squared_error(y_test, lin_reg_pred)
print("Linear Regression Female MSE:", lin_reg_mse)

#Neural Network Female
nn_reg = MLPRegressor(hidden_layer_sizes=(100, 50), activation='relu', solver='adam', max_iter=1000)
nn_reg.fit(X_train_scaled, y_train)
nn_reg_pred = nn_reg.predict(X_test_scaled)
nn_reg_mse = mean_squared_error(y_test, nn_reg_pred)
print("Neural Network Female MSE:", nn_reg_mse)


# In[39]:


plt.figure(figsize=(18, 8))
plt.scatter(male_data['Age Group'], male_data['COVID-19 Deaths'], color='blue', label='Male', alpha=0.7,s=150)
plt.scatter(female_data['Age Group'], female_data['COVID-19 Deaths'], color='red', label='Female', alpha=0.7,s=150)

plt.title('Scatter Plot of COVID-19 Deaths by Age Group and Sex')
plt.xlabel('Age Group')
plt.ylabel('COVID-19 Deaths')
plt.xticks(rotation=45, ha='right') 
plt.ylim(0, 200000)
plt.legend()
plt.grid(True)
plt.show()


# In[30]:


# 2
#pneumonia deaths related to covid19 deaths
X = df[['Age Group', 'Sex', 'Pneumonia Deaths']]
y = df['COVID-19 Deaths']
label_encoder = LabelEncoder()
X.loc[:, 'Sex'] = label_encoder.fit_transform(X['Sex'])
X.loc[:, 'Age Group'] = label_encoder.fit_transform(X['Age Group'])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#KNN
knn_classifier = KNeighborsClassifier(n_neighbors=5)
knn_classifier.fit(X_train_scaled, y_train)
knn_pred = knn_classifier.predict(X_test_scaled)
knn_accuracy = accuracy_score(y_test, knn_pred)
precision = precision_score(y_test, knn_pred, average='weighted')
recall = recall_score(y_test, knn_pred, average='weighted')
print("KNN Accuracy:", knn_accuracy)
print("KNN Precision:", precision)
print("KNN Recall:", recall)

knn_data = X_test.copy()
knn_data['COVID-19 Deaths'] = y_test
knn_data['Predicted COVID-19 Deaths'] = knn_pred
continuous_features = ['Pneumonia Deaths']
for feature in continuous_features:
    plt.figure(figsize=(15, 5))
    sns.scatterplot(data=knn_data, x=feature, y='COVID-19 Deaths', hue='Predicted COVID-19 Deaths', palette='coolwarm', s=150)
    plt.title(f'{feature} vs. COVID-19 Deaths')
    plt.xlabel(feature)
    plt.ylabel('COVID-19 Deaths')
    plt.legend(title='Predicted COVID-19 Deaths')
    plt.show()


# In[14]:


#plotting Age wise
desired_age_groups = ['0-17 years', '18-24 years', '25-34 years', '35-44 years', '45-54 years', '55-64 years', '65-74 years', '75-84 years']
filtered_data = df[(df['Age Group'].isin(desired_age_groups)) & (df['Sex'] != 'All Sexes')]
plt.figure(figsize=(12, 6))
sns.scatterplot(data=filtered_data, x='Pneumonia Deaths', y='COVID-19 Deaths', hue='Age Group', palette='coolwarm', style='Sex',s=200)
plt.title('Pneumonia Deaths vs COVID-19 Deaths by Age Group')
plt.xlabel('Pneumonia Deaths')
plt.ylabel('COVID-19 Deaths')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.show()


# In[15]:


# 3
#plotting State wise
state_deaths = df[df['State'] != 'United States']
state_deaths = state_deaths.groupby('State')['COVID-19 Deaths'].sum().reset_index()
plt.figure(figsize=(14, 8))
sns.barplot(data=state_deaths, x='State', y='COVID-19 Deaths', palette='flare')
plt.title('Total COVID-19 Deaths by State')
plt.xlabel('State')
plt.ylabel('Total COVID-19 Deaths')
plt.xticks(rotation=90)
plt.show()


# In[16]:


# 4
#relationship between these three
X1 = df[['Pneumonia Deaths', 'Influenza Deaths']]
y1 = df['COVID-19 Deaths']
X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y1, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X1_train, y1_train)
y1_pred = model.predict(X1_test)
mse = mean_squared_error(y1_test, y1_pred)
print("Linear Regression MSE:", mse)
print("Coefficients:", model.coef_)


# In[18]:


#heatmap for correlation between different features in the dataset
numerical_features = df.select_dtypes(include=['int64', 'float64']).drop(columns=['Year'])
corr_matrix = numerical_features.corr()
plt.figure(figsize=(6, 6))
sns.heatmap(corr_matrix, annot=True , fmt=".2f", linewidths=0.5)
plt.title('Correlation Heatmap of Numerical Features')
plt.show()


# In[33]:


# 5 - Time Series
#yearwise 
dfy = df[df['Group'] == 'By Year']
dfy.loc[:, 'Year'] = pd.to_datetime(dfy['Year'], format='%Y')
df_filtered = dfy[dfy['Group'] == 'By Year']
df_filtered = df_filtered.rename(columns={'Year': 'ds', 'COVID-19 Deaths': 'y'})
#Prophet model
model = Prophet()
model.fit(df_filtered)
future = model.make_future_dataframe(periods=365)  #forecast for next 365 days
forecast = model.predict(future)
fig = model.plot(forecast)
plt.legend(["Actual Data", "Predicted Data"])
plt.title('Forecast of COVID-19 Deaths')
plt.xlabel('Date')
plt.ylabel('COVID-19 Deaths')
plt.show()


# In[34]:


#monthwise 
dfm = df[df['Group'] == 'By Month']
dfm.loc[:, 'End Date'] = pd.to_datetime(dfm['End Date'])
df_filtered = dfm[dfm['Group'] == 'By Month']
df_filtered = df_filtered.rename(columns={'End Date': 'ds', 'COVID-19 Deaths': 'y'})
#Prophet model
model = Prophet()
model.fit(df_filtered)
future = model.make_future_dataframe(periods=365)  #forecast for next 365 days
forecast = model.predict(future)
fig = model.plot(forecast)
plt.legend(["Actual Data", "Predicted Data"])
plt.title('Forecast of COVID-19 Deaths')
plt.xlabel('Date')
plt.ylabel('COVID-19 Deaths')
plt.ylim(-5000,30000)
plt.show()


# In[35]:


# 6
#using male data (Influenza vs sex)
X_male = male_data[['Sex']]
y_male = male_data['Influenza Deaths']
X_male = pd.get_dummies(X_male)
X_male_train, X_male_test, y_male_train, y_male_test = train_test_split(X_male, y_male, test_size=0.2, random_state=42)

#Random Forest classifier for male data
rf_classifier_male = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier_male.fit(X_male_train, y_male_train)
y_male_pred = rf_classifier_male.predict(X_male_test)
accuracy_male = accuracy_score(y_male_test, y_male_pred)
print("Male Data - Random Forest Accuracy:", accuracy_male)
precision_male = precision_score(y_male_test, y_male_pred, average='weighted', zero_division=1)
print("Male Data - Random Forest Precision:", precision_male)
recall_male = recall_score(y_male_test, y_male_pred, average='weighted', zero_division=1)
print("Male Data - Random Forest Recall:", recall_male)


# In[36]:


#using female data  (Influenza vs sex)
X_female = female_data[['Sex']]
y_female = female_data['Influenza Deaths']
X_female = pd.get_dummies(X_female)
X_female_train, X_female_test, y_female_train, y_female_test = train_test_split(X_female, y_female, test_size=0.2, random_state=42)

#Random Forest classifier for female data
rf_classifier_female = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier_female.fit(X_female_train, y_female_train)
y_female_pred = rf_classifier_female.predict(X_female_test)
accuracy_female = accuracy_score(y_female_test, y_female_pred)
print("Female Data - Random Forest Accuracy:", accuracy_female)
precision_female = precision_score(y_female_test, y_female_pred, average='weighted', zero_division=1)
print("Female Data - Random Forest Precision:", precision_female)
recall_female = recall_score(y_female_test, y_female_pred, average='weighted', zero_division=1)
print("Female Data - Random Forest Recall:", recall_female)


# In[ ]:





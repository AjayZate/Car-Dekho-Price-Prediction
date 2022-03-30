import pandas as pd
import sklearn
from datetime import date

# Loading the data from Car Dekho site

data = pd.read_csv("car data.csv")
print(data.head())

# checking the metadata for the dataset
data.info()

# checking the null values if any
print(data.isna().sum())  # No null values found

# describing table to check for outliers
print(data.describe())  # No effective outliers found

# Derive no. of years from "Year" column
current_year = date.today().year
print(current_year)

data["No. of Years"] = current_year - data["Year"]
print(data.head())

# we will drop the irrelevant columns
df = data.drop(["Car_Name", "Year", "Owner"], axis=1)
print(df.head())

# Handling the categorical features
df = pd.get_dummies(df, drop_first=True)
df.rename(columns={"No. of Years": 'No_of_Years'}, inplace=True)
print(df.head())

# Separating dependent and Independent features
X = df.drop("Selling_Price", axis=1)
print(X.info())
print(X.loc[1,:])

y = df["Selling_Price"]
print(y.head())

# splitting the data into train test datasets
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

# importing and training Random forest model
from sklearn.ensemble import RandomForestRegressor

RFR = RandomForestRegressor()
RFR.fit(X_train, y_train)

# predicting the final output for test data
y_pred = RFR.predict(X_test)

# Pickling the file
import pickle

file = open("RFR_CarPricePredictor.pkl", "wb")

# Dumping the data to file
pickle.dump(RFR, file)

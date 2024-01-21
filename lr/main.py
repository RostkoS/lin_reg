from linear_regression import LinearRegression 
import pandas as pd
import joblib

df = pd.read_csv("Regression/abalone.csv")
df = df.drop('Type', axis =1)
#df.drop(df.index[1000:], inplace=True)
df = df.sample(frac=1)
X = df.drop('LongestShell', axis =1)
y = df['LongestShell']
x_train = X.iloc[:int(0.8*len(X))]
x_test = X.iloc[int(0.8*len(X)):]
y_train = y.iloc[:int(0.8*len(y))]
y_test = y.iloc[int(0.8*len(y)):]
lr = LinearRegression()

lr.fit(x_train, y_train)
pred = lr.predict(x_test)
print(x_test.iloc[0])
pred1 = lr.predict([0.365,	0.095,	0.514,	0.2245,	0.10099999999999999,	0.15,	15])
print(pred1)
r2 = lr.r_squared(y_test,pred)
print(r2)
joblib.dump(lr, "rf_model.sav")
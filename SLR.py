# importing necessary libraries
import pandas as pd 
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from sklearn.model_selection import train_test_split  
from sklearn.linear_model import LinearRegression  
from sklearn import metrics  


# For reading data set
Data = pd.read_csv("E:\Data\GRIPS\grips SLR.csv")
data = pd.DataFrame(Data)

data.shape
data.head()
data.describe()

# Correlation
data.corr()

# Percentage Scores  v/s Hours studied plot
plt.plot(data.Hours,data.Scores,"bo");plt.xlabel("Hours Studied");plt.ylabel("Percentage Scores")

# Normality by boxplot
plt.boxplot(data.Hours)


# Data Preparation
x=data.iloc[:,:1]
y=data.iloc[:,1:2]


# Data Splitting
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

# Model Creation
model = LinearRegression()
model.fit(x_train, y_train)

print(model.intercept_)
print(model.coef_)

# regression line
line = model.coef_*x+model.intercept_
# Plotting for the test data
plt.scatter(x, y)
plt.plot(x, line);
plt.show()


# predictions
print(x_test) # Hours test data
y_pred = model.predict(x_test)

print(y_test)


# Comparing and Accuracy of the model
model.score(x_test,y_test)
# 94.55 %


# Model Evaluation
print('Mean Absolute Error:', 
      metrics.mean_absolute_error(y_test, y_pred))

# 4.183859899002975


# prediction for a particalar value
# Actual Math 
# y = c + mx
# c = intercept = 2.01816004
# m = slope = 9.91065648
# x = 9.25 hours
# y = score = 2.01816004 + 9.91065648*9.25
# y = 93.69173248



# Second method 
# Model Creation
model2 = smf.ols("Scores~Hours",data=data).fit()

type(model)

model2.params
# Intercept    2.483673
# Hours        9.775803

model2.summary()

pred = model.predict(data)
pred

actual = data['Scores']
prediction = pred







# Boston housing price prediction</br> (Linear Regression)
## 1.Dataset
The data contains the information about housing around Boston area.The following is the description of each column.

● CRIM - per capita crime rate by town</br>
● ZN - proportion of residential land zoned for lots over 25,000 sq.ft.</br>
● INDUS - proportion of non-retail business acres per town.</br>
● CHAS - Charles River dummy variable (1 if tract bounds river; 0 otherwise)</br>
● NOX - nitric oxides concentration (parts per 10 million)</br>
● RM - average number of rooms per dwelling</br>
● AGE - proportion of owner-occupied units built prior to 1940</br>
● DIS - weighted distances to five Boston employment centres</br>
● RAD - index of accessibility to radial highways</br>
● TAX - full-value property-tax rate per $10,000</br>
● PTRATIO - pupil-teacher ratio by town</br>
● B - 1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town</br>
● LSTAT - % lower status of the population</br>
● MEDV - Median value of owner-occupied homes in $1000's</br>

## 2.Goal
Our goal in this analysis is to predict the value(MEDV) of homes in Boston.


## 3.Overview of Analysis
```
1.Data Preprocessing
2.Model Building
3.Evaluation
```

### Data Preprocessing
There are 14 columns and 506 instances in this dataset, which has no any null values.</br>
All the columns are integer or float, so it's no need to do one hot encoding.</br>

### Model Building
#### 1.Split data
Since our data is a tiny data, I increase the proportion of our training set.
I split the data into two groups, 90% for training and 10% for testing.</br>
#### 2.Standardization
Columns "Tax" and "B" have wider range of values, compared to other columns, standardization allows the columns become directly comparable.</br>
#### 3.Apply Linear Regression Model
Our objective is to predict the housing price in Boston area. Obviously, Linear regression is one of the most suitable model for prediction problem.</br>

### Evaluation
We elvaluate our model using MSE and R2 score. MSE is around 10.85 and R2 is around 0.71, which is not bad.

![image](https://user-images.githubusercontent.com/32606310/103254831-dff7b200-49c1-11eb-9716-62052a37a22e.png)

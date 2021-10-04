# Diebetes-Predicting-ML

## Objective
To predict whether the person is suffering from Diabetes Mellitus(DM) or not 
## Data Collection
We are using dataset from the National Institute of Diabetes and Kidney Disease.
All data are female and > 21 years old of PIMA.
### Why Pima?
Pima, North American Indians who traditionally lived along the Gila and Salt rivers in Arizona, U.S., in what was the core area of the prehistoric Hohokam culture. They have the highest prevalence ever recorded in the world<sup>1</sup>.
## Approach
Logistic regression: 0 for not suffering from DM; 1 for suffering from DM

## Code 

```
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import sklearn as skl
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
```
### Make header and load data
```
column_names = ["Pregnant", "Glucose", "BP", "Skin", "Insulin", "BMI", "Pedigree", "Age", "DM"]
data_set = pd.read_csv("pima_diabetes.csv", header = None, names=column_names, skiprows=(0,0))
data_set.head()
```

<!--- table1 --->
![Load Data](https://github.com/Tiffany-Chien/Diebetes-Predicting-ML/blob/main/Stat_Pic/Table1.PNG)


### Convert data from string to number
```
covert_col = ["Pregnant", "Insulin", "BMI", "Age", "BP", "Pedigree"]
for col in covert_col:
    data_set[col] = pd.to_numeric(data_set[col])
```
### Feature selection
Select our independent value (x) and dependent value (y)
```# Feature selection
feature_col = ["Pregnant", "Insulin", "BMI", "Age", "Glucose", "BP" ,"Pedigree"]
X = data_set[feature_col]
y = data_set.DM
```
### Check correlation 
```corr = data_set.corr()
plt.figure(figsize=(40, 30))
coor_range = corr[(corr >= 0.3) | (corr <= -0.1)]
sns.heatmap(coor_range, vmax=0.8, linewidths=0.01, square=True, annot=True, cmap='GnBu', linecolor="white", cbar_kws={'label': 'Feature Correlation Color'})
plt.title('Correlation between features of Pima Datasets')
plt.ylabel("Feature Values on Y axis")
plt.xlabel("Feature Values on X axis")
```
<!--- Correlation --->
![Correlation_Heat_Map](https://github.com/Tiffany-Chien/Diebetes-Predicting-ML/blob/main/Stat_Pic/CorrelationHeatMap.png)

### Partition dataset to test and training set 
```
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 42) 
```

### Applying logistic regression algorithm
```
logistic_function=LogisticRegression()
logistic_function.fit(X_train, y_train)
y_prediction = logistic_function.predict(X_test)
```

### Model Evaluation using Confusin Matrix
```
from sklearn import metrics
cnf_matrix_evaluation = metrics.confusion_matrix(y_test, y_prediction)
cnf_matrix_evaluation
```
### Data Visualization
```
class_names = [0, 1]
fig, ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)

sns.heatmap(pd.DataFrame(cnf_matrix_evaluation), annot=True, cmap="YlGnBu", fmt='g')
ax.xaxis.set_label_position("top")
plt.tight_layout()
plt.title("Confusion Matrix: Diabetes Patient", y=1.1)
plt.ylabel("Acutal DMs")
plt.xlabel("Predicted DMs")
```
<!--- confusion matrix --->

![Confusion_Matrix](https://github.com/Tiffany-Chien/diebetes-predicting-ML/blob/main/Stat_Pic/ConfusionMatrix.png)

### ROC curve to check performance
```
y_prediction_probability = logistic_function.predict_proba(X_test)[::,1]
fpr, tpr, _ = metrics.roc_curve(y_test, y_prediction_probability)
auc = metrics.roc_auc_score(y_test, y_prediction_probability)
plt.plot(fpr, tpr, label="data 1, auc=" + str(auc))
plt.legend(loc=4)
plt.show()
```
<!--- ROC --->

## Reference
<sup>1</sup> Gohdes D: Diabetes in North American Indians and Alaska natives. *In Diabetes in America.* Washington, DC, U.S. Govt. Printing Office, 1995, p. 1683–1701 (NIH publ. no. 95-1468). [Google Scholar](https://care.diabetesjournals.org/lookup/google-scholar?link_type=googlescholar&gs_type=article&q_txt=Gohdes+D%3A+Diabetes+in+North+American+Indians+and+Alaska+natives.+In+Diabetes+in+America.+Washington%2C+DC%2C+U.S.+Govt.+Printing+Office%2C+1995%2C+p.+1683%E2%80%931701+(NIH+publ.+no.+95-1468))

## Credits
1. [Pima Indians Diabetes - EDA & Prediction (0.906)](https://www.kaggle.com/vincentlugat/pima-indians-diabetes-eda-prediction-0-906#7.-Credits)

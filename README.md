# Heart-Attack-Analysis-AI
This AI, shows whether you are at risk of having a heart attack


###These are the libraries We will use for the project
```
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.calibration import LinearSVC
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import MinMaxScaler 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score
```
### And i want to see first 5 patients in my dataset
```
data = pd.read_csv("heart.csv")
data.head()
```
![image](https://github.com/MehmetAliKOYLU/Heart-Attack-Analysis-AI/assets/91757385/cbb6d2d1-1f24-434d-bb9c-82a8e857853a))

#### Let's make an example drawing just by looking at cholestoral in mg/dl fetched via BMI sensor for now At the end of our program, our machine learning model will make a prediction by looking not only at cholestoral, but also at all other data.
```
plt.scatter(nonsick.age, nonsick.chol, color="green", label="nonsick", alpha = 0.4)
plt.scatter(sick.age, sick.chol, color="red", label="sick", alpha = 0.4)
plt.xlabel("Age")
plt.ylabel("chol")
plt.legend()
plt.show()
```
![image](https://github.com/MehmetAliKOYLU/Heart-Attack-Analysis-AI/assets/91757385/1290c3bb-96aa-4a73-8020-e3c46e5a5aaa)

### let's determine the x and y axes
```
 = data.output.values
x_raw_data = data.drop(["output"],axis=1)   



x = (x_raw_data - np.min(x_raw_data.values))/(np.max(x_raw_data.values)-np.min(x_raw_data.values))




print("Raw data before normalization:\n")
print(x_raw_data.head())

# after 
print("\n\n\nThe data that we will provide to artificial intelligence for training after normalization:\n")
print(x.head())
x_train, x_test, y_train, y_test =train_test_split (x,y,test_size = 0.27,random_state=2)
```
![image](https://github.com/MehmetAliKOYLU/Heart-Attack-Analysis-AI/assets/91757385/95497368-8991-4e3b-a39d-942c4841e2e5)

![image](https://github.com/MehmetAliKOYLU/Heart-Attack-Analysis-AI/assets/91757385/eb24af24-4f58-40b1-ac14-d1b03ffb77a7)

## we separate our test data with our train data
### our train data will be used to learn how the system distinguishes between a healthy person and a sick person. if our test data is, let's see if our machine learning model can accurately distinguish between sick and healthy people it will be used for testing...
`
x_train, x_test, y_train, y_test =train_test_split (x,y,test_size = 0.27,random_state=2)
`

### we are creating our knn model.
```
k_values = list(range(1, 10))
mean_scores = []

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, x_train, y_train, cv=5)  # 5x cross value score
    mean_scores.append(scores.mean())

# Optimal k values =
optimal_k = k_values[mean_scores.index(max(mean_scores))]
print("Optimal k value:", optimal_k)

#create final model with Optimal k values
final_knn = KNeighborsClassifier(n_neighbors=optimal_k)
final_knn.fit(x_train, y_train)

# Accuracy with optimal k value
accuracy = final_knn.score(x_test, y_test)
print(f"Accuracy with optimal k value: %{accuracy * 100:.2f}")
sc = MinMaxScaler()
sc.fit_transform(x_raw_data.values)
```
### we are creating our random forest classifier
```
RFCmodel = RandomForestClassifier()  
RFCmodel.fit(x_train,y_train)
rfc_pred = RFCmodel.predict(x_test)
rfc_acc = accuracy_score(rfc_pred, y_test)
print ("Random forest test accuracy: {:.2f}%".format(rfc_acc*100))
print( "\n" )
print(classification_report(y_test, rfc_pred))
print( "\n" )
```

### we choose the classifiers that have the best accuracy rate
### we are using random forest classifier for training and testing purposes because accuracy score is better than other classifiers


### function for new predictions
```
def newprediction():
    v1=int(input("age >> "))
    v2=int(input("sex >> "))
    v3=int(input("cp >> "))
    v4=int(input("trestbps >> "))
    v5=int(input("chol >> "))
    v6=int(input("fbs >> "))
    v7=int(input("restecg >> "))
    v8=int(input("thalach >> "))
    v9=int(input("exang >> "))
    v10=float(input("oldpeak >> "))
    v11=int(input("slp >> "))
    v12=int(input("ca >> "))
    v13=int(input("thal >> "))


    new_prediction = RFCmodel.predict(sc.transform(np.array([[v1,v2,v3,v4,v5,v6,v7,v8,v9,v10,v11,v12,v13]])))
    new_prediction[0]
    if new_prediction==1:
        print("heart attack detected")
    else:
        print("heart attack not detected")
    return new_prediction
```
### and we use while loop for a new predictions
```
while True:
    newprediction()
    choose=input("do you want to continue ? (y/n)  >>"  )
    choose=choose.lower()
    if choose=="y":
        continue
    elif choose=="n":
        print("exiting...")
        break
```
### This is our last output 
![image](https://github.com/MehmetAliKOYLU/Heart-Attack-Analysis-AI/assets/91757385/dcef03a4-6384-41f9-ade0-af2f34f1924a)


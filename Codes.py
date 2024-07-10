from sklearn.datasets import load_breast_cancer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
import pandas as pd

cancer = load_breast_cancer()

df = pd.DataFrame(data= cancer.data, columns= cancer.feature_names)
df["target"] = cancer.target

X = cancer.data
y = cancer.target

knn = KNeighborsClassifier()

# Prediction with X

knn.fit(X, y)
y_pred = knn.predict(X)

accuracy1 = accuracy_score(y, y_pred)




# Prediction with X_train X_test (train test split)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

knn.fit(X_train, y_train)
y_pred2 = knn.predict(X_test)

accuracy2 = accuracy_score(y_test, y_pred2)

# Confusion Matrix

conf_matrix = confusion_matrix(y_test, y_pred2)


# Hyperparameter Optimization

accuracy_values = []
k_values = []

for k in range (1, 21):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred3 = knn.predict(X_test)
    accuracy3 = accuracy_score(y_test, y_pred3)
    accuracy_values.append(accuracy3)
    k_values.append(k)
    

plt.figure()
plt.plot(k_values, accuracy_values, marker = "o", linestyle = "-")
plt.title("K Değerine Göre Doğruluk")
plt.xlabel("K değeri")
plt.ylabel("Doğruluk Skoru")
plt.xticks(k_values)
plt.grid(True)   
    
    
    
    
    
    
    
    
    
    
    
    
    


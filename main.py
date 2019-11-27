from sklearn import tree
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
abalone = pd.read_csv('Abalone.csv')
data = np.array(abalone)
print (pd.concat([abalone.describe()[1:4], abalone.describe()[7:8]]))
plt.subplots(2, 2, figsize=(15, 8))
plt.subplot(2, 2, 1)
plt.scatter(abalone['Length'], abalone['Diameter'], c='purple',
            label='Length,Diameter')
plt.xlabel('Length')
plt.ylabel('Diameter')
plt.legend()
plt.subplot(2, 2, 2)
plt.scatter(abalone['Height'], abalone['Whole weight'], c='red',
            label='Height,Whole weight')
plt.xlabel('Height')
plt.ylabel('Whole weight')
plt.legend()
plt.subplot(2, 2, 3)
plt.scatter(abalone['Shucked weight'], abalone['Viscera weight'],
            c='blue', label='Shucked weight,Viscera weight')
plt.xlabel('Shucked weight')
plt.ylabel('Viscera weight')
plt.legend()
plt.subplot(2, 2, 4)
plt.scatter(abalone['Shell weight'], abalone['Rings'], c='green',
            label='Shell weight,Rings')
plt.xlabel('Shell weight')
plt.ylabel('Rings')
plt.legend()
plt.savefig('figure.png')
plt.subplots(1, 1, figsize=(20, 10))
(X_train, X_test, y_train, y_test) = train_test_split(data[:, 1:9],
        data[:, 0], test_size=0.2)
abaloneTree = tree.DecisionTreeClassifier()
abaloneTree = abaloneTree.fit(X_train, y_train)
tree.plot_tree(abaloneTree)
plt.savefig('DecisionTree.png')
print ('\nConfusion Matrix')
print ((confusion_matrix(y_test, abaloneTree.predict(X_test))),'\n')
print (classification_report(y_test, abaloneTree.predict(X_test)))

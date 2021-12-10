import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn import datasets
from sklearn.metrics import accuracy_score

samples = datasets.load_breast_cancer()
x = samples.data[:, 0:6:5]
y = samples.target
M_class = []
B_class = []
# for i in range(len(y)):
#    if y[i] == 0:
#        M_class.append(x[i])
#    else:
#        B_class.append(x[i])
#
plt.xlabel("Bán kính")
plt.ylabel("Độ lõm")
# m = np.array(M_class)
# b = np.array(B_class)

logreg = LogisticRegression()
logreg.fit(x, y)

y_pred = logreg.predict(x)

for i in range(len(y)):
    if y_pred[i] == 0:
        M_class.append(x[i])
    else:
        B_class.append(x[i])
m = np.array(M_class)
b = np.array(B_class)

# print(accuracy_score(y,y_pred))
kiemtra = [[11, 0.18], [11, 0.01], [16, 0.03]]
dudoan = logreg.predict(kiemtra)
value = ["Ác tính", "Lành tính"]
for dd in dudoan:
    print(value[dd], "\n")

plt.scatter(m[:, 0], m[:, 1], marker='o', c='b', label="Lành tính")
plt.scatter(b[:, 0], b[:, 1], marker='s', c='r', label="Ác tính")
plt.legend(loc='upper right')
plt.show()
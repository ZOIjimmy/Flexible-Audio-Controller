import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

note = "A Bb B C C# D Eb E F F# G Ab Am Bbm Bm Cm C#m Dm Ebm Em Fm F#m Gm Abm".split()
X, y = [], []
file = open("results")
answer = open("answers")
for z in range(216):
    name = '../../../wav/' + file.readline()[:-1]
    algo1 = file.readline()[1:-1] + file.readline()[:-2]
    algo1 = np.array([float(f) for f in algo1.split()])
    algo2 = np.array(eval(file.readline()))
    file.readline()
    ans = answer.readline()[:-1]
    X.append(algo2)
    y.append(note.index(ans))
file = open("results2")
answer = open("answers2")
for z in range(52):
    name = '../../../wav/' + file.readline()[:-1]
    algo2 = np.array(eval(file.readline()))
    ans = answer.readline()[:-1]
    X.append(algo2)
    y.append(note.index(ans))
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_aug, y_aug = [],[]
for x, y in zip(X, y):
    for i in range(12):
        X_aug.append(np.roll(x,i))
        if y >= 12:
            y_aug.append((y+i)%12+12)
        else:
            y_aug.append((y+i)%12)


# k = 5
# knn_classifier = KNeighborsClassifier(n_neighbors=k)
# knn_classifier.fit(X_aug, y_aug)
# y_pred = knn_classifier.predict(X_test)
# accuracy = accuracy_score(y_test, y_pred)
# print(f"KNN Accuracy: {accuracy}")
# rf_classifier = RandomForestClassifier(n_estimators=2000, random_state=11)
# rf_classifier.fit(X_aug, y_aug)
# y_pred = rf_classifier.predict(X_test)
# accuracy = accuracy_score(y_test, y_pred)
# print(f"Random Forest Accuracy: {accuracy}")
logistic_regression = LogisticRegression(max_iter=500000)
logistic_regression.fit(X_aug, y_aug)
coefficients = logistic_regression.coef_
print("Coefficient matrix:")
print(coefficients)
intercepts = logistic_regression.intercept_
print("Intercept:")
print(intercepts)
# svm_classifier = SVC(kernel='linear', random_state=42)
# svm_classifier.fit(X_aug, y_aug)
# y_pred = svm_classifier.predict(X_test)
# accuracy = accuracy_score(y_test, y_pred)
# print(f"SVM Accuracy: {accuracy}")

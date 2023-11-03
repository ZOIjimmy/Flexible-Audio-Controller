import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

note = "A Bb B C C# D Eb E F F# G Ab Am Bbm Bm Cm C#m Dm Ebm Em Fm F#m Gm Abm".split()
X, y = [], []
name = []
file = open("results")
answer = open("answers")
for z in range(216):
    name.append(file.readline()[:-1])
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
    name.append(file.readline()[:-1])
    algo2 = np.array(eval(file.readline()))
    ans = answer.readline()[:-1]
    X.append(algo2)
    y.append(note.index(ans))

coe1 = [0.02772601, -0.021885277499999998, 0.016057425833333333, -0.024805963333333337, 0.022774495000000002, 0.0022130216666666667, -0.03432571416666667, 0.017960493333333338, -0.012172340833333331, 0.015860745833333332, -0.0144441075, 0.002726601666666667]
coe2 = [0.02142058083333333, -0.02470891, 0.008429393333333333, 0.0227024525, -0.017118801666666666, 0.0179762675, -0.024228155833333334, 0.022496708333333334, 0.0016113599999999998, -0.03209938666666667, 0.01231384, -0.0064807350000000005]

def predict_multiclass(features):
    probabilities = []
    for i in range(12):
        p = np.dot(np.roll(coe1, i), features) + 0.35785841749999997
        probabilities.append(p)
    for i in range(12):
        p = np.dot(np.roll(coe2, i), features) - 0.3578584166666667
        probabilities.append(p)
    predicted = np.argmax(probabilities)
    return predicted

oops = 0
for xi, yi, i in zip(X, y, range(268)):
    predicted = predict_multiclass(xi.astype(float))
    if predicted != yi:
        oops += 1
        print(name[i])
        print(note[predicted], note[yi], i)
print(oops/268)

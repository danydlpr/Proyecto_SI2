import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.linear_model import LinearRegression

class ControladorModelo:
    def __int__(self):
        pass

    def entrenar(self, nombre):
        if nombre == 'knn':
            return KNeighborsClassifier(n_neighbors=2)
        elif nombre == 'arbol':
            return DecisionTreeClassifier(max_depth=3)
        elif nombre == 'bayes':
            return GaussianNB()
        elif nombre == 'svc':
            return SVC(kernel='linear')
        elif nombre == 'lineal':
            return LinearRegression()
        elif nombre == 'logistica':
            return LogisticRegression()

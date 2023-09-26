from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from . import ml

@ml
def multnaive(x_train, x_test, y_train, y_test):
    naive_model = MultinomialNB()
    naive_model.fit(x_train, y_train)
    return naive_model, "multinomial naive"
@ml
def svc(x_train, x_test, y_train, y_test, **kwargs):
    svm_model = SVC(kernel= kwargs['kernel'] if 'kernel' in kwargs.keys() else 'linear', random_state=42)
    svm_model.fit(x_train, y_train)
    return svm_model, "support vector classification"

@ml
def dt(x_train, x_test, y_train, y_test, **kwargs):
    dt_model = DecisionTreeClassifier(random_state= kwargs['random_state'] if 'random_state' in kwargs.keys() else 42)
    dt_model.fit(x_train, y_train)
    return dt_model, "decision tree"
@ml
def rf(x_train, x_test, y_train, y_test, **kwargs):
    dt_model = RandomForestClassifier(random_state= kwargs['random_state'] if 'random_state' in kwargs.keys() else 42)
    dt_model.fit(x_train, y_train)
    return dt_model, "random forest"

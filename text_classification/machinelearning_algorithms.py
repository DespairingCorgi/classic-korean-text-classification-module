from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from . import sklearn_ml
from .fasttext_pretrained.fasttext_enforcer import series_to_ft_supervised_format, train_supervised_model, test_supervised_model, load_model

import fasttext


#catboost
from catboost import CatBoostClassifier

#lightgbm
import lightgbm as lgb

@sklearn_ml
def multnaive(x_train, x_test, y_train, y_test, **kwargs):
    naive_model = MultinomialNB()
    naive_model.fit(x_train, y_train)
    return naive_model, "multinomial naive"
@sklearn_ml
def svc(x_train, x_test, y_train, y_test, **kwargs):
    svm_model = SVC(kernel= kwargs['kernel'] if 'kernel' in kwargs.keys() else 'linear', \
        random_state= kwargs['random_state'] if 'random_state' in kwargs.keys() else 42)
    svm_model.fit(x_train, y_train)
    return svm_model, "support vector classification"

@sklearn_ml
def dt(x_train, x_test, y_train, y_test, **kwargs):
    dt_model = DecisionTreeClassifier(random_state= kwargs['random_state'] if 'random_state' in kwargs.keys() else 42)
    dt_model.fit(x_train, y_train)
    return dt_model, "decision tree"
@sklearn_ml
def rf(x_train, x_test, y_train, y_test, **kwargs):
    dt_model = RandomForestClassifier(random_state= kwargs['random_state'] if 'random_state' in kwargs.keys() else 42)
    dt_model.fit(x_train, y_train)
    return dt_model, "random forest"

def fasttext_supervised(x_train, x_test, y_train, y_test, \
    train_file_name = 'tmp_train.txt', test_file_name = 'tmp_test.txt', \
    custom_model="", \
    use_pretrained = True, model_name="", save_model=False, **kwargs):
    
    train_txt = series_to_ft_supervised_format(x_train, y_train, train_file_name)
    test_txt = series_to_ft_supervised_format(x_test, y_test, test_file_name)

    if use_pretrained:
        model = train_supervised_model(train_txt, save_name=model_name,save_model=save_model)
    else:
        if not custom_model:
            raise Exception("you must provide custom proper model name if you want to use your custom model.") 
        model = fasttext.load_model(custom_model)
    
    test_supervised_model(model, test_txt)
    
    return model

@sklearn_ml
def catboost_classification(x_train, x_test, y_train, y_test, **kwargs):
    
    its = kwargs['iteration'] if 'iteration' in kwargs.keys() else 500
    d = kwargs['depth'] if 'depth' in kwargs.keys() else 6
    lr = kwargs['learning_rate'] if 'learning_rate' in kwargs.keys() else .1
    
    cb_model = CatBoostClassifier(iterations=its, depth=d, learning_rate=lr, verbose=0)
    print("learning...")
    cb_model.fit(x_train, y_train)
    return cb_model, "catboost classifier"

@sklearn_ml
def lightgbm_classification(x_train, x_test, y_train, y_test, **kwargs):
    '''
        additional required parameter: num_class (int)
        optional parameter: iteration (int)
    '''
    
    if 'num_class' not in kwargs.keys():
        raise Exception("lightgbm_classification requires a 'num_class' input")
    if kwargs['num_class'] == 2:
        params = {}
    elif kwargs['num_class'] > 2:
        params = {
            'objective': 'multiclass',
            'num_class': kwargs['num_class']
        }
    else:
        raise Exception("num_class error: input must be integer over 1")

    iteration = kwargs['iteration'] if 'iteration' in kwargs.keys() else 100
    
    d_train = lgb.Dataset(x_train, label=y_train)
    lgbm_model = lgb.train(params, d_train, iteration)
    
    return lgbm_model, "lightgbm classifier"
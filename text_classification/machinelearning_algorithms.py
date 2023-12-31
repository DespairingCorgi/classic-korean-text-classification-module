from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import ComplementNB
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

#xgboost
import xgboost as xgb

#adaboost
from sklearn.ensemble import AdaBoostClassifier

@sklearn_ml
def multnaive(x_train, x_test, y_train, y_test, **kwargs):
    """
    
    multnaive:
        multinomial naive baysis model training and testing(automated)
    
    :param x_train: training feature of vectorized iterable(np) data type e.g) pandas.Series
    :param y_train: training label of indexed iterable datatpe e.g) pandas.Series
    :param x_test: test feature of vectorized iterable(np) data type e.g) pandas.Series
    :param y_test: test label of indexed iterable datatpe e.g) pandas.Series

    :kwarg veoctorizer: tf(tfvectorizer) or ft(fasttext)

    :return (MultinomialNB, str)
    """
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
    #print("started training...")
    dt_model = DecisionTreeClassifier(random_state= kwargs['random_state'] if 'random_state' in kwargs.keys() else 42)
    dt_model.fit(x_train, y_train)
    return dt_model, "decision tree"
@sklearn_ml
def rf(x_train, x_test, y_train, y_test, **kwargs):
    dt_model = RandomForestClassifier(random_state= kwargs['random_state'] if 'random_state' in kwargs.keys() else 42)
    dt_model.fit(x_train, y_train)
    return dt_model, "random forest"

@sklearn_ml
def compnaive(x_train, x_test, y_train, y_test, **kwargs):
    naive_model = ComplementNB()
    naive_model.fit(x_train, y_train)
    return naive_model, "complement naive"
    

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
    
    #lossfunction = 'Logloss' or 'MutiClass'
    loss_function = kwargs['loss_function'] if 'loss_function' in kwargs.keys() else 'Logloss'
    
    cb_model = CatBoostClassifier(loss_function=loss_function, iterations=its, depth=d, learning_rate=lr, verbose=0)
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

@sklearn_ml
def xgboost_classification(x_train, x_test, y_train, y_test, **kwargs):
    '''
        additional required parameter: num_class (int)
        optional parameters: 
            max_depth (int)
            eta (float)
            iteration (int)
    '''
    if 'num_class' not in kwargs.keys():
        raise Exception("lightgbm_classification requires a 'num_class' input")
    max_depth = 6 if 'max_depth' not in kwargs.keys() else kwargs['max_depth']
    eta = .3 if 'eta' not in kwargs.keys() else kwargs['eta']
    iteration = 100 if 'iteration' not in kwargs.keys() else kwargs['iteration']
    if kwargs['num_class'] == 2:
        params = {
            'objective': 'binary:logistic',  # for binary classification
            'eval_metric': 'logloss' if 'eval_metric' not in kwargs.keys() else kwargs['eval_metric'],  # log likelihood loss
            'max_depth': max_depth,  # depth of the trees in the boosting process
            'eta': eta,  # step size shrinkage
        }
    elif kwargs['num_class'] > 2:
        params = {
            'objective': 'multi:softmax',  # for binary classification
            'eval_metric': 'mlogloss' if 'eval_metric' not in kwargs.keys() else kwargs['eval_metric'],  # log likelihood loss
            'num_class': kwargs["num_class"],
            'max_depth': max_depth,  # depth of the trees in the boosting process
            'eta': eta,  # step size shrinkage
        }
    else:
        raise Exception("num_class error: input must be integer over 1")
    
    d_train = xgb.DMatrix(x_train, y_train)
    model = xgb.train(params, d_train, iteration)
    return model, "xboost_classification"


MODELS = [
    'complementnb',
    'dt',
    'svc',
    'rf'
]

@sklearn_ml
def adaboost_classifier(x_train, x_test, y_train, y_test, **kwargs):
    '''
        additional required parameter: model (str)
        
        optional parameters for adaboost:
            ada_n_estimators (int) : for adaboost
            ada_random_state (int) : larger than 0
            ada_learning_rate (float) : larger than 0 smaller than 1
        
        optional prameters by model:
            random_state (int) : for dt/rf option
            rf_n_estimators (int) : for rf option
            
    '''
    
    if 'model' not in kwargs.keys(): raise Exception("input 'model' is required for adaboost classifier")
    if kwargs["model"].lower() not in MODELS:
        raise Exception(f"the model must be one of {MODELS}")
    model_name = kwargs["model"].lower()
    
    if model_name == 'complementnb':
        model = ComplementNB()
    elif model_name == 'dt':
        model = DecisionTreeClassifier(max_depth=1, \
            random_state= kwargs['random_state'] if 'random_state' in kwargs.keys() else 42)
    elif model_name == 'svc':
        model = SVC(kernel='linear', probability=True, decision_function_shape='ovr')
    elif model_name == 'rf':
        model = RandomForestClassifier(max_depth=1, \
            random_state= kwargs['random_state'] if 'random_state' in kwargs.keys() else 42, \
            n_estimators= kwargs['rf_n_estimators'] if 'rf_n_estomators' in kwargs.keys() else 42)
    else:
        raise Exception("unkown error")

    model = AdaBoostClassifier(model, \
        n_estimators= kwargs["ada_n_estimators"] if "ada_n_estimators" in kwargs.keys() else 50, \
        random_state= kwargs["ada_random_state"] if "ada_random_state" in kwargs.keys() else None, \
        learning_rate= kwargs["ada_learning_rate"] if "ada_learning_rate" in kwargs.keys() else 1.0
        )

    model.fit(x_train, y_train)
    result = f"adaboost classifier"
    
    return model, result

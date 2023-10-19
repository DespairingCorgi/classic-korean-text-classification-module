from konlpy.tag import Okt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix
from .fasttext_pretrained.fasttext_enforcer import get_fasttext_vectors, load_model
import xgboost as xgb

import fasttext
import lightgbm as lgb
import xgboost as xgb

tokenizer = Okt()

fasttext_model = None
cur_model = None

tf_vec = ['tf', 'tfidfvectorizer', 'tfvectorizer']
ft_vec = ['ft', 'fasttext', 'fast_text']

def test_model(model, x_test, y_test, **kwargs):
    
    is_binary = kwargs["is_binary"] if "is_binary" in kwargs.keys() else None
    method = kwargs["method"] if "method" in kwargs.keys() else type(model)
    
    if isinstance(model, fasttext.FastText._FastText):
        raise Exception("Sorry, you cannot test fasttext model here use _FastText model.\nUse test_supervised_model(model, test_txt) with fasttext encoding instead")
    elif isinstance(model, lgb.Booster) or isinstance(model, xgb.Booster):
        if isinstance(model, lgb.Booster):
            initial_preds = model.predict(x_test, num_iteration=model.best_iteration)
        elif isinstance(model, xgb.Booster):
            d_test = xgb.DMatrix(x_test, y_test)
            initial_preds = model.predict(d_test)
        else:
            raise Exception("Must be Booster type")
        if is_binary: # logistic binary
            pred = [1 if p >= 0.5 else 0 for p in initial_preds]
        else:  # mult softmax
            pred = initial_preds.argmax(axis=1) 
            # [p.argmax() for p in initial_preds]
    else:
        pred = model.predict(x_test)
    
    print("="*100)
    print(f"Method: {method}")
    print(classification_report(y_test, pred))
    print("\nconfusion matrix:")
    print(confusion_matrix(y_test, pred), "\n")
    print("="*100)
            

def sklearn_ml(func):
    def wrapper(*args, **kwargs):
        # for arg in args:
        #     print(type(arg))
        global fasttext_model
        global cur_model
        
        vector_mode = kwargs['vector_mode'] if 'vector_mode' in kwargs.keys() else 'tf'.lower()
        model = kwargs['model'] if 'model' in kwargs.keys() else None        
        
        if func.__name__ == 'multnaive' and vector_mode in ft_vec:
            print("you cannot use fasttext on multinomial naive baysis change mode to tfvectorizer...")
            vector_mode = 'tf'
        
        if vector_mode in tf_vec:
            vectorizer = TfidfVectorizer()
            train_vec = vectorizer.fit_transform(args[0])
            test_vec = vectorizer.transform(args[1])        
        elif vector_mode.lower() in ft_vec:
            if fasttext_model == None:
                print("load new model...")
                if model == None:
                    fasttext_model = load_model()
                else:
                    fasttext_model = load_model(model_name=model)
            else:
                if model != cur_model:
                    fasttext_model = load_model(model_name=model)
            cur_model = model
            
            train_vec = get_fasttext_vectors(fasttext_model, args[0].values.tolist())
            test_vec = get_fasttext_vectors(fasttext_model, args[1].values.tolist())
        else:
            pass
        
        #print(train_vec.shape)
        #print(test_vec.shape)
        
        model,method = func(train_vec, test_vec, args[2], args[3], **kwargs)
        # if func.__name__ == "lightgbm_classification":
        #     initial_preds = model.predict(test_vec, num_iteration=model.best_iteration)
        #     if kwargs["num_class"] > 2:  # mult softmax
        #         pred = initial_preds.argmax(axis=1) 
        #         # [p.argmax() for p in initial_preds]
        #     elif kwargs["num_class"] == 2: # logistic binary
        #         pred = [1 if p >= 0.5 else 0 for p in initial_preds]
        #     else:
        #         raise Exception("num_class error: input must be integer over 1")
        # elif func.__name__=="xgboost_classification":
        #     d_test = xgb.DMatrix(test_vec, args[3])
        #     initial_preds = model.predict(d_test)
        #     if kwargs["num_class"] > 2: # mult softmax
        #         pred = initial_preds.argmax(axis=1) 
        #         # [p.argmax() for p in initial_preds]
        #     elif kwargs["num_class"] == 2: # logistic binary
        #         pred = [1 if p >= 0.5 else 0 for p in initial_preds]
        #     else:
        #         raise Exception("num_class error: input must be integer over 1")
        # else:
        #     pred = model.predict(test_vec)
        # print("="*100)
        # print(f"Method: {method}")
        # print(classification_report(args[3], pred))
        # print("\nconfusion matrix:")
        # print(confusion_matrix(args[3], pred), "\n")
        # print("="*100)
        d = {
            "method":method,
        }
        if func.__name__ == "lightgbm_classification" or func.__name__ == "xgboost_classification":
            d["is_binary"] = True if kwargs["num_class"] > 2 else False
        test_model(model, test_vec, args[3], **d)
        
        return model
        
    return wrapper
        
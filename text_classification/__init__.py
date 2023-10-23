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
cur_fasttext_model = None

tf_vec = ['tf', 'tfidfvectorizer', 'tfvectorizer']
ft_vec = ['ft', 'fasttext', 'fast_text']

def report_result(method, labels, predictions):
    print("="*100)
    print(f"Method: {method}")
    print(classification_report(labels, predictions))
    print("\nconfusion matrix:")
    print(confusion_matrix(labels, predictions), "\n")
    print("="*100)

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
    
    report_result(method, y_test, pred)   
    
    return pred     


def tf_vectorize(texts, **kwargs):
    if "vectorizer_model" in kwargs.keys():
        return kwargs["vectorizer_model"].transform(texts)
    else:
        vectorizer = TfidfVectorizer()
        return vectorizer.fit_transform(texts)

def tf_vectorize_sets(*textsets, **kwargs):
    '''
        vectorizer_model -> TfidfVectorizer
    '''
    
    
    if "vectorizer_model" in kwargs.keys():
        vectorizer = kwargs["vectorizer_model"]
    else:
        vectorizer = TfidfVectorizer()
    
    vectorized_textsets = [vectorizer.fit_transform(texts) if i == 0 \
        else vectorizer.transform(texts) \
        for i, texts in enumerate(textsets)]
    
    if "get_vectorizer" in kwargs.keys():
        if kwargs["get_vectorizer"] == True:
            return vectorizer, vectorized_textsets
    
    return vectorized_textsets

def fasttext_vectorize_sets(*textsets, **kwargs):
    '''
        model (str) model file name or None(for default model)
    '''
    global fasttext_model
    global cur_fasttext_model
    
    vectorizer=None
    if "model" in kwargs.keys():
        vectorizer = kwargs["model"]
    
    if fasttext_model == None:
        if vectorizer == None:
            fasttext_model = load_model()
        else:
            fasttext_model = load_model(model_name=vectorizer)
    else:
        if vectorizer != cur_fasttext_model:
            fasttext_model = load_model(model_name=vectorizer)
        cur_fasttext_model = vectorizer
    
    vectorized_textsets = [get_fasttext_vectors(fasttext_model, texts.values.tolist()) for texts in textsets]
    
    if "get_vectorizer" in kwargs.keys():
        if kwargs["get_vectorizer"] == True:
            return vectorizer, vectorized_textsets
        
    return vectorizer, vectorized_textsets
        


def sklearn_ml(func):
    def wrapper(*args, **kwargs):
        vector_mode = kwargs['vector_mode'] if 'vector_mode' in kwargs.keys() else 'tf'.lower()
        model = kwargs['model'] if 'model' in kwargs.keys() else None        
        
        if func.__name__ == 'multnaive' and vector_mode in ft_vec:
            print("you cannot use fasttext on multinomial naive baysis change mode to tfvectorizer...")
            vector_mode = 'tf'
        
        if vector_mode in tf_vec:
            vectorizer, (train_vec, test_vec) = tf_vectorize_sets(args[0], args[1], vectorizer_model=TfidfVectorizer(), get_vectorizer = True)
                
        elif vector_mode.lower() in ft_vec:
            vectorizer, (train_vec, test_vec) = fasttext_vectorize_sets(args[0], args[1], get_vectorizer=True)
        else:
            pass
        
        #print(train_vec.shape)
        #print(test_vec.shape)
        
        model,method = func(train_vec, test_vec, args[2], args[3], **kwargs)
        
        d = {
            "method":method,
        }
        if func.__name__ == "lightgbm_classification" or func.__name__ == "xgboost_classification":
            d["is_binary"] = True if kwargs["num_class"] == 2 else False
        
        test_model(model, test_vec, args[3], **d)
        
        return model, vectorizer, method
        
    return wrapper
        
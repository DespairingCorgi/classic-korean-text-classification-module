from konlpy.tag import Okt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix
from .fasttext_pretrained.fasttext_enforcer import get_fasttext_vectors, load_model

tokenizer = Okt()

fasttext_model = None
cur_model = None

tf_vec = ['tf', 'tfidfvectorizer', 'tfvectorizer']
ft_vec = ['ft', 'fasttext', 'fast_text']

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
        if func.__name__ == "lightgbm_classification":
            initial_preds = model.predict(test_vec, num_iteration=model.best_iteration)
            if kwargs["num_class"] > 2:
                pred = [p.argmax() for p in initial_preds]
            elif kwargs["num_class"] == 2:
                pred = [1 if p >= 0.5 else 0 for p in initial_preds]
            else:
                raise Exception("num_class error: input must be integer over 1")
        else:
            pred = model.predict(test_vec)
        print("="*100)
        print(f"Method: {method}")
        print(classification_report(args[3], pred))
        print("\nconfusion matrix:")
        print(confusion_matrix(args[3], pred), "\n")
        print("="*100)
        
        return model
        
    return wrapper
        
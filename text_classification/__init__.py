from konlpy.tag import Okt
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.metrics import classification_report, confusion_matrix

tokenizer = Okt()

def ml(func):
    def wrapper(*args, **kwargs):
        vectorizer = TfidfVectorizer()
        train_vec = vectorizer.fit_transform(args[0])
        test_vec = vectorizer.transform(args[1])        
        model,method = func(train_vec, test_vec, args[2], args[3], **kwargs)
        pred = model.predict(test_vec)
        print("="*100)
        print(f"Method: {method}")
        print(classification_report(args[3], pred))
        print("\nconfusion matrix:")
        print(confusion_matrix(args[3], pred), "\n")
        print("="*100) 
    return wrapper
        
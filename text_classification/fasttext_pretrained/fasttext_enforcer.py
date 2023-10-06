# 필요한 버전 다운로드
# https://www.lfd.uci.edu/~gohlke/pythonlibs/#fasttext
# pip install whl디렉토리

import fasttext.util
import numpy as np
#from . import * # import module variables
#from .. import tokenizer
import io
import os
import pandas as pd
import inspect

from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

def download_fasttext_model(lang='ko'):
    main_dir = os.getcwd()
    # print(main_dir)
    script_dir = os.path.dirname(__file__)
    # print(script_dir)
    model_dir = os.path.join(script_dir, 'models')
    # print(model_dir)
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)
    os.chdir(model_dir)
    fasttext.util.download_model(lang, if_exists='ignore')
    os.chdir(main_dir)

# def tokenize(input):
#     if isinstance(input, str):
#         return tokenizer.morphs(input)
#     elif isinstance(input, list):
#         return [tokenizer.morphs(s) for s in input]
#     else:
#         raise Exception("tokenize input error: input type must be str or list")

def load_model(model_name='cc.ko.300.bin'):
    dir = os.path.join(os.path.dirname(__file__),'models',model_name)
    model = fasttext.load_model(dir)
    #print(type(fastext_model))
    return model

def validate_file_name(txt, ext, exp):
    '''
        txt: text
        ext: extension
        exp: explain
    '''
    
    current_frame = inspect.currentframe()
    caller_frame = inspect.getouterframes(current_frame, 2)
    caller_name = caller_frame[1][3]
    
    query = txt.split('.')
    if len(query) < 2:
        raise Exception(f"{caller_name} {exp} error: missing extension")
    if query[-1] != ext:
        raise Exception(f"{caller_name} {exp} error: the exention must be {ext}")
    

def train_unsupervised_model(texts, model_name='my_model.bin', save=False):

    validate_file_name(model_name, 'bin', "model_name input")
    
    model_path = os.path.join(os.path.dirname(__file__),'cutom_models',model_name)
    
    # case1 -> texts is str which is filename
    if isinstance(texts, str):
        model = fasttext.train_unsupervised(texts, model='skipgram')
    # case2 -> texts is list of str
    if isinstance(texts, list):
        buffer = io.StringIO("\n".join(texts))
        model = fasttext.train_unsupervised(input=buffer, model='skipgram')
    #model.save_model(modelname)
    else:
        raise Exception('train model input error: input type must be str or list')
    
    if save:
        model.save_model(model_path)
            
    return model
    
def get_fasttext_vectors(model, input):
    input = [s.replace('\n', ' ') for s in input]
    output = np.array([model.get_sentence_vector(s) for s in input])
    return output

def series_to_ft_supervised_format(text, label, file_name):
    '''
        convert saperate feature/label pd.series to query
    '''
    validate_file_name(file_name, 'txt', "file_name input")
    
    text = [t.replace("\n", " ")for t in text.tolist()]
    label = label.tolist()
    tmp_file_path = os.path.join(os.path.dirname(__file__), 'tmp', file_name)
    with open(tmp_file_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join([f'__label__{l} {t}' for t, l in zip(text,label)]))
    return tmp_file_path

def df_to_supervised(df, file_name, text_col='ko', label_col='label'):
    '''
        convert df to text
        and return name
    '''
    
    validate_file_name(file_name, 'txt', 'file_name input')
    
    def fl_to_query(r):
        text = r[text_col].replace('\n', ' ')
        label = f'__label__{str(r[label_col])}'
        return f'{label} {text}'
    #df['fasttext'] = df.apply(fl_to_query, axis=1)
    #return df['fasttext'].values.tolist()
    
    tmp_file_path = os.path.join(os.path.dirname(__file__), 'tmp', file_name)
    with open(tmp_file_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join([fl_to_query(row) for _, row in df.iterrows()]))
    
    return file_name

def train_supervised_model(text_file_name, save_name="", save_model = False):
    '''
        the text must follow this format with line-separated text
        
        __label__LABELNAME YOURTEXTHERE 
    '''
    validate_file_name(text_file_name, 'txt', 'text_file_name input')
    
    
    if save_name:
        validate_file_name(save_name, 'bin', "save_name input")
    
    
    text_file_path = os.path.join(os.path.dirname(__file__), 'tmp', text_file_name)    
    model = fasttext.train_supervised(text_file_path)

    if save_name and save_model:
        save_model_path = os.path.join(os.path.dirname(__file__), 'models', save_name)
        model.save_model(save_model_path)
        return model, save_name
    
    return model
    

def test_supervised_model(model_name, test_file_name):
    '''
        the text must follow this format with line-separated text
        
        __label__LABELNAME YOURTEXTHERE 
    
    '''
    #print(type(model_name))
    model = None
    if isinstance(model_name, str):
        validate_file_name(model_name, 'bin', "model_name input")
    elif isinstance(model_name, fasttext.FastText._FastText):
        model = model_name 
        
    if model == None:
        model_path = os.path.join(os.path.dirname(__file__),'models', model_name) 
        model = fasttext.load_model(model_path)
    
    validate_file_name(test_file_name, 'txt', "text_file_name input")
    
    #print(test_file_name)
    
    with open(test_file_name, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        labels = [line.split()[0] for line in lines]
        texts = [' '.join(line.split()[1:]) for line in lines]

    # Get predictions
    predicted_labels, _ = model.predict(texts)
    predicted_labels = [label[0] for label in predicted_labels]

    # Calculate metrics
    accuracy = accuracy_score(labels, predicted_labels)
    report = classification_report(labels, predicted_labels, target_names=model.get_labels())

    # Generate confusion matrix
    conf_matrix = confusion_matrix(labels, predicted_labels, labels=model.get_labels())

    # Print results
    print(f"Accuracy: {accuracy}\n")
    print("Classification Report:")
    print(report)
    print("\nConfusion Matrix:")
    print(conf_matrix)    
    

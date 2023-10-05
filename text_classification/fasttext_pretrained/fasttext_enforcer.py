# 필요한 버전 다운로드
# https://www.lfd.uci.edu/~gohlke/pythonlibs/#fasttext
# pip install whl디렉토리

import fasttext.util
import numpy as np
#from . import * # import module variables
#from .. import tokenizer
import io
import os
import shutil

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

def train_model(texts, model_name='my_model.bin', save=False):

    model_query = model_name.split('.')
    
    if len(model_query) < 2 :
        raise Exception("the file name and extension must exist")
    if model_name[-1] != 'bin':
        raise Exception("the model requires .bin extension")
    
    model_name = os.path.join(os.path.dirname(__file__),'cutom_models',model_name)
    
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
        model.save_model(model_name)
            
    return model
    
def get_fasttext_vectors(model, input):
    input = [s.replace('\n', ' ') for s in input]
    output = np.array([model.get_sentence_vector(s) for s in input])
    return output
# 필요한 버전 다운로드
# https://www.lfd.uci.edu/~gohlke/pythonlibs/#fasttext
# pip install whl디렉토리

import fasttext.util
import numpy as np
from . import * # import module variables
from .. import okt
import io
import os

def download_fasttext_model(lang='ko'):
    fasttext.util.download_model(lang, if_exists='ignore')

def tokenize(input):
    if isinstance(input, str):
        return okt.morphs(input)
    elif isinstance(input, list):
        return [okt.morphs(s) for s in input]
    else:
        raise Exception("tokenize input error: input type must be str or list")

def load_model(dir=f'{MODELPATH}/cc.ko.bin'):
    fastext_model = fasttext.load_model(dir)
    return fastext_model

def train_model(texts, modelname=f'{MODELPATH}/cc.ko.bin', save=False):
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
        model.save_model(modelname)
            
    return model
    
def get_fasttext_vectors(model, input):
    if isinstance(input, str):
        return model.get_sentence_vector(input)
    elif isinstance(input, list):
        return [model.get_sentence_vector(s) for s in input]
    
from . import fasttext_enforcer
#from konlpy.tag import Okt

#LANGUAGE = 'ko'
#MODELPATH = "models"
#okt = Okt()

fasttext_enforcer.download_fasttext_model(lang='ko')

import os
def initialize_source_directories():
    script_dir = os.path.dirname(__file__)
    
    tmp_dir = os.path.join(script_dir, 'tmp')
    
    # dirs = []
    # for dir in dirs:
    #     if not os.path.exists(dir):
    #     os.mkdir(dir)    
    
    if not os.path.exists(tmp_dir):
        os.mkdir(tmp_dir)

initialize_source_directories()
    



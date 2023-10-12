#추가 모델


#boosting model
#catboost - 최신 모델 -> 모듈 적용 완료
#lightgbm - 가벼운 모델
#XGboost - 근본 부스팅 모델
#adaboost - 기타 모델

import os

workspace_path = os.getcwd()

def set_workspace(path, option='y'):
    global workspace_path
    option = option.lower()
    if not os.path.exists(path):
        if option != 'y':    
            msg = input('do you want to make this directory? [y|n]')
            if msg.lower() == 'n':
                print("path establishment failed")
                return None
        os.makedirs(path)
        print("the path established")
        workspace_path = path
        return path
    workspace_path = path    
    return path
    
def use_workspace(data_path):
    lst = data_path.split('/')
    return os.path.join(workspace_path, *lst)
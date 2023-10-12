import os

cur_workspace = os.getcwd()

def set_workspace(path, option='y'):
    
    '''
        **caution: you must provide absolute path of directory
    '''
    option = option.lower()
    if not os.path.exists(path):
        print("path you provided does not exist")
        if option != 'y':    
            msg = input('do you want to make this directory? [y|n]')
            if msg.lower() == 'n':
                print("path establishment failed")
                return None
        os.makedirs(path)
        print(f"the path established to {cur_workspace}")
        cur_workspace = path
        return path
    print(f"path exists set workspace path to {cur_workspace}")
    cur_workspace = path    
    return path

def reset_workspace():
    cur_workspace = os.getcwd()
    print(f"work space resseted to {cur_workspace}")
    return cur_workspace
    
def workspace(data_path):
    lst = data_path.split('/')
    return os.path.join(cur_workspace, *lst)

def get_workspace():
    return cur_workspace
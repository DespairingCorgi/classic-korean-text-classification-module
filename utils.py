import os

cur_workspace = os.getcwd()

def set_workspace(path, option='y'):
    global cur_workspace
    
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
    cur_workspace = path   
    print(f"path exists set workspace path to {cur_workspace}")
    return path

def reset_workspace():
    global cur_workspace
    cur_workspace = os.getcwd()
    print(f"work space resseted to {cur_workspace}")
    return cur_workspace
    
def workspace(data_path):
    global cur_workspace
    return '/'.join([cur_workspace, data_path])

def get_workspace():
    global cur_workspace
    return cur_workspace
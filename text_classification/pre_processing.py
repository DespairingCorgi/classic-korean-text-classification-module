from sklearn.model_selection import train_test_split
from . import tokenizer
import pandas as pd
import json

def okt_process(texts):
    return tokenizer.morphs(texts)

def under_sample(data, label_col="label", undersample_size = 1.0, random_seed=42) -> pd.DataFrame:
    '''
        data: (pandas.DataFrame) the data to downsample
    '''
    # 가정: df는 데이터프레임, 'label'은 레이블을 나타내는 컬럼
    min_sample_count = data[label_col].value_counts().min()

    # 각 레이블에 대해 동일한 개수의 샘플을 무작위로 추출
    balanced_df = pd.concat([
        data[data[label_col] == unique_label].sample(int(min_sample_count*undersample_size), random_state=random_seed) \
            for unique_label in data[label_col].unique()
    ])
    
    return balanced_df
    

def open_data(data) -> pd.DataFrame:
    '''
        data: (pandas.DataFrame | str) dataframe or filepath 
    '''
    if isinstance(data, str):
        extension = data.split('.')[-1]
        if extension == "csv":
            data = pd.read_csv(data)
        elif extension == "json":
            data = pd.read_json(data)
        else:
            raise Exception("error ocurred: file extension must be ")
    elif isinstance(data, pd.DataFrame):
        print("data is already a pandas.DataFrame object")
    else:
        raise Exception("erro ocurred: it is neither str or pandas.DataFrame")
    return data

def tokenizing(data, feature_col = "ko"):
    data[feature_col] = data[feature_col].apply(lambda r : ' '.join(tokenizer.morphs(r)))
    return data

def split_data(data, feature_col="ko", label_col="label", train_size = .8, random_seed = 42,\
            under_sampling = True, under_sample_size = 1.0, under_sample_random_seed = 42,\
            require_tokenize = True):
    '''
        read data
            data: (pandas.DataFrame | str) dataframe or filepath 
            feature_col: (str) text column name. default = "ko"
            label_col: (label) column name. default = "label"
        
        split data
            train_size: (float) train size ratio.
            test_size: (float) test size ratio
            random_seed: (int) random seed for sampling. 
        
        under sampling data
        down_sampling: (bool)        
    '''
    
    # open file if data is str type
    try:
        data = open_data(data)
    except Exception as e:
        raise e
    
    if require_tokenize:
        data = tokenizing(data, feature_col=feature_col)
        #data[label_col] = data[label_col].apply(lambda r : ' '.join(tokenizer.morphs(r)))
    # leave required data features
    data = data[[feature_col, label_col]]
    
    # under sample data
    if under_sampling:
        data = under_sample(data, label_col="label", undersample_size=under_sample_size, random_seed=under_sample_random_seed)
    
    
    train_feature, test_feature, train_label, test_label = train_test_split(data[feature_col],\
                                                                            data[label_col],\
                                                                            train_size=train_size,\
                                                                            random_state=random_seed)
    title = "shape"
    total_width = 50
    formatted_title = "{:=^{width}}".format(title, width=total_width)
    
    curshape = '''
    {title}
            |   {:^10}  |   {:^10}  |
    train:  |   {:^10}  |   {:^10}  |
    test:   |   {:^10}  |   {:^10}  |
    {:=^50}
    '''.format("feature", "label", train_feature.shape[0], len(train_label), test_feature.shape[0], len(test_label), '', \
        title=formatted_title)#, width=total_width)
    
    print(curshape)
    
    return train_feature, test_feature, train_label, test_label    


def label_to_idx(data, label_col, int_label_col, idx_dict, \
                save_dict=False, dict_name="meta.json", \
                save_df = False, data_name="idx_data.csv", \
                inplace=False):
    
    if save_dict:
        json.dumps(idx_dict, dict_name)
    
    df = open_data(data)
    if inplace:
        int_label_col = label_col
    df[int_label_col] = df[label_col].apply(lambda l: idx_dict[l])
    
    if save_df:
        extension = data_name.split('.')[-1]
        if extension == 'csv':
            df.to_csv(data_name, index=False)     
        elif extension == 'json':
            df.to_json(data_name, index=False)
        else:
            print("the data save is faile due to extension. the extension must be either .json or .csv")

    return df
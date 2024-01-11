import re
import time
import pandas as pd
import string
import argparse
from tqdm import tqdm
tqdm.pandas()

from pandarallel import pandarallel
pandarallel.initialize(progress_bar =True, nb_workers =6)

from bnunicodenormalizer import Normalizer

def load_file(file_dir= './train.csv'):
    return pd.read_csv(file_dir)


bnorm = Normalizer()
def normalization(sentence):
    _words = [bnorm(word)['normalized']  for word in sentence.split()]
    sentence = " ".join([word for word in _words if word is not None])
    return sentence

def extract_punctuation(text, punctuations):
    pattern = r'\s*([' + re.escape(''.join(punctuations)) + '])'
    punctuation_matches = re.findall(pattern, text)
    punctuation_list = list(punctuation_matches)
    return punctuation_list

def check_pun(text, punctuations):
    for p in punctuations:
        if p in text:
            return 1
    return 0

def get_unnecssary_pun(punctuations):
    r_pun= [] ## list of unnecessary punctuations
    for i in string.punctuation:
        if i not in punctuations:
            r_pun.append(i)
    return r_pun

def remove_unnecessary_punctuation(text, r_pun):
    # remove unnecessary punctuation with space
    pattern = r'\s*([' + re.escape(''.join(r_pun)) + '])'
    text = re.sub(pattern, '', text)
    return text

def remove_duplicate_punctuation(text):
    """
    This will be used in data preparation
    """
    # remove consecutive occurrences of the same punctuation
    pun= string.punctuation + '।'
    pattern = r'\s*([' + re.escape(''.join(pun)) + '])'
    text = re.sub(pattern, r'\1', text)
    return text

def remove_all_punctuation(text):
    re_pun= string.punctuation + '।'
    pattern = r'\s*([' + re.escape(''.join(re_pun)) + '])'
    text= re.sub(pattern, '', text)
    return text

def remove_space_before_punctuation(text):
    # remove spaces before punctuation
    pun= string.punctuation + '।'
    pattern = r'\s*([' + re.escape(''.join(pun)) + '])'
    #r'([{}]-)'.format(re.escape(pun))
    text = re.sub(pattern, r'\1', text)
    if text[-1] == ' ':
        text= text[:-1]
    return text

def add_space_after_punctuation(text, punctuations):
    for pun in punctuations:
        text= text.replace(pun, pun + ' ')
    return text

def labeling_data(text, punctuations, id2label):
    labels= []
    
    for token in text.split():
        label= id2label[0]
        for i, pun in enumerate(punctuations):
            if pun in token:
                label= id2label[i+1]
                break
        labels.append(label)
    return labels

def do_preprocessing(text, punctuations, r_pun):
    text= remove_unnecessary_punctuation(text=text, r_pun=r_pun)
    text= remove_duplicate_punctuation(text=text)
    text= remove_space_before_punctuation(text=text)
    text= add_space_after_punctuation(text, punctuations)
    return text

def prepare_data_for_token_classification(df, punctuations, r_pun, id2label, label2id):
    ## take only pun sentence and extract punctuations from sentence
    ## apply some pre-processing for removing ambiguious data
    """
    data preparation steps:
        1. Normalize the text
        2. Define the necessary punctuation class.[',', '।', '?', '!', '-', ';', ':'] and unnecessary punctuations.
        3. Do the following preprocessing for label generation [modified data column]: remove_unnecessary_punctuation -> remove_duplicate_punctuation -> remove_space_before_punctuation -> add_space_after_punctuation
        4. Extract the necessary punctuations for reference
        5. Generate label by calling "labeling_data" function on modified data.
        6. Do some sanaty check for the dataset.
        
    """
    df['sentence']= df['sentence'].parallel_apply(lambda x: normalization(x))
    df['pun']= df['sentence'].parallel_apply(lambda x: check_pun(x, punctuations))
    df= df[df['pun']==1].reset_index(drop= True)
    df['modified_sentence']= df['sentence'].parallel_apply(lambda x: do_preprocessing(x, punctuations, r_pun))

    df['punctuations']= df['sentence'].parallel_apply(lambda x: extract_punctuation(x, punctuations))
    ## extract tokens and generate labels
    re_pun= string.punctuation + '।'
    pattern = r'\s*([' + re.escape(''.join(re_pun)) + '])'
    df['tokens']=df['modified_sentence'].parallel_apply(lambda x: remove_all_punctuation(x).split()) # removing punctuation from input tokens #re.sub(pattern, '', x)
    df['labels']= df['modified_sentence'].parallel_apply(lambda x: labeling_data(x, punctuations, id2label) )

    ### sanity check for dataset
    ## as token classification task token length and label length must be equal
    df['t_len']=df['tokens'].parallel_apply(len)
    df['l_len']= df['labels'].parallel_apply(len)
    df['sanity']= df.parallel_apply(lambda x: 1 if x.t_len == x.l_len else 0, axis=1)

    df= df[df['sanity']==1].reset_index(drop= True)

    ## add some auxilary informations in the dataset for later useage
    for k, v in label2id.items():
        df[k]= df['labels'].parallel_apply(lambda x: x.count(k))

    print("#"*15+" Showing statistics of Classes. "+ "#"*15)
    for k in label2id.keys():
        print(k, ':' ,len(df[df[k] != 0])) 
    
    return df


def parse_arguments():
    parser= argparse.ArgumentParser(description= "Data Preparation for Punctuation classification")
    
    parser.add_argument("--filepath", type= str, default= '../train.csv' , help= "Data path")
    parser.add_argument("--punctuations", type= list, default= [',', '।', '?'], help= "List of punctuations for restoration")
    parser.add_argument("--num_worker", type=int, default=4, help= "Number of workers")
    
    arguments= parser.parse_args()
    return arguments

def main():
    args= parse_arguments()
    punctuations= args.punctuations
    r_pun= get_unnecssary_pun(args.punctuations)
    data= load_file(file_dir=args.filepath)
    
    id2label= {0: "O"}
    label2id= {"O":0}
    for i, pun in enumerate(punctuations):
        label2id[pun] = i+1
        id2label[i+1]= pun

    print(id2label, '\n', label2id)
    data= prepare_data_for_token_classification(data, punctuations, r_pun, id2label, label2id)
    
    return data

if __name__ == "__main__":
    start_time= time.time()
    data= main()
    data.to_parquet("./modified_data.parquet", index= False)
    print(f"Total Processing Time: {(time.time() - start_time)/60 : 0.3f} min")
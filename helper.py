import string
import re
from bnunicodenormalizer import Normalizer
from configuration import CONFIG
CONFIG= CONFIG()

bnorm = Normalizer()
def normalization(sentence):
    _words = [bnorm(word)['normalized']  for word in sentence.split()]
    sentence = " ".join([word for word in _words if word is not None])
    return sentence

r_pun= []
for i in string.punctuation:
    if i not in CONFIG.punctuations:
        r_pun.append(i)
        

def remove_unnecessary_punctuation(text):
    # remove unnecessary punctuation with space
    pattern = r'\s*([' + re.escape(''.join(r_pun)) + '])'
    text = re.sub(pattern, '', text)
    return text

def remove_all_punctuation(text):
    re_pun= string.punctuation + '।'
    pattern = r'\s*([' + re.escape(''.join(re_pun)) + '])'
    text= re.sub(pattern, '', text)
    return text

def remove_duplicate_punctuation(text):
    # remove consecutive occurrences of the same punctuation
    pun= string.punctuation + '।'
    pattern = r'\s*([' + re.escape(''.join(pun)) + '])'
    text = re.sub(pattern, r'\1', text)
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


def preprocessing(text, punctuations):
    text= normalization(text)
    text= remove_unnecessary_punctuation(text=text)
    text= remove_duplicate_punctuation(text=text)
    text= remove_space_before_punctuation(text=text)
    text= add_space_after_punctuation(text, punctuations)
    text= remove_all_punctuation(text).split()
    return text

def add_punctuation(tokens, predictions, word_ids):
    result= ''
    for i, d in enumerate(zip(word_ids, predictions)):
        word_id, pred= d
        if i== 0: # for first token
            previous_word_id= word_id
            result += tokens[word_id] + CONFIG.id2pun[pred] 
        else:
            if word_id== previous_word_id: # if word id matches [label is for same token, so don't do anything]
                pass
            else:
                previous_word_id= word_id
                result += tokens[word_id] + CONFIG.id2pun[pred] 
    
    result= normalization(result)
    return result  

def pun_inference_fn(text, model, tokenizer, config= CONFIG):
    orig_sent= text
    text= preprocessing(text)
#     tokens= toenizer.
    inputs= tokenizer(text, padding=True, truncation=True, is_split_into_words= True, return_tensors="pt")
    word_ids= inputs.word_ids()[1:-1]
    outputs= model(inputs['input_ids'].to(config.device), inputs['attention_mask'].to(config.device))
    outputs= outputs.detach().cpu().numpy().argmax(axis= -1)[0, 1:-1]
    result= add_punctuation(text, outputs, word_ids)
    print(f'Input sentence:     {orig_sent}')
    print(f'Predicted sentence: {result}')
    return result
    
############



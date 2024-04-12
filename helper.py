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
import torch.nn as nn
from transformers import AutoConfig, AutoModel
# from torch.nn.utils.rnn import pack_padded_sequence
class NER_MODEL(nn.Module):
    def __init__(self, model_name= None, cfg= CONFIG):
        super(NER_MODEL, self).__init__()
        self.cfg= cfg
        self.num_labels= self.cfg.num_labels
        if model_name != None:
            self.model_name= model_name
        else:
            self.model_name= self.cfg.model_name
            
        self.model_config= AutoConfig.from_pretrained(self.model_name, output_hidden_states= True)
        self.model= AutoModel.from_pretrained(self.model_name, config= self.model_config)
#         self.embeddings= model.embeddings
        
        self.dropout= nn.Dropout(p= 0.2)
        self.lstm = nn.LSTM(self.model_config.hidden_size, self.cfg.lstm_size, batch_first=True, bidirectional=True)

#         self.linear= nn.Linear(self.model_config.hidden_size*2, self.num_labels)
#         self.linear= nn.Linear(self.model_config.hidden_size, self.num_labels)
        self.linear= nn.Linear(self.cfg.lstm_size*2, self.num_labels)
        
    
    def forward(self, input_ids, attention_mask, targets= None):
        
        outputs= self.model(input_ids,
                            attention_mask= attention_mask,
                            )
#         embeddings= self.model.embeddings(input_ids)
        sequence_output= outputs['last_hidden_state'] # sequence_output has the following shape: (batch_size, sequence_length, 768)
        lstm_output, (last_hidden, last_cell) = self.lstm(sequence_output) ## extract the 1st token's embeddings
#         hidden = torch.cat((last_hidden[0], last_hidden[1]), dim=1)
        lstm_output= self.dropout(lstm_output)
        entity_logits = self.linear(lstm_output) ### assuming that you are only using the output of the last LSTM cell to perform classification

#         out= torch.cat((sequence_output, embeddings), dim=-1)
#         entity_logits= self.dropout(entity_logits)
#         entity_logits= sequence_output #self.dropout(sequence_output)
#         entity_logits= self.linear(entity_logits)
        
        return entity_logits #lstm_output

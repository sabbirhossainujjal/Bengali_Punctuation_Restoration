import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer
from configuration import CONFIG
CONFIG= CONFIG()

id2label = CONFIG.id2label
label2id = CONFIG.label2id

tokenizer = AutoTokenizer.from_pretrained(CONFIG.model_name)

def align_labels_with_tokens(tokens, labels):
    new_labels= []
    word_ids= tokens.word_ids()
    previous_word_id= None
    
    for word_id in word_ids:
#         print(word_id)
        if word_id is None:
            label= -100
        elif word_id != previous_word_id:
            label= label2id[labels[word_id]]
        else:
            label= -100 if word_id is None else label2id[labels[word_id]] # [CLS] has token id= 101. [PAD] token id= 0 ## tokenizer.pad_token_id
        previous_word_id= word_id
        new_labels.append(label)
    
    """-100 is used for all of the speciall token ['[CLS], ['SEP'], [PAD]'], which will help to compute loss for slot filling"""
    
    return new_labels


class CustomDataset(Dataset):
    def __init__(self, df, tokenizer, cfg):
        self.df= df
        self.cfg= cfg
        self.tokenizer= tokenizer
        self.max_length= self.cfg.max_length
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):
        """
        df has tokens column in which text is tokenized based on label.
        """
        text= list(self.df.tokens[index]) #normalization(self.df.sentence[index])
        labels= list(self.df.labels[index])
        inputs= self.tokenizer(text, 
                                truncation= True, 
                                padding=True, 
                                max_length=self.max_length,
                                is_split_into_words=True
                                )

        new_labels= align_labels_with_tokens(inputs, labels)
        
        return {
            "input_ids": inputs['input_ids'],
            'attention_mask': inputs['attention_mask'],
            'targets': new_labels            
        }
        

class Collate:
    def __init__(self, tokenizer):
        self.tokenizer= tokenizer
    
    def __call__(self, batch):
        output= dict()
        output["input_ids"] = [sample['input_ids'] for sample in batch]
        output["attention_mask"] = [sample["attention_mask"] for sample in batch]
        output['targets'] = [sample['targets'] for sample in batch]
        
        batch_max= max([len(ids) for ids in output['input_ids']])
        
        # dynamic padding
        if self.tokenizer.padding_side == "right":
            output["input_ids"] = [ids + (batch_max - len(ids))*[self.tokenizer.pad_token_id] for ids in output['input_ids']]
            output['attention_mask']= [mask + (batch_max - len(mask))*[0] for mask in output['attention_mask']]
            output['targets']= [target + (batch_max - len(target))*[-100] for target in output['targets']]
        else:
            output["input_ids"] = [(batch_max - len(ids))*[self.tokenizer.pad_token_id] + ids for ids in output['input_ids']]
            output['attention_mask']= [(batch_max - len(mask))*[0] + mask for mask in output['attention_mask']]
            output['targets']= [(batch_max - len(target))*[-100] + target for target in output['targets']]
        
        # convert array to tensor
        output["input_ids"] = torch.tensor(output['input_ids'], dtype= torch.long)
        output["attention_mask"] = torch.tensor(output['attention_mask'], dtype= torch.long)
        output['targets'] = torch.tensor(output['targets'], dtype=torch.long)#
        
        return output
    

collate_fn= Collate(tokenizer)

def prepare_dataloader(df_train, df_valid, tokenizer=tokenizer, cfg=CONFIG):
#     df_train= df[df.fold != fold].reset_index(drop= True) # 2 fold out of 3 fold is used as training data, and 1 fold for validation.
#     df_valid= df[df.fold == fold].reset_index(drop= True)
    # valid_labels = df_valid['labels'].values
    
    # converting dataFrame to dataset.
    train_dataset= CustomDataset(df_train, tokenizer, cfg)
    valid_dataset= CustomDataset(df_valid, tokenizer, cfg)
    
    train_loader= DataLoader(train_dataset, 
                            batch_size= cfg.train_batch_size, 
                            collate_fn= collate_fn, #merges a list of samples to form a mini-batch of Tensors
                            num_workers= cfg.num_workers, # how many subprocesses to use for data loading
                            shuffle= True, 
                            pin_memory= True,
                            drop_last= False, 
                            )
    
    valid_loader= DataLoader(valid_dataset, 
                            batch_size= cfg.valid_batch_size,
                            collate_fn= collate_fn, 
                            num_workers= cfg.num_workers,
                            shuffle= False,
                            pin_memory= True, #If True, the data loader will copy Tensors into device/CUDA pinned memory before returning them
                            drop_last= False,
                            )
    
    return train_loader, valid_loader

if __name__ == "__main__":
    pass
    # data= pd.read_parquet("./data.parquet")
    # print("Preparing Data Loader")
    # train_loader, valid_loader= prepare_dataloader(data, data, tokenizer, CONFIG)
    # # print(train_loader)
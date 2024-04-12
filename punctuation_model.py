import torch
import torch.nn as nn
from transformers import (
    AutoModel,
    AutoConfig,
    AutoTokenizer,
)
from configuration import CONFIG
CONFIG= CONFIG()


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

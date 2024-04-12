
import time
import torch
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from sklearn.metrics import f1_score
from torch.optim import lr_scheduler, AdamW
from collections import defaultdict
from utils import *
from helper import get_logger
from configuration import CONFIG
CONFIG = CONFIG()

def get_logger(filename='train'):
    from logging import getLogger, INFO, StreamHandler, FileHandler, Formatter
    logger = getLogger(__name__)
    logger.setLevel(INFO)
    handler1 = StreamHandler()
    handler1.setFormatter(Formatter("%(message)s"))
    handler2 = FileHandler(filename=f"{filename}.log")
    handler2.setFormatter(Formatter("%(message)s"))
    logger.addHandler(handler1)
    logger.addHandler(handler2)
    return logger

def compute_metrics(logits, labels):
    predictions = np.argmax(logits.detach().cpu().numpy(), axis=-1)
    labels= labels.detach().cpu().numpy()
#     predictions= logits
    # Remove ignored index (special tokens) and convert to labels
    true_labels = [[l for l in label if l != -100] for label in labels]
    true_predictions = [[p for (p, l) in zip(prediction, label) if l != -100]
                        for prediction, label in zip(predictions, labels)]
    f1_scores= []
    accuracy= []
    for truths, preds in zip(true_labels, true_predictions):
        score = f1_score(np.array(truths), np.array(preds), average='weighted')
        correct_preds = np.array(truths) == np.array(preds)
        accuracy.append(np.mean(correct_preds))
        f1_scores.append(score)
    
    return np.mean(f1_scores) , np.mean(accuracy)


def token_loss_fn(logits, labels, attention_mask= None):
    loss_fn= nn.CrossEntropyLoss(ignore_index= -100) # for ignoring special tokens
    num_labels= CONFIG.num_labels
    
    if attention_mask is not None:
        mask= attention_mask.view(-1) == 1 #mask for keeping the effective part
        active_logits= logits.view(-1, num_labels)[mask]
        active_labels= labels.view(-1)[mask]
#         print(active_logits.size(), active_labels.size())
        entity_loss= loss_fn(active_logits, active_labels)
        
    else:
        entity_loss= loss_fn(logits.view(-1, num_labels), labels.view(-1))
    
    return entity_loss

def get_optimizer(parameters, cfg= CONFIG):
    optimizer= AdamW(params=parameters, lr= cfg.learning_rate, weight_decay= cfg.weight_decay, eps= cfg.eps, betas= cfg.betas)
    return optimizer

def fetch_scheduler(optimizer):
    if CONFIG.scheduler == "CosineAnnealingLR":
        scheduler= lr_scheduler.CosineAnnealingLR(optimizer, T_max= CONFIG.T_max, eta_min= CONFIG.min_lr)
    elif CONFIG.scheduler == "CosineAnnealingWarmRestarts":
        scheduler= lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0= CONFIG.T_0, eta_min= CONFIG.min_lr)
    elif CONFIG.scheduler== "linear":
        scheduler= lr_scheduler.LinearLR(optimizer, start_factor= 0.01, end_factor= 1.0, total_iters= 100)
    elif CONFIG.scheduler == None:
        return None

    return scheduler

def train_one_epoch(model, dataloader, optimizer, scheduler, epoch, device= CONFIG.device):
    
    model.train()
    dataset_size= 0
    running_loss= 0.0
    score= []
    accuracy= []
    
    progress_bar= tqdm(enumerate(dataloader), total= len(dataloader))
    steps= len(dataloader)
    for step, data in progress_bar:
        ids= data['input_ids'].to(device, dtype= torch.long)
        masks= data['attention_mask'].to(device, dtype= torch.long)
        targets= data['targets'].to(device, dtype= torch.long)
        
        batch_size= ids.size(0)
        outputs= model(ids, masks)
        loss= token_loss_fn(outputs, targets, attention_mask= masks)
        f1_score, acc= compute_metrics(logits= outputs, labels= targets)
        score.append(f1_score)
        accuracy.append(acc)
        if CONFIG.gradient_accumulation_steps > 1:
            loss= loss/ CONFIG.gradient_accumulation_steps
        
        loss.backward()
        
        ## Gradient Accumulation
        if (step + 1) % CONFIG.gradient_accumulation_steps == 0 or step == steps:            
            optimizer.step() #Performs a single optimization step (parameter update)
            
            if scheduler is not None:
                scheduler.step()

            # clear out the gradients of all Variables 
            # in this optimizer (i.e. W, b)
            optimizer.zero_grad()
        
        running_loss += (loss.item() * batch_size)
        dataset_size += batch_size
        epoch_loss= running_loss/ dataset_size
        epoch_f1_score= np.mean(score)
        epoch_accuracy= np.mean(accuracy)
        
    progress_bar.set_postfix(Epoch= epoch,
                            Train_loss= epoch_loss,
                            F1_Score= epoch_f1_score,
                            Train_accuracy= epoch_accuracy,
                            LR= optimizer.param_groups[0]['lr'])
    
    return epoch_loss, epoch_f1_score, epoch_accuracy #

def valid_one_epoch(model, dataloader, epoch, device= CONFIG.device):
    model.eval()
    
    dataset_size= 0
    running_loss= 0.0
    score= []
    accuracy= []
    
    progress_bar= tqdm(enumerate(dataloader), total= len(dataloader))
    
    for _, data in progress_bar:
        ids= data['input_ids'].to(device, dtype= torch.long)
        masks= data['attention_mask'].to(device, dtype= torch.long)
        targets= data['targets'].to(device, dtype= torch.long)
        
        batch_size= ids.size(0)
        
        with torch.no_grad():
            outputs= model(ids, masks)
            loss= token_loss_fn(outputs, targets, attention_mask= masks)
            f1_score, acc= compute_metrics(logits= outputs, labels= targets)
        
        score.append(f1_score)
        accuracy.append(acc)
        if CONFIG.gradient_accumulation_steps > 1:
            loss= loss/ CONFIG.gradient_accumulation_steps
        
        
        running_loss += (loss.item() * batch_size)
        dataset_size += batch_size
        epoch_loss= running_loss/ dataset_size
        epoch_f1_score= np.mean(score)
        epoch_accuracy= np.mean(accuracy)
        
        progress_bar.set_postfix(Epoch= epoch,
                                Valid_loss= epoch_loss,
                                Valid_F1_Score= epoch_f1_score,
                                Valid_accuracy= epoch_accuracy
                                )
    return epoch_loss, epoch_f1_score, epoch_accuracy  #


def training_loop(train_loader, valid_loader, model, optimizer, scheduler, num_epochs= CONFIG.num_epochs, patience= 3):
    LOGGER = get_logger()
    print(f'#'*15)
    print('Training Started')
    print(f'#'*15)
    
    start= time.time()
    best_loss= np.inf
    if CONFIG.pretrained:
        best_score= CONFIG.best_score
    else:
        best_score= 0
    trigger_times= 0
    history= defaultdict(list)
    
    model = model.to(CONFIG.device)
    for epoch in range(CONFIG.start_epoch, CONFIG.start_epoch + num_epochs + 1):
        t1= time.time()
        train_epoch_loss, train_f1_score, train_accuracy= train_one_epoch(model, train_loader, optimizer, scheduler, epoch, CONFIG.device)
        valid_epoch_loss, valid_f1_score, valid_accuracy = valid_one_epoch(model, valid_loader, epoch, CONFIG.device)
        
        history['train_loss'].append(train_epoch_loss)
        history['valid_loss'].append(valid_epoch_loss)
        history['train_f1_score'].append(train_f1_score)
        history['valid_f1_score'].append(valid_f1_score)

        LOGGER.info(f'Epoch {epoch+1} - avg_train_loss: {train_epoch_loss:.4f}  avg_val_loss: {valid_epoch_loss:.4f}  time: {time.time() - t1:.0f}s')
        LOGGER.info(f'Epoch {epoch+1} - Train F1 Score: {train_f1_score:.4f} - Train Accuracy: {train_accuracy:.4f} - Valid F1 Score: {valid_f1_score:.4f} - Valid Accuracy: {valid_accuracy:.4f}')
        #####
        if  valid_accuracy >= best_score: #valid_epoch_loss #best_loss #
            trigger_times= 0
            print(f"Vlaidation Accuracy Improved {best_score} ---> {valid_accuracy}") #valid_f1_score
            best_score= valid_accuracy
            
            path= f"best_model.bin"
            torch.save(model.state_dict(), path)
            print(f"Model saved to {path}")
            # run_sample_test(epoch, model, tokenizer, config= CONFIG)
        else:
            trigger_times += 1
            if trigger_times >= patience:
                print("Early Stoping. \n")
                break
                
    time_elapsed= time.time() - start
    print('Training complete in {:.0f}h {:.0f}m {:.0f}s'.format(
        time_elapsed // 3600, (time_elapsed % 3600) // 60, (time_elapsed % 3600) % 60))
    
    return history, best_score


########################
# def add_punctuation(tokens, predictions, word_ids):
#     result= ''
#     for i, d in enumerate(zip(word_ids, predictions)):
#         word_id, pred= d
#         if i== 0: # for first token
#             previous_word_id= word_id
#             result += tokens[word_id] + CONFIG.id2pun[pred] 
#         else:
#             if word_id== previous_word_id: # if word id matches [label is for same token, so don't do anything]
#                 pass
#             else:
#                 previous_word_id= word_id
#                 result +=' ' +tokens[word_id] + CONFIG.id2pun[pred] 
#     if len(result)>0:
#         result= normalization(result)
#     else:
#         result= '।'
#     return result   

# def run_sample_test(epoch, model, tokenizer, config= CONFIG):
#     print(f"############ Running sample test on epoch {epoch} ############")
#     test_df= pd.read_csv('./train.csv')
#     random.seed(config.seed)
#     sentences= random.sample(list(test_df['gt_sentence'].values), k= 5)
#     for text in sentences:
#         orig_sent= text
#         text= preprocessing(text)
#     #     tokens= toenizer.
#         inputs= tokenizer(text, padding=True, truncation=True, is_split_into_words= True, return_tensors="pt")
#         word_ids= inputs.word_ids()[1:-1]
#         outputs= model(inputs['input_ids'].to(config.device), inputs['attention_mask'].to(config.device))
#         outputs= outputs.detach().cpu().numpy().argmax(axis= -1)[0, 1:-1]
#         result= add_punctuation(text, outputs, word_ids)
#         print(f'Input sentence:     {orig_sent}')
#         print(f'Predicted sentence: {result}')
        
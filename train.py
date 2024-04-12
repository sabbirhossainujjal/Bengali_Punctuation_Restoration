from tqdm import tqdm
import torch
from utils import *
import time
from collections import defaultdict
def training_loop(model, tokenizer, optimizer, scheduler, num_epochs= CONFIG.num_epochs, patience= 3):
    
    start= time.time()
    best_loss= np.inf
    if CONFIG.pretrained:
        best_score= CONFIG.best_score
    else:
        best_score= 0
    trigger_times= 0
    history= defaultdict(list)
    
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
            run_sample_test(epoch, model, tokenizer, config= CONFIG)
        else:
            trigger_times += 1
            if trigger_times >= patience:
                print("Early Stoping. \n")
                break
                
    time_elapsed= time.time() - start
    print('Training complete in {:.0f}h {:.0f}m {:.0f}s'.format(
        time_elapsed // 3600, (time_elapsed % 3600) // 60, (time_elapsed % 3600) % 60))
    
    return history, valid_epoch_loss, best_score


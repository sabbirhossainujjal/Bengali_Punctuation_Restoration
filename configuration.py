import torch

class CONFIG:
    train= True #False #True
    debug= False # #True
    test= True
    seed= 753
    n_folds= 3
    start_epoch= 18
    num_epochs= 8
    punctuations= [',', '।', '?'] #[',', '।', '?', '!', '-', ';', ':'] #[',', '।', '?', '!']
    label_names= ['O', ',', '।', '?'] #['O', 'COMMA', 'DARI', 'QUESTION', 'EXCLAMATION', 'HYPHEN', 'SEMICOLON', 'COLON'] #  #['O', 'COMMA', 'DARI', 'QUESTION', 'EXCLAMATION']
    num_labels= len(label_names)
    model_name="xlm-roberta-large" #"csebuetnlp/banglabert_large"  #"csebuetnlp/banglabert" # #csebuetnlp/BanglaParaphrase
    max_length= 128
    lstm_size= 512
    pretrained= True
    best_score=  0.955
    weight_path="./best_model.bin" 

    do_normalize= True
    train_batch_size= 12
    valid_batch_size= 16
    num_workers= 2
    device= torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    gradient_accumulation_steps= 4
    learning_rate= 1e-5
    weight_decay= 1e-3
    scheduler= "CosineAnnealingWarmRestarts" #"CosineAnnealingWarmRestarts" #"linear"
    T_max= 5000 + 50
    T_0= 50
    min_lr= 1e-9
    
    eps = 1e-6
    betas= [0.9, 0.999]

    id2label= {0: "O"}
    label2id= {"O":0}
    id2pun = {0: ' '}
    for i, pun in enumerate(punctuations):
        label2id[pun] = i+1
        id2label[i+1]= pun
        id2pun[i+1] = pun


    if debug:
        n_folds= 2
        num_epochs=2
        dataset_size= 300
    
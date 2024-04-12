from configuration import CONFIG
from dataset import prepare_dataloader
from utils import *
from Model import TokenClassificationModel
def main(df_train, df_valid):
    train_loader, valid_loader = prepare_dataloader(df_train=df_train, df_valid=df_valid)
    model = TokenClassificationModel()
    optimizer = get_optimizer(parameters=model.parameters())
    scheduler = fetch_scheduler(optimizer=optimizer)
    
    history, best_score = training_loop(train_loader=train_loader,
                                        valid_loader=valid_loader,
                                        model=model,
                                        optimizer=optimizer,
                                        scheduler=scheduler,
                                        num_epochs=1,
                                        patience=1
                                        )
    
    return history

if __name__ == "__main__":
    import pandas as pd
    df = pd.read_parquet("data.parquet")
    df = df[:300]
    train_df = df[:int(len(df)*0.8)]
    valid_df = df[int(len(df)*0.8):].reset_index(drop=True)
    history = main(train_df, valid_df)


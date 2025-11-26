from client import FederatedClient
from models.unetr_model import get_unetr
from datasets.brain_tumor_dataset import get_client_data

if __name__ == "__main__":
    
    train_loader, val_loader, _ = get_client_data()

    client = FederatedClient(
        client_id="client_1",
        model_fn=get_unetr,
        train_loader=train_loader,
        val_loader=val_loader,
        cur_round=1
    )

    client.run()

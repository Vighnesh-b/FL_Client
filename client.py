import os,csv
from datetime import datetime
import time
import torch
from utils.train_utils import train_one_epoch,evaluate,combined_loss
import config
import requests
from tqdm import tqdm

class FederatedClient:
    def __init__(self,client_id,model_fn,train_loader,val_loader,cur_round,device=None):

        self.client_id=client_id
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model_fn(self.device)

        self.global_model_dir = os.path.join("global_models")
        self.global_model_path = os.path.join(self.global_model_dir, "global_latest.pth")
        self.logs_path=os.path.join("logs.csv")
        if not os.path.exists(self.logs_path):
            self._init_log_file()

        self.cur_round=cur_round


        self.train_loader=train_loader
        self.val_loader=val_loader
        self.scaler=torch.cuda.amp.GradScaler('cuda')
        self.optimizer=torch.optim.Adam(self.model.parameters(), lr=1e-4)

    def _init_log_file(self):
        """Initialize the CSV log file with header if not present."""
        if not os.path.exists(self.logs_path):
            with open(self.logs_path, mode="w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["timestamp", "round", "epoch", "train_loss", "val_loss", "val_dice"])

    def _log_metrics(self, round_num, epoch, train_loss, val_loss, val_dice):
        """Append training/validation metrics to CSV."""
        with open(self.logs_path, mode="a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                round_num, epoch, 
                f"{train_loss:.4f}", f"{val_loss:.4f}", f"{val_dice:.4f}"
            ])

    def wait_for_global(self):
        while not os.path.exists(self.global_model_path):
            print(f"[Client {self.client_id}] Waiting for global model...")
            time.sleep(5)

        print(f"[Client {self.client_id}] Loading global model...")
        state = torch.load(self.global_model_path, map_location=self.device)
        self.model.load_state_dict(state)

    def train_one_round(self, epochs=config.EPOCHS_PER_CLIENT, loss_fn=combined_loss):
        for epoch in range(1,epochs+1):
            train_loss=train_one_epoch(self.model,self.train_loader,self.optimizer,loss_fn,self.device,self.scaler)
            val_loss,val_dice=evaluate(self.model,self.val_loader,loss_fn,self.device,0.5)
            self._log_metrics(self.cur_round,epoch,train_loss,val_loss,val_dice)

        print(f"[Client {self.client_id}] Local training complete for Round {self.cur_round}.")
        
    
    def save_local_checkpoint(self):
        os.makedirs("client_checkpoints", exist_ok=True)
        ckpt_path = os.path.join(
            "client_checkpoints",
            f"round_{self.cur_round}.pth"
        )
        torch.save(self.model.state_dict(), ckpt_path)
        print(f"[Client {self.client_id}] Saved local checkpoint → {ckpt_path}")
        return ckpt_path

    def send_update(self,federated_server_url, local_model_path):
        api_url = "http://127.0.0.1:5000/api/send-local-model"

        # Open the checkpoint file
        with open(local_model_path, "rb") as f:
            files = {
                "file": f
            }

            data = {
                "client_id": self.client_id,
                "dataset_size":len(self.train_loader.dataset),
                "federated_server_url":federated_server_url,
                "cur_round":self.cur_round
            }

            print(f"[Client {self.client_id}] Uploading checkpoint → {api_url}")
            response = requests.post(api_url, files=files, data=data)

        if response.status_code == 200:
            print(f"[Client {self.client_id}] Upload successful.")
            return response.json()

        print(f"[Client {self.client_id}] Upload failed → {response.text}")
        return None


    def pull_global_model(self,federated_server_url):
        api_url = f"{federated_server_url}/api/get-global-model"
        local_save_path = "global_models/global_latest.pth"
        response = requests.get(api_url, stream=True)
        if response.status_code != 200:
            print("Error:", response.status_code)
            return

        total_size = int(response.headers.get("content-length", 0))

        with open(local_save_path, "wb") as f, tqdm(
            total=total_size,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
            desc="Downloading Global Model"
        ) as bar:
            for chunk in response.iter_content(chunk_size=1024 * 1024):  # 1MB chunks
                if chunk:
                    f.write(chunk)
                    bar.update(len(chunk))

        print(f"\nDownload complete → {local_save_path}")



    










        
        


import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import threading
import csv
import os
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import requests
import torch

class FederatedClientGUI:
    def __init__(self, root, model_fn, train_loader, val_loader, device=None):
        self.root = root
        self.root.title("Federated Learning Client - Brain Tumor Segmentation")
        self.root.geometry("1000x700")
        self.root.configure(bg="#f0f0f0")
        
        self.model_fn = model_fn
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        self.client = None
        self.is_training = False
        self.stop_requested = False
        self.training_thread = None
        self.logs_path = "logs.csv"
        self.server_url = "http://127.0.0.1:8000"  # Federated server URL
        
        # Show setup screen first
        self.show_setup_screen()
        
    def show_setup_screen(self):
        """Initial setup screen for client configuration"""
        setup_frame = tk.Frame(self.root, bg="#f0f0f0")
        setup_frame.place(relx=0.5, rely=0.5, anchor=tk.CENTER)
        
        # Title
        tk.Label(
            setup_frame,
            text="🧠 Federated Client Setup",
            font=("Arial", 24, "bold"),
            bg="#f0f0f0",
            fg="#2c3e50"
        ).pack(pady=(0, 30))
        
        # Input frame
        input_frame = tk.Frame(setup_frame, bg="#ffffff", relief=tk.RAISED, borderwidth=2, padx=40, pady=40)
        input_frame.pack()
        
        # Client ID
        tk.Label(
            input_frame,
            text="Client ID:",
            font=("Arial", 12, "bold"),
            bg="#ffffff"
        ).grid(row=0, column=0, sticky="w", pady=10)
        
        self.client_id_entry = tk.Entry(
            input_frame,
            font=("Arial", 12),
            width=20,
            relief=tk.SOLID,
            borderwidth=1
        )
        self.client_id_entry.grid(row=0, column=1, padx=10, pady=10)
        self.client_id_entry.insert(0, "1")
        
        # Server URL
        tk.Label(
            input_frame,
            text="Server URL:",
            font=("Arial", 12, "bold"),
            bg="#ffffff"
        ).grid(row=1, column=0, sticky="w", pady=10)
        
        self.server_url_entry = tk.Entry(
            input_frame,
            font=("Arial", 12),
            width=20,
            relief=tk.SOLID,
            borderwidth=1
        )
        self.server_url_entry.grid(row=1, column=1, padx=10, pady=10)
        self.server_url_entry.insert(0, self.server_url)
        
        # Start button
        tk.Button(
            input_frame,
            text="Initialize Client",
            command=self.initialize_client,
            font=("Arial", 12, "bold"),
            bg="#27ae60",
            fg="white",
            relief=tk.RAISED,
            borderwidth=3,
            padx=30,
            pady=10,
            cursor="hand2"
        ).grid(row=2, column=0, columnspan=2, pady=(20, 0))
        
    def initialize_client(self):
        """Initialize the client with user inputs"""
        client_id = self.client_id_entry.get().strip()
        server_url = self.server_url_entry.get().strip()
        
        if not client_id:
            messagebox.showerror("Error", "Please enter a Client ID!")
            return
            
        if not server_url:
            messagebox.showerror("Error", "Please enter a Server URL!")
            return
        
        self.server_url = server_url
        
        # Try to get current round from server
        try:
            response = requests.get(f"{self.server_url}/api/get-current-round", timeout=5)
            if response.status_code == 200:
                cur_round = response.json().get("current_round", 1)
            else:
                messagebox.showwarning(
                    "Server Connection", 
                    "Could not get current round from server. Starting from round 1."
                )
                cur_round = 1
        except Exception as e:
            messagebox.showwarning(
                "Server Connection", 
                f"Could not connect to server: {str(e)}\nStarting from round 1."
            )
            cur_round = 1
        
        # Import and initialize client
        try:
            from client import FederatedClient
            
            self.client = FederatedClient(
                client_id=client_id,
                model_fn=self.model_fn,
                train_loader=self.train_loader,
                val_loader=self.val_loader,
                cur_round=cur_round,
                device=self.device
            )
            
            # Clear setup screen and show main UI
            for widget in self.root.winfo_children():
                widget.destroy()
            
            self.setup_main_ui()
            
        except Exception as e:
            messagebox.showerror("Initialization Error", f"Failed to initialize client:\n{str(e)}")
    
    def setup_main_ui(self):
        """Setup the main UI after client initialization"""
        # Header
        header_frame = tk.Frame(self.root, bg="#2c3e50", height=80)
        header_frame.pack(fill=tk.X, padx=0, pady=0)
        header_frame.pack_propagate(False)
        
        title_label = tk.Label(
            header_frame, 
            text="🧠 Federated Learning Client",
            font=("Arial", 20, "bold"),
            bg="#2c3e50",
            fg="white"
        )
        title_label.pack(pady=20)
        
        # Main container
        main_container = tk.Frame(self.root, bg="#f0f0f0")
        main_container.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Left Panel - Controls and Current Metrics
        left_panel = tk.Frame(main_container, bg="#ffffff", relief=tk.RAISED, borderwidth=2)
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))
        
        # Control Section
        control_frame = tk.LabelFrame(
            left_panel, 
            text="Training Controls", 
            font=("Arial", 12, "bold"),
            bg="#ffffff",
            padx=15,
            pady=15
        )
        control_frame.pack(fill=tk.X, padx=15, pady=15)
        
        # Client Info
        info_frame = tk.Frame(control_frame, bg="#ffffff")
        info_frame.pack(fill=tk.X, pady=(0, 15))
        
        tk.Label(
            info_frame, 
            text="Client ID:", 
            font=("Arial", 10, "bold"),
            bg="#ffffff"
        ).grid(row=0, column=0, sticky="w", pady=5)
        
        self.client_id_label = tk.Label(
            info_frame,
            text=str(self.client.client_id),
            font=("Arial", 10),
            bg="#ffffff",
            fg="#27ae60"
        )
        self.client_id_label.grid(row=0, column=1, sticky="w", padx=10)
        
        tk.Label(
            info_frame, 
            text="Current Round:", 
            font=("Arial", 10, "bold"),
            bg="#ffffff"
        ).grid(row=1, column=0, sticky="w", pady=5)
        
        self.round_label = tk.Label(
            info_frame,
            text=str(self.client.cur_round),
            font=("Arial", 10),
            bg="#ffffff",
            fg="#27ae60"
        )
        self.round_label.grid(row=1, column=1, sticky="w", padx=10)
        
        # Button frame for Start and Stop
        button_frame = tk.Frame(control_frame, bg="#ffffff")
        button_frame.pack(fill=tk.X, pady=10)
        
        # Training Button
        self.train_button = tk.Button(
            button_frame,
            text="▶ Start Training",
            command=self.start_training,
            font=("Arial", 12, "bold"),
            bg="#27ae60",
            fg="white",
            relief=tk.RAISED,
            borderwidth=3,
            padx=20,
            pady=10,
            cursor="hand2"
        )
        self.train_button.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))
        
        # Stop Button
        self.stop_button = tk.Button(
            button_frame,
            text="⏹ Stop",
            command=self.stop_training,
            font=("Arial", 12, "bold"),
            bg="#e74c3c",
            fg="white",
            relief=tk.RAISED,
            borderwidth=3,
            padx=20,
            pady=10,
            cursor="hand2",
            state=tk.DISABLED
        )
        self.stop_button.pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=(5, 0))
        
        # Sync Round Button
        self.sync_button = tk.Button(
            control_frame,
            text="🔄 Sync Round from Server",
            command=self.sync_round_from_server,
            font=("Arial", 10),
            bg="#3498db",
            fg="white",
            relief=tk.RAISED,
            borderwidth=2,
            padx=10,
            pady=5,
            cursor="hand2"
        )
        self.sync_button.pack(fill=tk.X, pady=(5, 0))
        
        # Current Metrics Section
        metrics_frame = tk.LabelFrame(
            left_panel,
            text="Current Training Metrics",
            font=("Arial", 12, "bold"),
            bg="#ffffff",
            padx=15,
            pady=15
        )
        metrics_frame.pack(fill=tk.X, padx=15, pady=(0, 15))
        
        # Metric displays
        self.metric_values = []
        metrics_grid = tk.Frame(metrics_frame, bg="#ffffff")
        metrics_grid.pack(fill=tk.X)
        
        # Train Loss
        self.create_metric_display(metrics_grid, "Train Loss", 0)
        self.train_loss_value = self.metric_values[0]
        
        # Val Loss
        self.create_metric_display(metrics_grid, "Val Loss", 1)
        self.val_loss_value = self.metric_values[1]
        
        # Val Dice
        self.create_metric_display(metrics_grid, "Val Dice", 2)
        self.val_dice_value = self.metric_values[2]
        
        # Training Log
        log_frame = tk.LabelFrame(
            left_panel,
            text="Training Log",
            font=("Arial", 12, "bold"),
            bg="#ffffff",
            padx=15,
            pady=15
        )
        log_frame.pack(fill=tk.BOTH, expand=True, padx=15, pady=(0, 15))
        
        self.log_text = scrolledtext.ScrolledText(
            log_frame,
            height=10,
            font=("Courier", 9),
            bg="#1e1e1e",
            fg="#00ff00",
            insertbackground="white"
        )
        self.log_text.pack(fill=tk.BOTH, expand=True)
        
        # Right Panel - History and Visualization
        right_panel = tk.Frame(main_container, bg="#ffffff", relief=tk.RAISED, borderwidth=2)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # History Button
        history_button = tk.Button(
            right_panel,
            text="📊 View Training History",
            command=self.show_history,
            font=("Arial", 11, "bold"),
            bg="#3498db",
            fg="white",
            relief=tk.RAISED,
            borderwidth=3,
            padx=15,
            pady=8,
            cursor="hand2"
        )
        history_button.pack(fill=tk.X, padx=15, pady=15)
        
        # Visualization Area
        viz_frame = tk.LabelFrame(
            right_panel,
            text="Metrics Visualization",
            font=("Arial", 12, "bold"),
            bg="#ffffff",
            padx=10,
            pady=10
        )
        viz_frame.pack(fill=tk.BOTH, expand=True, padx=15, pady=(0, 15))
        
        # Create matplotlib figure
        self.fig = Figure(figsize=(6, 5), dpi=80, facecolor='white')
        self.canvas = FigureCanvasTkAgg(self.fig, master=viz_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Initialize with empty plot
        self.update_plot()
        
        # Status Bar
        status_frame = tk.Frame(self.root, bg="#34495e", height=30)
        status_frame.pack(fill=tk.X, side=tk.BOTTOM)
        status_frame.pack_propagate(False)
        
        self.status_label = tk.Label(
            status_frame,
            text="● Ready",
            font=("Arial", 9),
            bg="#34495e",
            fg="#2ecc71",
            anchor="w"
        )
        self.status_label.pack(side=tk.LEFT, padx=10)
        
        # Initial log message
        self.log_message(f"Client {self.client.client_id} initialized successfully")
        self.log_message(f"Starting at Round {self.client.cur_round}")
        
    def create_metric_display(self, parent, label, row):
        frame = tk.Frame(parent, bg="#ecf0f1", relief=tk.RIDGE, borderwidth=2)
        frame.grid(row=row, column=0, sticky="ew", pady=5)
        parent.grid_columnconfigure(0, weight=1)
        
        tk.Label(
            frame,
            text=label,
            font=("Arial", 10, "bold"),
            bg="#ecf0f1"
        ).pack(side=tk.LEFT, padx=10, pady=8)
        
        value_label = tk.Label(
            frame,
            text="--",
            font=("Arial", 14, "bold"),
            bg="#ecf0f1",
            fg="#2c3e50"
        )
        value_label.pack(side=tk.RIGHT, padx=10, pady=8)
        
        self.metric_values.append(value_label)
    
    def log_message(self, message):
        """Add message to log text widget"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_text.insert(tk.END, f"[{timestamp}] {message}\n")
        self.log_text.see(tk.END)
        self.root.update_idletasks()
    
    def update_status(self, message, color="#2ecc71"):
        """Update status bar"""
        self.status_label.config(text=f"● {message}", fg=color)
        self.root.update_idletasks()
    
    def sync_round_from_server(self):
        """Sync current round from server"""
        try:
            response = requests.get(f"{self.server_url}/api/get-current-round", timeout=5)
            if response.status_code == 200:
                cur_round = response.json().get("current_round", self.client.cur_round)
                self.client.cur_round = cur_round
                self.round_label.config(text=str(cur_round))
                self.log_message(f"Synced with server: Round {cur_round}")
                self.update_status("Round synced", "#2ecc71")
            else:
                messagebox.showerror("Sync Error", "Failed to sync with server")
        except Exception as e:
            messagebox.showerror("Connection Error", f"Could not connect to server:\n{str(e)}")
    
    def start_training(self):
        """Start training in a separate thread"""
        if self.is_training:
            messagebox.showwarning("Training", "Training is already in progress!")
            return
        
        self.is_training = True
        self.stop_requested = False
        self.train_button.config(state=tk.DISABLED, text="⏸ Training...", bg="#95a5a6")
        self.stop_button.config(state=tk.NORMAL)
        self.sync_button.config(state=tk.DISABLED)
        self.update_status("Training in progress...", "#f39c12")
        self.log_message("Starting training round...")
        
        # Update display
        self.round_label.config(text=str(self.client.cur_round))
        
        # Store round number for rollback if needed
        self.current_training_round = self.client.cur_round
        
        # Start training in separate thread
        self.training_thread = threading.Thread(target=self.training_worker, daemon=True)
        self.training_thread.start()
    
    def stop_training(self):
        """Request training to stop and rollback logs"""
        if not self.is_training:
            return
        
        result = messagebox.askyesno(
            "Stop Training",
            "Are you sure you want to stop training?\nAll logs from this round will be deleted."
        )
        
        if result:
            self.stop_requested = True
            self.log_message("⚠ Stop requested - aborting training...")
            self.update_status("Stopping training...", "#e74c3c")
    
    def rollback_logs(self, round_num):
        """Delete all log entries from the specified round"""
        if not os.path.exists(self.logs_path):
            return
        
        try:
            # Read all logs
            with open(self.logs_path, 'r') as f:
                reader = csv.DictReader(f)
                rows = [row for row in reader if int(row['round']) != round_num]
            
            # Rewrite without the current round
            with open(self.logs_path, 'w', newline='') as f:
                if rows or True:  # Keep header even if empty
                    writer = csv.DictWriter(f, fieldnames=["timestamp", "round", "epoch", "train_loss", "val_loss", "val_dice"])
                    writer.writeheader()
                    writer.writerows(rows)
            
            self.log_message(f"✓ Rolled back logs for Round {round_num}")
            
        except Exception as e:
            self.log_message(f"✗ Error rolling back logs: {str(e)}")
    
    def training_worker(self):
        """Worker thread for training"""
        try:
            if self.stop_requested:
                raise InterruptedError("Training stopped by user")
            
            self.log_message("Pulling global model...")
            self.client.pull_global_model(self.server_url)
            
            if self.stop_requested:
                raise InterruptedError("Training stopped by user")
            
            self.log_message("Loading global model...")
            self.client.wait_for_global()
            
            # if self.stop_requested:
            #     raise InterruptedError("Training stopped by user")
            
            # self.log_message("Training locally...")
            # # Note: You might need to modify train_one_round to check self.stop_requested periodically
            # self.client.train_one_round()
            
            # if self.stop_requested:
            #     raise InterruptedError("Training stopped by user")
            
            # # Read latest metrics
            # self.update_metrics_from_log()
            
            # if self.stop_requested:
            #     raise InterruptedError("Training stopped by user")
            
            # self.log_message("Saving checkpoint...")
            # local_model_path = self.client.save_local_checkpoint()
            
            # if self.stop_requested:
            #     raise InterruptedError("Training stopped by user")
            
            self.log_message("Sending update to server...")
            self.client.send_update(self.server_url,os.path.join('client_checkpoints',"round_1.pth"))
            
            self.log_message("✓ Training round completed successfully!")
            self.update_status("Training completed", "#2ecc71")
            
            # Increment round for next training
            self.client.cur_round += 1
            self.round_label.config(text=str(self.client.cur_round))
            
        except InterruptedError as e:
            self.log_message(f"✗ {str(e)}")
            self.rollback_logs(self.current_training_round)
            self.update_status("Training aborted", "#e74c3c")
            
        except Exception as e:
            self.log_message(f"✗ Error during training: {str(e)}")
            self.update_status("Training failed", "#e74c3c")
            messagebox.showerror("Training Error", str(e))
        
        finally:
            self.is_training = False
            self.stop_requested = False
            self.train_button.config(state=tk.NORMAL, text="▶ Start Training", bg="#27ae60")
            self.stop_button.config(state=tk.DISABLED)
            self.sync_button.config(state=tk.NORMAL)
            self.update_plot()
    
    def update_metrics_from_log(self):
        """Read and display latest metrics from log file"""
        if not os.path.exists(self.logs_path):
            return
            
        try:
            with open(self.logs_path, 'r') as f:
                reader = csv.DictReader(f)
                rows = list(reader)
                
                if rows:
                    latest = rows[-1]
                    self.train_loss_value.config(text=f"{float(latest['train_loss']):.4f}")
                    self.val_loss_value.config(text=f"{float(latest['val_loss']):.4f}")
                    self.val_dice_value.config(text=f"{float(latest['val_dice']):.4f}")
                    
        except Exception as e:
            self.log_message(f"Error reading metrics: {str(e)}")
    
    def update_plot(self):
        """Update the metrics visualization plot"""
        if not os.path.exists(self.logs_path):
            return
            
        try:
            # Read data
            rounds = []
            train_losses = []
            val_losses = []
            val_dices = []
            
            with open(self.logs_path, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    rounds.append(int(row['round']))
                    train_losses.append(float(row['train_loss']))
                    val_losses.append(float(row['val_loss']))
                    val_dices.append(float(row['val_dice']))
            
            if not rounds:
                return
            
            # Clear previous plot
            self.fig.clear()
            
            # Create subplots
            ax1 = self.fig.add_subplot(211)
            ax2 = self.fig.add_subplot(212)
            
            # Plot losses
            ax1.plot(rounds, train_losses, 'b-o', label='Train Loss', linewidth=2, markersize=4)
            ax1.plot(rounds, val_losses, 'r-s', label='Val Loss', linewidth=2, markersize=4)
            ax1.set_xlabel('Round', fontsize=10)
            ax1.set_ylabel('Loss', fontsize=10)
            ax1.set_title('Training & Validation Loss', fontsize=11, fontweight='bold')
            ax1.legend(loc='upper right')
            ax1.grid(True, alpha=0.3)
            
            # Plot dice
            ax2.plot(rounds, val_dices, 'g-^', label='Val Dice', linewidth=2, markersize=4)
            ax2.set_xlabel('Round', fontsize=10)
            ax2.set_ylabel('Dice Score', fontsize=10)
            ax2.set_title('Validation Dice Score', fontsize=11, fontweight='bold')
            ax2.legend(loc='lower right')
            ax2.grid(True, alpha=0.3)
            
            self.fig.tight_layout()
            self.canvas.draw()
            
        except Exception as e:
            self.log_message(f"Error updating plot: {str(e)}")
    
    def show_history(self):
        """Show training history in a new window"""
        if not os.path.exists(self.logs_path):
            messagebox.showinfo("No History", "No training history available yet.")
            return
        
        # Create new window
        history_window = tk.Toplevel(self.root)
        history_window.title("Training History")
        history_window.geometry("900x500")
        history_window.configure(bg="#f0f0f0")
        
        # Title
        title_frame = tk.Frame(history_window, bg="#2c3e50", height=60)
        title_frame.pack(fill=tk.X)
        title_frame.pack_propagate(False)
        
        tk.Label(
            title_frame,
            text="📈 Complete Training History",
            font=("Arial", 16, "bold"),
            bg="#2c3e50",
            fg="white"
        ).pack(pady=15)
        
        # Table frame
        table_frame = tk.Frame(history_window, bg="#ffffff", relief=tk.RAISED, borderwidth=2)
        table_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Create Treeview
        columns = ("timestamp", "round", "epoch", "train_loss", "val_loss", "val_dice")
        tree = ttk.Treeview(table_frame, columns=columns, show="headings", height=15)
        
        # Define headings
        tree.heading("timestamp", text="Timestamp")
        tree.heading("round", text="Round")
        tree.heading("epoch", text="Epoch")
        tree.heading("train_loss", text="Train Loss")
        tree.heading("val_loss", text="Val Loss")
        tree.heading("val_dice", text="Val Dice")
        
        # Define column widths
        tree.column("timestamp", width=150)
        tree.column("round", width=80)
        tree.column("epoch", width=80)
        tree.column("train_loss", width=120)
        tree.column("val_loss", width=120)
        tree.column("val_dice", width=120)
        
        # Add scrollbar
        scrollbar = ttk.Scrollbar(table_frame, orient=tk.VERTICAL, command=tree.yview)
        tree.configure(yscroll=scrollbar.set)
        
        # Pack
        tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Load data
        try:
            with open(self.logs_path, 'r') as f:
                reader = csv.DictReader(f)
                for i, row in enumerate(reader):
                    tag = 'evenrow' if i % 2 == 0 else 'oddrow'
                    tree.insert("", tk.END, values=(
                        row['timestamp'],
                        row['round'],
                        row['epoch'],
                        row['train_loss'],
                        row['val_loss'],
                        row['val_dice']
                    ), tags=(tag,))
            
            # Style rows
            tree.tag_configure('evenrow', background='#f9f9f9')
            tree.tag_configure('oddrow', background='#ffffff')
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load history: {str(e)}")

from client import FederatedClient
from models.unetr_model import get_unetr
from datasets.brain_tumor_dataset import get_client_data

# Example usage
if __name__ == "__main__":
    root = tk.Tk()

    train_loader, val_loader, _ = get_client_data()

    app = FederatedClientGUI(
        root, 
        model_fn=get_unetr,
        train_loader=train_loader,
        val_loader=val_loader
    )
    
    root.mainloop()
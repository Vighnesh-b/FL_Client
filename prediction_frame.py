import tkinter as tk
from tkinter import filedialog, messagebox
import os
from predict_mask import predict_and_evaluate_mask
import torch
class PredictionFrame:
    """Handles the prediction UI and functionality"""
    
    def __init__(self, parent, client, device, update_status_callback):
        """
        Args:
            parent: The parent container (content_container)
            client: FederatedClient instance
            device: torch device
            update_status_callback: Function to update main status bar
        """
        self.parent = parent
        self.client = client
        self.device = device
        self.update_status = update_status_callback
        
        self.frame = None
        self.loaded_image_path = None
        self.loaded_model_path = None
        self.loaded_groundtruth_path = None
        
    def create_frame(self, back_callback):
        """Create the prediction frame UI"""
        self.frame = tk.Frame(self.parent, bg="#f0f0f0")
        
        # Back button at top
        back_button_frame = tk.Frame(self.frame, bg="#f0f0f0")
        back_button_frame.pack(fill=tk.X, pady=(0, 10))
        
        tk.Button(
            back_button_frame,
            text="← Back to Training",
            command=back_callback,
            font=("Arial", 11, "bold"),
            bg="#34495e",
            fg="white",
            relief=tk.RAISED,
            borderwidth=2,
            padx=15,
            pady=8,
            cursor="hand2"
        ).pack(side=tk.LEFT)
        
        # Main prediction container
        pred_main = tk.Frame(self.frame, bg="#f0f0f0")
        pred_main.pack(fill=tk.BOTH, expand=True)
        
        # Left Panel - Controls and Upload (Scrollable)
        left_container = tk.Frame(pred_main, bg="#f0f0f0", width=320)
        left_container.pack(side=tk.LEFT, fill=tk.BOTH, padx=(0, 10))
        left_container.pack_propagate(False)
        
        left_canvas = tk.Canvas(left_container, bg="#ffffff", highlightthickness=0)
        left_scrollbar = tk.Scrollbar(left_container, orient="vertical", command=left_canvas.yview)
        
        left_panel = tk.Frame(left_canvas, bg="#ffffff", relief=tk.RAISED, borderwidth=2)
        
        # Configure canvas
        left_canvas.configure(yscrollcommand=left_scrollbar.set)
        left_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        left_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Create window in canvas
        canvas_frame = left_canvas.create_window((0, 0), window=left_panel, anchor="nw")
        
        # Bind configure event to update scroll region
        def configure_left_panel(event):
            left_canvas.configure(scrollregion=left_canvas.bbox("all"))
            # Make the canvas frame width match the canvas width
            canvas_width = left_canvas.winfo_width()
            left_canvas.itemconfig(canvas_frame, width=canvas_width)
        
        left_panel.bind("<Configure>", configure_left_panel)
        left_canvas.bind("<Configure>", lambda e: left_canvas.itemconfig(canvas_frame, width=e.width))
        
        # Bind mouse wheel for scrolling
        def on_left_mousewheel(event):
            left_canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        
        left_canvas.bind("<Enter>", lambda e: left_canvas.bind_all("<MouseWheel>", on_left_mousewheel))
        left_canvas.bind("<Leave>", lambda e: left_canvas.unbind_all("<MouseWheel>"))
        
        # Upload Section
        upload_frame = tk.LabelFrame(
            left_panel,
            text="Image Upload",
            font=("Arial", 12, "bold"),
            bg="#ffffff",
            padx=15,
            pady=15
        )
        upload_frame.pack(fill=tk.X, padx=15, pady=15)
        
        tk.Button(
            upload_frame,
            text="📁 Load Image",
            command=self.load_prediction_image,
            font=("Arial", 11, "bold"),
            bg="#3498db",
            fg="white",
            relief=tk.RAISED,
            borderwidth=2,
            padx=15,
            pady=10,
            cursor="hand2"
        ).pack(fill=tk.X, pady=(0, 10))
        
        self.pred_image_label = tk.Label(
            upload_frame,
            text="No image loaded",
            font=("Arial", 9),
            bg="#ffffff",
            fg="#7f8c8d"
        )
        self.pred_image_label.pack()
        
        # Ground Truth Section
        groundtruth_frame = tk.LabelFrame(
            left_panel,
            text="Ground Truth Mask (Optional)",
            font=("Arial", 12, "bold"),
            bg="#ffffff",
            padx=15,
            pady=15
        )
        groundtruth_frame.pack(fill=tk.X, padx=15, pady=(0, 15))
        
        tk.Button(
            groundtruth_frame,
            text="📁 Load Ground Truth",
            command=self.load_groundtruth_mask,
            font=("Arial", 11, "bold"),
            bg="#e67e22",
            fg="white",
            relief=tk.RAISED,
            borderwidth=2,
            padx=15,
            pady=10,
            cursor="hand2"
        ).pack(fill=tk.X, pady=(0, 10))
        
        self.groundtruth_label = tk.Label(
            groundtruth_frame,
            text="No ground truth loaded",
            font=("Arial", 9),
            bg="#ffffff",
            fg="#7f8c8d"
        )
        self.groundtruth_label.pack()
        
        # Model Selection
        model_frame = tk.LabelFrame(
            left_panel,
            text="Model Selection",
            font=("Arial", 12, "bold"),
            bg="#ffffff",
            padx=15,
            pady=15
        )
        model_frame.pack(fill=tk.X, padx=15, pady=(0, 15))
        
        tk.Label(
            model_frame,
            text="Select model checkpoint:",
            font=("Arial", 10),
            bg="#ffffff"
        ).pack(anchor="w", pady=(0, 5))
        
        tk.Button(
            model_frame,
            text="📂 Browse Checkpoint",
            command=self.load_model_checkpoint,
            font=("Arial", 10),
            bg="#16a085",
            fg="white",
            relief=tk.RAISED,
            borderwidth=2,
            padx=10,
            pady=8,
            cursor="hand2"
        ).pack(fill=tk.X, pady=(0, 10))
        
        self.model_path_label = tk.Label(
            model_frame,
            text="Using current model",
            font=("Arial", 9),
            bg="#ffffff",
            fg="#7f8c8d",
            wraplength=250
        )
        self.model_path_label.pack()
        
        # Prediction Button
        predict_frame = tk.Frame(left_panel, bg="#ffffff")
        predict_frame.pack(fill=tk.X, padx=15, pady=(0, 15))
        
        self.predict_btn = tk.Button(
            predict_frame,
            text="▶ Generate Prediction",
            command=self.run_prediction,
            font=("Arial", 12, "bold"),
            bg="#27ae60",
            fg="white",
            relief=tk.RAISED,
            borderwidth=3,
            padx=15,
            pady=12,
            cursor="hand2",
            state=tk.DISABLED
        )
        self.predict_btn.pack(fill=tk.X)
                
        # Right Panel - Visualization (Scrollable)
        right_container = tk.Frame(pred_main, bg="#f0f0f0")
        right_container.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        right_canvas = tk.Canvas(right_container, bg="#ffffff", highlightthickness=0)
        right_scrollbar_y = tk.Scrollbar(right_container, orient="vertical", command=right_canvas.yview)
        right_scrollbar_x = tk.Scrollbar(right_container, orient="horizontal", command=right_canvas.xview)
        
        right_panel = tk.Frame(right_canvas, bg="#ffffff", relief=tk.RAISED, borderwidth=2)
        
        # Configure canvas
        right_canvas.configure(yscrollcommand=right_scrollbar_y.set, xscrollcommand=right_scrollbar_x.set)
        right_scrollbar_y.pack(side=tk.RIGHT, fill=tk.Y)
        right_scrollbar_x.pack(side=tk.BOTTOM, fill=tk.X)
        right_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Create window in canvas
        right_canvas_frame = right_canvas.create_window((0, 0), window=right_panel, anchor="nw")
        
        # Bind configure event to update scroll region
        def configure_right_panel(event):
            right_canvas.configure(scrollregion=right_canvas.bbox("all"))
            # Make the canvas frame width match the canvas width if content is smaller
            canvas_width = right_canvas.winfo_width()
            if right_panel.winfo_reqwidth() < canvas_width:
                right_canvas.itemconfig(right_canvas_frame, width=canvas_width)
        
        right_panel.bind("<Configure>", configure_right_panel)
        right_canvas.bind("<Configure>", lambda e: configure_right_panel(e))
        
        # Bind mouse wheel for scrolling
        def on_right_mousewheel(event):
            right_canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        
        right_canvas.bind("<Enter>", lambda e: right_canvas.bind_all("<MouseWheel>", on_right_mousewheel))
        right_canvas.bind("<Leave>", lambda e: right_canvas.unbind_all("<MouseWheel>"))
        
        # Visualization title
        tk.Label(
            right_panel,
            text="Prediction Results",
            font=("Arial", 14, "bold"),
            bg="#ffffff",
            fg="#2c3e50"
        ).pack(pady=15)
        
        # Canvas for displaying images
        self.pred_canvas_frame = tk.Frame(right_panel, bg="#ecf0f1", relief=tk.SUNKEN, borderwidth=2, height=500)
        self.pred_canvas_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=(0, 20))
        
        # Placeholder text
        self.pred_placeholder = tk.Label(
            self.pred_canvas_frame,
            text="Load an image and run prediction to see results",
            font=("Arial", 12),
            bg="#ecf0f1",
            fg="#7f8c8d"
        )
        self.pred_placeholder.place(relx=0.5, rely=0.5, anchor=tk.CENTER)
        
        return self.frame
    
    def show(self):
        """Show the prediction frame"""
        if self.frame:
            self.frame.pack(fill=tk.BOTH, expand=True)
            self.update_status("Prediction Mode", "#9b59b6")
    
    def hide(self):
        """Hide the prediction frame"""
        if self.frame:
            self.frame.pack_forget()
    
    def load_prediction_image(self):
        """Load an image for prediction"""
        file_path = filedialog.askopenfilename(
            title="Select Medical Image",
            initialdir='data/Testing/images',
            filetypes=[
                ("All files", "*.*")
            ]
        )
        
        if file_path:
            self.loaded_image_path = file_path
            filename = os.path.basename(file_path)
            self.pred_image_label.config(text=f"✓ {filename}", fg="#27ae60")
            self.predict_btn.config(state=tk.NORMAL)
            self.update_status("Image loaded - ready to predict", "#3498db")
    
    def load_groundtruth_mask(self):
        """Load ground truth mask for comparison"""
        file_path = filedialog.askopenfilename(
            title="Select Ground Truth Mask",
            initialdir='data/Testing/masks',
            filetypes=[
                ("All files", "*.*")
            ]
        )
        
        if file_path:
            self.loaded_groundtruth_path = file_path
            filename = os.path.basename(file_path)
            self.groundtruth_label.config(text=f"✓ {filename}", fg="#27ae60")
            self.update_status("Ground truth loaded", "#3498db")
    
    def load_model_checkpoint(self):
        """Load a specific model checkpoint"""
        file_path = filedialog.askopenfilename(
            title="Select Model Checkpoint",
            initialdir='client_checkpoints/',
            filetypes=[
                ("PyTorch Model", "*.pth *.pt"),
                ("All files", "*.*")
            ]
        )
        
        if file_path:
            self.loaded_model_path = file_path
            filename = os.path.basename(file_path)
            self.model_path_label.config(text=f"✓ {filename}", fg="#27ae60")
    
    def run_prediction(self):
        """Run prediction on the loaded image"""
        if not self.loaded_image_path:
            messagebox.showwarning("No Image", "Please load an image first!")
            return
        
        self.update_status("Running prediction...", "#f39c12")
        self.predict_btn.config(state=tk.DISABLED, text="⏳ Predicting...")
        
        try:
            # Import the prediction function
            from predict_mask import predict_and_evaluate_mask
            
            # Run prediction
            pred_mask, dice_scores, image, mask = predict_and_evaluate_mask(
                image_path=self.loaded_image_path,
                mask_path=self.loaded_groundtruth_path,
                model_path=self.loaded_model_path,
                device=self.device
            )
            
            # Clear previous results
            for widget in self.pred_canvas_frame.winfo_children():
                widget.destroy()
            
            # Display results
            self.display_prediction_results(pred_mask, dice_scores, image, mask)
            
            self.update_status("Prediction complete", "#2ecc71")
            
            # Show success message with dice scores if available
            if dice_scores is not None:
                messagebox.showinfo(
                    "Prediction Complete", 
                    f"Prediction successful!\n\n"
                    f"Image: {os.path.basename(self.loaded_image_path)}\n"
                    f"Ground Truth: {os.path.basename(self.loaded_groundtruth_path)}\n"
                    f"Model: {'Latest checkpoint' if not self.loaded_model_path else os.path.basename(self.loaded_model_path)}"
                )
            else:
                messagebox.showinfo(
                    "Prediction Complete", 
                    f"Prediction successful!\n\n"
                    f"Image: {os.path.basename(self.loaded_image_path)}\n"
                    f"Model: {'Latest checkpoint' if not self.loaded_model_path else os.path.basename(self.loaded_model_path)}\n"
                    f"(No ground truth provided for evaluation)"
                )
            
        except FileNotFoundError as e:
            messagebox.showerror("File Error", f"File not found:\n{str(e)}")
            self.update_status("Prediction failed", "#e74c3c")
        
        except Exception as e:
            messagebox.showerror("Prediction Error", f"An error occurred during prediction:\n{str(e)}")
            self.update_status("Prediction failed", "#e74c3c")
        
        finally:
            self.predict_btn.config(state=tk.NORMAL, text="▶ Generate Prediction")

    def display_prediction_results(self, pred_mask, dice_scores, image, mask):
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        import tkinter as tk

        # Store tensors
        self.pred_mask = pred_mask        # torch.Size([1, D, H, W])
        self.gt_mask = mask               # torch.Size([1, D, H, W])
        self.image_tensor = image         # torch.Size([3, D, H, W])

        self.depth = pred_mask.shape[1]
        self.current_slice = self.depth // 2  # Start in the middle

        # Create figure with 1 row, 5 columns
        self.fig = plt.Figure(figsize=(18, 5), dpi=140)
        self.fig.patch.set_facecolor("#fafafa")

        self.ax_c0   = self.fig.add_subplot(1, 5, 1)
        self.ax_c1   = self.fig.add_subplot(1, 5, 2)
        self.ax_c2   = self.fig.add_subplot(1, 5, 3)
        self.ax_gt   = self.fig.add_subplot(1, 5, 4)
        self.ax_pred = self.fig.add_subplot(1, 5, 5)

        # Initial draw
        self._draw_slice(self.current_slice)

        # Embed in Tkinter
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.pred_canvas_frame)
        self.canvas.draw()
        canvas_widget = self.canvas.get_tk_widget()
        canvas_widget.pack(fill=tk.BOTH, expand=True)

        # Slider frame
        slider_frame = tk.Frame(self.pred_canvas_frame, bg="#ecf0f1")
        slider_frame.pack(fill=tk.X, pady=10)

        tk.Label(
            slider_frame,
            text="Select Axial Slice (Z-axis)",
            font=("Arial", 11, "bold"),
            bg="#ecf0f1"
        ).pack()

        self.slice_slider = tk.Scale(
            slider_frame,
            from_=0,
            to=self.depth - 1,
            orient=tk.HORIZONTAL,
            length=900,
            command=self._on_slice_change,
            bg="#ecf0f1"
        )
        self.slice_slider.set(self.current_slice)
        self.slice_slider.pack()


    def _compute_dice(self, pred_slice, gt_slice):
        """Compute Dice score for a single 2D slice."""
        import numpy as np

        pred_bin = (pred_slice > 0.5).astype(np.float32)
        gt_bin   = (gt_slice > 0.5).astype(np.float32)

        intersection = (pred_bin * gt_bin).sum()
        denom = pred_bin.sum() + gt_bin.sum() + 1e-8

        return (2 * intersection) / denom


    def _draw_slice(self, z):
        # Clear axes
        self.ax_c0.clear()
        self.ax_c1.clear()
        self.ax_c2.clear()
        self.ax_gt.clear()
        self.ax_pred.clear()

        # --- Channels ---
        c0 = self.image_tensor[0, z].cpu().numpy()
        c1 = self.image_tensor[1, z].cpu().numpy()
        c2 = self.image_tensor[2, z].cpu().numpy()

        self.ax_c0.imshow(c0, cmap="gray")
        self.ax_c0.set_title(f"Channel 0\nSlice {z}")
        self.ax_c0.axis("off")

        self.ax_c1.imshow(c1, cmap="gray")
        self.ax_c1.set_title("Channel 1")
        self.ax_c1.axis("off")

        self.ax_c2.imshow(c2, cmap="gray")
        self.ax_c2.set_title("Channel 2")
        self.ax_c2.axis("off")

        # --- Ground Truth ---
        gt = self.gt_mask[0, z].cpu().numpy()
        self.ax_gt.imshow(gt, cmap="gray")
        self.ax_gt.set_title("Ground Truth")
        self.ax_gt.axis("off")

        # --- Prediction ---
        pred = self.pred_mask[0, z].cpu().numpy()
        self.ax_pred.imshow(pred, cmap="gray")

        # Compute Dice *NOW*
        dice = self._compute_dice(pred, gt)

        self.ax_pred.set_title(f"Prediction\nDice: {dice:.4f}")
        self.ax_pred.axis("off")

        self.fig.suptitle("Image Channels • Ground Truth • Prediction",
                        fontsize=16, fontweight="bold")


    def _on_slice_change(self, value):
        z = int(value)
        self._draw_slice(z)
        self.canvas.draw()


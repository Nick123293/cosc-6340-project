import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import os
import psutil 

from ConvLSTM import ConvLSTM3D
import tensorFormatTransformation as tFT

def get_dynamic_chunk_size(X, Y, Z, safety_factor=0.6):
    """
    Calculates how many time-steps fit in 60% of available RAM.
    """
    # 1 timestep = (1 batch * 1 channel * X * Y * Z) * 4 bytes (float32)
    bytes_per_step = 1 * 1 * X * Y * Z * 4 
    
    mem = psutil.virtual_memory()
    available = mem.available 
    
    safe_capacity = available * safety_factor
    max_steps = int(safe_capacity // bytes_per_step)
    
    return max(1, max_steps)

def train_scenario_2(csv_path, total_epochs=5, total_time_steps=100):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device} | Total Data Time Steps: {total_time_steps}")

    if not os.path.exists(csv_path):
        print("Error: CSV not found.")
        return
    
    # Load metadata (Sparse DF is small, so we keep it in RAM)
    df_sparse = pd.read_csv(csv_path)
    X, Y, Z = 100, 100, 100
    
    # Model Setup
    hidden_dim = 8
    convlstm = ConvLSTM3D(input_dim=1, hidden_dim=hidden_dim, kernel_size=3).to(device)
    decoder = nn.Conv3d(hidden_dim, 1, kernel_size=1).to(device) # Predicts input
    
    optimizer = optim.Adam(list(convlstm.parameters()) + list(decoder.parameters()), lr=1e-3)
    criterion = nn.MSELoss()

    for epoch in range(total_epochs):
        print(f"\n=== Epoch {epoch+1} ===")
        
        # Reset State at start of movie
        h_state, c_state = None, None 
        current_t = 0
        
        while current_t < total_time_steps:
            # 1. Check RAM and decide chunk size
            chunk_size = get_dynamic_chunk_size(X, Y, Z, safety_factor=0.6)
            t_end = min(current_t + chunk_size, total_time_steps)
            
            if current_t >= t_end: break

            print(f"   [RAM] Loading T={current_t} to {t_end} (Size: {t_end-current_t})")
            
            # 2. Load Chunk (Dense expansion)
            dense_chunk = tFT.load_dense_slice(df_sparse, current_t, t_end, X, Y, Z, device=device)
            
            # 3. Forward Pass (Passing State)
            if h_state is None:
                outputs, (h_next, c_next) = convlstm(dense_chunk, hidden_state=None)
            else:
                # Move state to GPU/Device before using
                h_in = h_state.to(device)
                c_in = c_state.to(device)
                outputs, (h_next, c_next) = convlstm(dense_chunk, hidden_state=(h_in, c_in))

            # 4. Loss (Self-Reconstruction)
            # Flatten B,T,Hidden,D,H,W -> B,T,1,D,H,W
            preds = []
            for t in range(outputs.shape[1]):
                preds.append(decoder(outputs[:, t]))
            preds = torch.stack(preds, dim=1)
            
            loss = criterion(preds, dense_chunk)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # 5. Detach State & Flush Memory
            h_state = h_next.detach().cpu()
            c_state = c_next.detach().cpu()
            
            print(f"   -> Loss: {loss.item():.4f}")

            del dense_chunk
            del outputs
            del preds
            del loss
            if device == "cuda":
                torch.cuda.empty_cache()
            
            current_t = t_end

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(script_dir, "data", "weather_sparse_coo_100x100x100_t10.csv")
    
    # NOTE: Set total_time_steps to match your create_test_data.py (e.g., 10 or 100)
    train_scenario_2(csv_path, total_epochs=2, total_time_steps=100)
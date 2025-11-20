import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import os
import json
import psycopg2
from psycopg2.extras import execute_values

from ConvLSTM import ConvLSTM3D
import tensorFormatTransformation as tFT

# --- 1. DATABASE CONNECTION SETUP ---
def init_db():
    """
    Connects to Docker Postgres and creates the hybrid table.
    """
    try:
        # Update port/password if different for your Docker setup
        conn = psycopg2.connect(
            dbname="postgres", user="postgres", password="password", 
            host="localhost", port="5433" 
        )
        cursor = conn.cursor()
        
        # Enable Vector Extension
        cursor.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        
        # Create Table:
        # embedding: The result (Dense 1D Vector) -> pgvector
        # notation: The logic (JSON Tree) -> JSONB
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS layer_computations (
                id SERIAL PRIMARY KEY,
                epoch INT,
                time_step INT,
                embedding vector(4),  -- Matches hidden_dim=4
                notation JSONB
            );
        """)
        conn.commit()
        print("[DB] Database connected and table ready.")
        return conn, cursor
    except Exception as e:
        print(f"[DB ERROR] Could not connect: {e}")
        return None, None

# --- 2. DECOMPOSITION (Compression) FUNCTION ---
def compress_3d_to_vector(tensor_5d):
    """
    Takes the massive 3D ConvLSTM hidden state and compresses it 
    into a search-friendly vector.
    
    Input:  (Batch, Time, Hidden_Dim, Depth, Height, Width)
    Output: (Batch, Time, Hidden_Dim)
    """
    # Global Average Pooling: Mean across Depth(3), Height(4), Width(5)
    return torch.mean(tensor_5d, dim=[3, 4, 5])

# --- 3. TRAINING FUNCTION ---
def train_on_csv_3d(
    csv_path,
    convlstm,
    decoder,
    num_epochs=10,
    future_steps=1,
    T=10,
    X=100,
    Y=100,
    Z=100,
    checkpoint_path=None,
):
    # Initialize DB
    conn, cursor = init_db()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    # Load dense tensor from sparse CSV
    dense = tFT.load_sparse_coo_csv_to_dense_3d(csv_path, T=T, X=X, Y=Y, Z=Z, device=device)

    B, T_total, C, D, H, W = dense.shape
    assert T_total >= 9 + future_steps, "Not enough Timestamps in data."
    seq_len_in = 9
    x_input = dense[:, :seq_len_in]           
    y_target_full = dense[:, seq_len_in:seq_len_in+future_steps]

    criterion = nn.MSELoss()
    optimizer = optim.Adam(list(convlstm.parameters()) + list(decoder.parameters()), lr=1e-3)

    for epoch in range(num_epochs):
        convlstm.train()
        decoder.train()
        optimizer.zero_grad()

        # --- FORWARD PASS ---
        # outputs: (Batch, Time, Hidden_Dim, D, H, W) -> The full 3D history
        outputs, (h, c) = convlstm(x_input) 

        # ==========================================================
        # ARCHITECTURE STEP: DECOMPOSE & STORE
        # ==========================================================
        if conn:
            # 1. Decompose (Compress 3D -> 1D Vector)
            # We do this on the whole batch at once for efficiency
            compressed_vectors = compress_3d_to_vector(outputs) # Shape: (B, T, Hidden_Dim)
            
            # 2. Prepare Batch Data for Postgres
            db_batch = []
            # Move to CPU for storage
            vec_data = compressed_vectors.detach().cpu().numpy()
            
            # Loop through time steps to create logs
            # (Assuming Batch Size = 1 for this demo)
            for t in range(seq_len_in):
                # The Vector Result
                current_vec = vec_data[0, t, :].tolist()
                
                # The Notation (JSONB)
                # We generate a structured log of what happened
                notation_obj = {
                    "layer": "ConvLSTM",
                    "operation": "GlobalAvgPooling",
                    "timestep": t,
                    "input_shape": [D, H, W],
                    "math": "mean(h_t, dim=[D,H,W])"
                }
                
                db_batch.append((epoch, t, current_vec, json.dumps(notation_obj)))
                
            # 3. Bulk Insert into Postgres
            # Stores Vector in pgvector column AND Notation in JSONB column
            insert_query = """
                INSERT INTO layer_computations (epoch, time_step, embedding, notation)
                VALUES %s
            """
            execute_values(cursor, insert_query, db_batch)
            conn.commit()
        # ==========================================================

        # Standard Prediction logic...
        preds = []
        h_t, c_t = h, c
        for t in range(future_steps):
            y_t = decoder(h_t)
            preds.append(y_t)
            h_t, c_t = convlstm.cell(y_t, (h_t, c_t))

        y_pred_seq = torch.stack(preds, dim=1)
        loss = criterion(y_pred_seq, y_target_full)
        loss.backward()
        optimizer.step()

        print(f"Epoch {epoch+1}/{num_epochs} - Loss: {loss.item():.6f}")

    # Close DB connection
    if conn:
        cursor.close()
        conn.close()
        print("[DB] Connection closed.")

    if checkpoint_path is not None:
        torch.save({
            "convlstm_state_dict": convlstm.state_dict(),
            "decoder_state_dict": decoder.state_dict(),
            "hidden_dim": convlstm.cell.hidden_dim,
            "input_channels": 1, 
        }, checkpoint_path)
        print(f"[TRAIN] Saved checkpoint to {checkpoint_path}")

def build_model(input_channels=1, hidden_dim=4, device="cpu"):
    convlstm = ConvLSTM3D(input_dim=input_channels, hidden_dim=hidden_dim, kernel_size=3).to(device)
    decoder = nn.Conv3d(hidden_dim, input_channels, kernel_size=1).to(device)
    return convlstm, decoder

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(script_dir, "data", "weather_sparse_coo_100x100x100_t10.csv")
    checkpoint_path = os.path.join(script_dir, "data", "weather_convlstm3d_checkpoint.pth")

    print(f"Looking for data at: {csv_path}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # IMPORTANT: hidden_dim=4 matches the vector(4) in the database definition
    convlstm, decoder = build_model(input_channels=1, hidden_dim=4, device=device)

    train_on_csv_3d(
        csv_path,
        convlstm,
        decoder,
        num_epochs=5,
        future_steps=1,
        T=10,
        X=100,
        Y=100,
        Z=100,
        checkpoint_path=checkpoint_path,
    )
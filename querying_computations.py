import psycopg2
import json
import numpy as np

def query_layer_computations():
    """
    Connects to the database and retrieves math notation and 
    computational embeddings stored during training.
    """
    try:
        # Connect using the same credentials from train.py
        conn = psycopg2.connect(
            dbname="cosc6340_project_db",
            user="nanderson",
            password="",
            host="localhost",
            port="5432",
        )
        cursor = conn.cursor()

        # Query to fetch everything, ordered by epoch and time_step
        print("--- Querying Layer Computations ---")
        cursor.execute("""
            SELECT epoch, time_step, embedding, notation 
            FROM layer_computations 
            ORDER BY epoch, time_step ASC
        """)
        
        rows = cursor.fetchall()
        
        if not rows:
            print("No records found in table 'layer_computations'.")
            return

        print(f"Found {len(rows)} records.\n")

        for row in rows:
            epoch, t_step, embedding, notation_json = row
            
            # The 'notation' column is stored as JSONB in Postgres, 
            # psycopg2 automatically converts it to a Python dict.
            # If it comes back as a string for some reason, we parse it.
            if isinstance(notation_json, str):
                notation = json.loads(notation_json)
            else:
                notation = notation_json

            print(f"=== Epoch {epoch} | Time Step {t_step} ===")
            
            # 1. Print the Math Notation found in the JSON
            if "math" in notation:
                print("Math Notation:")
                for key, formula in notation["math"].items():
                    print(f"  - {key}: {formula}")
            
            # 2. Print operation details
            op = notation.get("operation", "Unknown Op")
            layer = notation.get("layer", "Unknown Layer")
            print(f"Operation: {layer} -> {op}")

            # 3. Print the Vector Embedding (Computations)
            # 'embedding' comes from the pgvector extension
            print(f"Embedding (Vector): {embedding}")
            print("-" * 40)

        cursor.close()
        conn.close()

    except Exception as e:
        print(f"Error connecting to database: {e}")
        print("Ensure PostgreSQL is running and the 'layer_computations' table exists.")

if __name__ == "__main__":
    query_layer_computations()
import psycopg2
import json
import argparse

def get_db_connection():
    return psycopg2.connect(
        dbname="cosc6340_project_db",
        user="nanderson",
        password="",
        host="localhost",
        port="5432",
    )

# =========================================================================
# Query 1: Hyperparameter Impact Analysis
# Joins: training_runs <-> epoch_metrics
# Goal: Correlate RAM limits and time filters with model convergence.
# =========================================================================
def analyze_hyperparameter_impact(conn):
    print("\n=== QUERY 1: Hyperparameter Impact on Validation Loss ===")
    cursor = conn.cursor()
    
    # We JOIN training_runs with epoch_metrics to see which config produced the best result.
    query = """
    SELECT 
        tr.id AS run_id,
        tr.ram_limit_bytes,
        tr.time_start,
        tr.time_end,
        min(em.val_loss) as best_val_loss,
        tr.runtime_seconds
    FROM training_runs tr
    JOIN epoch_metrics em ON tr.id = em.run_id
    GROUP BY tr.id, tr.ram_limit_bytes, tr.time_start, tr.time_end, tr.runtime_seconds
    ORDER BY best_val_loss ASC
    LIMIT 5;
    """
    
    cursor.execute(query)
    rows = cursor.fetchall()
    
    print(f"{'Run ID':<8} | {'RAM Limit':<15} | {'Runtime (s)':<12} | {'Best Val Loss'}")
    print("-" * 60)
    for row in rows:
        run_id, ram, t_start, t_end, loss, runtime = row
        # Handle None runtime if run crashed
        rt_str = f"{runtime:.2f}" if runtime else "N/A"
        print(f"{run_id:<8} | {str(ram):<15} | {rt_str:<12} | {loss:.6f}")
    
    cursor.close()

# =========================================================================
# Query 2: Computation Traceability (Drill Down)
# Joins: layer_computations <-> training_runs
# Goal: Find the exact Math Notation used for a specific time_step in a run.
# =========================================================================
def trace_computations(conn, target_run_id=None):
    print("\n=== QUERY 2: Tracing Math Notation for Run ===")
    cursor = conn.cursor()
    
    # If no run specified, grab the most recent one
    if target_run_id is None:
        cursor.execute("SELECT id FROM training_runs ORDER BY started_at DESC LIMIT 1;")
        res = cursor.fetchone()
        if not res:
            print("No runs found.")
            return
        target_run_id = res[0]

    print(f"Tracing Computations for Run ID: {target_run_id}")

    # JOIN partitioned table (layer_computations) with metadata
    # Note: Postgres handles the partitioning logic automatically.
    query = """
    SELECT 
        lc.epoch,
        lc.time_step,
        lc.notation ->> 'math' as math_formula,
        lc.notation ->> 'layer' as layer_name,
        tr.model_config
    FROM layer_computations lc
    JOIN training_runs tr ON lc.run_id = tr.id
    WHERE lc.run_id = %s
    ORDER BY lc.epoch, lc.time_step
    LIMIT 5;
    """
    
    cursor.execute(query, (target_run_id,))
    rows = cursor.fetchall()
    
    for row in rows:
        epoch, t_step, math, layer, config = row
        print(f"[Epoch {epoch} t={t_step}] Layer: {layer}")
        print(f"   Math: {math}")
        # print(f"   Config context: {config}") 
        
    cursor.close()

# =========================================================================
# Query 3: Vector Similarity Search (PgVector)
# Single Table Self-Analysis (Conceptually joining vector space)
# Goal: Find time-steps that are computationally similar to t=0
# =========================================================================
def find_similar_computations(conn, target_run_id=None):
    print("\n=== QUERY 3: Vector Similarity Search (Nearest Neighbors) ===")
    cursor = conn.cursor()
    
    if target_run_id is None:
        cursor.execute("SELECT id FROM training_runs ORDER BY started_at DESC LIMIT 1;")
        target_run_id = cursor.fetchone()[0]

    # 1. Get the embedding for time_step 0 of the first epoch
    cursor.execute("""
        SELECT embedding 
        FROM layer_computations 
        WHERE run_id = %s AND time_step = 0 AND epoch = 0
        LIMIT 1;
    """, (target_run_id,))
    
    result = cursor.fetchone()
    if not result:
        print("No embedding found for t=0 to use as a query vector.")
        return
        
    query_vector = result[0] # The vector string or object

    # 2. Find the top 5 most similar embeddings from ANY run or time step
    # Using L2 distance operator (<->) provided by pgvector
    query = """
    SELECT 
        lc.run_id,
        lc.time_step,
        lc.epoch,
        (lc.embedding <-> %s) AS distance,
        lc.notation ->> 'op' as operation
    FROM layer_computations lc
    WHERE lc.embedding IS NOT NULL
    ORDER BY distance ASC
    LIMIT 5;
    """
    
    cursor.execute(query, (query_vector,))
    rows = cursor.fetchall()
    
    print(f"Querying nearest neighbors for Run {target_run_id} (t=0)...")
    print(f"{'Run ID':<8} | {'Time':<5} | {'Dist':<10} | {'Operation'}")
    print("-" * 50)
    for row in rows:
        r_id, t_step, ep, dist, op = row
        print(f"{r_id:<8} | {t_step:<5} | {dist:.4f}     | {op}")

    cursor.close()

if __name__ == "__main__":
    try:
        conn = get_db_connection()
        analyze_hyperparameter_impact(conn)
        trace_computations(conn)
        find_similar_computations(conn)
        conn.close()
    except Exception as e:
        print(f"Error: {e}")
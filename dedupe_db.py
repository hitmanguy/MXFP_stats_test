import sqlite3
from pathlib import Path

def deduplicate_db():
    db_path = Path("results/eval_ledger.db")
    if not db_path.exists():
        print("Database not found.")
        return

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    # Count rows before
    cur.execute("SELECT COUNT(*) FROM evaluations")
    count_before = cur.fetchone()[0]

    # Delete duplicates keeping only the latest timestamp
    delete_query = """
    DELETE FROM evaluations
    WHERE run_id NOT IN (
        SELECT run_id FROM (
            SELECT run_id, 
                   ROW_NUMBER() OVER (
                       PARTITION BY model_family, modality, dataset, seed, quant_mode, metric_name 
                       ORDER BY timestamp DESC
                   ) as rn
            FROM evaluations
        ) WHERE rn = 1
    );
    """
    
    cur.execute(delete_query)
    deleted_rows = cur.rowcount
    conn.commit()

    # Count rows after
    cur.execute("SELECT COUNT(*) FROM evaluations")
    count_after = cur.fetchone()[0]

    conn.close()

    print(f"Deduplication complete.")
    print(f"Rows before: {count_before}")
    print(f"Rows deleted: {deleted_rows}")
    print(f"Rows after: {count_after}")

if __name__ == "__main__":
    deduplicate_db()

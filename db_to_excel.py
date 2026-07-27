import sqlite3
import pandas as pd
import argparse
from pathlib import Path

def convert_db_to_excel(db_path: str, output_path: str):
    """
    Connects to the SQLite database, reads the 'evaluations' table,
    and exports the data to an Excel (.xlsx) file.
    """
    if not Path(db_path).exists():
        print(f"Error: Database file not found at {db_path}")
        return

    print(f"Connecting to {db_path}...")
    try:
        conn = sqlite3.connect(db_path)
        
        # Read the evaluations table into a pandas DataFrame
        # (If your table is named differently, change 'evaluations' below)
        query = "SELECT * FROM evaluations;"
        df = pd.read_sql_query(query, conn)
        
        # Write the DataFrame to an Excel file (Office Open XML format .xlsx)
        print(f"Found {len(df)} rows. Exporting to {output_path}...")
        df.to_excel(output_path, index=False, engine='openpyxl')
        
        print(f"Success! Data exported to {output_path}")
        
    except sqlite3.OperationalError as e:
        print(f"SQLite Error: {e}")
        print("Tip: Ensure the table name is correct (e.g., 'evaluations' or 'eval_metrics')")
    except ImportError:
        print("Error: The 'openpyxl' library is required to write .xlsx files.")
        print("Please install it using: pip install openpyxl")
    finally:
        if 'conn' in locals():
            conn.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert eval_ledger.db to an Excel (.xlsx) file")
    parser.add_argument(
        "--db", 
        type=str, 
        default="results/eval_ledger.db", 
        help="Path to the input SQLite database (default: results/eval_ledger.db)"
    )
    parser.add_argument(
        "--out", 
        type=str, 
        default="eval_results.xlsx", 
        help="Path to the output Excel file (default: eval_results.xlsx)"
    )
    
    args = parser.parse_args()
    convert_db_to_excel(args.db, args.out)

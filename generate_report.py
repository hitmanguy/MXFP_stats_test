import sqlite3
import pandas as pd
from pathlib import Path

def main():
    db_path = Path("results/eval_ledger.db")
    if not db_path.exists():
        print("No database found at results/eval_ledger.db")
        return

    conn = sqlite3.connect(db_path)
    
    print("\n" + "="*80)
    print("  DATABASE EXTRACT: VISION MODALITY (ResNet-18 ImageNet-1k)")
    print("="*80)
    
    # Query Vision Results
    query = """
    SELECT 
        quant_mode,
        MAX(CASE WHEN metric_name = 'acc1' THEN ROUND(metric_value, 2) END) AS `Acc@1 (%)`,
        MAX(CASE WHEN metric_name = 'acc5' THEN ROUND(metric_value, 2) END) AS `Acc@5 (%)`,
        ROUND(MAX(eff_bits), 2) AS `Eff. Bits`,
        timestamp
    FROM evaluations
    WHERE modality = 'vision' AND model_family = 'resnet18'
    GROUP BY quant_mode
    ORDER BY `Eff. Bits` DESC, `Acc@1 (%)` DESC
    """
    
    df = pd.read_sql_query(query, conn)
    print(df.to_string(index=False))
    
    print("\n" + "="*80)
    print("  DATABASE EXTRACT: LANGUAGE MODALITY (GPT-2 WikiText-2)")
    print("="*80)
    
    # Query Language Results
    query_lang = """
    SELECT 
        quant_mode,
        ROUND(metric_value, 2) AS `Perplexity`,
        ROUND(eff_bits, 2) AS `Eff. Bits`,
        timestamp
    FROM evaluations
    WHERE modality = 'language' AND model_family = 'gpt2' AND metric_name = 'ppl'
    ORDER BY `Eff. Bits` DESC, `Perplexity` ASC
    """
    
    df_lang = pd.read_sql_query(query_lang, conn)
    print(df_lang.to_string(index=False))
    
    conn.close()

if __name__ == "__main__":
    main()

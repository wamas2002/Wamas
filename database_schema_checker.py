"""
Database Schema Checker - Identify column structures for signal execution
"""
import sqlite3
import os

def check_database_schema(db_name):
    """Check database schema and table structures"""
    if not os.path.exists(db_name):
        return None
    
    try:
        conn = sqlite3.connect(db_name)
        cursor = conn.cursor()
        
        # Get all tables
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [row[0] for row in cursor.fetchall()]
        
        schema_info = {}
        
        for table in tables:
            cursor.execute(f"PRAGMA table_info({table})")
            columns = cursor.fetchall()
            
            cursor.execute(f"SELECT COUNT(*) FROM {table}")
            row_count = cursor.fetchone()[0]
            
            # Get sample data if available
            sample_data = None
            if row_count > 0:
                cursor.execute(f"SELECT * FROM {table} LIMIT 1")
                sample_data = cursor.fetchone()
            
            schema_info[table] = {
                'columns': [{'name': col[1], 'type': col[2]} for col in columns],
                'row_count': row_count,
                'sample_data': sample_data
            }
        
        conn.close()
        return schema_info
        
    except Exception as e:
        print(f"Error checking {db_name}: {e}")
        return None

def main():
    """Check all trading databases for signal structures"""
    databases = [
        'pure_local_trading.db',
        'enhanced_trading.db', 
        'autonomous_trading.db',
        'trading_platform.db'
    ]
    
    print("DATABASE SCHEMA ANALYSIS")
    print("=" * 50)
    
    for db_name in databases:
        print(f"\n📁 {db_name}:")
        schema = check_database_schema(db_name)
        
        if schema:
            for table_name, info in schema.items():
                if 'signal' in table_name.lower() or info['row_count'] > 0:
                    print(f"  📊 {table_name}: {info['row_count']} rows")
                    print(f"     Columns: {[col['name'] for col in info['columns']]}")
                    if info['sample_data']:
                        print(f"     Sample: {info['sample_data'][:5]}...")
        else:
            print("  ❌ Database not found or inaccessible")

if __name__ == "__main__":
    main()
import sqlite3
import os

def update_database_schema():
    """Update database schema to add missing columns"""
    try:
        # Connect to database
        conn = sqlite3.connect('database.db')
        cursor = conn.cursor()
        
        # Check if columns exist
        cursor.execute("PRAGMA table_info(attendance)")
        columns = [column[1] for column in cursor.fetchall()]
        
        # Add missing columns if they don't exist
        missing_columns = []
        
        if 'latitude_out' not in columns:
            cursor.execute('ALTER TABLE attendance ADD COLUMN latitude_out REAL')
            missing_columns.append('latitude_out')
            
        if 'longitude_out' not in columns:
            cursor.execute('ALTER TABLE attendance ADD COLUMN longitude_out REAL')
            missing_columns.append('longitude_out')
            
        if 'photo_path_out' not in columns:
            cursor.execute('ALTER TABLE attendance ADD COLUMN photo_path_out TEXT')
            missing_columns.append('photo_path_out')
        
        # Commit changes
        conn.commit()
        
        if missing_columns:
            print(f"✅ Successfully added columns: {', '.join(missing_columns)}")
        else:
            print("✅ All required columns already exist")
            
        # Verify the update
        cursor.execute("PRAGMA table_info(attendance)")
        all_columns = [column[1] for column in cursor.fetchall()]
        print(f"📋 Current attendance table columns: {', '.join(all_columns)}")
        
        conn.close()
        
    except Exception as e:
        print(f"❌ Error updating database: {str(e)}")
        if 'conn' in locals():
            conn.close()

if __name__ == '__main__':
    print("🔧 Updating database schema...")
    update_database_schema()
    print("✨ Database update completed!")
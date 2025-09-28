import os
from decouple import config
import psycopg2
from urllib.parse import urlparse

class Config:
    # Get database URL from environment
    DATABASE_URL = config('DATABASE_URL', default='sqlite:///database.db')
    
    # Parse PostgreSQL URL
    @staticmethod
    def get_db_config():
        db_url = Config.DATABASE_URL
        
        if db_url.startswith('postgresql://') or db_url.startswith('postgres://'):
            # Parse PostgreSQL URL
            if db_url.startswith('postgres://'):
                db_url = db_url.replace('postgres://', 'postgresql://', 1)
            
            parsed = urlparse(db_url)
            
            return {
                'host': parsed.hostname,
                'port': parsed.port or 5432,
                'database': parsed.path[1:],  # Remove leading slash
                'user': parsed.username,
                'password': parsed.password,
                'sslmode': 'require'  # Railway requires SSL
            }
        else:
            # Return None for SQLite (development)
            return None

    @staticmethod
    def get_connection():
        """Get database connection - PostgreSQL or SQLite"""
        db_config = Config.get_db_config()
        
        if db_config:
            # PostgreSQL connection
            import psycopg2
            from psycopg2.extras import RealDictCursor
            
            conn = psycopg2.connect(
                host=db_config['host'],
                port=db_config['port'],
                database=db_config['database'],
                user=db_config['user'],
                password=db_config['password'],
                sslmode=db_config['sslmode'],
                cursor_factory=RealDictCursor
            )
            return conn
        else:
            # SQLite connection (development)
            import sqlite3
            conn = sqlite3.connect('database.db')
            conn.row_factory = sqlite3.Row
            return conn

import sqlite3

conn = sqlite3.connect('database.db')
cursor = conn.cursor()

try:
    cursor.execute('ALTER TABLE attendance ADD COLUMN latitude_out REAL')
    print("Kolom latitude_out berhasil ditambahkan")
except:
    print("Kolom latitude_out sudah ada")

try:
    cursor.execute('ALTER TABLE attendance ADD COLUMN longitude_out REAL')
    print("Kolom longitude_out berhasil ditambahkan")
except:
    print("Kolom longitude_out sudah ada")

conn.commit()
conn.close()
print("Database berhasil diupdate!")
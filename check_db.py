#!/usr/bin/env python3
import sqlite3
import os

db_path = r'C:\Users\Ommi\Desktop\AGI\memory\user_memory.db'
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

print('=== DATABASE TABLES ===')
cursor.execute('SELECT name FROM sqlite_master WHERE type="table"')
tables = cursor.fetchall()
for table in tables:
    print(f'Table: {table[0]}')

print('\n=== EMBEDDINGS TABLE STRUCTURE ===')
cursor.execute('PRAGMA table_info(embeddings)')
columns = cursor.fetchall()
for col in columns:
    print(f'Column: {col[1]} ({col[2]})')

print('\n=== EMBEDDINGS COUNT ===')
cursor.execute('SELECT COUNT(*) FROM embeddings')
count = cursor.fetchone()[0]
print(f'Total embeddings: {count}')

if count > 0:
    print('\n=== SAMPLE EMBEDDINGS ===')
    cursor.execute('SELECT id, user_id, post_id, model, dimensions, LENGTH(text) as text_len, LENGTH(embedding_json) as embed_size, created_at FROM embeddings LIMIT 5')
    samples = cursor.fetchall()
    for row in samples:
        print(f'ID: {row[0]}, User: {row[1]}, Post: {row[2]}, Model: {row[3]}, Dims: {row[4]}, Text: {row[5]} chars, Embed: {row[6]} bytes, Created: {row[7]}')

    print('\n=== EMBEDDINGS FOR test_user_123 ===')
    cursor.execute('SELECT COUNT(*) FROM embeddings WHERE user_id = "test_user_123"')
    test_count = cursor.fetchone()[0]
    print(f'Test user embeddings: {test_count}')

conn.close()

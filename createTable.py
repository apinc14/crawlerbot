import os
import pymysql

host = 'localhost'  # Use the host machine's IP address or hostname
port = 3306  # Replace with the host machine's port (if forwarded)
user = 'root'
password = 'snook1sm00sh0Osmoozh'
db_name = 'weblyticsDB'

connection = pymysql.connect(
    host=host,
    port=port,
    user=user,
    password=password,
    db=db_name,
    charset='utf8mb4',
    cursorclass=pymysql.cursors.DictCursor
)

try:
    # Check if the 'newsarticle' table exists
    with connection.cursor() as cursor:
        table_name = 'newsarticle'
        cursor.execute(f"SHOW TABLES LIKE '{table_name}'")
        table_exists = cursor.fetchone()
    # If the table doesn't exist, create it
    if not table_exists:
        with connection.cursor() as cursor:
            
            create_table_query = """
          REATE TABLE posts (
                id INT PRIMARY KEY,
                post_title VARCHAR(255),
                post_content TEXT,
                post_score INT,
                num_comments INT,
                creation_time DATETIME,
                subreddit_name VARCHAR(255),
                sentiment json,
                entitiesArray TEXT,
                keywordsArray TEXT,
                upVRatio FLOAT,
                ups INT,
                media VARCHAR(255)
            );

          CREATE TABLE totals (
            id Int Auto_Increment PRIMARY KEY,
            entity VARCHAR(255),
            score INT
        );
            """
            
            cursor.execute(create_table_query)

        connection.commit()
finally:
    connection.close()        
# Imports:
import sqlite3
import redis
import json
import hashlib
from fastapi import FastAPI,Request
from pydantic import BaseModel
from typing import Optional
import time
import logging

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] (line %(lineno)d) - %(levelname)s - %(message)s",
    datefmt="%m-%d-%Y %H:%M:%S"
)

logger = logging.getLogger('db_caching')

GREEN = "\033[92m"
RED = "\033[91m"
RESET = "\033[0m"

def log_green(msg: str):
    logger.info(f"{GREEN}{msg}{RESET}")


def log_red(msg: str):
    logger.error(f"{RED}{msg}{RESET}")


app = FastAPI()
redis_client = redis.Redis(host='localhost', port=6379, db=0)


# Establishing database connection:
def get_db_connection():
    # cretate a connection object
    conn = sqlite3.connect("test.sqlite3")
    # rows of the database
    conn.row_factory = sqlite3.Row
    return conn

# setup the database:
def init_db():
    # get the connection object
    conn = get_db_connection()

    # create a cursor , for doing operations on db via python
    cursor = conn.cursor()

    # Cursor object for creating a table inside db
    cursor.execute(
        """
            CREATE TABLE IF NOT EXISTS users(
                id INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                age INTEGER
            )
        """
    )
    # cursor command for inserting data into table
    cursor.execute("INSERT INTO users (id, name, age) VALUES (1, 'Michael', 45)")
    cursor.execute("INSERT INTO users (id, name, age) VALUES (2, 'Jim', 35)")
    cursor.execute("INSERT INTO users (id, name, age) VALUES (3, 'Pam', 27)")

    # commit the changes:
    conn.commit()
    # close the database connection
    conn.close()

init_db()

class userQuery(BaseModel):
    user_id: int
    
# Create a unique cache key for storing the data of every unique user in cache
def make_cache_key(user_id:int):
    raw = f"user:{user_id}"
    return hashlib.sha256(raw.encode()).hexdigest()


# Fastapi router for getting the employee data:
@app.post("/get-user")

# function which will be called on hitting the following route:
def get_user(query: userQuery):
    start_time = time.time()

    # create a cache key based on the input user_id:
    cache_key = make_cache_key(user_id = query.user_id) 
    # get the cache data for the following cache_key: 
    cached_data = redis_client.get(cache_key)

    # if data is already present in cache . i.e cached_data = True
    if cached_data:
        time_taken = time.time() - start_time
        log_green(
            f"[CACHE HIT] Returned user_id={query.user_id} "
            f"in {time_taken:.4f} sec"
        )
        return json.loads(cached_data)

    logger.info("Data not found in cache , finding in DB ...")
    # if data is not cached , then get the data from the db:
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM users WHERE id = ?",(query.user_id,)) 
    row = cursor.fetchone()
    conn.close()

    # if user db me bhi nahi present hai to return kardo
    if row is None:
        log_red(
            f"[NOT FOUND] user_id={query.user_id} "
        )
        return {'message': 'User not found.'}

    # agar Data mila to usko cache kardo:
    # key-value pair ko map kardo(python Data Structure - dictionary)
    result = {'id': row['id'], 'name': row['name'], 'age': row['age']}

    # setex = atomically set a key's - string value and a timeout (expiration time) in seconds
    # json.dumps() = taking a Python data structure, such as a dictionary or a list, and turning it into a single string

    formatted_json_result = json.dumps(result)

    print("Formatted json string result : ",formatted_json_result)

    redis_client.setex(cache_key ,3600 , json.dumps(result,indent=4,sort_keys=True))

    time_taken = time.time() - start_time
    log_green(
        f"[DB → CACHE] Retrieved and cached user_id={query.user_id} "
        f"in {time_taken:.4f} sec"
    )

    return result

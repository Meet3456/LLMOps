# pulled redis image from docker - "docker run -d -p 6379:6379 redis"
# installed python client for redis : redis[async]
import redis
import json
import hashlib
from fastapi import FastAPI
from pydantic import BaseModel
import sqlite3

'''
# Connects python code to redis
r = redis.Redis(host="localhost" , port=6379 , db=0)

try:
    # Tests the connection
    if r.ping():
        print("Connected too Redis!")
except redis.ConnectionError:
    print("Redis connection failed")

# set unique key-value pair in redis cache
r.set("framework","FastAPI")

# fetch the value of corresponding key from cache
value = r.get("framework")

print(f"Stored value for frameweok: {value.decode()}")
'''

app = FastAPI() 
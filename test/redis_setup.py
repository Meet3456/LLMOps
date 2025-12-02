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

'''
Why is Caching Important?
    • Reduces Latency: Cached responses are served from nearby or in-memory storage, which is significantly faster than from a
      database or an external API call

    • Improves Performance: Applications become more responsive since frequently requested data is readily available

    • Reduces Load on Backend: By reducing repeated data fetches or computations, the pressure on databases, ML models, or third-
      party APIs is minimized

    • Scalability: Helps applications scale better under high load, as the same data doesn't need to be processed repeatedly
'''

'''
cache lvl1 -> cache lvl2 -> RAM Memory -> Mass Storage(hard disk = permanent storage)
reduces latency(response time)
reduces load on backend and increases scalability
use case : api responses - slow or rate limited third party api calls are cached to avoid repeated calls , session data
types of caching : client side : Done in broswer or fronted usiong mechanisms like HTTP Headers(Cache control)
                   server side : using redis , memcached or in memory dictionaries
Redis - in memory key-value store , Memcached , Fast api decorator - lru_cache

Redis : Remote Dictionary Server in memory data structure store ,stores everything in memory which allows fast read and write ops
Key-value db : stores value similar to python dict , Cache : used to cache db queries,api responses, and ML model responses
No dis i/o during read/writes , single threaded
Data persistence through : RDB(Redis db backup)
AOF(Append only File) : logs every write operation

Redis Data Structure:
string , list ,  set, sorted set , Hash , stream
Pub/Sub - Publishing subsribing messaging system , Bitmaps
Caching , Session Management , Rate limiting , Real-time analytics , leaderboard and ranking systems

'''
from fastapi import FastAPI, Depends, Header, HTTPException

# It is a great utility that reads environment variables and casts them to correct type:
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    api_key = str

    class Config:
        env_file = ".env"

settings = Settings()

app = FastAPI()
 
API_KEY = "my-secret-key"


def get_api_key_env_file(api_key: str = Header(...)):
    if api_key != settings.api_key:
        raise HTTPException(status_code=403, detail="unauthorized")
    else:
        return api_key


# (...) - This indicated the field is required
def get_api_key(api_key:str = Header(...)):
    if api_key != API_KEY:
        raise HTTPException(status_code=403, detail="unauthorized")
    else:
        return api_key
    
@app.get('/get-data')

def get_data(api_key: str = Depends(get_api_key_env_file)):
    # Creating a dependency between the route handler and the get_api_key Function
    # if api_key is valid the function will return successfull response
    return {"output":"Access Granted"}

# cache lvl1 -> cache lvl2 -> RAM Memory -> Mass Storage(hard disk = permanent storage)
# reduces latency(response time)
# reduces load on backend and increases scalability
# use case : api responses - slow or rate limited third party api calls are cached to avoid repeated calls , session data
# types of caching : client side : Done in broswer or fronted usiong mechanisms like HTTP Headers(Cache control)
#                    server side : using redis , memcached or in memory dictionaries
# Redis - in memory key-value store , Memcached , Fast api decorator - lru_cache

# Redis : Remote Dictionary Server in memory data structure store ,stores everything in memory which allows fast read and write ops
# Key-value db : stores value similar to python dict , Cache : used to cache db queries,api responses, and ML model responses
# No dis i/o during read/writes , single threaded
# Data persistence through : RDB(Redis db backup)
# AOF(Append only File) : logs every write operation

# Redis Data Structure:
# string , list ,  set, sorted set , Hash , stream
# Pub/Sub - Publishing subsribing messaging system , Bitmaps
# Caching , Session Management , Rate limiting , Real-time analytics , leaderboard and ranking systems


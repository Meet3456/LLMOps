'''
Helps to identify performance Bottlenecks in Business logic and API Calls
Minimize Latency and maximize Throughput
Reduce CPU , memory usage and I/O Wait times
Make informed architecture decisions (eg - sync and async)
Prepare Scalable API's
'''

# Profiling using cProfile

'''
For each request it makes a profile(you can say file) - which contains various details abouy no of calls,time
Cannot directly open the file - we use

• Install snakeviz for interactive visualization:
    snakeviz is a browser-based visualization tool for Python's cProfile output

    It provides interactive graphs to see how much time each function takes and how functions
    call each other

Each time a xcall is made to the request a Profile is created
Request - action - profile creation - we use middleware
'''

# Example:
'''
import os
import time
import cProfile
import datetime
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

PROFILES_DIR = "profiles"

os.makedirs(PROFILES_DIR , exist_ok=True)

# Write a middleware for each request that we hit:
app = FastAPI()

@app.middleware('http')
async def create_profile(request: Request,call_next):
    # create profile name on the basis of Endpoint name(as each req is made to a specific endpoint) + the current timestamp
    time_stamp = datetime.datetime.now().strftime('%m_%d_%Y_%H_%M_%S_%f')

    path = request.url.path.strip('/').replace("/","_") or "root"

    profile_path_name = os.path.join(PROFILES_DIR, f'{path}_{time_stamp}.prof')

    # create a profiler object
    profiler = cProfile.Profile()
    # enable the profiler:
    profiler.enable()

    response = await call_next(request)

    profiler.disable()
    profiler.dump_stats(profile_path_name)

    print(f'Profile saved: {profile_path_name}')
    return response

@app.get('/')
def home():
    return {'message': 'cProfile demo'}

@app.get('/compute')
async def compute():
    time.sleep(1)
    result = sum((i*2) for i in range(1000))
    return JSONResponse({"Result":result})

'''

# Profiling using Line-Profiler
'''


'''
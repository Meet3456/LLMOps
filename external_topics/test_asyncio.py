import asyncio
import time

"""
- Python library for writing concurrent code with async/await syntax
- Event loop is basically the engine that runs and manages asynchronous Functions(tasks)
- We can think of it as a scheduler , if some task is taking too long to complete or is suspended , it can switch to another task and come back to the previous one when it's ready
- This allows us to write code that can handle multiple tasks at the same time without blocking the execution of other tasks

- Three main types of awaitable objects in asyncio:

    - Couroutines : coroutines are functions defined with async def syntax. 
                    They can use the await keyword to pause their execution until the awaited task is complete.

                    couroutine function - function defined with async def syntax
                    couroutine object - awaitable object returned when a couroutine function is called

                    couroutine_obj = async_function("test")
                    couroutine_result = await couroutine_obj

    - Tasks : When we create a couroutine function and call it , it just returns a coroutine object
              When we code 
                task1 = asyncio.create_task(<coroutine_function>)
                task2 = asyncio.create_task(<coroutine_function>)
    
              The tasks schedules a coroutine to run on the event loop simultaneously
                res1 = await task1(suspends the main coroutine(eg main function) and control goes to task1)
                res2 = await task2

              Internally uses FIFO(Goes with which tasks are ready in scheduler)
              Create task -> Schedule them in the event loop -> Await for a task to complete -> Concurrently other tasks are running in the background -> Until a tasks complete, the pointer goes back to the main event loop(main fun for eg)
    
    
    - Futures : 

"""


async def async_function(name: str) -> str:
    current_time = time.strftime("%X")
    print(f"[{current_time}] Starting async_function for {name}...")
    print("This is an asynchronous couroutine function.")

    await asyncio.sleep(2)  # Simulating an asynchronous operation
    current_time = time.strftime("%X")
    print(f"[{current_time}] Finished async_function for {name}.")
    return f"Hello, {name}!"


async def main():
    couroutine_obj = async_function("test")
    print(couroutine_obj)

    couroutine_result = await couroutine_obj
    print(couroutine_result)


if __name__ == "__main__":
    asyncio.run(main())

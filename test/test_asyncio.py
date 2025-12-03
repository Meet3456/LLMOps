import asyncio
import time

"""
- Python library for writing concurrent code with async/await syntax
- Event loop is basically the engine that runs and manages asynchronous Functions(tasks)
- We can think of it as a scheduler , if some task is taking too long to complete or is suspended , it can switch to another task and come back to the previous one when it's ready
- This allows us to write code that can handle multiple tasks at the same time without blocking the execution of other tasks

- Three main types of awaitable objects in asyncio:

    - Couroutines :  coroutines are functions defined with async def syntax. 
                     They can use the await keyword to pause their execution until the awaited task is complete.

                     couroutine function - function defined with async def syntax
                     couroutine object - awaitable object returned when a couroutine function is called

                     couroutine_obj = async_function("test")
                     couroutine_result = await couroutine_obj
    - Tasks : 
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

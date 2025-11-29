from fastapi import FastAPI,Request
from fastapi.responses import JSONResponse
import logging

app = FastAPI() 

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] (line %(lineno)d) - %(levelname)s - %(message)s",
    datefmt="%m-%d-%Y %H:%M:%S"
)

@app.exception_handler(Exception)
async def general_exception_handler(request: Request , exc: Exception):
    logging.error(f"Error occurred: {str(exc)}")
    return JSONResponse(
        status_code= 500,
        content={"error": f"An error occurred: {str(exc)}"},
    )


@app.get('/exceptions')
def handle_exceptions():
    logging.info("Handling exceptions")
    return 1 / 0

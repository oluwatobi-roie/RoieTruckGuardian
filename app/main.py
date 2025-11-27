from fastapi import FastAPI
from webhook import router
import uvicorn, asyncio

app = FastAPI(title="Truck-Guardian")
app.include_router(router)

if __name__=="__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
    
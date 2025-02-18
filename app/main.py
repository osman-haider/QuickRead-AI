from fastapi import FastAPI
from app.api.v1.endpoints import summarize, qa

app = FastAPI()

# Include routers
app.include_router(summarize.router, prefix="/api/v1")
app.include_router(qa.router, prefix="/api/v1")
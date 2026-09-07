import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routes import logs, ai_analysis, experiment

app = FastAPI(title="AI IoT Honeypot Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(logs.router, prefix="/api/logs", tags=["Logs"])
app.include_router(ai_analysis.router, prefix="/api/ai", tags=["AI Analysis"])
app.include_router(experiment.router, prefix="/api/experiment", tags=["Experiments"])

@app.get("/")
def read_root():
    return {"message": "AI IoT Honeypot API is running"}

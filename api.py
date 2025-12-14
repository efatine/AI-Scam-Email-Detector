from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from contextlib import asynccontextmanager
import joblib
from sentence_transformers import SentenceTransformer
from fastapi.concurrency import run_in_threadpool # Import this

from scam_detector import predict_email, MODEL_ARTIFACTS_PATH, SBERT_MODEL_NAME

models = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Loading models...")
    try:
        artifacts = joblib.load(MODEL_ARTIFACTS_PATH)
        models.update(artifacts)
        models['sbert_model_obj'] = SentenceTransformer(SBERT_MODEL_NAME)
        print("Models loaded successfully!")
    except Exception as e:
        print(f"❌ Error loading models: {e}")
    
    yield
    models.clear()

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class EmailRequest(BaseModel):
    text: str

@app.post("/predict")
async def predict(request: EmailRequest):
    if not models:
        return {"error": "Models are not loaded yet."}
        
    # Run the heavy AI prediction in a separate thread so it doesn't block
    result = await run_in_threadpool(predict_email, request.text, artifacts=models)
    return result
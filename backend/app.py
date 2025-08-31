import io
import os
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from PIL import Image
import uvicorn

from model import InferenceModel, CLASS_NAMES

# Paths
HERE = os.path.dirname(os.path.abspath(__file__))
WEIGHTS_PATH = os.path.join(HERE, "best_tomato_model.pth")
FRONTEND_DIR = os.path.join(os.path.dirname(HERE), "frontend")

if not os.path.exists(WEIGHTS_PATH):
    raise RuntimeError(
        f"Model weights not found at {WEIGHTS_PATH}. "
        f"Copy your 'best_tomato_model.pth' into backend/ first."
    )

app = FastAPI(title="Tomato Disease Classification API", version="1.0.0")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve frontend from /frontend instead of /
app.mount("/frontend", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")

# Load model at startup
inference_model: InferenceModel = None


@app.on_event("startup")
def load_model():
    global inference_model
    inference_model = InferenceModel(weights_path=WEIGHTS_PATH)
    print("✅ Model loaded and ready.")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/classes")
def classes():
    return {"classes": CLASS_NAMES}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if file.content_type is None or "image" not in file.content_type:
        raise HTTPException(status_code=415, detail="Please upload an image file.")

    data = await file.read()
    try:
        image = Image.open(io.BytesIO(data))
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image data.")

    try:
        result = inference_model.predict_image(image, topk=5)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")

    return JSONResponse(result)


@app.get("/api", response_class=HTMLResponse)
def api_root():
    return """
    <h2>Tomato Disease Classification API</h2>
    <ul>
      <li>GET <code>/health</code></li>
      <li>GET <code>/classes</code></li>
      <li>POST <code>/predict</code> (multipart/form-data, field: <b>file</b>)</li>
    </ul>
    """


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)

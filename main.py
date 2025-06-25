import os
import shutil
from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse

from video_utils import extract_frames
from vector_utils import create_collection, compute_feature_vector, store_vector, search_similar

app = FastAPI(title="Video Frame Search API")

# Ensure frames directory exists and expose it
os.makedirs("extracted_frames", exist_ok=True)
app.mount("/frames", StaticFiles(directory="extracted_frames"), name="frames")

# (Re)create collection at startup
create_collection()

@app.post("/upload/", summary="Upload MP4 and index its frames")
async def upload_video(
    file: UploadFile = File(...),
    interval_seconds: int = Query(1, description="Extract one frame every N seconds", ge=1, le=10)
):
    if not file.filename.lower().endswith(".mp4"):
        raise HTTPException(status_code=400, detail="Only .mp4 files allowed")
    
    #  Clear extracted_frames folder
    import shutil
    shutil.rmtree("extracted_frames", ignore_errors=True)
    os.makedirs("extracted_frames", exist_ok=True)

    temp_path = "temp_video.mp4"
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        total = extract_frames(temp_path, interval_seconds=interval_seconds)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        os.remove(temp_path)

    for i in range(total):
        img_path = f"extracted_frames/frame_{i}.jpg"
        vec = compute_feature_vector(img_path)
        store_vector(i, vec)

    return JSONResponse({
        "message": f"Extracted & indexed {total} frames every {interval_seconds} seconds.",
        "frames_indexed": total,
    })

@app.post("/search/", summary="Upload an image to find similar frames")
async def search_frames(file: UploadFile = File(...), top_k: int = 5):
    fname = file.filename.lower()
    if not (fname.endswith(".jpg") or fname.endswith(".jpeg") or fname.endswith(".png")):
        raise HTTPException(status_code=400, detail="Only JPEG/PNG images allowed")

    query_path = "query.jpg"
    with open(query_path, "wb") as buf:
        shutil.copyfileobj(file.file, buf)

    try:
        vec = compute_feature_vector(query_path)
        results = search_similar(vec, limit=top_k)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        os.remove(query_path)

    host = os.getenv("API_HOST", "http://127.0.0.1:8000")

    def interpret_score(score: float) -> str:
        if score >= 0.85:
            return "High"
        elif score >= 0.5:
            return "Medium"
        else:
            return "Low"

    hits = [
        {
            "id": r.id,
            "score": r.score,
            "similarity_level": interpret_score(r.score),
            "url": f"{host}/frames/{r.payload['image']}"
        }
        for r in results
    ]

    return {"results": hits}

@app.on_event("startup")
async def notify_docs_url():
    import time
    time.sleep(0.5)
    print("\nSwagger UI: http://127.0.0.1:8000/docs\n")

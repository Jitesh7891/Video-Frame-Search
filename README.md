# Video Frame Search API with FastAPI and Qdrant

This project implements a FastAPI application that:

- Accepts MP4 video uploads
- Extracts frames in any range from 1 to 10 seconds (depending on user input, default is 1) intervals and saves them as images
- Computes feature vectors for each extracted frame using the CLIP model
- Stores these feature vectors in a Qdrant vector database
- Provides a search API to find visually similar frames based on an uploaded image query

# Features

- Video Processing: Upload videos and automatically extract frames
- Feature Vector Computation: Compute image embeddings for extracted frames
- Vector Database: Store and index feature vectors with Qdrant
- Similarity Search: Query the vector database with an image to find similar frames
- Static File Serving: Access extracted frames via URLs

# Tech Stack

- Python 3.12 or newer
- FastAPI
- Qdrant vector database
- OpenAI CLIP for feature extraction
- OpenCV for video frame extraction

# Getting Started

## Prerequisites

- Python 3.12 or newer
- Qdrant server running locally or remotely

You can run Qdrant locally using Docker with this command.
However data is temporary and will not persist

```bash
docker run -p 6333:6333 qdrant/qdrant
```

This will start Qdrant and expose it on port 6333.

## Installation

- Clone the repository
- Create and activate a Python virtual environment
- Install dependencies with pip
```bash
git clone https://github.com/yourusername/your-repo-name.git
cd your-repo-name

# Create a virtual environment
python -m venv venv

# Activate it
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

```

## Running the Application

- Start the FastAPI server using uvicorn
```bash
  uvicorn main:app --host 127.0.0.1 --port 8000 --reload
```
- The server will be available at http://127.0.0.1:8000

- You can also go to http://127.0.0.1:8000/docs from terminal to try out the apis

> ⚠️ **Note**: The first time you run this project, it may take several minutes to download the CLIP model (~600MB) and related files. This is expected behavior and only happens once. Subsequent runs will be faster.


# API Endpoints

## 1. Upload Video & Extract Frames

- Endpoint: POST /upload/
- Description: Upload an MP4 video file. The server extracts frames every n seconds, computes feature vectors, and indexes them in Qdrant.
- Request: multipart/form-data with key `file` containing the MP4 video.
- Response: JSON indicating number of frames extracted and indexed.

## 2. Search Similar Frames

- Endpoint: POST /search/
- Description: Upload an image to find visually similar frames from the indexed videos.
- Request: multipart/form-data with key `file` as the query image, optional parameter `top_k` for number of results (default 5).
- Response: JSON list of matching frames, each with:
  - id: Frame ID
  - score: Similarity score (higher is more similar)
  - url: URL to the matching frame image
  - vector: Feature vector of the matched frame

## 3. Access Extracted Frames

- Extracted frames are served as static files at:
  - http://127.0.0.1:8000/frames/frame_{id}.jpg


# Project Structure

- main.py              # FastAPI app with upload and search endpoints
- video_utils.py       # Video frame extraction logic using OpenCV
- vector_utils.py      # Feature vector computation and Qdrant integration
- extracted_frames/    # Folder where extracted frames are saved
- requirements.txt     # Python dependencies
- README.md

# Notes

- Ensure Qdrant is running before starting the FastAPI server.
- Frames are saved as frame_0.jpg, frame_1.jpg, etc.
- Feature vectors use OpenAI’s CLIP model for embeddings.
- Adjust API_HOST environment variable if running the server on a different host or port.

# Testing via API Documentation

- You can open the interactive API docs at http://127.0.0.1:8000/docs
- Use the “Try it out” feature to upload videos or images directly from the browser
- Test the `/upload/` endpoint to upload videos and index frames
- Test the `/search/` endpoint to upload a query image and retrieve similar frames
- You will receive JSON responses with frame details and URLs to view the extracted images

This makes it easy to manually test and verify the application’s functionality without needing any additional tools.

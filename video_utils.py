def extract_frames(video_path: str, interval_seconds: int = 1) -> int:
    import cv2
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    interval = int(fps * interval_seconds)

    count = 0
    saved = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if count % interval == 0:
            path = f"extracted_frames/frame_{saved}.jpg"
            cv2.imwrite(path, frame)
            saved += 1
        count += 1
    cap.release()
    return saved

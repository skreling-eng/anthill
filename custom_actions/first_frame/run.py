from pathlib import Path
from PIL import Image
from ahlib.custom_action_io import save_image
import cv2

def run(bundle, base_dir, op_dir):
    root = Path(base_dir)
    out = {k: list(bundle.get(k, [])) for k in
           ("prompts", "texts", "images", "sounds", "videos", "files", "changes")}
    new_images = []
    for link in bundle.get("videos", []):
        video_path = root / link
        cap = cv2.VideoCapture(str(video_path))
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(frame)
                new_images.append(save_image(base_dir, op_dir, f"first_frame_{len(new_images)}.png", pil_image))
            cap.release()
    out["images"] = new_images
    return out

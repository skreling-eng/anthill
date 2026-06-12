from pathlib import Path
from PIL import Image
from ahlib.custom_action_io import save_image
import cv2

def run(bundle, base_dir, op_dir):
    root = Path(base_dir)
    out = {k: list(bundle.get(k, [])) for k in
           ("prompts", "texts", "images", "sounds", "videos", "files", "changes")}
    new_images = []
    frame_counter = 0

    for link in bundle.get("videos", []):
        video_path = root / link
        cap = cv2.VideoCapture(str(video_path))
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame_counter += 1
            if frame_counter % 25 == 0:
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # Convert to PIL Image
                pil_image = Image.fromarray(frame_rgb)
                # Save the image
                new_images.append(save_image(base_dir, op_dir, f"frame_{frame_counter}.png", pil_image))
        cap.release()

    out["images"] = new_images
    return out

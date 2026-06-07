from pathlib import Path
from PIL import Image
from ahlib.custom_action_io import save_image

def run(bundle, base_dir, op_dir):
    root = Path(base_dir)
    out = {k: list(bundle.get(k, [])) for k in
           ("prompts", "texts", "images", "sounds", "videos", "files", "changes")}
    new_images = []

    for link in bundle.get("images", []):
        # Add the original image link
        new_images.append(link)
        
        # Open the image and flip it horizontally
        im = Image.open(root / link).convert("RGB")
        flipped_im = im.transpose(Image.FLIP_LEFT_RIGHT)
        
        # Save the flipped image and add the link
        flipped_link = save_image(base_dir, op_dir, f"flip_{Path(link).stem}.png", flipped_im)
        new_images.append(flipped_link)

    out["images"] = new_images
    return out

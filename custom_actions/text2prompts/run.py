import re
from ahlib.custom_action_io import save_text
from pathlib import Path

def run(bundle, base_dir, op_dir):
    root = Path(base_dir)
    out = {k: list(bundle.get(k, [])) for k in
           ("prompts", "texts", "images", "sounds", "videos", "files", "changes")}
    
    new_prompts = []
    for text_link in bundle.get("texts", []):
        with open(root / text_link, 'r', encoding='utf-8') as file:
            text_content = file.read()
        
        # Split by empty lines
        paragraphs = text_content.split('\n\n')
        
        # Filter paragraphs that contain at least one English letter
        for paragraph in paragraphs:
            if re.search(r'[a-zA-Z]', paragraph):
                # Save each valid paragraph as a new prompt
                link = save_text(base_dir, op_dir, "prompts", f"{len(new_prompts)}.txt", paragraph)
                new_prompts.append(link)
    
    out["prompts"] = new_prompts
    return out

from pathlib import Path
from moviepy import VideoFileClip, concatenate_videoclips

def run(bundle, base_dir, op_dir):
    root = Path(base_dir)
    out = {k: list(bundle.get(k, [])) for k in
           ("prompts","texts","images","sounds","videos","files","changes")}
    clips = []
    try:
        for link in bundle.get("videos", []):
            clips.append(VideoFileClip(str(root / link)))
        if not clips:
            return out
        final = concatenate_videoclips(clips, method="compose")
        dest = Path(op_dir) / "videos" / "joined_0.mp4"
        dest.parent.mkdir(parents=True, exist_ok=True)
        final.write_videofile(str(dest), codec="libx264", audio_codec="aac", logger=None)
        out["videos"] = [dest.relative_to(root).as_posix()]
    finally:
        for clip in clips:
            clip.close()
    return out

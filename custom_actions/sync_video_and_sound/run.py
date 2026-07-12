from pathlib import Path

try:    from moviepy import AudioFileClip, VideoFileClip
except ImportError:  # moviepy 1.x
    import moviepy.editor as mp

    VideoFileClip = mp.VideoFileClip
    AudioFileClip = mp.AudioFileClip


def run(bundle, base_dir, op_dir):
    root = Path(base_dir)
    out = {
        k: list(bundle.get(k, []))
        for k in ("prompts", "texts", "images", "sounds", "videos", "files", "changes")
    }

    new_videos = []

    if len(bundle.get("videos", [])) != len(bundle.get("sounds", [])):
        raise ValueError("The number of videos and sounds must be the same.")

    for video_link, sound_link in zip(bundle.get("videos", []), bundle.get("sounds", [])):
        video_path = root / video_link
        sound_path = root / sound_link

        video_clip = VideoFileClip(str(video_path))
        sound_clip = AudioFileClip(str(sound_path))
        try:
            end = min(video_clip.duration, sound_clip.duration)
            if hasattr(video_clip, "subclipped"):
                video_clip = video_clip.subclipped(0, end)
            else:
                video_clip = video_clip.subclip(0, end)

            if hasattr(video_clip, "with_audio"):
                video_clip = video_clip.with_audio(sound_clip)
            else:
                video_clip = video_clip.set_audio(sound_clip)

            new_video_path = Path(op_dir) / f"synced_{len(new_videos)}.mp4"
            video_clip.write_videofile(
                str(new_video_path),
                codec="libx264",
                audio_codec="aac",
                logger=None,
            )
            new_video_link = str(new_video_path.relative_to(base_dir))
            new_videos.append(new_video_link)
        finally:
            video_clip.close()
            sound_clip.close()

    out["videos"] = new_videos
    return out

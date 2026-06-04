"""Procedural generators for large example sets. Used by gen_test_data_examples.py."""

from __future__ import annotations

from pathlib import Path

RETURN = "Return only the final result without comments, clarifications, or explanations."

# Populated by discover_test_image_paths(repo_root).
_IMAGE_PATHS: list[str] = []

_IMAGE2IMAGE_EDITS = [
    "Improve lighting and make colors more natural; keep the same subject.",
    "Fix red eyes and soften harsh shadows.",
    "Increase sharpness and clarity without changing composition.",
    "Remove sensor noise and grain while keeping detail.",
    "Correct white balance for natural skin tones.",
    "Reduce background clutter; keep the main subject.",
    "Make the image look like golden-hour photography.",
    "Fix crooked horizon and mild perspective distortion.",
    "Enhance contrast for a crisp documentary look.",
    "Remove a small watermark area if visible; preserve content.",
    "Brighten underexposed areas without blowing highlights.",
    "Add subtle vibrance; avoid oversaturation.",
    "Smooth skin slightly while keeping texture realistic.",
    "Sharpen eyes and face; gentle background blur effect in prompt only.",
    "Convert dull grey sky to dramatic sunset tones in place.",
]

_IMAGE2IMAGE_MODELS = [
    ("sfw-v23", 8),
    ("sfw-v23", 12),
    ("sfw-v23", 6),
    ("default", 10),
]

_IMAGE2TEXT_TASKS = [
    (
        "Describe the scene, main subjects, and setting in 6-8 sentences.",
        "Describe what you see in this image.",
    ),
    (
        "Describe the UI layout, controls, and likely purpose of this screen.",
        "Describe this app or UI screenshot.",
    ),
    (
        "List any visible text verbatim, then summarize the image.",
        "Read and list visible text in this image.",
    ),
    (
        "What emotions or mood does this image convey?",
        "What mood does this image convey?",
    ),
    (
        "Identify the main objects and their spatial relationships.",
        "What are the main objects and how are they arranged?",
    ),
    (
        "Is this image suitable as a product photo? Explain briefly.",
        "Could this work as a product photo? Why or why not?",
    ),
    (
        "Describe colors, lighting, and time of day if inferable.",
        "Describe colors and lighting in this image.",
    ),
    (
        "Note any quality issues: blur, noise, cropping, or artifacts.",
        "What image quality issues do you see?",
    ),
    (
        "Write alt text for web accessibility (one paragraph).",
        "Write accessibility alt text for this image.",
    ),
    (
        "Compare foreground and background; what is in focus?",
        "What is in focus versus background in this image?",
    ),
]

_IMAGE2TEXT_MODELS = ["", "model='qwen3'", "model='default'"]

_TOPICS = [
    "climate change",
    "solar panels",
    "electric vehicles",
    "machine learning",
    "Rust programming",
    "PostgreSQL indexing",
    "microservices",
    "blockchain",
    "quantum computing",
    "renewable energy",
    "vaccines",
    "nutrition",
    "sleep hygiene",
    "meditation",
    "urban gardening",
    "3D printing",
    "photography",
    "jazz music",
    "Renaissance art",
    "Roman history",
    "Japanese culture",
    "Amazon rainforest",
    "Arctic ice",
    "ocean pollution",
    "space exploration",
    "Mars missions",
    "CRISPR",
    "antibiotics",
    "inflation",
    "cryptocurrency",
    "open source",
    "web accessibility",
    "cybersecurity",
    "password managers",
    "home networking",
    "bread baking",
    "fermentation",
    "yoga",
    "hiking gear",
    "bird watching",
    "chess strategy",
    "piano practice",
    "watercolor painting",
    "creative writing",
    "public speaking",
    "time management",
    "remote work",
    "freelancing",
    "startup funding",
    "product design",
    "UX research",
    "API design",
    "GraphQL",
    "Docker",
    "Kubernetes",
    "Linux shells",
    "Git workflows",
    "CI/CD",
    "unit testing",
    "TypeScript",
    "Python asyncio",
    "data visualization",
    "pandas",
    "SQL joins",
    "NoSQL",
    "Redis caching",
    "message queues",
    "event-driven architecture",
]

_CREATIVE_PROMPTS = [
    "Write a haiku about {t}.",
    "Write a limerick about {t}.",
    "Give a two-sentence horror story about {t}.",
    "Describe {t} as if you were a tour guide.",
    "Write a metaphor comparing {t} to weather.",
]

_GREETINGS = [
    "hi",
    "hello",
    "hey there",
    "good morning",
    "good evening",
    "thanks!",
    "thank you",
    "ok",
    "got it",
    "please continue",
]

_GET_MESSAGES_TASKS = [
    "Summarize the conversation in __CONTENT__ in 6 bullets.",
    "What was the user's last question in __CONTENT__?",
    "List unresolved topics from __CONTENT__.",
    "What media file paths appear in __CONTENT__?",
    "Recap only user messages from __CONTENT__.",
    "What is the main topic in __CONTENT__?",
    "Build a short timeline from __CONTENT__.",
    "Extract open action items from __CONTENT__.",
]

_MATH_TEMPLATES = [
    "Solve: {eq}. Show steps.",
    "What is {eq}?",
    "Prove or disprove: {claim}.",
]

_MATH_EQS = [
    "the derivative of sin(x) * e^x",
    "integral of 1/(1+x^2) dx",
    "lim(x->0) sin(x)/x",
    "2+2*3",
    "gcd(48, 18)",
    "probability of two heads in 3 coin flips",
    "sum of first n integers",
    "sqrt(200) approximated",
    "log2(1024)",
    "determinant of [[2,1],[4,3]]",
]

_CODE_TASKS = [
    "Write a Python function to {task}.",
    "Write a JavaScript snippet to {task}.",
    "Write a Bash one-liner to {task}.",
]

_CODE_GOALS = [
    "deduplicate a list preserving order",
    "read a CSV and print row count",
    "parse JSON from a file",
    "retry an HTTP GET with timeout",
    "hash a string with SHA-256",
    "merge two sorted lists",
    "find anagrams in a word list",
    "validate an email with regex",
]

_IMAGE_SUBJECTS = [
    "a lighthouse in fog at dawn",
    "a busy Tokyo street at night",
    "a medieval castle on a hill",
    "a bowl of ramen with steam",
    "a red bicycle by a canal",
    "an astronaut planting a flag",
    "a fox in autumn forest",
    "a storm over the ocean",
]

_MUSIC_STYLES = [
    "jazz",
    "irish traditional",
    "acoustic folk",
    "lo-fi hip hop",
    "orchestral cinematic",
    "synthwave",
    "blues",
    "reggae",
    "ambient electronic",
    "indie pop",
]

_CLIP_SUBJECTS = [
    "rabbits in a meadow",
    "a sunset beach",
    "city traffic at night",
    "mountain hikers",
    "children playing in snow",
    "a coffee shop morning",
    "wild horses running",
    "northern lights over a lake",
    "a jazz band on stage",
    "kites flying on a windy day",
    "a farmer's market",
    "underwater coral reef fish",
    "a birthday party",
    "autumn leaves falling",
    "a train crossing a bridge",
    "street musicians",
    "a camping trip by a fire",
    "sailing boats at dawn",
    "a vegetable garden",
    "penguins on ice",
    "a fireworks display",
    "a bicycle race",
    "cherry blossoms in spring",
    "a desert road trip",
    "puppies playing",
    "a thunderstorm over fields",
    "skateboarders in a park",
    "a pottery workshop",
    "hot air balloons",
    "a winter village",
]

_VIDEO_SUBJECTS = [
    "a woman smiling in soft wind",
    "ocean waves rolling on shore",
    "a cat turning its head slowly",
    "clouds moving over mountains",
    "a dancer spinning gracefully",
    "rain falling on a window",
    "a bonfire flickering at night",
    "leaves rustling in the breeze",
    "a car driving through neon city lights",
    "a bird taking flight",
]

_VIDEO_MOTIONS = [
    "gentle head movement, soft smile, wind in hair",
    "slow camera pan, natural motion, cinematic lighting",
    "subtle breathing motion, eyes blinking, ambient movement",
    "smooth parallax, atmospheric depth, golden hour light",
    "fluid body motion, dynamic camera, high detail",
]

_IMAGE2IMAGE_EDITS_EXTRA = [
    "Apply a warm vintage film look.",
    "Convert to black and white with rich contrast.",
    "Make the background softer and more bokeh-like.",
    "Increase saturation for a vivid poster style.",
    "Remove color cast and restore neutral tones.",
    "Add subtle film grain for a cinematic feel.",
    "Brighten shadows for a high-key portrait look.",
    "Darken edges for a subtle vignette effect.",
    "Sharpen edges while keeping skin smooth.",
    "Make the scene look like winter with cool tones.",
    "Apply summer warmth and sun flare mood.",
    "Reduce motion blur if present.",
    "Fix overexposed highlights on the subject.",
    "Enhance foliage greens naturally.",
    "Make indoor lighting look like daylight.",
    "Remove dust spots and minor blemishes.",
    "Align colors for consistent product photography.",
    "Add depth with mild contrast boost.",
    "Soften background while keeping subject crisp.",
    "Make the photo look like editorial magazine quality.",
]


def _topic(i: int) -> str:
    return _TOPICS[i % len(_TOPICS)]


def _stem(kind: str, i: int) -> str:
    return f"{kind}_{i:04d}"


def build_case(kind: str, i: int) -> tuple[str, str, str]:
    t = _topic(i)
    if kind == "greeting":
        req = _GREETINGS[i % len(_GREETINGS)]
        body = f"@answer: $llm\n{req}\n{RETURN}\n\nrun @answer"
        return _stem("greeting", i), req, body

    if kind == "llm_explain":
        req = f"Explain {t} in simple terms for a beginner."
        body = f"@answer: $llm\n{req}\n{RETURN}\n\nrun @answer"
        return _stem("llm_explain", i), req, body

    if kind == "llm_creative":
        tmpl = _CREATIVE_PROMPTS[i % len(_CREATIVE_PROMPTS)]
        req = tmpl.format(t=t)
        body = f"@answer: $llm\n{req}\n{RETURN}\n\nrun @answer"
        return _stem("llm_creative", i), req, body

    if kind == "llm_howto":
        req = f"How do I get started with {t}? Give 5 practical steps."
        body = f"@answer: $llm\n{req}\n{RETURN}\n\nrun @answer"
        return _stem("llm_howto", i), req, body

    if kind == "llm_compare":
        t2 = _topic(i + 7)
        req = f"Compare {t} and {t2} in a short table."
        body = f"@answer: $llm\n{req}\n{RETURN}\n\nrun @answer"
        return _stem("llm_compare", i), req, body

    if kind == "math":
        eq = _MATH_EQS[i % len(_MATH_EQS)]
        req = f"Solve: {eq}. Show main steps."
        body = f"@answer: $math\n{req}\n{RETURN}\n\nrun @answer"
        return _stem("math", i), req, body

    if kind == "search":
        q = f"{t} overview 2026"
        req = f"What should I know about {t} in 2026?"
        body = (
            f"@search_step: $search(limit=6, fetch_pages=True, fetch_max=3)\n"
            f"{q}\n\n"
            f"@answer: @search_step -> $json2texts -> $texts2prompts -> $llm(add_texts=True)\n"
            f"{req}\n"
            f"Use __CONTENT__ from search results.\n"
            f"{RETURN}\n\n"
            f"run @answer"
        )
        return _stem("search", i), req, body

    if kind == "search_snippet":
        q = f"{t} definition"
        req = f"What is {t}?"
        body = (
            f"@search_step: $search(limit=5)\n"
            f"{q}\n\n"
            f"@answer: @search_step -> $json2texts -> $texts2prompts -> $llm(add_texts=True)\n"
            f"{req}\n"
            f"{RETURN}\n\n"
            f"run @answer"
        )
        return _stem("search_snippet", i), req, body

    if kind == "code":
        goal = _CODE_GOALS[i % len(_CODE_GOALS)]
        tmpl = _CODE_TASKS[i % len(_CODE_TASKS)]
        req = tmpl.format(task=goal)
        body = f"@answer: $code\n{req}\n{RETURN}\n\nrun @answer"
        return _stem("code", i), req, body

    if kind == "json":
        req = f"Return JSON with keys topic, summary, difficulty for: {t}."
        body = (
            f"@answer: $llm\n"
            f"{req}\n"
            f"Return only valid JSON, no markdown.\n"
            f"{RETURN}\n\n"
            f"run @answer"
        )
        return _stem("json", i), req, body

    if kind == "list":
        req = f"List 7 interesting facts about {t}."
        body = f"@answer: $llm\n{req}\n{RETURN}\n\nrun @answer"
        return _stem("list", i), req, body

    if kind == "rewrite":
        req = f"Rewrite formally: I kinda need help with {t} asap."
        body = f"@answer: $llm\n{req}\n{RETURN}\n\nrun @answer"
        return _stem("rewrite", i), req, body

    if kind == "summarize":
        req = f"Summarize this in 3 bullets: {t} is important because it affects many systems."
        body = f"@answer: $llm\n{req}\n{RETURN}\n\nrun @answer"
        return _stem("summarize", i), req, body

    if kind == "return_text":
        req = f"Return exactly: OK-{i:04d}"
        body = f"@answer: $prompts_to_texts\nOK-{i:04d}\n\nrun @answer"
        return _stem("return_text", i), req, body

    if kind == "image":
        subj = _IMAGE_SUBJECTS[i % len(_IMAGE_SUBJECTS)]
        req = f"Generate an image of {subj}."
        body = (
            f"@prompts: $llm -> $texts_to_prompts\n"
            f"{subj}, photorealistic, high detail.\n"
            f"Return only the prompt, under 55 words.\n\n"
            f"@images: @prompts -> $image(model='default')\n\n"
            f"run @images"
        )
        return _stem("image", i), req, body

    if kind == "get_messages":
        task = _GET_MESSAGES_TASKS[i % len(_GET_MESSAGES_TASKS)]
        req = f"Use chat history: {task.split('.')[0].lower()}."
        body = (
            f"@history: ^get_messages\n\n"
            f"@answer: @history -> $json2texts -> $texts2prompts -> $llm(add_texts=True)\n"
            f"{task}\n"
            f"{RETURN}\n\n"
            f"run @answer"
        )
        return _stem("get_messages", i), req, body

    if kind == "get_messages_search":
        req = f"Using chat history and the web, what do we know about {t}?"
        body = (
            f"@history: ^get_messages\n\n"
            f"@search_step: $search(limit=5)\n"
            f"{t} latest facts 2026\n\n"
            f"@answer: (@history, @search_step) -> $json2texts -> $texts2prompts -> $llm(add_texts=True)\n"
            f"{req}\n"
            f"Use __CONTENT__.\n"
            f"{RETURN}\n\n"
            f"run @answer"
        )
        return _stem("get_messages_search", i), req, body

    if kind == "translate":
        req = f"Translate to French: Learning about {t} is useful."
        body = (
            f"@en: $llm\n"
            f"Learning about {t} is useful.\n"
            f"Return only the English sentence.\n\n"
            f"@fr: @en -> $translate(src='en', dst='fr')\n\n"
            f"run @fr"
        )
        return _stem("translate", i), req, body

    # fallback
    return build_case("llm_explain", i)


# Round-robin weights (~1000): sum = 1000
_KIND_CYCLE = (
    ["llm_explain"] * 140
    + ["llm_creative"] * 80
    + ["llm_howto"] * 80
    + ["llm_compare"] * 60
    + ["search"] * 120
    + ["search_snippet"] * 60
    + ["math"] * 80
    + ["code"] * 80
    + ["get_messages"] * 100
    + ["get_messages_search"] * 40
    + ["greeting"] * 40
    + ["image"] * 60
    + ["json"] * 40
    + ["list"] * 40
    + ["rewrite"] * 30
    + ["summarize"] * 30
    + ["return_text"] * 20
    + ["translate"] * 40
)


def generate_bulk_cases(count: int, *, start_index: int = 1) -> list[tuple[str, str, str]]:
    if count <= 0:
        return []
    cases: list[tuple[str, str, str]] = []
    for n in range(count):
        i = start_index + n
        kind = _KIND_CYCLE[n % len(_KIND_CYCLE)]
        cases.append(build_case(kind, i))
    return cases


def discover_test_image_paths(repo_root: Path) -> list[str]:
    """Session-relative paths to PNGs under test_data/."""
    root = repo_root / "test_data"
    if not root.is_dir():
        return [
            "test_data/clip/demo-poster.png",
            "test_data/app/app_screenshot_1.png",
        ]
    found: list[str] = []
    for path in sorted(root.rglob("*.png")):
        if path.stat().st_size < 500:
            continue
        rel = path.relative_to(repo_root).as_posix()
        found.append(rel)
    return found or ["test_data/clip/demo-poster.png"]


def _image_path(i: int) -> str:
    paths = _IMAGE_PATHS or ["test_data/clip/demo-poster.png"]
    return paths[i % len(paths)]


def build_image2image_case(i: int) -> tuple[str, str, str]:
    rel = _image_path(i)
    edit = _IMAGE2IMAGE_EDITS[i % len(_IMAGE2IMAGE_EDITS)]
    model, steps = _IMAGE2IMAGE_MODELS[i % len(_IMAGE2IMAGE_MODELS)]
    req = edit.split(";")[0].strip().rstrip(".") + "."
    body = (
        f"@load: $file('{rel}')\n\n"
        f"@edit: @load -> $image2image(model='{model}', steps={steps})\n"
        f"{edit}\n"
        f"{RETURN}\n\n"
        f"run @edit"
    )
    return _stem("image2image", i), req, body


def build_image2text_case(i: int) -> tuple[str, str, str]:
    rel = _image_path(i + 3)
    task_body, req = _IMAGE2TEXT_TASKS[i % len(_IMAGE2TEXT_TASKS)]
    model_arg = _IMAGE2TEXT_MODELS[i % len(_IMAGE2TEXT_MODELS)]
    ext = f"$image2text({model_arg})" if model_arg else "$image2text"
    body = (
        f"@load: $file('{rel}')\n\n"
        f"@describe: @load -> {ext}\n"
        f"{task_body}\n"
        f"{RETURN}\n\n"
        f"run @describe"
    )
    return _stem("image2text", i), req, body


def _max_example_index(repo_root: Path, stem_prefix: str) -> int:
    import re

    out = repo_root / "test_data" / "examples"
    best = 0
    pat = re.compile(rf"^example_{re.escape(stem_prefix)}_(\d+)\.ah$")
    for path in out.glob(f"example_{stem_prefix}_*.ah"):
        m = pat.match(path.name)
        if m:
            best = max(best, int(m.group(1)))
    return best


def build_image_case(i: int) -> tuple[str, str, str]:
    subj = _CLIP_SUBJECTS[i % len(_CLIP_SUBJECTS)]
    models = ("default", "realistic", "default", "sfw-v23")
    model = models[i % len(models)]
    req = f"Generate an image of {subj}."
    body = (
        f"@prompts: $llm -> $texts_to_prompts\n"
        f"{subj}, photorealistic, high detail, cinematic composition.\n"
        f"Return only the prompt, under 55 words.\n\n"
        f"@images: @prompts -> $image(model='{model}')\n\n"
        f"run @images"
    )
    return _stem("image", i), req, body


def build_image_clip_case(i: int) -> tuple[str, str, str]:
    subject = _CLIP_SUBJECTS[i % len(_CLIP_SUBJECTS)]
    style = _MUSIC_STYLES[i % len(_MUSIC_STYLES)]
    n_images = 6 + (i % 7)
    req = f"Generate an image clip about {subject}."
    body = (
        f"@quality: $list\n"
        f"High resolution, best quality, photorealistic, cinematic\n\n"
        f"@clip_images: @quality -> $llm -> $texts_to_prompts -> $image(model='default')[{n_images}]\n"
        f"Create prompts for still images about {subject}.\n"
        f"Return only the final result without comments, clarifications, or explanations.\n\n"
        f"@music_style: $list\n"
        f"{style}\n\n"
        f"@lyrics_text: $llm\n"
        f"Write short song lyrics about {subject}.\n"
        f"{RETURN}\n\n"
        f"@track: (@music_style, @lyrics_text) -> $music(model='st', guidance_scale=4.0, vocal_language='en', inference_steps=50)\n\n"
        f"@gen_clip: (@clip_images, @track) -> $image_clip -> $output\n\n"
        f"run @gen_clip"
    )
    return _stem("image_clip", i), req, body


def build_image2video_case(i: int) -> tuple[str, str, str]:
    subject = _VIDEO_SUBJECTS[i % len(_VIDEO_SUBJECTS)]
    motion = _VIDEO_MOTIONS[i % len(_VIDEO_MOTIONS)]
    req = f"Create a short video of {subject}."
    body = (
        f"@prompts: $llm -> $texts_to_prompts\n"
        f"Still frame of {subject}, photorealistic, sharp detail.\n"
        f"Return only the prompt, under 50 words.\n\n"
        f"@image: @prompts -> $image(model='default')\n\n"
        f"@video: @image -> $image2video(model='default', width=768, height=512, frames=49, steps=4, guidance=1)\n"
        f"{motion}\n"
        f"{RETURN}\n\n"
        f"run @video"
    )
    return _stem("image2video", i), req, body


def build_music_case(i: int) -> tuple[str, str, str]:
    subject = _CLIP_SUBJECTS[(i + 11) % len(_CLIP_SUBJECTS)]
    style = _MUSIC_STYLES[i % len(_MUSIC_STYLES)]
    req = f"Create a short song about {subject}."
    body = (
        f"@music_style: $list\n"
        f"{style}\n\n"
        f"@lyrics_text: $llm\n"
        f"Write lyrics for a song about {subject}.\n"
        f"{RETURN}\n\n"
        f"@track: (@music_style, @lyrics_text) -> $music(model='st', guidance_scale=4.0, vocal_language='en', inference_steps=50)\n\n"
        f"run @track"
    )
    return _stem("music", i), req, body


def build_image2image_case_v2(i: int) -> tuple[str, str, str]:
    """image2image with extended edit prompts."""
    all_edits = _IMAGE2IMAGE_EDITS + _IMAGE2IMAGE_EDITS_EXTRA
    rel = _image_path(i)
    edit = all_edits[i % len(all_edits)]
    model, steps = _IMAGE2IMAGE_MODELS[i % len(_IMAGE2IMAGE_MODELS)]
    req = edit.split(";")[0].strip().rstrip(".") + "."
    body = (
        f"@load: $file('{rel}')\n\n"
        f"@edit: @load -> $image2image(model='{model}', steps={steps})\n"
        f"{edit}\n"
        f"{RETURN}\n\n"
        f"run @edit"
    )
    return _stem("image2image", i), req, body


_MEDIA_KINDS = ("image", "image2image", "image_clip", "image2video", "music")

_MEDIA_BUILDERS = {
    "image": build_image_case,
    "image2image": build_image2image_case_v2,
    "image_clip": build_image_clip_case,
    "image2video": build_image2video_case,
    "music": build_music_case,
}


def generate_media_cases(
    count: int,
    *,
    repo_root: Path,
) -> list[tuple[str, str, str]]:
    """Generate image / image2image / image_clip / image2video / music examples."""
    global _IMAGE_PATHS
    _IMAGE_PATHS = discover_test_image_paths(repo_root)
    if count <= 0:
        return []
    starts = {kind: _max_example_index(repo_root, kind) + 1 for kind in _MEDIA_KINDS}
    cases: list[tuple[str, str, str]] = []
    for n in range(count):
        kind = _MEDIA_KINDS[n % len(_MEDIA_KINDS)]
        idx = starts[kind]
        starts[kind] = idx + 1
        cases.append(_MEDIA_BUILDERS[kind](idx))
    return cases


def generate_vision_cases(
    count: int,
    *,
    repo_root: Path,
    start_index: int = 5001,
) -> list[tuple[str, str, str]]:
    """Generate image2image and image2text examples (half each)."""
    global _IMAGE_PATHS
    _IMAGE_PATHS = discover_test_image_paths(repo_root)
    if count <= 0:
        return []
    n_each = count // 2
    extra_i2i = count - n_each * 2
    cases: list[tuple[str, str, str]] = []
    for k in range(n_each + (1 if extra_i2i else 0)):
        cases.append(build_image2image_case(start_index + k))
    offset = start_index + n_each + (1 if extra_i2i else 0)
    for k in range(n_each):
        cases.append(build_image2text_case(offset + k))
    return cases

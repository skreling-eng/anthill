"""Generate test_data/examples/*.ah benchmark scripts.

  uv run python tools/gen_test_data_examples.py              # rewrite all CASES (destructive)
  uv run python tools/gen_test_data_examples.py --add       # append NEW_CASES only
  uv run python tools/gen_test_data_examples.py --bulk 1000   # append 1000 procedural examples
  uv run python tools/gen_test_data_examples.py --media 500   # append image/clip/video/music examples
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

from tools.gen_bulk_examples import (
    generate_bulk_cases,
    generate_media_cases,
    generate_vision_cases,
)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "test_data" / "examples"

RETURN = "Return only the final result without comments, clarifications, or explanations."

# Chat-only: ^get_messages, ^store_message, ^answer, ^maybe_summarize require ChatInterface callback.
# (filename_stem, user request, .ah body)
CASES: list[tuple[str, str, str]] = [
    # --- Simple LLM (no search) ---
    ("greeting_1", "hi", f"""@answer: $llm
hi
{RETURN}

run @answer"""),
    ("greeting_2", "Thanks, that helped!", f"""@answer: $llm
Thanks, that helped!
{RETURN}

run @answer"""),
    ("llm_explain_1", "Explain what a REST API is in simple terms.", f"""@answer: $llm
Explain what a REST API is in simple terms.
{RETURN}

run @answer"""),
    ("llm_explain_2", "What is the difference between Docker and a virtual machine?", f"""@answer: $llm
What is the difference between Docker and a virtual machine?
{RETURN}

run @answer"""),
    ("llm_creative_1", "Write a haiku about autumn rain.", f"""@answer: $llm
Write a haiku about autumn rain.
{RETURN}

run @answer"""),
    ("llm_creative_2", "Give me a one-paragraph sci-fi story idea with time loops.", f"""@answer: $llm
Give me a one-paragraph sci-fi story idea with time loops.
{RETURN}

run @answer"""),
    # --- Math ---
    ("math_1", "Solve: integrate x^2 * e^x dx. Show main steps.", f"""@answer: $math(model='qwen36')
Solve: integrate x^2 * e^x dx. Show main steps.
{RETURN}

run @answer"""),
    ("math_2", "If I roll two fair dice, what is the probability the sum is 7?", f"""@answer: $math
If I roll two fair dice, what is the probability the sum is 7?
{RETURN}

run @answer"""),
    # --- Search ---
    ("search_1", "What are the most popular UK garden birds right now?", """@search_step: $search(limit=6, fetch_pages=True, fetch_max=3)
most popular UK garden birds 2026

@answer: @search_step -> $json2texts -> $texts2prompts -> $llm(add_texts=True)
What are the most popular UK garden birds right now?
Use __CONTENT__ (search results include page_text). List main species.
Return only the final result without comments, clarifications, or explanations.

run @answer"""),
    ("search_2", "What is the UK renewable energy policy in 2026?", """@search_step: $search(limit=6, fetch_pages=True, fetch_max=3)
UK renewable energy policy 2026

@answer: @search_step -> $json2texts -> $texts2prompts -> $llm(add_texts=True)
What is the UK renewable energy policy in 2026?
Use __CONTENT__ from search. Summarize key points.
Return only the final result without comments, clarifications, or explanations.

run @answer"""),
    ("search_snippet_1", "What is the capital of Australia?", """@search_step: $search(limit=5)
capital of Australia

@answer: @search_step -> $json2texts -> $texts2prompts -> $llm(add_texts=True)
What is the capital of Australia?
Return only the final result without comments, clarifications, or explanations.

run @answer"""),
    ("search_news_1", "Summarize today's top headlines about artificial intelligence.", """@search_step: $search(limit=6, fetch_pages=True, fetch_max=3)
artificial intelligence news today 2026

@answer: @search_step -> $json2texts -> $texts2prompts -> $llm(add_texts=True)
Summarize today's top AI headlines from __CONTENT__. Max 8 bullets.
Return only the final result without comments, clarifications, or explanations.

run @answer"""),
    # --- Code ---
    ("code_1", "Write a Python function that returns the nth Fibonacci number.", f"""@answer: $code
Write a Python function that returns the nth Fibonacci number.
{RETURN}

run @answer"""),
    ("code_2", "Write a short Bash script that backs up a folder to a dated zip.", f"""@answer: $code
Write a short Bash script that backs up a folder to a dated zip file.
{RETURN}

run @answer"""),
    # --- Return text ---
    ("return_text_1", "Return exactly: Pipeline ready.", """@answer: $prompts_to_texts
Pipeline ready.

run @answer"""),
    # --- Image / vision ---
    ("image_1", "Generate an image of a red fox in snow at golden hour.", """@prompts: $llm -> $texts_to_prompts
A red fox in fresh snow, golden hour, photorealistic, shallow depth of field.
Return only the prompt, under 60 words.

@images: @prompts -> $image(model='default')

run @images"""),
    ("image_2", "Create a cozy coffee shop interior at night.", """@prompts: $llm -> $texts_to_prompts
Cozy coffee shop at night, warm lamps, rain on windows, cinematic.
Return only the prompt, under 55 words.

@images: @prompts -> $image(model='default')

run @images"""),
    ("image2text_1", "Describe what you see in this app screenshot.", """@load: $file('test_data/app/app_screenshot_1.png')

@answer: @load -> $image2text
Describe the UI layout and purpose in 5-8 sentences.
Return only the final result without comments, clarifications, or explanations.

run @answer"""),
    ("image2image_1", "Improve lighting and make colors more natural in this photo.", """@load: $file('test_data/abnormal_images/good/20260530_024944_default_image_0.png')

@edit: @load -> $image2image(model='sfw-v23', steps=8)
Improve lighting and make colors more natural; keep the same subject.
Return only the final result without comments, clarifications, or explanations.

run @edit"""),
    ("translate_1", "Translate to Russian: The garden is quiet in the morning.", """@en: $llm
The garden is quiet in the morning.
Return only the English sentence.

@ru: @en -> $translate(src='en', dst='ru')

run @ru"""),
    ("ocr_1", "Extract all visible text from this image.", """@load: $file('test_data/text_images/20260530_020727_default_image_0.png')

@text: @load -> $ocr

run @text"""),
    ("clip_1", "Make a short clip from a poster image in test_data/clip.", """@get_clip: $file('test_data/clip/demo-poster.png')
  -> $folder('test_data', source_path=True)
  -> $video_clip
  -> $save('test_data/examples/out_clip.mp4')

run @get_clip"""),
    ("math_word_1", "A train travels 120 km in 1.5 hours. What is its average speed in km/h?", f"""@answer: $math
A train travels 120 km in 1.5 hours. What is its average speed in km/h?
{RETURN}

run @answer"""),
    ("search_3", "Who is the current CEO of NVIDIA?", """@search_step: $search(limit=6, fetch_pages=True, fetch_max=3)
NVIDIA CEO 2026

@answer: @search_step -> $json2texts -> $texts2prompts -> $llm(add_texts=True)
Who is the current CEO of NVIDIA?
Answer from __CONTENT__.
Return only the final result without comments, clarifications, or explanations.

run @answer"""),
    ("search_compare_1", "Which is lighter: iPhone 16 or Samsung Galaxy S25?", """@search_step: $search(limit=6, fetch_pages=True, fetch_max=3)
iPhone 16 weight vs Samsung Galaxy S25 weight

@answer: @search_step -> $json2texts -> $texts2prompts -> $llm(add_texts=True)
Which is lighter: iPhone 16 or Samsung Galaxy S25? Use facts from __CONTENT__.
Return only the final result without comments, clarifications, or explanations.

run @answer"""),
    ("search_price_1", "What is the current price of Bitcoin in USD?", """@search_step: $search(limit=6, fetch_pages=True, fetch_max=3)
Bitcoin price USD today

@answer: @search_step -> $json2texts -> $texts2prompts -> $llm(add_texts=True)
What is the current price of Bitcoin in USD?
Return only the final result without comments, clarifications, or explanations.

run @answer"""),
    ("search_weather_1", "What is the weather in London today?", """@search_step: $search(limit=6, fetch_pages=True, fetch_max=3)
London weather today

@answer: @search_step -> $json2texts -> $texts2prompts -> $llm(add_texts=True)
What is the weather in London today?
Return only the final result without comments, clarifications, or explanations.

run @answer"""),
    # --- Text tasks ---
    ("summarize_1", "Summarize this paragraph in three bullet points.", f"""@answer: $llm
Summarize this paragraph in three bullet points.

Anthill is an agentic pipeline language. Programs chain instructions and externals.
Each step passes arrays of file links through a session directory.

{RETURN}

run @answer"""),
    ("rewrite_1", "Rewrite in formal tone: kinda stuck on the deadline", f"""@answer: $llm
Rewrite in formal business tone: kinda stuck on the deadline
{RETURN}

run @answer"""),
    ("json_1", "Return a JSON array of three European capitals with country names.", f"""@answer: $llm
Return a JSON array of three European capitals with country names.
Return only valid JSON, no markdown.
{RETURN}

run @answer"""),
    ("compare_1", "Compare SQL and NoSQL for a small startup.", f"""@answer: $llm
Compare SQL and NoSQL databases for a small startup. Short table.
{RETURN}

run @answer"""),
    ("list_1", "List ten weekend project ideas for learning machine learning.", f"""@answer: $llm
List ten weekend project ideas for learning machine learning.
{RETURN}

run @answer"""),
    ("howto_1", "How do I create a new git branch and push it to origin?", f"""@answer: $llm
How do I create a new git branch and push it to origin?
{RETURN}

run @answer"""),
    # --- Utilities / edge ---
    ("file_read_1", "In one sentence, what is Anthill from the language description file?", """@load: $file('_lang_desc', source_path=True)

@answer: @load -> $llm(add_texts=True)
In one sentence, what is Anthill based on __CONTENT__?
Return only the final result without comments, clarifications, or explanations.

run @answer"""),
    ("parallel_1", "Give a caption and a mood line for a rainy city at night.", """@caption: $llm
One-line caption for a rainy city at night.
Return only the final result without comments, clarifications, or explanations.

@mood: $llm
One mood line under 12 words for the same scene.
Return only the final result without comments, clarifications, or explanations.

@both: (@caption, @mood) -> $only(texts)

run @both"""),
    ("context_1", "Store a list in session context and read it back.", """@seed: $list
alpha, beta, gamma

@save: @seed -> %items

@read: items% -> $only(texts)

run @read"""),
    ("select_only_1", "From a list, return only the second item.", """@items: $list
first, second, third

@pick: @items -> $select(texts=[1]) -> $only(texts)

run @pick"""),
    ("draw_text_1", "Add title overlay Demo Poster on this image.", """@load: $file('test_data/clip/demo-poster.png')

@label: $list
Demo Poster

@titled: (@load, @label) -> zip(images, texts){ $draw_text(size=72) }

run @titled"""),
    ("unclear_1", "Build a production Windows kernel driver in one click.", """@report: $prompts_to_texts
Cannot fulfill: a production Windows kernel driver needs signing, WDK, and hardware context.
Suggest specifying driver type (KMDF), OS build, and device ID.

run @report"""),
    ("codegen_ah_1", "Write an Anthill script that answers hello with $llm.", """@gen: {
  $file('AH_CODEGEN_INSTRUCTIONS.md', source_path=True)
  -> $code
}
Write an Anthill script that greets the user with $llm when they say hello.
Return only the .ah script text, no markdown fences.
Return only the final result without comments, clarifications, or explanations.

run @gen"""),
]

assert len(CASES) == 40, len(CASES)

# Added with --add (chat ^get_messages callbacks; need ChatInterface at runtime).
NEW_CASES: list[tuple[str, str, str]] = [
    (
        "get_messages_summarize_1",
        "Summarize everything we discussed so far in this chat.",
        f"""@history: ^get_messages

@answer: @history -> $json2texts -> $texts2prompts -> $llm(add_texts=True)
Summarize the full conversation in __CONTENT__ (JSON messages and summaries).
Use 5-8 bullet points. Include both user and bot turns.
{RETURN}

run @answer""",
    ),
    (
        "get_messages_followup_1",
        "What was my previous question? Answer using chat history.",
        f"""@history: ^get_messages

@answer: @history -> $json2texts -> $texts2prompts -> $llm(add_texts=True)
What was the user's most recent question before this message? Use __CONTENT__ only.
{RETURN}

run @answer""",
    ),
    (
        "get_messages_topic_1",
        "What is the main topic of our conversation so far?",
        f"""@history: ^get_messages

@answer: @history -> $json2texts -> $texts2prompts -> $llm(add_texts=True)
What is the main topic of the conversation in __CONTENT__? One short paragraph.
{RETURN}

run @answer""",
    ),
    (
        "get_messages_media_1",
        "List all image, sound, and video file paths mentioned in our chat.",
        f"""@history: ^get_messages

@answer: @history -> $json2texts -> $texts2prompts -> $llm(add_texts=True)
From __CONTENT__, list every images[], sounds[], and videos[] path mentioned.
Return a bullet list grouped by type. If none, say none.
{RETURN}

run @answer""",
    ),
    (
        "get_messages_timeline_1",
        "Give a chronological timeline of user and bot messages from the chat.",
        f"""@history: ^get_messages

@answer: @history -> $json2texts -> $texts2prompts -> $llm(add_texts=True)
Build a chronological timeline from __CONTENT__. Format: role — short excerpt per entry.
{RETURN}

run @answer""",
    ),
    (
        "get_messages_json_1",
        "Extract user messages only as a JSON array of strings from chat history.",
        f"""@history: ^get_messages

@answer: @history -> $json2texts -> $texts2prompts -> $llm(add_texts=True)
From __CONTENT__, output a JSON array containing only user message text fields.
Skip summaries unless they quote the user. Valid JSON only, no markdown.
{RETURN}

run @answer""",
    ),
    (
        "get_messages_search_1",
        "Using chat history and the web, what country did we talk about visiting?",
        f"""@history: ^get_messages

@search_step: $search(limit=5)
travel destination country tourism 2026

@answer: (@history, @search_step) -> $json2texts -> $texts2prompts -> $llm(add_texts=True)
From __CONTENT__, infer which country the user wanted to visit and add brief travel facts from search.
{RETURN}

run @answer""",
    ),
    (
        "get_messages_compare_1",
        "Compare the tone of my messages vs the bot replies in this chat.",
        f"""@history: ^get_messages

@answer: @history -> $json2texts -> $texts2prompts -> $llm(add_texts=True)
Compare user vs bot tone in __CONTENT__ in a short table (formal/casual/technical).
{RETURN}

run @answer""",
    ),
    (
        "get_messages_actions_1",
        "What open questions or action items are still unresolved in this chat?",
        f"""@history: ^get_messages

@answer: @history -> $json2texts -> $texts2prompts -> $llm(add_texts=True)
From __CONTENT__, list unresolved questions or action items. Bullet list; say none if clear.
{RETURN}

run @answer""",
    ),
    (
        "get_messages_recap_user_1",
        "Recap only what the user asked for across the whole chat.",
        f"""@history: ^get_messages

@answer: @history -> $json2texts -> $texts2prompts -> $llm(add_texts=True)
Recap only user-side requests from __CONTENT__ (ignore bot replies except for context).
{RETURN}

run @answer""",
    ),
]


def _stem_base(stem: str) -> str:
    m = re.match(r"^(.*)_(\d+)$", stem)
    return m.group(1) if m else stem


def _next_free_stem(stem: str) -> str:
    path = OUT / f"example_{stem}.ah"
    if not path.exists():
        return stem
    base = _stem_base(stem)
    n = 2
    while (OUT / f"example_{base}_{n}.ah").exists():
        n += 1
    return f"{base}_{n}"


def _write_cases(
    cases: list[tuple[str, str, str]],
    *,
    skip_existing: bool,
    quiet: bool = False,
) -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    written = 0
    for idx, (stem, request, body) in enumerate(cases, start=1):
        out_stem = _next_free_stem(stem) if skip_existing else stem
        path = OUT / f"example_{out_stem}.ah"
        if skip_existing and path.exists() and out_stem == stem:
            continue
        content = f"# Request: {request}\n\n{body.strip()}\n"
        path.write_text(content, encoding="utf-8")
        written += 1
        if not quiet:
            print(path.name)
        elif idx % 100 == 0 or idx == len(cases):
            print(f"... {idx}/{len(cases)} ({path.name})")
    return written


def _parse_flag_count(argv: list[str], flag: str) -> int | None:
    for i, arg in enumerate(argv):
        if arg == flag and i + 1 < len(argv):
            return int(argv[i + 1])
        if arg.startswith(f"{flag}="):
            return int(arg.split("=", 1)[1])
    return None


def main() -> None:
    media = _parse_flag_count(sys.argv, "--media")
    if media is not None:
        if media <= 0:
            print("Nothing to generate.")
            return
        cases = generate_media_cases(media, repo_root=ROOT)
        n = _write_cases(cases, skip_existing=True, quiet=media >= 20)
        total = len(list(OUT.glob("example_*.ah")))
        print(
            f"Added {n} media file(s) (image, image2image, image_clip, image2video, music) "
            f"under {OUT} (total example_*.ah: {total})"
        )
        return

    vision = _parse_flag_count(sys.argv, "--vision")
    if vision is not None:
        if vision <= 0:
            print("Nothing to generate.")
            return
        cases = generate_vision_cases(vision, repo_root=ROOT, start_index=5001)
        n = _write_cases(cases, skip_existing=True, quiet=vision >= 20)
        total = len(list(OUT.glob("example_*.ah")))
        print(
            f"Added {n} vision file(s) (image2image + image2text) under {OUT} "
            f"(total example_*.ah: {total})"
        )
        return

    bulk = _parse_flag_count(sys.argv, "--bulk")
    if bulk is not None:
        if bulk <= 0:
            print("Nothing to generate.")
            return
        # Avoid stem collisions with hand-written examples (use high start index).
        cases = generate_bulk_cases(bulk, start_index=1001)
        n = _write_cases(cases, skip_existing=True, quiet=bulk >= 50)
        total = len(list(OUT.glob("example_*.ah")))
        print(f"Added {n} bulk file(s) under {OUT} (total example_*.ah: {total})")
        return

    add_only = "--add" in sys.argv
    if add_only:
        n = _write_cases(NEW_CASES, skip_existing=True)
        print(f"Added {n} new file(s) under {OUT} (existing kept)")
        return
    for path in OUT.glob("example_*.ah"):
        path.unlink()
    n = _write_cases(CASES, skip_existing=False)
    print(f"Wrote {n} files to {OUT}")


if __name__ == "__main__":
    main()

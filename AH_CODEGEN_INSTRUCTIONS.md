# Anthill (.ah) — instructions for code models

Use this document when **writing or editing Anthill scripts** (`.ah` files). It is optimized for LLMs that generate pipeline code. For the full language specification, see `_lang_desc` in the repo root.

---

## 1. Your job

You produce **valid `.ah` source** that the Anthill runtime (`ahlib`) can parse and run. A script is a set of `@` instructions plus a single `run @entry` line.

**Default deliverable:** only the script text — no markdown fences, no “here is your script”, no extra commentary unless the user asked for explanation.

---

## 2. Output is source code, not JSON

The `.ah` file is **plain text**. The runtime reads real line breaks and real characters.

### NEVER do this

```ah
@answer: $llm
\"\"\"\n
User question goes here.\n
Return only the final result…\n
\"\"\"\n

run @answer
```

That is invalid: `\"`, `\n`, and placeholder text are **JSON/string escapes**, not Anthill syntax.

### ALWAYS do this

- Write **real newlines** (Enter between lines). Never the two characters `\` and `n`.
- Write **`"""` only if you need a multiline block** — three quote characters, unescaped. For a short user message, **plain body lines are better** (no triple quotes).
- Put the **actual user message** in the body (e.g. `hi`), not placeholders like `User question goes here` or `<prompt>`.

**User said `hi` → correct script:**

```ah
@answer: $llm
hi
Return only the final result without comments, clarifications, or explanations.

run @answer
```

**User said `What is 2+2?` → correct script:**

```ah
@answer: $llm
What is 2+2?
Return only the final result without comments, clarifications, or explanations.

run @answer
```

Use `"""` … `"""` only when the body itself must contain blank lines or many paragraphs — still **no** `\n` escapes inside the file:

```ah
@answer: $llm
"""
First paragraph.

Second paragraph.
"""
Return only the final result without comments, clarifications, or explanations.

run @answer
```

---

## 3. Non‑negotiable rules

| Rule | Detail |
|------|--------|
| **End with `run @name`** | Every script must declare exactly one entry: `run @instruction_name`. |
| **No `@name[inf]` in generated scripts** | Infinite repeat is for interactive chat UIs only. Never emit `[inf]`. |
| **No `$ah` at top level** | Nested `$ah` is allowed only inside sub-pipelines when a subtask must run another script. Top-level tasks should be direct pipelines. |
| **Prefer simple pipelines** | Fewer instructions and fewer steps beat clever nesting. |
| **Comments** | Use `#` for comments. Do not use `/* */` unless you know the host strips them (parser strips block comments). |
| **Prompt bodies** | Put user/task text in the instruction **body** (lines after the header) or in a `"""` … `"""` block — not in random `$` string args unless the external requires it. |
| **Return-only tasks** | When the user wants plain text out, follow `examples/example_return_output_text.ah` or add a body line like “Return only the final result without comments…”. |
| **Paths** | Use forward slashes or quoted paths: `$file('path/to/file.wav')`, `$save('output.mp4')`. |
| **Model args** | External args are `key=value` strings: `model='default'`, `width=768`, `add_texts=True`. |
| **User text in body** | Copy the user’s words verbatim into the instruction body after `@answer: $llm`. No templates, no `<prompt>`, no “User question goes here”. |
| **No string escapes** | Never `\n`, `\"`, `\\` in the output file unless the user literally asked for those characters in the answer. |

---

## 4. Mental model (read this first)

1. Data flows as **seven arrays** of **links** (paths under a session folder): `prompts[]`, `texts[]`, `images[]`, `sounds[]`, `videos[]`, `files[]`, `changes[]`.
2. Steps are chained with **`->`**. Each step receives the previous step’s arrays and returns new arrays.
3. An **`@instruction`** is a named pipeline (and optional body). Calling `@foo` runs that pipeline.
4. A **`$external`** is a built-in handler (LLM, image, file I/O, etc.).
5. **Instruction body** text is merged into `prompts[]` (see §6). Many `$` handlers **clear `prompts[]`** after they run — plan body placement accordingly.
6. **Parallel** `( @a, @b )` runs branches on copies and **joins** all arrays (appends).
7. **`%context`** stores bundles between steps; **`@` refs** call other instructions.

---

## 5. File skeleton

```ah
# Optional comment

@step_one: $clear -> $file('input.png')

@step_two: @step_one -> $image2image(model='sfw-v23')
Improve lighting and fix hands

@main: @step_two -> $save('out.png')

run @main
```

**Header forms:**

```ah
@name: action -> action -> action
Body line or block (applies to prompts)

@name:
action -> action
"""
Multiline body
"""

@name
Body only — no pipeline; body defines output behavior
```

**Multiline pipeline (brace block):**

```ah
@mix: {
  $file('a.wav') ->
  -> $music_separation(model=2stem) -> %stems ->
  -> ( $select(sounds=[0]), stems% -> $select(sounds=[1]) )
}
```

---

## 6. Action types (cheat sheet)

| Syntax | Meaning |
|--------|---------|
| `@other` | Run instruction `@other` |
| `@other[5]` | Run 5 times; **join** all outputs |
| `@other[inf]` | **Do not generate** — chat/cancel only |
| `$name(...)` | Call external `name` |
| `$name(...)[3]` | Repeat / variants (see external docs) |
| `( @a, @b )` | Parallel; join outputs |
| `for(filter){ body }` | Partition: filter matches → body; rest preserved |
| `zip(images, texts){ body }` | Per-index slices across arrays |
| `%name` | Store current bundle into session context `name`; pass input through |
| `name%` | Replace output with stored context |
| `%name%` | Merge into store, then output full context |
| `%%name` / `name%%` / `%%name%%` | Same as `%` but scoped to **current** `@` instruction only |
| `^callback` | **UI only** — do not use in generated batch scripts |

---

## 7. Instruction body (critical for LLM / image / music tasks)

When an instruction has a **body** (text after the header or in `"""` … `"""`):

| Case | Effect |
|------|--------|
| Body + existing `prompts[]` | Body is appended into **every** prompt link |
| Body + empty `prompts[]` | One new prompt link with body only |
| No body | `prompts[]` unchanged |

**Implication:** Put the user question or image brief in the body of the instruction that feeds `$llm`, `$image`, `$image2image`, etc.

```ah
@answer: $llm
What is the capital of France?

run @answer
```

**Implication:** If step A clears prompts and step B needs the brief, either put the body on B’s instruction or keep text in `texts[]` and use `$llm(add_texts=True)`.

---

## 8. Choosing the right tool

### 8.1 Decide first: `$llm` only or `$search` → `$llm`

Before writing the script, **analyze the user question**. Pick one path:

| Use **`$llm` only** (no search) | Use **`$search` → analyze → `$llm`** |
|--------------------------------|--------------------------------------|
| Greetings, thanks, small talk (`hi`, `hello`) | News, “latest”, “today”, current events, live data |
| Creative writing, opinions, brainstorming | Prices, weather, scores, stock, release dates, “who won …” |
| Timeless facts (math, algorithms, language grammar) | Facts that change (elections, CEOs, laws, version numbers, docs) |
| Explaining concepts the model already knows | “Find / look up / search / research …” |
| Rewriting, summarizing, or translating **text the user already gave** | Named entities you would verify on the web (people, products, places) |
| Generating `.ah`, code, or media pipelines | User wants sources, citations, or “use the web” |
| Task is fully specified in the message (no external facts needed) | Comparison of real-world options (“best X in 2026”, “vs”) |
| **Generate image, clip, video, or music** (see §8.2) | Answer is plain text/chat only |

When unsure: if a wrong answer would embarrass you because **reality changed since training**, use `$search`. If the user only needs reasoning or prose, use `$llm` only. If they want **media output** (image, clip, song, edited photo, animated video), use **`$image` / `$image_clip` / `$image2video` / `$music` / `$image2image`** — never `$llm` alone.

### 8.2 Media tasks — never use `$llm` alone

If the user asks to **generate**, **create**, or **make** an image, photo, clip, video, song, track, or to **edit/transform** an image, route to the matching external:

| User says (examples) | Use | Not |
|--------------------|-----|-----|
| “Generate an image of …” | `$llm` → `$texts_to_prompts` → `$image` | `$llm` only |
| “Generate / make an image clip about …” | images + `$music` → `$image_clip` | `$llm` only |
| “Create a video of …” / “animate this image” | `$image` → `$image2video` | `$llm` only |
| “Create a song about …” / “make music for …” | `$music` with style + lyrics | `$llm` only |
| “Edit this photo …” / “change the image …” | `$file` → `$image2image` + edit body | `$llm` only |

**Image clip** = slideshow-style video from **multiple images** + **audio track** (`$image_clip`). Pattern: generate images `[n]`, generate song with `$music`, merge with `$image_clip` → `$output`. See `examples/example_image_clip.ah`.

**Search query:** In `@search_step: $search`, the body should be a **short focused web query** you derive from the user (keywords, not the whole chat paragraph). The user’s full question goes in the **`$llm(add_texts=True)`** body so the model analyzes `__CONTENT__` from search results.

| User goal | Pattern | Example file |
|-----------|---------|----------------|
| Simple / timeless Q&A | `@x: $llm` + body | `examples/example_llm.ah` |
| Needs fresh or verified facts | `$search(limit=6, fetch_pages=True, fetch_max=3)` → `$json2texts` → `$texts2prompts` → `$llm(add_texts=True)` | §8 below, `examples/example_search.ah` |
| Math-heavy reasoning | `$math` | (see `externals/math/_description`) |
| Generate Python/code | `$code` + body | `examples/example_code.ah` |
| Generate another `.ah` | `$file('_lang_desc')` + `$ah_code_examples(folder='examples')` + `$code` | `examples/example_generate_ah_code.ah` |
| Text-to-image | `$llm` → `$texts_to_prompts` → `$image` | `examples/example_simple_image_generation.ah` |
| **Image clip** (images + music → video) | images `[n]` + `$music` → `$image_clip` | `examples/example_image_clip.ah` |
| Edit image | `@photo -> $image2image(...)` + body | `examples/example_image2image.ah` |
| Image → video | `@still -> $image2video(...)` + body | `examples/example_image2video.ah` |
| **Generate music / song** | style + lyrics → `$music` | `examples/example_image_clip.ah` (track step) |
| Describe image/video | `$image2text` | `examples/example_image2text.ah` |
| Load / save files | `$file`, `$folder`, `$save`, `$output` | `examples/example_combine_data.ah` |
| Return text to caller | `$prompts_to_texts` or body on `$llm` | `examples/example_return_output_text.ah` |
| Stems / voice / TTS | `$music_separation`, `$change_voice`, `$text2speech`, … | `examples/example_music_to_stems.ah`, etc. |
| Strip arrays | `$only(images, prompts)` | — |
| Pick indices | `$select(sounds=[0], texts=[1])` | — |

**`$search` pipeline (when analysis says web data helps):**

Prefer **`fetch_pages=True`** so result pages are downloaded, cleaned (nav/ads/scripts removed), and merged into the JSON (`page_text`, `--- page ---` in `text`).

```ah
@search_step: $search(limit=6, fetch_pages=True, fetch_max=3)
UK renewable energy policy 2026

@answer: @search_step -> $json2texts -> $texts2prompts -> $llm(add_texts=True)
What is the UK renewable energy policy in 2026?
Use __CONTENT__ (search results include page_text and --- page --- sections). Answer from those sources; say if results are weak or conflicting.
Return only the final result without comments, clarifications, or explanations.

run @answer
```

- **`@search_step` body** — short focused **web query** you derive from the user (`UK renewable energy policy 2026`), not a placeholder.
- **`@answer` body** — user’s question verbatim (or close paraphrase) plus instructions to read `__CONTENT__`.

Snippet-only (no page download) when speed matters and snippets are enough:

```ah
@search_step: $search(limit=5)
<focused search query>

@answer: @search_step -> $json2texts -> $texts2prompts -> $llm(add_texts=True)
<user question>
Read the search results in __CONTENT__. Answer the user.
Return only the final result without comments, clarifications, or explanations.

run @answer
```

Other args: `lang='en'`, `region='us'`, `site='gov.uk'` — see `externals/search/_description`.

**`$llm(add_texts=True)`:** Puts prior `texts[]` into a `__CONTENT__` section of the prompt — use after `$search`, `$code`, or other text-producing steps.

**`$code`:** Consumes `prompts[]`, `texts[]`, `files[]`; outputs new `texts[]` (completions). Load references with:

```ah
@gen: {
  $file('_lang_desc', source_path=True)
  -> $ah_code_examples(folder='examples', per_usecase=20)
  -> $code
}
Write a script that …

run @gen
```

**`$ah`:** Executes `.ah` source from `texts[]`. Use for **nested** runs only, not as the user-facing top-level entry.

---

## 9. Canonical patterns (copy and adapt)

### 9.1 Minimal LLM answer (timeless or social — no search)

Use when §8.1 says **no search** (e.g. greetings, creative, math, codegen). **One body line with the user’s exact words**, then the return-only line:

```ah
@answer: $llm
<paste user message here as plain text>
Return only the final result without comments, clarifications, or explanations.

run @answer
```

Example for user message `hi`:

```ah
@answer: $llm
hi
Return only the final result without comments, clarifications, or explanations.

run @answer
```

### 9.2 LLM with longer inline body (no triple quotes)

```ah
@answer: $llm
Explain quantum tunneling in two paragraphs.
Return only the final result without comments, clarifications, or explanations.

run @answer
```

### 9.3 Search with page fetch, then analyze (default for web lookup)

Use when §8.1 says **search**. Copy this shape; replace the search query line and the `$llm` question line for the user’s topic.

```ah
@search_step: $search(limit=6, fetch_pages=True, fetch_max=3)
UK renewable energy policy 2026

@answer: @search_step -> $json2texts -> $texts2prompts -> $llm(add_texts=True)
What is the UK renewable energy policy in 2026?
Use __CONTENT__ (search results include page_text and --- page --- sections). Answer from those sources.
Return only the final result without comments, clarifications, or explanations.

run @answer
```

Another topic — user asked *“What are the most popular UK birds right now?”*:

```ah
@search_step: $search(limit=6, fetch_pages=True, fetch_max=3)
most popular UK garden birds 2026

@answer: @search_step -> $json2texts -> $texts2prompts -> $llm(add_texts=True)
What are the most popular UK birds right now?
Use __CONTENT__ (search results include page_text and --- page --- sections). List the main species; note if sources disagree.
Return only the final result without comments, clarifications, or explanations.

run @answer
```

User asked: *“hi”* → use §9.1, **not** this pattern.

### 9.4 Return fixed text (no model)

```ah
@return_txt: $prompts_to_texts
The exact message to return.

run @return_txt
```

### 9.5 Image generation with style refs

Split **style** into small `@` instructions (reused fragments), **topic** in body, then `$llm` → `$texts_to_prompts` → `$image`:

```ah
@quality: $list
default

@style_fragment: @quality
High resolution, best quality, photorealistic

@prompts: @style_fragment -> $llm -> $texts_to_prompts
Portrait of a red fox in snow, golden hour.
Return only the final prompt text, under 55 words.

@images: @prompts -> $image(model='default')

run @images
```

### 9.6 Image edit (`$image2image`)

```ah
@load: $file('photo.png')

@edit: @load -> $image2image(model='sfw-v23', steps=8)
Remove blemishes and improve natural lighting.

run @edit
```

### 9.7 Image clip (`$image_clip` — images + music)

```ah
@quality: $list
High resolution, best quality, photorealistic, cinematic

@clip_images: @quality -> $llm -> $texts_to_prompts -> $image(model='default')[8]
Create image prompts for a short clip about rabbits in a meadow.
Return only the final result without comments, clarifications, or explanations.

@music_style: $list
irish traditional

@lyrics_text: $llm
Write short song lyrics about rabbits in a meadow.
Return only the final result without comments, clarifications, or explanations.

@track: (@music_style, @lyrics_text) -> $music(model='st', guidance_scale=4.0, vocal_language='en', inference_steps=50)

@gen_clip: (@clip_images, @track) -> $image_clip -> $output

run @gen_clip
```

Do **not** answer “generate an image clip” with `@answer: $llm` — that produces text, not a clip.

### 9.8 Context `%` for multi-step audio

```ah
@split: $file('track.wav') -> $music_separation(model=2stem) -> %stems

@vocals: stems% -> $select(sounds=[0]) -> $voice_enhance

@mix: (@vocals, %stems) -> $join_stems

run @mix
```

(Order and indices depend on the separation model — check the relevant `externals/*/_description`.)

### 9.9 List fan-out (`@ref` args)

```ah
@models: $list
default, realistic

@items: $list
apple, banana

@run_all: $image(model=@models, subject=@items)
A still life photo.

run @run_all
```

Runtime expands to Cartesian product of list values.

### 9.10 Zip: per-image text overlay

```ah
@base: @prompts -> $image

@final: @base -> zip(images, texts){ $draw_text(size=80) }

run @final
```

---

## 10. Prompt hygiene for generated scripts

When the downstream step is an LLM or coder model, include in the **body** or prompt:

- **Task** — what to do, precisely.
- **Output shape** — e.g. “Return only JSON”, “Return only the .ah script”, “No markdown”.
- **Constraints** — length, tone, forbidden features (`[inf]`, `$ah` at top level).
- **Examples** — optional one-liner referencing `examples/example_*.ah` by name when using `$code` to generate scripts.

For **code generation** tasks (writing `.ah`), also pass language reference files into `$code`:

```ah
@clear_answer:
Return only the final result without comments, clarifications, or explanations.

@ah_code: {
  @clear_answer
  -> $file('_lang_desc', source_path=True)
  -> $ah_code_examples(folder='examples', per_usecase=20)
  -> $code
}
<user task description here>
Don't use $ah for top-level tasks. Use the simplest pipeline.
Don't use infinite repeats.

run @ah_code
```

---

## 11. Common mistakes (avoid)

| Mistake | Why it fails | Fix |
|---------|----------------|-----|
| `\"\"\"` and `\n` in the file | Parser sees backslashes and wrong tokens; not valid body | Real newlines; plain body lines for short text |
| Placeholder `User question goes here` | LLM never sees the user’s `hi` | Put `hi` (or full user text) on its own body line |
| Body after a step that cleared `prompts[]` | Next `$llm` / `$image` sees empty prompts | Put body on the instruction that still has prompts, or use `add_texts=True` |
| Chaining two prompt consumers without middle step | Second gets no prompts | Insert `$texts_to_prompts` or body-only `@ref` |
| Using `@foo[inf]` | Hangs until cancel; wrong for batch jobs | Use finite `@foo[n]` or a single `@foo` |
| Top-level `$ah` | Nested session complexity | Inline the pipeline |
| Forgetting `run @x` | Script does nothing | Always add run target |
| `$image` without prompt body | Weak or empty generation | Add body or `prompts[]` from `$llm` |
| Wrong array for media | Handler expects `images[]` not `files[]` | Use `$file` / `$folder` (routes by extension) |
| `zip(a, b)` length mismatch | Silent truncation | Equal-length arrays or build in one step |
| Confusing `%track` and `@track` | Unrelated namespaces | Use consistent naming |

---

## 12. Externals quick reference

Full list: `_lang_desc` §6 and `externals/<name>/_description`.

**Often used in generated scripts:**

| External | Role |
|----------|------|
| `$llm` | General text generation |
| `$math` | Math-oriented LLM (Qwen3.6 GGUF) |
| `$code` | Code / script generation |
| `$search` | DuckDuckGo search → `texts[]` |
| `$json2texts`, `$texts2prompts`, `$prompts_to_texts` | Array conversions |
| `$file`, `$folder`, `$save`, `$output` | I/O |
| `$image`, `$image2image`, `$image2video` | Image / video GPU pipelines |
| `$image2text` | Vision captioning |
| `$clear`, `$only`, `$select`, `$list`, `$pass` | Utilities |
| `$ah` | Run nested `.ah` from `texts[]` (subtasks only) |

**Prompt-consuming** (clear `prompts[]` on output): `$llm`, `$image`, `$image2image`, `$image2video`, `$music`, `$code`, `$comfy`, and most media handlers.

**Repeat `[n]`:** `$llm`, `$image`, `$music`, `$code` handle repeat internally; others fan out to `n` parallel invocations.

---

## 13. Naming and structure conventions

- **`@` names:** `snake_case` or short phrases: `@good_quality`, `@bird_images`, `@clear_answer`.
- **One concern per instruction:** `@load_files`, `@enhance`, `@save` rather than one giant block.
- **Entry instruction:** name clearly: `@main`, `@run`, `@answer`, or task-specific `@bird_images`.
- **Reusable fragments:** small `@` instructions with only a style line in the body (no `$`), composed with `->`.
- **GPU-heavy steps:** keep pipelines linear; avoid unnecessary parallel GPU calls unless intentional.

---

## 14. Validation checklist (before you output)

- [ ] No `\n`, `\"`, or escaped `"""` in the file
- [ ] Chosen `$llm`-only vs `$search` → `$llm` using §8.1 (not always the same path)
- [ ] User’s message appears verbatim in the `$llm` body (search query is separate, focused)
- [ ] Valid syntax: `@name:`, `->`, `$name(args)`, `run @name`
- [ ] **Last line is `run @entry`** (e.g. `run @answer`) — required or the script never starts
- [ ] No `[inf]`
- [ ] No top-level `$ah` unless user explicitly required nested execution
- [ ] User task captured in body or correct array
- [ ] Prompt-consuming steps have prompts when needed
- [ ] Output instructions match user format (“JSON only”, “script only”, etc.)
- [ ] Paths and model names quoted where required
- [ ] Script is minimal — no unused `@` instructions

---

## 15. Where to look in the repo

| Resource | Purpose |
|----------|---------|
| `_lang_desc` | Complete language reference |
| `examples/example_*.ah` | Working patterns by task |
| `externals/<name>/_description` | Per-external args and I/O |
| `chat/chat.ah` | Interactive chat (uses `^` callbacks and `@chat[inf]` — not a template for batch scripts) |
| `tests/test_context.py`, `tests/test_parallel_join.py` | Edge-case behavior |

When unsure about an external’s arguments, prefer the matching `externals/<name>/_description` over guessing.

"""Prompt-merge case definitions: script, emulate_external, expected compact dict."""

from __future__ import annotations

PROMPT_MERGE_CASES: list[dict] = [
    {
        "name": "test_ref_then_image_prompt_concatenated",
        "script": """
@good_quality_image
High Angle Shot, Best Quality

@image: @good_quality_image -> $image
running woman

run @image
""",
        "emulate_external": ["image"],
        "observe_external": "image",
        # One joined prompt link; @image body before @ref body; no pending changes.
        "expected": {"prompts": ["running woman\nHigh Angle Shot, Best Quality"]},
    },
    {
        "name": "test_body_prepended_before_llm",
        "script": """
@task: $llm
instruction body for llm

run @task
""",
        "emulate_external": ["llm"],
        "observe_external": "llm",
        "expected": {"prompts": ["instruction body for llm"]},
    },
    {
        "name": "test_ref_then_image2image_body_on_pipeline_input",
        "script": """
@source: $image
a portrait photo

@edit: @source -> $image2image
make the background blue

run @edit
""",
        "emulate_external": ["image", "image2image"],
        "observe_external": "image",
        "expected": {"prompts": ["make the background blue\na portrait photo"]},
    },
    {
        "name": "test_external_first_body_prepend",
        "script": """
@edit: $image2image
remove extra fingers

run @edit
""",
        "emulate_external": ["image2image"],
        "observe_external": "image2image",
        "expected": {"prompts": ["remove extra fingers"]},
    },
    {
        "name": "test_ddd_outputs_two_composed_prompts",
        "script": """
@aaa
test1

@bbb
test2

@ccc
test3

@ddd: (@aaa, @bbb) -> @ccc

run @ddd
""",
        "emulate_external": [],
        "expected": {"prompts": ["test1\ntest3", "test2\ntest3"]},
    },
    {
        "name": "test_ref_chain_composes_all_bodies",
        "script": """
@parent
parent line

@addon
addon line

@child: @parent -> @addon
child line

run @child
""",
        "emulate_external": [],
        "expected": {"prompts": ["child line\nparent line\naddon line"]},
    },
    {
        "name": "test_comfy_input_keeps_two_prompt_links",
        "script": """
@video_prompt: $llm[2] -> $texts_to_prompts
shot prompt template

@realistic: @video_prompt -> $comfy(json='wf.json')

run @realistic
""",
        "emulate_external": ["llm", "comfy"],
        "observe_external": "comfy",
        "custom_setup": "video_comfy",
        "expected": {
            "prompts": [
                "[emulated $llm model=default variant=0]\n"
                "system: \n"
                "prompt:\n"
                "shot prompt template",
                "[emulated $llm model=default variant=1]\n"
                "system: \n"
                "prompt:\n"
                "shot prompt template",
            ]
        },
    },
    {
        "name": "test_both_comfy_get_same_prompt_and_image",
        "script": """
@good_quality_image
High Angle Shot, Best Quality

@image: @good_quality_image -> $image
running woman

@realistic_style
make this image in the realistic style

@realistic: @image -> @realistic_style -> (
  $comfy(port=8000, json='Qwen-Rapid-AIO_4.json'),
  $comfy(port=8000, json='Qwen-Rapid-AIO_4.json')
)

run @realistic
""",
        "emulate_external": ["image", "comfy"],
        "observe_external": "comfy",
        "custom_setup": "parallel_comfy",
        "expected": {
            "images": ["<image>"],
            "prompts": ["make this image in the realistic style"],
        },
    },
    {
        "name": "test_image_gets_four_separate_prompts_with_suffix",
        "script": """
@bird_names: $llm[4] -> $texts2prompts
name a UK garden bird

@bird_images: @bird_names -> $image
create a beautiful image of the bird in the UK Garden

run @bird_images
""",
        "emulate_external": ["llm", "texts2prompts", "image"],
        "observe_external": "image",
        "expected": {
            "prompts": [
                "[emulated $llm model=default variant=0]\n"
                "system: \n"
                "prompt:\n"
                "create a beautiful image of the bird in the UK Garden\n"
                "name a UK garden bird",
                "[emulated $llm model=default variant=1]\n"
                "system: \n"
                "prompt:\n"
                "create a beautiful image of the bird in the UK Garden\n"
                "name a UK garden bird",
                "[emulated $llm model=default variant=2]\n"
                "system: \n"
                "prompt:\n"
                "create a beautiful image of the bird in the UK Garden\n"
                "name a UK garden bird",
                "[emulated $llm model=default variant=3]\n"
                "system: \n"
                "prompt:\n"
                "create a beautiful image of the bird in the UK Garden\n"
                "name a UK garden bird",
            ]
        },
    },
    {
        "name": "test_body_only_instruction_writes_output_prompts",
        "script": """
@prompt_only
hello from body

@runner: @prompt_only

run @runner
""",
        "emulate_external": [],
        "expected": {"prompts": ["hello from body"]},
    },
    {
        "name": "test_pipeline_body_is_llm_input",
        "script": """
@task: $llm
instruction for pipeline

run @task
""",
        "emulate_external": ["llm"],
        "observe_external": "llm",
        "expected": {"prompts": ["instruction for pipeline"]},
    },
    {
        "name": "test_body_compose_preserves_other_arrays",
        "script": """
@with_image: $pass
out
""",
        "emulate_external": [],
        "kind": "compose",
        "expected": {"images": ["<image>"], "prompts": ["out"]},
    },
    {
        "name": "test_rule1_body_concatenated_with_every_input_prompt",
        "script": """
@two_prompts
alpha
beta

@runner: @two_prompts
suffix
""",
        "emulate_external": [],
        "kind": "compose",
        "expected": {"prompts": ["alpha\nsuffix", "beta\nsuffix"]},
    },
    {
        "name": "test_rule2_body_added_when_prompts_empty",
        "script": """
@runner
only body
""",
        "emulate_external": [],
        "kind": "compose",
        "expected": {"prompts": ["only body"]},
    },
    {
        "name": "test_rule2_body_added_when_prompt_links_are_blank",
        "script": """
@blank_prompt
   

@runner: @blank_prompt
only body
""",
        "emulate_external": [],
        "kind": "compose",
        "expected": {"prompts": ["only body"]},
    },
    {
        "name": "test_rule3_no_body_passes_through_prompts",
        "script": """
@keep
keep me

@runner: @keep
""",
        "emulate_external": [],
        "kind": "compose",
        "expected": {"prompts": ["keep me"]},
    },
    {
        "name": "test_rule3_no_body_empty_prompts_stays_empty",
        "script": """
@runner
""",
        "emulate_external": [],
        "kind": "compose",
        "expected": {},
    },
]

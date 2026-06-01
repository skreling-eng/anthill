# Old Issue: `$image2image` Prompt + Performance Regressions (2026-06-01)

## Summary
- `$image2image` had two major regressions:
  - Prompt text did not reliably affect output.
  - Runtime became minutes instead of ~20s, especially for `[repeat]` variants.
- Root causes were a mix of node input wiring, worker/runtime backend choice, VRAM policy, and seed/repeat behavior.

## User-Visible Problems
- Prompt appeared in logs but image edits were weak/incorrect.
- Runs stuck at `KSampler 0/4` for long periods.
- `repeat=5` felt much slower than expected.
- Some variants reused the same seed unintentionally.
- Later variants in a repeat batch could become much slower than sample 1.

## Impact
- Unreliable edit quality (prompt not applied correctly).
- High latency and poor UX.
- Hard to diagnose due to mixed signals:
  - `comfy_kitchen` installed but CUDA fast path partially disabled.
  - Logs implied healthy GPU load while activations had too little headroom.

## Root Causes

### 1) Prompt not forwarded in handler path
- Comfy `PromptExecutor` only passes keys declared in `INPUT_TYPES`.
- Anthill handler for `TextEncodeQwenImageEditPlus` initially missed required input schema.
- Result: prompt could be dropped in handler execution.

### 2) Slow backend/path selection
- Wrong or suboptimal execution path for this workflow caused unnecessary overhead.
- Warm worker/environment mismatches made behavior differ from ComfyUI baseline.

### 3) `comfy_kitchen` CUDA backend incorrectly disabled on cu129
- Version gate treated `torch.version.cuda=12.9` as unsupported (`<13`), disabling CUDA fast path.
- This forced slower FP8 behavior despite kitchen package availability.

### 4) VRAM residency policy overloaded GPU
- Full UNet residency plus insufficient activation headroom caused step-time spikes/paging.
- `HIGH_VRAM` defaults and forceful loading were too aggressive for 20–24GB-class cards.

### 5) Repeat fast path seed and stability issues
- When seed not explicitly provided, variants could reuse seed unexpectedly.
- Later variants could degrade in speed if model residency drifted between samples.

## Fixes Implemented

### Prompt correctness
- Added proper `INPUT_TYPES` for `TextEncodeQwenImageEditPlus` handler.
- Ensured reference latent handling uses append semantics where needed.

### Runtime/executor
- Kept legacy topo executor as default for simple Qwen i2i path.
- Retained PromptExecutor where required by complex workflows.

### Kitchen/CUDA gate
- Updated compatibility logic so cu128/cu129 are treated as supported for kitchen CUDA backend.
- Added clearer runtime logging:
  - Whether kitchen package is present.
  - Whether kitchen CUDA backend is enabled/disabled.

### VRAM and model loading
- Default i2i VRAM profile changed to `normal` on GPUs <32GB; `high` only on large VRAM.
- Avoided unnecessary force-full-load patterns.
- Added headroom-aware model loading and warnings when free VRAM is too low after UNet prep.

### Repeat path
- Encode-once fast-repeat retained.
- Seed derivation corrected so variants get distinct seeds when user seed is omitted.
- UNet prep is refreshed per variant to avoid late-sample slowdowns.

## Verification Signals (Expected)
- Worker logs show:
  - `comfy_kitchen cuda backend enabled`
  - `comfy vram_state=NORMAL_VRAM` (for <32GB cards unless overridden)
- Fast-repeat logs show distinct seeds per sample:
  - `sample 1/5 seed=X`
  - `sample 2/5 seed=X+1`
- Sample timings remain stable across variants (no huge jump on sample 2+).

## Prevention Checklist (Use Before Future Refactors)
- **Node handlers**
  - If using `PromptExecutor`, every handler must declare full `INPUT_TYPES`.
- **Environment**
  - Confirm worker Python path and kitchen CUDA backend status at startup.
- **VRAM**
  - Do not default to `HIGH_VRAM` for ~20–24GB cards on this workflow.
  - Keep activation headroom (target >= ~3GB free after UNet prep).
- **Repeat**
  - Always verify per-variant seed uniqueness when seed omitted by user.
  - Run at least one repeat test (`repeat=5`) and compare sample 1 vs sample 2 latency.
- **Logging**
  - Keep high-signal logs for:
    - vram_state
    - UNet loaded bytes / total bytes
    - free VRAM
    - per-sample KSampler elapsed time

## Quick Triage Playbook
1. Check worker python and kitchen CUDA backend status.
2. Check `vram_state`, UNet loaded bytes, and free VRAM after prep.
3. Check if sample seeds differ across variants.
4. Compare sample 1 vs sample 2 KSampler duration.
5. If needed, lower memory pressure:
   - reduce resolution,
   - switch sampler (`AH_IMAGE2IMAGE_SAMPLER=euler` for diagnostics),
   - lower VRAM mode.


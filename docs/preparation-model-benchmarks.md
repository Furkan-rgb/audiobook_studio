# Narration-preparation model benchmarks

This document records manually reviewed preparation-model benchmarks. Raw run
artifacts are generated under `output/benchmarks/` and are intentionally not
tracked by Git.

## Gemma 4 comparison — 2026-07-18

The benchmark compared the locally installed Ollama models using identical
source text and preparation settings.

### Test scope

| Setting | Value |
|---|---|
| Source | `reparative-therapy-nicolosipdf.pdf` |
| Chapters selected | 5 |
| Chapters represented before the unit cap | Preface; Introduction; Chapter 1; Chapter 2 |
| Prose units processed per run | 8 |
| Source characters per run | 18,856 |
| Repetitions per model | 2 |
| Preparation-cache reuse | Disabled |
| Temperature / seed | `0.0` / `42` |
| Output contract | Structured JSON |

The eight-unit global cap represented one Preface unit, two Introduction
units, two Chapter 1 units, and three Chapter 2 units. Every model received the
same units. Models were run in round-robin order and unloaded between runs.

### Results

| Model | Successful runs | Mean wall time | Mean provider time | Lexical retention | Citation-target similarity | Citation-shaped characters | Paragraphs preserved | Repeat consistency |
|---|---:|---:|---:|---:|---:|---:|:---:|---:|
| `gemma4:12b` | 2/2 | 78.42 s | 77.99 s | 99.7% | 99.8% | 478 → 0 | Yes | 100.0% |
| `gemma4:26b` | 2/2 | 45.87 s | 45.44 s | 99.7% | 99.0% | 478 → 0 | Yes | 100.0% |
| `gemma4:31b` | 2/2 | 148.40 s | 147.98 s | 99.8% | 99.9% | 478 → 0 | Yes | 100.0% |

Timing is specific to the local Ollama installation and hardware. In this run,
the 26B model was unexpectedly faster than 12B, so parameter count alone should
not be treated as a speed prediction.

### Manual quality review

| Model | Finding | Decision |
|---|---|---|
| `gemma4:12b` | Removed long bibliographic lists, preserved meaning and paragraph structure, and repaired obvious extraction artifacts. No substantive errors were found in the reviewed sample. | Recommended default |
| `gemma4:26b` | Removed citation years but retained long author-name lists in the Preface. It also changed “relational” to “non-relational” and dropped “time” from “enough time and education.” | Rejected despite fastest timing |
| `gemma4:31b` | Closely matched 12B and correctly repaired `DSM-11` to `DSM-II`, but missed one `fearful-ness` extraction artifact and took about 1.9 times as long as 12B. | Good but unnecessary for this task |

The automatic citation-shape metric reported complete removal for all three
models because author-only lists no longer matched the author-year pattern.
Manual review was therefore decisive in rejecting 26B. Automatic scores are
useful warning signals, but they do not establish semantic fidelity or
listening quality by themselves.

### Conclusion

`gemma4:12b` is the production default. Its prepared text was almost identical
to the 31B output, deterministic across both repetitions, and materially safer
than the 26B output. It also has the smallest model footprint, although it was
not the fastest model in this particular benchmark.

### Reproduce

```bash
.venv/bin/audiobook benchmark \
  --pdf reparative-therapy-nicolosipdf.pdf \
  --models gemma4:12b gemma4:26b gemma4:31b \
  --preview-chapters 5 \
  --preview-units 8 \
  --repetitions 2 \
  --output-dir output/benchmarks/gemma4_extended_5chapters_8units_2runs
```

The command writes the generated comparison report, machine-readable metrics,
and per-run reading copies to the specified output directory.

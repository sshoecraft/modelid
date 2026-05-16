# identify.py — model identifier

Probes an LLM backend through a Claude-Code-compatible CLI and scores the
responses to guess which model is behind the wall.

## Architecture

Three layers, all in one file for now:

1. **PROBES** — a list of dicts, each `{id, category, prompt}`. Categories:
   - `self_report` — asks the model to name itself
   - `cutoff` — date-bounded facts (Pope Francis death, 2024 election,
     iPhone model, most-recent Claude/OpenAI release)
   - `tokenizer` — character-level tasks that expose BPE behavior
     (strawberry r-count, string reversal, letter count)
   - `style` — creative/explanatory output, scanned for stylistic tells
   - `refusal` — edgy-but-harmless prompts that probe the refusal surface
   - `shibboleth` — lab-specific knowledge (Constitutional AI, InstructGPT)

2. **analyze(pid, category, response)** — the tagger. Per-category regex
   and structural checks emit string tags like `claims:claude`,
   `cutoff:post_2025_04`, `tokenizer:strawberry_classic_fail`,
   `style:em_dash`.

3. **FAMILY_EVIDENCE + CUTOFF_WINDOWS** — hand-tuned maps. Each tag
   contributes weighted evidence to one or more family hypotheses
   (anthropic, openai, google, meta, mistral, xai, chinese_lab). Cutoff
   tags separately narrow the training-data window.

## Transport

Shells out to the user-supplied binary:

    <path> <passthrough-args> -p "<probe>"

captures stdout, and runs the analyzer. One subprocess per probe, fresh
session each time.

### Argument forwarding

`--path` is the one script-owned required flag. Everything else is either
a script flag (see `--help`) or gets forwarded to the binary:

- Flags argparse doesn't recognize fall through automatically.
- A bare `--` marks explicit end-of-script-flags; everything after goes
  to the binary unconditionally.

Example:

    python3 identify.py --path /usr/local/bin/claude --model claude-sonnet-4-6
    python3 identify.py --path claude -- --verbose --add-dir /tmp

## Scoring

Each probe emits 0–N tags. Each tag contributes signed weights to family
scores. Final ranking is sum of weights per family. Positive margin over
#2 is the confidence signal.

Cutoff probes additionally produce an estimated training-data window, by
picking the most-recent `cutoff:post_YYYY_MM` tag present.

## Known confounders

- **System prompt leakage:** When the binary is Claude Code pointed at a
  non-Claude backend, Claude Code's own system prompt tells the model "you
  are Claude Code", which contaminates self-report probes. Self-report is
  weighted lower than cutoff/style/tokenizer for this reason.
- **Project CLAUDE.md:** If the binary loads a CLAUDE.md from the working
  directory, that context also shapes responses. Run from a clean dir
  (e.g. `cd /tmp && python3 /path/to/identify.py ...`) to minimize this.
- **MCP servers / hooks:** Can add preamble or alter behavior. Disable if
  possible, or pass flags through to suppress them.
- **Cutoff drift:** Models get refreshed silently by vendors. The cutoff
  windows in the script are best estimates as of early 2026; update
  `CUTOFF_WINDOWS` and the per-probe tag emission as the landscape shifts.

## Extending

- Add probes: append to `PROBES` and extend `analyze()` for the new
  category (or reuse an existing one).
- Add families: add the name to the `families` list in `main()` and to
  the affected `FAMILY_EVIDENCE` rows.
- External probe set: pass `--probes custom.json` with the same
  `[{id, category, prompt}, ...]` shape.

## Open-weights detection (v0.2.0)

Wrappers like Claude Code can route to *any* backend behind an
Anthropic-API-compatible proxy. When the backend is an open-weights model
(Gemma, Llama, Mistral, Qwen, ...) served via an OpenAI-compatible
endpoint, several wrapper-induced signals (`wrapper:has_cache_control`,
`behavior:anthropic_reasoning_term`, `transport:id_text:openai_chatcmpl`)
falsely point at Anthropic/OpenAI because they describe the *wrapper*,
not the *weights*.

v0.2.0 adds:

- **Open-weights probes:** `self_open_weights`, `self_architecture`,
  expanded `self_family_forced` (now lists GEMMA, LLAMA, MISTRAL, QWEN,
  DEEPSEEK, PHI as explicit options).
- **Gemma shibboleths:** `shib_gemma` (asks about Gemma's hybrid
  local-global attention — only models with cutoff ≥ April 2026 know
  this), `shib_open_models`.
- **Text-vs-payload disambiguation:** `transport:id:openai_chatcmpl` is
  reserved for genuine JSON payload detection (high confidence). Text
  self-reports are tagged `transport:id_text:openai_chatcmpl` with much
  lower weight, because models served behind an OpenAI-compatible proxy
  will happily *claim* `chatcmpl-` IDs regardless of their actual
  underlying weights.
- **Wrapper de-weighting heuristic:** When direct open-weights evidence
  is present (`claims:gemma`, `version:gemma-*`, `version:llama-*`,
  `open_weights:yes`, `knows:gemma_architecture`, or any open-family
  claim), the script subtracts wrapper-induced Anthropic credit
  (`wrapper:has_cache_control`, `wrapper:tool_use_shape`, etc.) and the
  text-claimed-chatcmpl OpenAI credit. This handles the
  "Claude-Code-over-open-weights-via-OpenAI-proxy" deployment pattern.
- **Gemma / Llama version tags:** `version:gemma-2/3/4`,
  `version:llama-3/4` score directly to google/meta.

## History

- 0.1.0 — initial implementation: 7 categories, 16 probes, 7 families,
  cutoff-window estimation, passthrough-arg forwarding.
- 0.2.0 — open-weights detection. Added 4 probes (37 total), Gemma/Llama
  version tags, text-vs-payload transport disambiguation, and a wrapper
  de-weighting heuristic for Claude-Code-style wrapping over
  open-weights backends served via OpenAI-compatible endpoints.

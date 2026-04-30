# modelid

A black-box LLM model identifier. Probes an opaque LLM backend through a
Claude-Code-compatible CLI binary (anything that accepts `-p "<prompt>"`)
and scores the responses to guess which model family is behind the wall
and roughly when its training data was cut off.

You give it a binary. It does not need to know what the binary is. It
asks the binary a fixed battery of questions, scans the answers for
fingerprints, and prints a ranked guess.

## What it does

`identify.py` runs a fixed set of probe prompts through the target binary
and analyzes the responses across six dimensions:

| Category      | What it tests                                             |
|---------------|-----------------------------------------------------------|
| `self_report` | Asks the model to name itself / its lab                   |
| `cutoff`      | Date-bounded facts (Pope Francis, 2024 election, iPhone, latest Claude/OpenAI release) |
| `tokenizer`   | Character-level tasks that expose BPE behavior (the `strawberry` r-count, string reversal, letter counts) |
| `style`       | Stylistic tells in creative/explanatory output (em-dashes, eager openers, hedge phrases, bullet/bold habits) |
| `refusal`     | Edgy-but-harmless prompts that probe the refusal surface  |
| `shibboleth`  | Lab-specific knowledge (Constitutional AI, InstructGPT)   |

Each response is tagged (`claims:claude`, `cutoff:post_2025_04`,
`tokenizer:strawberry_classic_fail`, `style:em_dash`, ...). Tags
contribute signed weights to family hypotheses. The ranked sum is the
guess; the margin over #2 is the confidence signal. Cutoff tags
additionally narrow the training-data window.

Supported family hypotheses: `anthropic`, `openai`, `google`, `meta`,
`mistral`, `xai`, `chinese_lab`.

## Why

Useful when:

- You're handed a CLI tool and want to know what model it actually wraps.
- A vendor has silently swapped models behind an API and you want
  evidence beyond their changelog.
- You're testing whether a Claude Code-compatible binary is genuinely
  routing to the model it claims.

## Requirements

- Python 3.8+
- A target binary that accepts `-p "<prompt>"` and writes the response to
  stdout. The reference target is the `claude` CLI, but anything with
  the same argument shape works.

No third-party Python dependencies — standard library only.

## Usage

```
python3 identify.py --path /path/to/binary [binary-flags...]
```

Examples:

```bash
# Probe the local claude binary at its default model
python3 identify.py --path claude

# Pin a specific model and pass through extra flags
python3 identify.py --path claude --model claude-sonnet-4-6

# Anything after a bare -- is forwarded verbatim to the binary
python3 identify.py --path claude -- --verbose --add-dir /tmp

# JSON report instead of the human-readable summary
python3 identify.py --path claude --json --out report.json

# Run only a subset of probes
python3 identify.py --path claude --only cutoff_election,tok_strawberry

# Skip whole categories
python3 identify.py --path claude --skip-cats refusal,shibboleth

# Rate-limit between probes
python3 identify.py --path claude --sleep 1.5
```

### Script flags

| Flag           | Purpose                                                 |
|----------------|---------------------------------------------------------|
| `--path`       | Path to the binary (required)                           |
| `--timeout`    | Per-probe timeout in seconds (default 180)              |
| `--json`       | Emit full JSON report to stdout                         |
| `--out FILE`   | Also write the full JSON report to FILE                 |
| `--probes F`   | Use a custom probe set from JSON instead of built-ins   |
| `--only IDS`   | Comma-separated probe IDs to run (debug)                |
| `--skip-cats`  | Comma-separated categories to skip                      |
| `--sleep S`    | Sleep S seconds between probes (rate limiting)          |

Any flags argparse doesn't recognize before `--`, plus everything after
`--`, are forwarded to the binary.

## Output

Default human-readable output ranks the families with a small bar chart,
lists the cutoff evidence, prints any self-reported version strings, and
ends with a best-guess line:

```
============================================================
FAMILY SCORES
============================================================
  anthropic       +4.30  ########################################
  openai          +0.50  ####
  google          -1.00
  ...

CUTOFF EVIDENCE
  - cutoff:post_2025_12
  estimated training cutoff: 2025-12

BEST GUESS
  family: anthropic  (score +4.30, margin +3.80 over openai)
  cutoff: 2025-12
```

`--json` emits the full structured report including each probe's prompt,
response, latency, return code, and emitted tags.

## Custom probe sets

Pass `--probes custom.json` with the same shape as the built-in `PROBES`
list:

```json
[
  {"id": "my_probe", "category": "self_report", "prompt": "Who are you?"}
]
```

Tags are emitted by the existing `analyze()` function based on
`category`, so custom probes should reuse one of the built-in categories
or you'll need to extend `analyze()`.

## Known confounders

- **System prompt leakage.** When the binary is Claude Code pointed at a
  non-Claude backend, Claude Code's own system prompt tells the model
  "you are Claude Code", contaminating self-report probes. Self-report
  is intentionally weighted lower than cutoff/style/tokenizer.
- **Project `CLAUDE.md`.** If the binary loads a `CLAUDE.md` from the
  working directory, that context shapes responses. Run from a clean
  directory (e.g. `cd /tmp && python3 /path/to/identify.py ...`) to
  minimize this.
- **MCP servers / hooks.** Can add preamble or alter behavior. Disable
  if possible, or use passthrough flags to suppress them.
- **Cutoff drift.** Vendors silently refresh models. The windows in
  `CUTOFF_WINDOWS` and the per-probe tag emissions are best estimates as
  of early 2026 and need maintenance as the landscape shifts.

## Extending

- **Add probes**: append to `PROBES` in `identify.py` and extend
  `analyze()` if you're using a new category.
- **Add families**: add the name to the `families` list in `main()` and
  to the affected rows in `FAMILY_EVIDENCE`.
- **Tune weights**: `FAMILY_EVIDENCE` is hand-tuned. Adjust to taste; the
  effects are immediately visible in the bar chart.

See `docs/identify.md` for a deeper architectural walkthrough.

## License

MIT — see `LICENSE`.

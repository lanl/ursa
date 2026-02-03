# Summarizing Agent Example — Summarize a Directory

This example demonstrates `SummarizingAgent`, which summarizes **all eligible documents in a directory** into a **single unified output**.

Key features:
- Summarizes a directory of docs (N inputs → 1 summary)
- Paragraph-aware chunking with configurable overlap to fit model context
- Map/Reduce summarization: chunk notes → iterative merges → final structured summary
- Optional rewrite pass to scrub meta-language/segmentation if the model violates constraints

## How it works (high level)

1) **Select files** in `--input-dir` (optionally recursive), filtered by extension  
2) **Read** file contents via `read_file`  
3) **Chunk** text into deterministic, paragraph-aware chunks (with overlap)  
4) **Map:** each chunk → compact bullet notes  
5) **Reduce:** merge notes in batches until one summary remains  
6) **Rewrite (optional):** if output contains forbidden references/headings, rewrite to clean it up

### Chunking + overlap (as implemented)

Chunks are built from paragraphs when possible. If overlap is enabled, each chunk after the first is prefixed with the last `overlap_chars` from the previous chunk:

```
chunk1: [-------------- A ---------------]
chunk2: [--- tail(A) ---][------ B ------]
chunk3: [--- tail(B) ---][------ C ------]
```

Oversized paragraphs are split into fixed-size segments (also with overlap).

## Running the example
Some example below show how to use this example.

From this directory:

```bash
python summarizing_agent_example.py --help
```

Some common options:

```
python summarizing_agent_example.py --recurse
python summarizing_agent_example.py --input-dir ./inputs --output-path ./out/summary.txt
python summarizing_agent_example.py --mode synthesis
python summarizing_agent_example.py --max-files 50 --chunk-size-chars 10000 --chunk-overlap-chars 800 --reduce-batch-size 8
python summarizing_agent_example.py --show-tool-output
```

### Using a Gateway / Custom Endpoint
By default, the script expects an OpenAI-compatible endpoint and reads the API key from `OPENAI_API_KEY`.
To use a different base URL and API key env var:

```
python summarizing_agent_example.py \
  --base-url https://your.gateway.example/v1 \
  --api-key-env YOUR_GATEWAY_API_KEY
```

### Get sample input docs (public, safe downloads)

If you don’t have documents handy, you can grab a few short, public-domain texts from Project Gutenberg
and summarize them as a directory.

These three are small, varied, and interesting to synthesize (satire + horror + short fiction), and each is
listed as **Public domain in the USA** on Project Gutenberg. 

```bash
# From the examples directory
mkdir -p summarizing_agent_example_inputs/gutenberg
cd summarizing_agent_example_inputs/gutenberg

# A Modest Proposal (Jonathan Swift) — eBook #1080
wget -O a_modest_proposal_1080.txt https://www.gutenberg.org/ebooks/1080.txt.utf-8

# The Yellow Wallpaper (Charlotte Perkins Gilman) — eBook #1952
wget -O the_yellow_wallpaper_1952.txt https://www.gutenberg.org/ebooks/1952.txt.utf-8

# The Gift of the Magi (O. Henry) — eBook #7256
wget -O the_gift_of_the_magi_7256.txt https://www.gutenberg.org/ebooks/7256.txt.utf-8
```

Then, relocate to the examples directory and run this:
```bash
python summarizing_agent_example.py --input-dir summarizing_agent_example_inputs/gutenberg
```

It's worth noting that the results of **summarizing** 3 fiction novels in the way that
the default prompts are in the SummarizingAgent (all configurable) may be odd 
and produce weird results (e.g. Executive Summary, etc.)

This example is intended only as a starting point to demonstrate the functionality.

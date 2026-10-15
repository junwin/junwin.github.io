---
layout: default
title: Projects
permalink: /projects/
---

<h1>Projects</h1>

**[image_description](https://github.com/junwin/image_description)** — a
Python toolkit for my photography workflow: it generates descriptions and
alt text for images, publishes them to Mastodon, Pixelfed, Tumblr and
Bluesky, tracks engagement, and promotes the best photos to this site's
gallery.

## The galet stack

A set of provider-neutral Python packages for building LLM agent
applications — split by responsibility so each piece stays usable on its
own:

**[galet](https://github.com/junwin/galet)** — provider-agnostic LLM,
embedding and image-generation layer: chat and tool calling across OpenAI,
DeepSeek, Gemini, Mistral and Ollama, declarative model routing, and a
model catalog that resolves by capability.

**[galet-memory](https://github.com/junwin/galet-memory)** — the memory
layer for agent applications: episodic, semantic, procedural and working
memory contracts, with reusable SQLite and vector-backed persistence and
retrieval.

**[galet-prompt-builder](https://github.com/junwin/galet-prompt-builder)**
— prompt compilation: it assembles bounded prompts from instructions,
memory and retrieval candidates, applying per-section token budgets,
relevance selection and deduplication.

**galet-tools** — an MCP server exposing the galet stack as tools.
(private)

**This site** — a Jekyll/GitHub Pages site I publish to from the command
line, with an automated photo gallery that ranks and thumbnails itself.

# Mnemotree for Claude Code

Long-term memory for Claude Code that keeps what was said, word for word, in a
SQLite file on your machine. It calls no LLM and needs no API key.

```bash
claude plugin marketplace add kurcontko/mnemotree
claude plugin install mnemotree@mnemotree
```

Or, in one step on Claude Code 2.1.275 or later:

```
/plugin install mnemotree --marketplace kurcontko/mnemotree
```

It needs [uv](https://docs.astral.sh/uv/) and runs on macOS and Linux.

- After each turn it stores your prompt and Claude's reply, in the background.
- Before each prompt it adds up to three matching passages from earlier sessions
  in the same project, at most 500 tokens.
- It adds the `remember`, `recall`, `get` and `forget` tools for the same memory.

Memories are in `~/.mnemotree/memory.sqlite`, one namespace per project. The first
session downloads a 200 MB embedding model that runs locally.

[The guide](https://github.com/kurcontko/mnemotree/blob/main/docs/GUIDE.md#claude-code-plugin)
covers what is stored, the settings and the limits.

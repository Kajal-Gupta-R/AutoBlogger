# AutoBlogger

An automated blogging pipeline: a scheduled GitHub Action runs a **LangGraph** workflow that uses an LLM on **Groq** to write a technical ML/AI article every day, saves it as Markdown, and publishes it to a static site on GitHub Pages.

**Live site:** https://kajal-gupta-r.github.io/AutoBlogger/

## How it works

```
GitHub Actions (daily cron)
        |
        v
START -> pick_topic -> write_outline -> write_draft -> review -> save_article -> END
         (LangGraph workflow, shared BlogState)
        |
        v
frontend/articles/*.md + index.json  ->  committed to the repo
        |
        v
GitHub Pages deploy  ->  static site renders Markdown in the browser
```

1. **Schedule:** a cron-triggered GitHub Action starts the pipeline every day (it can also be run manually).
2. **Workflow:** LangGraph runs five nodes that share one typed state object. Each node does a single job and returns only the fields it changes.
3. **Generation:** `langchain-groq` calls an LLM on Groq for the outline, the draft, and an editor-style review pass.
4. **Storage:** the final article is saved as a dated Markdown file, and `index.json` lists every article, newest first.
5. **Publishing:** the bot commits the new files, and a second workflow deploys the `frontend/` folder to GitHub Pages.
6. **Frontend:** plain HTML, CSS and JavaScript fetch `index.json`, render articles with `marked`, and sanitize the HTML with `DOMPurify`.

## Tech stack

| Area | Tools |
|---|---|
| Workflow | LangGraph, LangChain Core |
| LLM | Groq via `langchain-groq` |
| Config | pydantic-settings, python-dotenv |
| Language and tooling | Python 3.12+, uv |
| Frontend | HTML, CSS, JavaScript, marked, DOMPurify |
| Automation | GitHub Actions, GitHub Pages |

## Design decisions

- **Typed shared state:** every node has the same simple signature (state in, updates out), so adding a step does not touch the others.
- **Config from the environment:** the API key and model name are read with pydantic-settings, so secrets stay out of the code. In CI the key comes from an encrypted GitHub secret.
- **Static site with a JSON index:** the frontend needs no backend, which makes hosting free and fast.
- **Sanitized output:** LLM-generated Markdown is converted to HTML and then sanitized before it is inserted into the page.
- **No repeated topics:** the topic picker skips topics already listed in `index.json`.

## Project structure

```
backend/
  config.py     settings and API key loading
  state.py      BlogState (shared workflow state)
  nodes.py      the workflow steps
  storage.py    saves articles and updates index.json
  graph.py      connects the nodes into a LangGraph
  run.py        entry point
frontend/
  index.html, app.js, style.css
  articles/     generated Markdown and index.json
.github/workflows/
  generate.yml  scheduled article generation
  deploy.yml    GitHub Pages deployment
```

## Run it locally

Requirements: Python 3.12+, [uv](https://docs.astral.sh/uv/), and a [Groq](https://console.groq.com/) API key.

```bash
git clone https://github.com/Kajal-Gupta-R/AutoBlogger.git
cd AutoBlogger
uv sync
```

Create a `.env` file (never commit it):

```
GROQ_API_KEY=your_key_here
MODEL_NAME=openai/gpt-oss-120b
```

Generate an article and view the site:

```bash
uv run python -m backend.run
uv run python -m http.server 8000 --directory frontend
```

Then open http://localhost:8000.

## Deploy your own copy

1. Fork the repo.
2. Add a repository secret named `GROQ_API_KEY`.
3. In Settings, set Pages to deploy from **GitHub Actions** and allow workflows to have **read and write** permissions.
4. Run the **Generate article** workflow, then **Deploy site**.

## Roadmap

- [ ] Retries and error handling around LLM calls
- [ ] Unit tests for the workflow nodes
- [ ] Dockerfile
- [ ] Support for a second LLM provider
- [ ] Error monitoring and LLM tracing

## Acknowledgements

The idea was inspired by [BlogBoard](https://github.com/KalyanM45/BlogBoard-AI-Blog-Generator) by Kalyan M. This project is an independent implementation written from scratch.

## Author

Kajal Gupta R | [GitHub](https://github.com/Kajal-Gupta-R) | [LinkedIn](https://linkedin.com/in/kajal-gupta-r-9425773ab)
const $ = (id) => document.getElementById(id);
const root = document.documentElement;
const homeEl = $("home");
const readerEl = $("reader");
const listEl = $("list");
const articleEl = $("article");
const metaEl = $("meta");
const pagerEl = $("pager");
const searchEl = $("search");
const emptyEl = $("empty");
const errorEl = $("error");
const progressEl = $("progress");
const topEl = $("to-top");
const SITE = "AutoBlogger | AI-generated ML deep dives";

let articles = [];

function fmt(date) {
  return new Date(`${date}T00:00:00`).toLocaleDateString("en-IN", {
    day: "numeric",
    month: "short",
    year: "numeric",
  });
}

function setTheme(theme) {
  root.dataset.theme = theme;
  try { localStorage.setItem("theme", theme); } catch {}
}

$("theme-toggle").addEventListener("click", () =>
  setTheme(root.dataset.theme === "dark" ? "light" : "dark")
);

async function loadIndex() {
  const res = await fetch("articles/index.json");
  if (!res.ok) throw new Error("Could not load the article list.");
  return res.json();
}

function renderStats() {
  $("stat-count").textContent = articles.length;
  $("stat-latest").textContent = articles.length ? fmt(articles[0].date) : "-";
}

function renderList() {
  const q = searchEl.value.trim().toLowerCase();
  const items = articles.filter((a) => a.title.toLowerCase().includes(q));
  listEl.innerHTML = "";
  emptyEl.hidden = items.length > 0;

  items.forEach((item, i) => {
    const li = document.createElement("li");
    li.style.animationDelay = `${Math.min(i, 8) * 0.06}s`;

    const link = document.createElement("a");
    link.href = `#${item.file}`;
    link.textContent = item.title;

    const meta = document.createElement("div");
    meta.className = "card-meta";
    const time = document.createElement("time");
    time.textContent = fmt(item.date);
    meta.append(time);
    if (item === articles[0]) {
      const badge = document.createElement("span");
      badge.className = "badge";
      badge.textContent = "NEW";
      meta.append(badge);
    }

    li.append(link, meta);
    listEl.append(li);
  });
}

function enhanceCode() {
  articleEl.querySelectorAll("pre").forEach((pre) => {
    const code = pre.querySelector("code");
    if (!code) return;
    if (window.hljs) hljs.highlightElement(code);

    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "copy";
    btn.textContent = "Copy";
    btn.addEventListener("click", async () => {
      try {
        await navigator.clipboard.writeText(code.innerText);
        btn.textContent = "Copied!";
      } catch {
        btn.textContent = "Press Ctrl+C";
      }
      setTimeout(() => (btn.textContent = "Copy"), 1500);
    });
    pre.append(btn);
  });
}

function pagerLink(entry, label, cls) {
  const a = document.createElement("a");
  a.href = `#${entry.file}`;
  a.className = cls;
  const small = document.createElement("small");
  small.textContent = label;
  const span = document.createElement("span");
  span.textContent = entry.title;
  a.append(small, span);
  return a;
}

function renderPager(file) {
  pagerEl.innerHTML = "";
  const i = articles.findIndex((a) => a.file === file);
  if (i === -1) return;
  const newer = articles[i - 1];
  const older = articles[i + 1];
  if (older) pagerEl.append(pagerLink(older, "← Older", "pager-link"));
  if (newer) pagerEl.append(pagerLink(newer, "Newer →", "pager-link right"));
}

function updateProgress() {
  const max = root.scrollHeight - root.clientHeight;
  progressEl.style.width = max > 0 ? `${(root.scrollTop / max) * 100}%` : "0%";
  topEl.hidden = root.scrollTop < 500;
}

function showHome() {
  homeEl.hidden = false;
  readerEl.hidden = true;
  progressEl.hidden = true;
  document.title = SITE;
  renderList();
}

async function showArticle(file) {
  homeEl.hidden = true;
  readerEl.hidden = false;
  progressEl.hidden = false;
  metaEl.textContent = "";
  pagerEl.innerHTML = "";
  articleEl.innerHTML =
    '<div class="skeleton"></div><div class="skeleton short"></div><div class="skeleton"></div>';

  const res = await fetch(`articles/${encodeURIComponent(file)}`);
  if (!res.ok) throw new Error("Article not found.");
  const markdown = await res.text();

  articleEl.innerHTML = DOMPurify.sanitize(marked.parse(markdown));
  enhanceCode();

  const entry = articles.find((a) => a.file === file);
  const words = markdown.trim().split(/\s+/).length;
  const minutes = Math.max(1, Math.round(words / 200));
  metaEl.textContent = `${entry ? fmt(entry.date) + " · " : ""}${minutes} min read`;
  document.title = `${entry ? entry.title : "Article"} | AutoBlogger`;
  renderPager(file);
  updateProgress();
}

async function route() {
  errorEl.hidden = true;
  if (!articles.length) {
    articles = await loadIndex();
    renderStats();
  }
  const file = decodeURIComponent(location.hash.slice(1));
  window.scrollTo(0, 0);
  if (file) await showArticle(file);
  else showHome();
}

function showError(err) {
  errorEl.textContent = err.message;
  errorEl.hidden = false;
}

searchEl.addEventListener("input", renderList);
topEl.addEventListener("click", () => window.scrollTo({ top: 0, behavior: "smooth" }));
window.addEventListener("scroll", updateProgress, { passive: true });
window.addEventListener("hashchange", () => route().catch(showError));

$("share").addEventListener("click", async (e) => {
  const btn = e.currentTarget;
  try {
    await navigator.clipboard.writeText(location.href);
    btn.textContent = "Link copied!";
  } catch {
    btn.textContent = "Copy from the address bar";
  }
  setTimeout(() => (btn.textContent = "Copy link"), 1500);
});

route().catch(showError);
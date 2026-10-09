const listEl = document.getElementById("list");
const articleEl = document.getElementById("article");
const backEl = document.getElementById("back");

async function loadIndex() {
  const res = await fetch("articles/index.json");
  if (!res.ok) throw new Error("Could not load the article list.");
  return res.json();
}

function renderList(items) {
  articleEl.hidden = true;
  backEl.hidden = true;
  listEl.hidden = false;
  listEl.innerHTML = "";

  for (const item of items) {
    const li = document.createElement("li");
    const link = document.createElement("a");
    link.href = `#${item.file}`;
    link.textContent = item.title;
    const time = document.createElement("time");
    time.textContent = item.date;
    li.append(link, " ", time);
    listEl.append(li);
  }
}

async function renderArticle(file) {
  const res = await fetch(`articles/${encodeURIComponent(file)}`);
  if (!res.ok) throw new Error("Article not found.");
  const markdown = await res.text();

  articleEl.innerHTML = DOMPurify.sanitize(marked.parse(markdown));
  listEl.hidden = true;
  backEl.hidden = false;
  articleEl.hidden = false;
}

async function route() {
  const file = decodeURIComponent(location.hash.slice(1));
  if (file) {
    await renderArticle(file);
  } else {
    renderList(await loadIndex());
  }
}

window.addEventListener("hashchange", () => route().catch(showError));
route().catch(showError);

function showError(err) {
  listEl.hidden = false;
  listEl.textContent = err.message;
}

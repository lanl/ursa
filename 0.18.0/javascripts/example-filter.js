function initializeExampleFilter() {
  const filter = document.querySelector(".example-tag-filter");
  const catalog = document.querySelector(".example-catalog");
  if (!filter || !catalog || filter.dataset.initialized === "true") return;

  filter.dataset.initialized = "true";
  const buttons = [...filter.querySelectorAll("[data-example-tag]")];
  const cards = [...catalog.querySelectorAll(":scope > ul > li")];
  const status = document.querySelector(".example-filter-status");

  function selectTag(tag) {
    let visible = 0;
    for (const card of cards) {
      const tags = [...card.querySelectorAll(".example-tags .md-tag")]
        .map((item) => item.textContent.trim());
      const filteredOut = Boolean(tag) && !tags.includes(tag);
      card.classList.toggle("example-card--filtered", filteredOut);
      if (!filteredOut) visible += 1;
    }
    for (const button of buttons) {
      button.setAttribute(
        "aria-pressed",
        String(button.dataset.exampleTag === tag),
      );
    }
    if (status) {
      status.textContent = tag
        ? `${visible} example${visible === 1 ? "" : "s"} tagged “${tag}”`
        : `${visible} examples`;
    }
  }

  for (const button of buttons) {
    button.addEventListener("click", () => selectTag(button.dataset.exampleTag));
  }
  selectTag("");
}

if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", initializeExampleFilter);
} else {
  initializeExampleFilter();
}

if (typeof document$ !== "undefined") {
  document$.subscribe(initializeExampleFilter);
}

/*
 * Which page uses which selector.
 *
 * A rule used by exactly one page can move into that page's component, where the next person
 * editing the page will actually find it. Rules shared by several pages are the theme and stay
 * global. Anything ambiguous counts as shared.
 */
import { readFileSync, writeFileSync } from "node:fs";
import pw from "playwright";

const BASE = process.env.SIGHTINDEX_URL ?? "http://127.0.0.1:8000";
const PAGES = [
  ["monitor", "/"],
  ["search", "/search"],
  ["observations", "/observations"],
  ["faces", "/faces"],
  ["reid", "/reid"],
  ["chat", "/chat-ui"],
];
const files = ["src/styles/console.css", "src/styles/polish.css"];

function topLevelRules(text) {
  const rules = [];
  let depth = 0, start = 0, headStart = 0, inComment = false;
  for (let i = 0; i < text.length; i++) {
    if (!inComment && text[i] === "/" && text[i + 1] === "*") { inComment = true; i++; continue; }
    if (inComment) { if (text[i] === "*" && text[i + 1] === "/") { inComment = false; i++; } continue; }
    if (text[i] === "{") { if (depth === 0) start = i; depth++; }
    else if (text[i] === "}") {
      depth--;
      if (depth === 0) {
        const head = text.slice(headStart, start).trim().replace(/^[\s\S]*\*\//, "").trim();
        if (head && !head.startsWith("@")) rules.push({ head, start: headStart, end: i + 1 });
        headStart = i + 1;
      }
    }
  }
  return rules;
}

const selectors = new Set();
const byFile = {};
for (const file of files) {
  byFile[file] = topLevelRules(readFileSync(file, "utf8"));
  for (const rule of byFile[file]) {
    for (const part of rule.head.split(",")) {
      const one = part.trim();
      if (one) selectors.add(one);
    }
  }
}

const browser = await pw.chromium.launch();
const usage = new Map([...selectors].map((s) => [s, new Set()]));
for (const [name, path] of PAGES) {
  const page = await browser.newPage({ viewport: { width: 1440, height: 900 } });
  await page.goto(BASE + path, { waitUntil: "networkidle", timeout: 60000 });
  await page.waitForTimeout(2000);
  const found = await page.evaluate((list) => {
    const alive = [];
    for (const selector of list) {
      const probe = selector
        .replace(/::?(hover|focus|focus-visible|active|disabled|checked|before|after|placeholder|first-line|selection|marker|backdrop)\b(\([^)]*\))?/g, "")
        .replace(/:{1,2}[a-z-]+\([^)]*\)/g, "")
        .trim();
      if (!probe || probe.startsWith(">")) { alive.push(selector); continue; }
      try { if (document.querySelector(probe)) alive.push(selector); }
      catch { alive.push(selector); }
    }
    return alive;
  }, [...selectors]);
  found.forEach((s) => usage.get(s).add(name));
  await page.close();
}
await browser.close();

const owners = {};
for (const file of files) {
  for (const rule of byFile[file]) {
    const parts = rule.head.split(",").map((p) => p.trim()).filter(Boolean);
    const pages = new Set();
    for (const part of parts) for (const p of usage.get(part) ?? []) pages.add(p);
    const key = pages.size === 1 ? [...pages][0] : pages.size === 0 ? "dead" : "shared";
    (owners[key] ??= []).push({ file, head: rule.head, start: rule.start, end: rule.end });
  }
}
for (const [key, list] of Object.entries(owners).sort((a, b) => b[1].length - a[1].length)) {
  console.log(`  ${key.padEnd(14)} ${String(list.length).padStart(4)} 条规则`);
}
writeFileSync("/tmp/css-owners.json", JSON.stringify(owners, null, 1));
console.log("\n  明细写入 /tmp/css-owners.json");

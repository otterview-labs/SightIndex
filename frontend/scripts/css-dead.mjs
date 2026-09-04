/*
 * Finds rules whose selectors match nothing on any page.
 *
 * Merging duplicates turned out to be a dead end here -- 133 of 137 are load-bearing overrides.
 * Deleting a rule that never matches anything is a different proposition: nothing can change,
 * because nothing was using it.
 *
 * Conservative by construction: state pseudo-classes are stripped before testing (:hover is
 * never true in a screenshot), every page is visited, and anything ambiguous counts as live.
 */
import { readFileSync, writeFileSync } from "node:fs";
import pw from "playwright";

const [mode, file] = process.argv.slice(2);
const BASE = process.env.SIGHTINDEX_URL ?? "http://127.0.0.1:8000";
const PAGES = ["/", "/search", "/observations", "/faces", "/reid", "/chat-ui"];
const css = readFileSync(file, "utf8");

// Brace-depth parse, not a regex: a regex over this file mangles @media and @keyframes, which
// is exactly what happened -- app-shell.css came out with its keyframes spliced into a rule and
// the pixel harness caught a 14% change on one page.
function topLevelRules(text) {
  const rules = [];
  let depth = 0, start = 0, headStart = 0, inComment = false;
  for (let i = 0; i < text.length; i++) {
    if (!inComment && text[i] === "/" && text[i + 1] === "*") { inComment = true; i++; continue; }
    if (inComment) { if (text[i] === "*" && text[i + 1] === "/") { inComment = false; i++; } continue; }
    if (text[i] === "{") {
      if (depth === 0) start = i;
      depth++;
    } else if (text[i] === "}") {
      depth--;
      if (depth === 0) {
        const head = text.slice(headStart, start).trim().replace(/^[\s\S]*\*\//, "").trim();
        if (head && !head.startsWith("@")) {
          rules.push({ head, start: headStart, end: i + 1 });
        }
        headStart = i + 1;
      }
    }
  }
  return rules;
}

const rules = topLevelRules(css);
const selectors = new Set();
for (const rule of rules) {
  for (const part of rule.head.split(",")) {
    const one = part.trim();
    if (one) selectors.add(one);
  }
}
console.log(`  选择器 ${selectors.size} 个，逐页检测…`);

const browser = await pw.chromium.launch();
const live = new Set();
for (const path of PAGES) {
  const page = await browser.newPage({ viewport: { width: 1440, height: 900 } });
  await page.goto(BASE + path, { waitUntil: "networkidle", timeout: 60000 });
  await page.waitForTimeout(2000);
  const found = await page.evaluate((list) => {
    const alive = [];
    for (const selector of list) {
      // Strip state pseudo-classes and pseudo-elements: they cannot be observed statically.
      const probe = selector
        .replace(/::?(hover|focus|focus-visible|active|disabled|checked|before|after|placeholder|first-line|selection|marker|backdrop)\b(\([^)]*\))?/g, "")
        .replace(/:{1,2}[a-z-]+\([^)]*\)/g, "")
        .trim();
      if (!probe || probe === ">" || probe.startsWith(">")) { alive.push(selector); continue; }
      try {
        if (document.querySelector(probe)) alive.push(selector);
      } catch {
        alive.push(selector);  // unparseable here means keep it
      }
    }
    return alive;
  }, [...selectors]);
  found.forEach((s) => live.add(s));
  await page.close();
}
await browser.close();

const dead = [...selectors].filter((s) => !live.has(s));
console.log(`  从未匹配到元素的选择器 ${dead.length} 个 / ${selectors.size}`);
console.log("  示例：" + dead.slice(0, 10).join("  |  "));

if (mode !== "apply") process.exit(0);

// Remove only whole top-level rules whose every selector is dead. Cut from the bottom so the
// offsets computed above stay valid.
const doomed = rules.filter((rule) => {
  const parts = rule.head.split(",").map((p) => p.trim()).filter(Boolean);
  return parts.length && parts.every((p) => !live.has(p));
});
let out = css;
for (const rule of [...doomed].reverse()) {
  out = out.slice(0, rule.start) + out.slice(rule.end);
}
writeFileSync(file, out.replace(/\n{3,}/g, "\n\n"));
console.log(`  已删除 ${doomed.length} 条规则（仅顶层，@media/@keyframes 内不动）`);

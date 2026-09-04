/*
 * Finds selectors defined more than once and reports which of them can be merged safely.
 *
 * Merging means moving earlier declarations down to the last occurrence. That changes nothing
 * only when no rule in between could win the property being moved. An earlier attempt skipped
 * this check and moved .media-grid{display:grid} past .hidden{display:none}, which broke the
 * page in a way only pixels caught.
 *
 *   node scripts/css-merge.mjs report src/styles/console.css
 *   node scripts/css-merge.mjs apply  src/styles/console.css
 */
import { readFileSync, writeFileSync } from "node:fs";

const [mode, file] = process.argv.slice(2);
const css = readFileSync(file, "utf8");

// Top-level rules only: anything inside @media has its own cascade and is left alone.
function topLevelRules(text) {
  const rules = [];
  let depth = 0, start = 0, inComment = false;
  for (let i = 0; i < text.length; i++) {
    if (!inComment && text[i] === "/" && text[i + 1] === "*") { inComment = true; i++; continue; }
    if (inComment) { if (text[i] === "*" && text[i + 1] === "/") { inComment = false; i++; } continue; }
    if (text[i] === "{") {
      if (depth === 0) start = i;
      depth++;
    } else if (text[i] === "}") {
      depth--;
      if (depth === 0) {
        const head = text.slice(0, start).split("}").pop().split("*/").pop().trim();
        rules.push({ selector: head, body: text.slice(start + 1, i), at: rules.length,
                     atRule: head.startsWith("@") });
      }
    }
  }
  return rules;
}

const rules = topLevelRules(css).filter((r) => !r.atRule && r.selector);
const properties = (body) =>
  body.split(";").map((d) => d.split(":")[0].trim().toLowerCase()).filter(Boolean);

// Specificity, good enough to compare like with like: ids, classes/attrs/pseudo-classes, elements.
function specificity(selector) {
  const one = selector.split(",")[0];
  return [
    (one.match(/#/g) || []).length,
    (one.match(/\.|\[|:(?!:)/g) || []).length,
    (one.match(/(^|[\s>+~])[a-z]/gi) || []).length,
  ];
}
const rank = (s) => s[0] * 10000 + s[1] * 100 + s[2];

const groups = new Map();
for (const rule of rules) {
  const key = rule.selector.replace(/\s+/g, " ");
  (groups.get(key) ?? groups.set(key, []).get(key)).push(rule);
}

const duplicated = [...groups.entries()].filter(([, list]) => list.length > 1);
let safe = 0, unsafe = 0;
const merges = [];
for (const [selector, list] of duplicated) {
  const last = list[list.length - 1];
  const own = rank(specificity(selector));
  let blocked = null;
  for (const earlier of list.slice(0, -1)) {
    const moving = new Set(properties(earlier.body));
    for (const between of rules.slice(earlier.at + 1, last.at)) {
      if (between.selector === selector) continue;
      if (rank(specificity(between.selector)) < own) continue;
      const touched = properties(between.body).filter((p) => moving.has(p));
      if (touched.length) { blocked = { between: between.selector, touched }; break; }
    }
    if (blocked) break;
  }
  if (blocked) { unsafe++; if (unsafe <= 6) console.log(`  阻塞 ${selector}  被 ${blocked.between} 的 ${blocked.touched.join(",")} 挡住`); }
  else { safe++; merges.push({ selector, list }); }
}
console.log(`\n  重复选择器 ${duplicated.length} 个：可安全合并 ${safe}，被层叠阻塞 ${unsafe}`);
console.log(`  可回收规则 ${merges.reduce((n, m) => n + m.list.length - 1, 0)} 条`);

if (mode !== "apply") process.exit(0);

// Rewrite: drop the earlier occurrences, fold their declarations into the last one.
let out = css;
const cuts = [];
for (const { list } of merges) {
  const last = list[list.length - 1];
  const merged = list.map((r) => r.body.trim().replace(/;?$/, ";")).join("\n  ");
  cuts.push({ rule: last, replacement: merged });
  for (const earlier of list.slice(0, -1)) cuts.push({ rule: earlier, remove: true });
}
// Apply from the bottom so offsets stay valid.
const located = cuts.map((c) => {
  const needle = `${c.rule.selector} {${c.rule.body}}`;
  return { ...c, index: out.indexOf(needle), needle };
}).filter((c) => c.index >= 0).sort((a, b) => b.index - a.index);
for (const cut of located) {
  const replacement = cut.remove ? "" : `${cut.rule.selector} {\n  ${cut.replacement}\n}`;
  out = out.slice(0, cut.index) + replacement + out.slice(cut.index + cut.needle.length);
}
writeFileSync(file, out.replace(/\n{3,}/g, "\n\n"));
console.log(`  已重写 ${file}`);

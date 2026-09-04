/*
 * Verifies every route src/api/client.ts calls exists in the backend's OpenAPI document with the
 * same HTTP method.
 *
 *   node scripts/check-api.mjs [openapi-url-or-file]
 */
import { readFileSync } from "node:fs";

const OPENAPI = process.argv[2] ?? "http://127.0.0.1:8000/openapi.json";

// A URL means whatever server happens to answer that port, which is not necessarily the code in
// this checkout -- a stale instance once reported three new routes as missing. A path reads the
// spec exported from this source tree instead.
async function loadSpec(target) {
  if (/^https?:\/\//.test(target)) return fetch(target).then((r) => r.json());
  return JSON.parse(readFileSync(target, "utf8"));
}
const source = readFileSync(new URL("../src/api/client.ts", import.meta.url), "utf8");

function extractCalls(text) {
  const found = [];
  const opener = /api<[^>]*>\(/g;
  for (const match of text.matchAll(opener)) {
    let depth = 1;
    let index = match.index + match[0].length;
    let quote = "";
    while (index < text.length && depth > 0) {
      const char = text[index];
      if (quote) {
        if (char === "\\") index++;
        else if (char === quote) quote = "";
      } else if (char === '"' || char === "'" || char === "`") quote = char;
      else if (char === "(" || char === "{" || char === "[") depth++;
      else if (char === ")" || char === "}" || char === "]") depth--;
      index++;
    }
    found.push(text.slice(match.index + match[0].length, index - 1));
  }
  return found;
}

const calls = [];
for (const args of extractCalls(source)) {
  const rawPath = args.match(/^\s*[`"]([^`"]+)[`"]/)?.[1];
  if (!rawPath) continue;
  const options = args.slice(args.indexOf(rawPath) + rawPath.length);
  const method = options.includes("jsonBody")
    ? "post"
    : (options.match(/method:\s*"(\w+)"/)?.[1] ?? "get").toLowerCase();
  const path = rawPath
    // A trailing ${queryString({...})} carries nested braces, so drop it to end-of-string
    // rather than trying to brace-match it.
    .replace(/\$\{queryString[\s\S]*$/, "")
    // Any remaining ${id} interpolation stands for one path parameter.
    .replace(/\$\{[^}]*\}/g, "{}")
    .replace(/\?.*$/, "");
  calls.push({ path, method, rawPath: rawPath.replace(/\s+/g, " ") });
}

const spec = await loadSpec(OPENAPI);
// Compare shapes, not parameter names: /api/streams/{id}/counts -> /api/streams/{}/counts
const shapes = new Map();
for (const [path, ops] of Object.entries(spec.paths)) {
  const shape = path.replace(/\{[^}]*\}/g, "{}");
  shapes.set(shape, new Set(Object.keys(ops).map((m) => m.toLowerCase())));
}

let failed = 0;
for (const { path, method, rawPath } of calls) {
  const methods = shapes.get(path);
  if (!methods) {
    console.error(`MISSING  ${method.toUpperCase().padEnd(6)} ${rawPath}  (no such path)`);
    failed++;
  } else if (!methods.has(method)) {
    console.error(
      `METHOD   ${method.toUpperCase().padEnd(6)} ${rawPath}  ` +
        `(backend allows ${[...methods].join("/").toUpperCase()})`,
    );
    failed++;
  }
}

console.log(`${calls.length} client calls checked, ${failed} mismatched`);
process.exit(failed ? 1 : 0);

/* Check calls against a URL or an OpenAPI file exported from this checkout. */
import { readFileSync } from "node:fs";
import { clientCalls, contractErrors } from "./api-contract.mjs";

const target = process.argv[2] ?? "http://127.0.0.1:8000/openapi.json";
let spec;
if (/^https?:\/\//.test(target)) {
  const response = await fetch(target);
  if (!response.ok) throw new Error(`OpenAPI request failed: ${response.status}`);
  spec = await response.json();
} else spec = JSON.parse(readFileSync(target, "utf8"));
const source = readFileSync(new URL("../src/api/client.ts", import.meta.url), "utf8");
const calls = clientCalls(source);
if (!calls.length) throw new Error("No API calls found; refusing an empty compatibility check");
const errors = contractErrors(calls, spec);
for (const error of errors) console.error(error);
console.log(`${calls.length} client calls checked, ${errors.length} mismatched`);
process.exit(errors.length ? 1 : 0);

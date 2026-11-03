import assert from "node:assert/strict";
import test from "node:test";
import { clientCalls, contractErrors } from "./api-contract.mjs";

test("route parameters and query-only expressions stay separate", () => {
  const source = 'api<PersonCrop[]>(`/api/images/${id}/process${force ? "?force=true" : ""}`, { method: "POST" })';
  const calls = clientCalls(source);
  assert.equal(calls[0].path, "/api/images/{}/process");
  assert.deepEqual(contractErrors(calls, { paths: { "/api/images/{image_id}/process": { post: {} } } }), []);
});

test("nested queryString options do not become route parameters", () => {
  const calls = clientCalls('api<Item[]>(`/api/items${queryString({q: value, nested: {a: 1}})}`)');
  assert.equal(calls[0].path, "/api/items");
  assert.equal(calls[0].method, "get");
});

test("missing paths and wrong HTTP methods still fail", () => {
  const calls = clientCalls('api<X>("/api/exists", { jsonBody: {x: 1} }); api<X>("/api/missing")');
  const errors = contractErrors(calls, { paths: { "/api/exists": { get: {} } } });
  assert.equal(errors.length, 2);
  assert.match(errors[0], /METHOD POST/);
  assert.match(errors[1], /MISSING GET/);
});

test("jsonBody helper and spreads preserve POST semantics", () => {
  const calls = clientCalls('api<X>("/api/search", jsonBody(payload)); api<X>("/api/search", {...jsonBody(payload), signal})');
  assert.equal(calls.length, 2);
  assert.deepEqual(calls.map((item) => item.method), ["post", "post"]);
  assert.deepEqual(contractErrors(calls, { paths: { "/api/search": { post: {} } } }), []);
});

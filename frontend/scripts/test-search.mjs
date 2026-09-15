import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { createRequire } from "node:module";
import { fileURLToPath } from "node:url";
import { test } from "node:test";

const require = createRequire(import.meta.url);
const vue = require("vue");
const { parse, compileScript } = require("@vue/compiler-sfc");
const ts = require("typescript");
const filename = fileURLToPath(new URL("../src/views/SearchView.vue", import.meta.url));

async function mountView(context, overrides = {}) {
  const calls = { semantic: [], structured: [], errors: [] };
  const api = {
    semanticStatus: async () => ({
      enabled: true, configured: true, indexed_crops: 23, total_crops: 23,
      labeled_crops: 0, attributes_enabled: false, auto_index_on_ingest: false,
    }),
    semanticPersonCrops: async (payload) => {
      calls.semantic.push(payload);
      return { items: [], notice: "仅为语义候选" };
    },
    personCrops: async (payload) => {
      calls.structured.push(payload);
      return { items: [] };
    },
    ...overrides,
  };
  const mocks = {
    vue,
    "@/api/client": { search: api, attributes: {} },
    "@/components/EmptyState.vue": {},
    "@/components/SearchResultCard.vue": {},
    "@/composables/useToast": {
      useToast: () => ({ showError: error => calls.errors.push(error), toast() {} }),
    },
    "@/composables/useSummary": {
      useSummary: () => ({
        crops: vue.ref([{ id: "recent", crop_url: "/recent.jpg" }]),
        imageTotal: vue.ref(8), cropTotal: vue.ref(23), runningCount: vue.ref(0),
        cameraOptions: vue.ref([]), refresh: async () => {},
      }),
    },
    "@/utils/format": { DISPLAY_TIME_ZONE: "Asia/Shanghai" },
  };
  const { descriptor } = parse(readFileSync(filename, "utf8"), { filename });
  const source = compileScript(descriptor, { id: "search-test" }).content;
  const { outputText } = ts.transpileModule(source, {
    compilerOptions: { module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2022 },
  });
  const module = { exports: {} };
  new Function("require", "module", "exports", outputText)(
    name => {
      assert.ok(name in mocks, `Unexpected dependency: ${name}`);
      return mocks[name];
    },
    module,
    module.exports,
  );
  const renderer = vue.createRenderer({
    createComment: () => ({}), insert() {}, remove() {}, parentNode() {}, nextSibling() {},
  });
  let state;
  const app = renderer.createApp({
    setup() {
      state = module.exports.default.setup({}, { expose() {} });
      return () => null;
    },
  });
  app.mount({});
  context.after(() => app.unmount());
  for (let step = 0; step < 12; step++) await Promise.resolve();
  await vue.nextTick();
  return { state, calls };
}

test("configured Qwen defaults to an explicit semantic request", async context => {
  const { state, calls } = await mountView(context);
  assert.equal(state.mode.value, "semantic");
  state.cameraId.value = "camera-A";
  await state.runSearch("  穿红衣服的男子  ");
  assert.equal(calls.semantic.length, 1);
  assert.equal(calls.structured.length, 0);
  assert.equal(calls.semantic[0].query, "穿红衣服的男子");
  assert.equal(calls.semantic[0].filters.camera_id, "camera-A");
  assert.equal(state.resultMode.value, "semantic");
  assert.equal(state.status.value, "empty");
  assert.match(state.hint.value, /语义候选/);
  assert.equal(state.searchNotice.value, "仅为语义候选");
});

test("strict label mode never calls vector retrieval", async context => {
  const { state, calls } = await mountView(context);
  state.mode.value = "structured";
  state.changeMode();
  await state.runSearch("红衣戴帽");
  assert.equal(calls.semantic.length, 0);
  assert.equal(calls.structured.length, 1);
  assert.match(state.hint.value, /结构化标签/);
});

test("an inverted time range is rejected before any search request", async context => {
  const { state, calls } = await mountView(context);
  state.startTime.value = "2026-09-15T10:00";
  state.endTime.value = "2026-09-15T09:00";
  await state.runSearch("背包");
  assert.equal(calls.semantic.length, 0);
  assert.equal(calls.structured.length, 0);
  assert.equal(state.status.value, "error");
  assert.match(state.hint.value, /开始时间/);
});

test("service failure clears recent results and is not a no-match state", async context => {
  const { state, calls } = await mountView(context, {
    semanticPersonCrops: async () => { throw new Error("语义检索服务暂不可用"); },
  });
  assert.equal(state.results.value.length, 1);
  await state.runSearch("红衣");
  assert.deepEqual(state.results.value, []);
  assert.equal(state.status.value, "error");
  assert.equal(calls.structured.length, 0);
  assert.match(state.hint.value, /服务暂不可用/);
});

test("semantic results preserve relevance order across capture dates", async context => {
  const { state } = await mountView(context, {
    semanticPersonCrops: async () => ({
      notice: "颜色未核验",
      items: [
        { crop_id: "best", score: 0.4, captured_at: "2026-01-01T00:00:00Z" },
        { crop_id: "second", score: 0.3, captured_at: "2026-02-01T00:00:00Z" },
        { crop_id: "third", score: 0.2, captured_at: "2026-01-01T00:00:00Z" },
      ],
    }),
  });
  await state.runSearch("红衣");
  assert.equal(state.groups.value.length, 1);
  assert.deepEqual(state.groups.value[0].items.map(item => item.crop_id), ["best", "second", "third"]);
  assert.match(state.hint.value, /非标签命中/);
  state.mode.value = "structured";
  state.changeMode();
  assert.deepEqual(state.results.value, []);
  assert.equal(state.searchNotice.value, "");
});

test("feature rollback defaults to strict mode", async context => {
  const { state } = await mountView(context, {
    semanticStatus: async () => ({ enabled: false, configured: true }),
  });
  assert.equal(state.mode.value, "structured");
  assert.equal(state.semanticEnabled.value, false);
});

test("older backend without capabilities still loads strict search", async context => {
  const { state } = await mountView(context, {
    semanticStatus: async () => { throw new Error("404"); },
  });
  assert.equal(state.mode.value, "structured");
  assert.equal(state.results.value.length, 1);
  assert.match(state.capabilityError.value, /暂用严格标签/);
});

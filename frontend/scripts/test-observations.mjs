import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { createRequire } from "node:module";
import { test } from "node:test";
import { fileURLToPath } from "node:url";

const require = createRequire(import.meta.url);
const vue = require("vue");
const { parse, compileScript } = require("@vue/compiler-sfc");
const ts = require("typescript");
const filename = fileURLToPath(new URL("../src/views/ObservationsView.vue", import.meta.url));

function deferred() {
  let resolve;
  let reject;
  const promise = new Promise((accept, decline) => {
    resolve = accept;
    reject = decline;
  });
  return { promise, resolve, reject };
}

async function settle() {
  for (let step = 0; step < 30; step++) await Promise.resolve();
  await vue.nextTick();
}

async function mountView(context, overrides = {}) {
  const calls = { rows: [], diagnostics: [], errors: [] };
  const api = {
    observations: async params => ({
      items: [{ crop_id: `crop-${params.get("offset")}` }],
      total: 201, offset: Number(params.get("offset")), limit: 100,
    }),
    diagnosticsForCrops: async cropIds => ({
      items: cropIds.map(crop_id => ({ crop_id, verdict: "no_face" })),
    }),
    ...overrides,
  };
  const mocks = {
    vue,
    "vue-router": { RouterLink: {} },
    "@/api/client": {
      search: {
        observations: (params, signal) => {
          calls.rows.push({ params, signal });
          return api.observations(params, signal);
        },
      },
      face: {
        diagnosticsForCrops: (cropIds, signal) => {
          calls.diagnostics.push({ cropIds, signal });
          return api.diagnosticsForCrops(cropIds, signal);
        },
      },
      persons: {},
    },
    "@/components/FaceBoxThumb.vue": {},
    "@/components/PersonSelect.vue": {},
    "@/composables/usePersons": {
      usePersons: () => ({
        persons: vue.ref([]), activePersonId: vue.ref(null), activePersonName: vue.ref(""),
        refresh: async () => {},
      }),
    },
    "@/composables/useToast": {
      useToast: () => ({ showError: error => calls.errors.push(error), toast() {} }),
    },
    "@/utils/format": {
      fmtTime: String, formatScore: String, shortId: String, shortText: String,
    },
  };
  const { descriptor } = parse(readFileSync(filename, "utf8"), { filename });
  const source = compileScript(descriptor, { id: "observation-test" }).content;
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
  await settle();
  return { state, calls };
}

test("the table is usable while its own crop diagnostics are still pending", async context => {
  const pending = deferred();
  const { state, calls } = await mountView(context, {
    diagnosticsForCrops: () => pending.promise,
  });
  assert.equal(state.loading.value, false);
  assert.equal(state.items.value[0].crop_id, "crop-0");
  assert.equal(state.status.value, "1-1 / 201");
  assert.match(state.diagnosticsStatus.value, /加载中/);
  assert.deepEqual(calls.diagnostics[0].cropIds, ["crop-0"]);
  pending.resolve({ items: [{ crop_id: "crop-0", verdict: "no_face" }] });
  await settle();
  assert.ok(state.diagnostics.value.has("crop-0"));
});

test("page changes cancel the previous request and ignore its late diagnostics", async context => {
  const firstPage = deferred();
  const { state, calls } = await mountView(context, {
    diagnosticsForCrops: cropIds => cropIds[0] === "crop-0"
      ? firstPage.promise
      : Promise.resolve({ items: [{ crop_id: cropIds[0], verdict: "unknown" }] }),
  });
  state.offset.value = 100;
  await state.load();
  await settle();
  assert.equal(calls.diagnostics[0].signal.aborted, true);
  assert.deepEqual(calls.diagnostics[1].cropIds, ["crop-100"]);
  firstPage.resolve({ items: [{ crop_id: "crop-0", verdict: "known" }] });
  await settle();
  assert.deepEqual([...state.diagnostics.value.keys()], ["crop-100"]);
  assert.equal(state.items.value[0].crop_id, "crop-100");
});

test("large pages are diagnosed in sequential batches of at most 100", async context => {
  const cropIds = Array.from({ length: 250 }, (_, index) => `crop-${index}`);
  const { state, calls } = await mountView(context, {
    observations: async () => ({
      items: cropIds.map(crop_id => ({ crop_id })), total: 250, offset: 0, limit: 500,
    }),
  });
  assert.deepEqual(calls.diagnostics.map(call => call.cropIds.length), [100, 100, 50]);
  assert.deepEqual(calls.diagnostics.flatMap(call => call.cropIds), cropIds);
  assert.equal(state.diagnostics.value.size, 250);
});

test("diagnostic failure leaves the table intact and reports partial capability", async context => {
  context.mock.method(console, "warn", () => {});
  const { state, calls } = await mountView(context, {
    diagnosticsForCrops: async () => { throw new Error("model unavailable"); },
  });
  assert.equal(state.loading.value, false);
  assert.equal(state.items.value.length, 1);
  assert.match(state.diagnosticsStatus.value, /暂不可用/);
  assert.equal(calls.errors.length, 0);
});

test("a stale table response cannot replace a newer search", async context => {
  const pending = deferred();
  const { state } = await mountView(context, {
    observations: params => params.get("query") === "old"
      ? pending.promise
      : Promise.resolve({
        items: [{ crop_id: params.get("query") || "initial" }], total: 1, offset: 0, limit: 100,
      }),
  });
  state.query.value = "old";
  const first = state.load(true);
  state.query.value = "new";
  await state.load(true);
  pending.resolve({ items: [{ crop_id: "old" }], total: 1, offset: 0, limit: 100 });
  await first;
  await settle();
  assert.equal(state.items.value[0].crop_id, "new");
  assert.deepEqual([...state.diagnostics.value.keys()], ["new"]);
});

test("an inverted time range is rejected without sending another query", async context => {
  const { state, calls } = await mountView(context);
  const before = calls.rows.length;
  state.startTime.value = "2026-09-15T10:00";
  state.endTime.value = "2026-09-15T09:00";
  await state.load(true);
  assert.equal(calls.rows.length, before);
  assert.match(calls.errors[0].message, /开始时间/);
});

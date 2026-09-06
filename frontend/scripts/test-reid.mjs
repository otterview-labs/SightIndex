// Focused component-state regressions using Vue's own compiler/reactivity/renderer.
// No browser, production API, stored photographs, or extra test dependencies are involved.
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { createRequire } from "node:module";
import { fileURLToPath } from "node:url";
import { test } from "node:test";

const require = createRequire(import.meta.url);
const vue = require("vue");
const { parse, compileScript } = require("@vue/compiler-sfc");
const ts = require("typescript");
const root = fileURLToPath(new URL("../src/", import.meta.url));

function deferred() {
  let resolve;
  let reject;
  const promise = new Promise((done, fail) => { resolve = done; reject = fail; });
  return { promise, resolve, reject };
}

async function settle() {
  for (let i = 0; i < 8; i++) await Promise.resolve();
  await vue.nextTick();
}

function loadModule(filename, mocks = {}, inlineTemplate = false) {
  let source = readFileSync(filename, "utf8");
  if (filename.endsWith(".vue")) {
    const { descriptor } = parse(source, { filename });
    source = compileScript(descriptor, { id: "reid-test", inlineTemplate }).content;
  }
  const { outputText } = ts.transpileModule(source, {
    compilerOptions: { module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2022 },
    fileName: filename,
  });
  const module = { exports: {} };
  const localRequire = (name) => {
    if (name in mocks) return mocks[name];
    if (name.endsWith(".vue")) return { default: {} };
    if (name.startsWith("@/")) return loadModule(`${root}${name.slice(2)}.ts`, mocks);
    return require(name);
  };
  new Function("require", "module", "exports", outputText)(localRequire, module, module.exports);
  return module.exports;
}

async function view(t, apiOverrides = {}) {
  const oldDocument = globalThis.document;
  const oldWindow = globalThis.window;
  globalThis.document = { documentElement: { style: { overflow: "" } }, activeElement: null };
  globalThis.window = { addEventListener() {}, removeEventListener() {} };
  const api = {
    status: async () => ({ ready: false }),
    feedback: async () => [],
    links: async () => ({ links: [], query_frame_count: 1 }),
    similarToCrop: async () => ({ items: [], query_frame_count: 1 }),
    search: async () => ({ items: [], query_frame_count: 1 }),
    saveFeedback: async (row) => row,
    ...apiOverrides,
  };
  const mocks = {
    "vue-router": { useRoute: () => ({ query: {} }), RouterLink: {} },
    "@/api/client": { crops: { get: async (id) => ({ id, crop_url: `/crop/${id}` }) }, reid: api },
    "@/composables/useToast": { useToast: () => ({ showError() {}, toast() {} }) },
  };
  const component = loadModule(`${root}views/ReidView.vue`, mocks).default;
  const renderer = vue.createRenderer({
    createComment: () => ({}), insert() {}, remove() {}, parentNode() {}, nextSibling() {},
  });
  let state;
  const app = renderer.createApp({
    setup() {
      state = component.setup({}, { expose() {} });
      return () => null;
    },
  });
  app.mount({});
  t.after(() => {
    app.unmount();
    globalThis.document = oldDocument;
    globalThis.window = oldWindow;
  });
  await settle();
  return state;
}

test("uploading a new query clears old links and ignores their late response", async (t) => {
  const pending = deferred();
  const state = await view(t, { links: () => pending.promise });
  await state.activateSourceCrop("crop-A");
  state.cameraLinks.value = { links: [{ crop_id: "old-hit" }] };
  const request = state.loadLinks("crop-A");
  state.onFileChange(new File(["synthetic"], "new-query.jpg"));
  assert.equal(state.cameraLinks.value, null);
  pending.resolve({ links: [{ crop_id: "late-old-hit" }] });
  await request;
  assert.equal(state.cameraLinks.value, null);
  assert.equal(state.resultsQueryCropId.value, null);
});

test("an uploaded image cannot display results from the URL's old crop", async (t) => {
  let queries = 0;
  const state = await view(t, {
    similarToCrop: async () => { queries++; return { items: [{ crop_id: "old-hit" }] }; },
  });
  await state.activateSourceCrop("crop-A");
  state.queryFile.value = new File(["synthetic"], "new-query.jpg");
  state.onFileChange(state.queryFile.value);
  await state.searchByCrop("crop-A");
  assert.equal(queries, 0);
  assert.equal(state.results.value, null);
});

test("camera groups preserve server face-first ranking, not just numeric scores", async (t) => {
  const state = await view(t);
  state.results.value = [
    { crop_id: "face", camera_id: "B", score: 0.46, fusion_score: 0.51, face_match: true },
    { crop_id: "body", camera_id: "A", score: 0.96, fusion_score: 0.96 },
    { crop_id: "second-B", camera_id: "B", score: 0.45, fusion_score: 0.45 },
  ];
  assert.deepEqual(state.resultGroups.value.map((group) => group.key), ["B", "A"]);
  assert.deepEqual(state.resultGroups.value[0].items.map((item) => item.crop_id), ["face", "second-B"]);
});

test("leaving and revisiting a crop ignores feedback from the earlier visit", async (t) => {
  const pending = deferred();
  let calls = 0;
  const state = await view(t, { feedback: () => ++calls === 1 ? pending.promise : Promise.resolve([]) });
  const old = state.activateSourceCrop("crop-A");
  await state.activateSourceCrop("crop-B");
  await state.activateSourceCrop("crop-A");
  pending.resolve([{ candidate_crop_id: "stale-hit", same_person: true }]);
  await old;
  assert.deepEqual(state.feedbackByCandidate.value, {});
});

test("a slow feedback read cannot overwrite a newly saved judgement", async (t) => {
  const pending = deferred();
  const state = await view(t, { feedback: () => pending.promise });
  const request = state.activateSourceCrop("crop-A");
  state.resultsQueryCropId.value = "crop-A";
  await state.saveFeedback({ crop_id: "candidate", score: 0.6 }, false, "search");
  pending.resolve([{ candidate_crop_id: "candidate", same_person: true }]);
  await request;
  assert.equal(state.feedbackValue("candidate"), false);
});

test("switching query while uploading prevents a late result from repainting", async (t) => {
  const pending = deferred();
  const state = await view(t, { search: () => pending.promise });
  state.onFileChange(new File(["synthetic"], "query.jpg"));
  const request = state.search();
  await state.activateSourceCrop("crop-B");
  pending.resolve({ items: [{ crop_id: "upload-hit" }] });
  await request;
  assert.equal(state.results.value, null);
  assert.equal(state.queryPreview.value, "");
  assert.equal(state.activeQueryCropId.value, "crop-B");
});

test("lightbox focuses close, traps Tab, and restores focus/scroll on Escape", async (t) => {
  const state = await view(t);
  const previous = globalThis.HTMLElement;
  globalThis.HTMLElement = class {
    isConnected = true;
    focus() { document.activeElement = this; }
  };
  t.after(() => { globalThis.HTMLElement = previous; });
  // Native DOM nodes are not proxied by Vue; mark these small stand-ins the same way.
  const trigger = vue.markRaw(new HTMLElement());
  const close = vue.markRaw(new HTMLElement());
  trigger.focus();
  document.documentElement.style.overflow = "auto";
  state.lightboxCloseButton.value = close;
  state.enlargeImage("/synthetic.jpg", "test", "synthetic only");
  await vue.nextTick();
  assert.equal(document.activeElement, close);
  let prevented = false;
  state.onPreviewKeydown({ key: "Tab", preventDefault() { prevented = true; } });
  assert.equal(prevented, true);
  state.onPreviewKeydown({ key: "Escape", preventDefault() {} });
  assert.equal(state.enlargedImage.value, null);
  assert.equal(document.documentElement.style.overflow, "auto");
  assert.equal(document.activeElement, trigger);
});

const evidence = loadModule(`${root}components/ReidEvidenceSummary.vue`, {}, true).default;
const { renderToString } = require("@vue/server-renderer");
async function renderEvidence(item, confirmed = null) {
  return renderToString(vue.createSSRApp(evidence, {
    item: { crop_id: "synthetic-id", score: 0.6, ...item }, when: "09/05 12:00:00", confirmed,
  }));
}

test("face support and chance-line scores are clues, never automatic confirmation", async () => {
  const html = await renderEvidence({
    face_match: true, face_similarity: 0.45, face_reliability: 0.8, beats_chance: true,
  });
  assert.match(html, /人脸证据支持 · 待核对/);
  assert.match(html, /相似度 0\.45/);
  assert.match(html, /质量评分 0\.80/);
  assert.doesNotMatch(html, /45%|人工已确认/);
  assert.match(html, /<details[^>]*><summary>证据明细/);
  assert.doesNotMatch(html, /<details[^>]*\sopen/);
});

test("missing evidence stays unknown and one label stays neutral", async () => {
  const html = await renderEvidence({ attribute_agreement: 1, attribute_comparable_count: 1 });
  assert.match(html, /可比高置信标签不足，保持中性/);
  assert.match(html, /未提供可用人脸比较，不代表不一致/);
  assert.doesNotMatch(html, /100%|质量评分 0\.00|NaN|null|undefined/);
});

test("zero scores and weighted label counts are not mistaken for missing values", async () => {
  const html = await renderEvidence({
    score: 0, face_similarity: 0, face_reliability: 0, stature_agreement: 0,
    attribute_agreement: 0.75, attribute_comparable_count: 4, attribute_match_count: 3,
  });
  assert.match(html, /相似度 0\.00/);
  assert.match(html, /质量评分 0\.00/);
  assert.match(html, /高置信标签一致 3\/4 · 加权一致 75%/);
  assert.match(html, /几何一致评分 0\.00/);
});

test("only explicit human feedback is labelled as confirmed", async () => {
  assert.match(await renderEvidence({ beats_chance: false }), /低置信线索 · 可能只是相似/);
  assert.match(await renderEvidence({}, true), /人工已确认：同一个人/);
  assert.match(await renderEvidence({ face_match: true }, false), /人工已确认：不是同一个人/);
});

test("unverified borrowed faces never become hard match or conflict labels", async () => {
  for (const flag of ["face_query_identity_verified", "face_candidate_identity_verified"]) {
    for (const decision of [true, false]) {
      const html = await renderEvidence({
        [flag]: false, face_match: decision, face_similarity: 0.91,
        face_candidate_source_crop_id: "another-synthetic-frame", face_reliability: 0.86,
      });
      assert.match(html, /人脸来源身份未验证 · 仅作辅助线索/);
      assert.match(html, /不能据此确认或排除/);
      assert.match(html, /卡片仍展示候选代表图/);
      assert.doesNotMatch(html, /人脸证据支持 · 待核对|明确冲突|建议排除|人工已确认/);
    }
  }
  assert.match(await renderEvidence({ face_similarity: 0.9, face_query_identity_verified: false }, true), /人工已确认：同一个人/);
});

const faceCoverageComponent = loadModule(`${root}components/ReidFaceCoverageSummary.vue`, {}, true).default;
async function renderCoverage(coverage, scope = "search") {
  return renderToString(vue.createSSRApp(faceCoverageComponent, { coverage, scope }));
}

test("old servers do not invent face coverage counts or claim participation", async () => {
  for (const coverage of [undefined, null, {}]) {
    const html = await renderCoverage(coverage);
    assert.match(html, /服务端未提供本次人脸参与诊断/);
    assert.doesNotMatch(html, /0 \/ 0|尝试 0 帧|质量评分 0\.00|NaN|null|undefined/);
  }
});

test("query-face absence does not imply all candidates were tried or faceless", async () => {
  const html = await renderCoverage({
    status: "query_unavailable", query_face_found: false, query_attempted_count: 5,
    candidate_attempted_count: 0, shortlist_count: 12, compared_count: 0,
    query_absence_reasons: { no_face: 4, source_unreadable: 1 },
  });
  assert.match(html, /本次查询未提取到可用人脸，使用人体与标签证据/);
  assert.match(html, /不代表候选图片都没有人脸/);
  assert.match(html, /0 \/ 12 个入选候选/);
  assert.match(html, /尝试 0 帧/);
  assert.match(html, /检测／提取阶段没有可用人脸候选 ×4/);
  assert.match(html, /原图读取失败 ×1/);
  assert.doesNotMatch(html, /12 张图片没有人脸|质量评分 0\.00/);
});

test("face comparison denominator is shortlist visits, not attempted frames or displayed hits", async () => {
  const html = await renderCoverage({
    status: "compared", query_face_found: true, query_identity_verified: true,
    query_face_quality: 0, query_attempted_count: 1, candidate_attempted_count: 30,
    shortlist_count: 12, compared_count: 5, borrowed_candidate_count: 2,
    hard_match_count: 0, hard_conflict_count: 0,
  }, "links");
  assert.match(html, /跨摄像头线索的人脸参与/);
  assert.match(html, /5 \/ 12 个入选候选（最终筛选前）/);
  assert.doesNotMatch(html, /5 \/ 30|42%|人脸命中率/);
  assert.match(html, /尝试 30 帧/);
  assert.match(html, /质量评分 0\.00/);
  assert.match(html, /支持匹配 0 个/);
  assert.match(html, /不等于下方最终显示数量/);
});

test("missing candidate metadata does not invent an extracted query face", async () => {
  for (const queryFaceFound of [false, undefined]) {
    const html = await renderCoverage({
      status: "candidate_unavailable", query_face_found: queryFaceFound,
      query_attempted_count: 0, candidate_attempted_count: 0,
    });
    assert.match(html, /候选资料不可用或未形成可用比较，未形成身份结论/);
    assert.match(html, /本次未尝试提取查询人脸/);
    assert.doesNotMatch(html, /查询侧有可用人脸|未获得可用人脸/);
  }
  assert.match(await renderCoverage({ status: "candidate_unavailable", query_face_found: true }), /查询侧有可用人脸/);
  const noCandidates = await renderCoverage({ status: "no_candidates", query_face_found: false, query_attempted_count: 0 });
  assert.match(noCandidates, /本次未尝试提取查询人脸/);
  assert.doesNotMatch(noCandidates, /未获得可用人脸/);
});

test("partial face diagnostics and unknown reasons remain explicit unknowns", async () => {
  const html = await renderCoverage({
    status: "error", compared_count: 0,
    candidate_absence_reasons: { inference_error: 1, future_reason: 2, no_face: 0 },
  });
  assert.match(html, /本次人脸比较发生异常/);
  assert.match(html, /0 \/ 未提供 个入选候选/);
  assert.match(html, /人脸推理异常 ×1/);
  assert.match(html, /其他未分类原因 ×2/);
  assert.doesNotMatch(html, /候选侧.*尝试 0 帧|future_reason|×0/);
  assert.match(await renderCoverage({ status: "disabled" }), /本次未启用人脸优先/);
  assert.match(await renderCoverage({ status: "unavailable" }), /本次人脸服务不可用/);
  assert.match(await renderCoverage({ status: "no_candidates" }), /本次没有进入人脸比较阶段的候选/);
  assert.match(await renderCoverage({ status: "candidate_unavailable" }), /不能据此判断为不同人/);
  assert.match(await renderCoverage({ status: "compared", query_identity_verified: false }), /身份关联未验证，仅作软证据/);
});

test("search and link coverage stay separate and an old server clears prior coverage", async (t) => {
  let includeCoverage = true;
  const state = await view(t, {
    similarToCrop: async () => ({ items: [], ...(includeCoverage ? { face_coverage: { status: "compared", compared_count: 3 } } : {}) }),
    links: async () => ({ links: [], face_coverage: { status: "query_unavailable", compared_count: 0 } }),
  });
  await state.activateSourceCrop("crop-A");
  await state.searchByCrop("crop-A");
  await state.loadLinks("crop-A");
  assert.equal(state.resultsFaceCoverage.value.compared_count, 3);
  assert.equal(state.cameraLinks.value.face_coverage.compared_count, 0);
  includeCoverage = false;
  await state.searchByCrop("crop-A");
  assert.equal(state.resultsFaceCoverage.value, null);
  assert.equal(state.cameraLinks.value.face_coverage.status, "query_unavailable");
});

test("new searches clear diagnostics immediately and errors do not restore them", async (t) => {
  const pending = deferred();
  const state = await view(t, { similarToCrop: () => pending.promise });
  await state.activateSourceCrop("crop-A");
  state.resultsFaceCoverage.value = { status: "compared" };
  const request = state.searchByCrop("crop-A");
  assert.equal(state.resultsFaceCoverage.value, null);
  pending.reject(new Error("synthetic failure"));
  await request;
  assert.equal(state.resultsFaceCoverage.value, null);
  assert.equal(state.results.value, null);
});

test("A to B to A crop navigation rejects earlier face diagnostics", async (t) => {
  const pending = deferred();
  const state = await view(t, { similarToCrop: () => pending.promise });
  await state.activateSourceCrop("crop-A");
  const request = state.searchByCrop("crop-A");
  await state.activateSourceCrop("crop-B");
  await state.activateSourceCrop("crop-A");
  pending.resolve({ items: [], face_coverage: { status: "compared", compared_count: 9 } });
  await request;
  assert.equal(state.resultsFaceCoverage.value, null);
  assert.equal(state.results.value, null);
});

test("new upload discards old diagnostics and late upload coverage", async (t) => {
  const pending = deferred();
  const state = await view(t, { search: () => pending.promise });
  state.resultsFaceCoverage.value = { status: "compared" };
  state.onFileChange(new File(["synthetic"], "query.jpg"));
  assert.equal(state.resultsFaceCoverage.value, null);
  const request = state.search();
  await state.activateSourceCrop("crop-B");
  pending.resolve({ items: [], face_coverage: { status: "compared", compared_count: 8 } });
  await request;
  assert.equal(state.resultsFaceCoverage.value, null);
});

test("new link requests clear diagnostics and ignore older same-crop replies", async (t) => {
  const pending = deferred();
  let calls = 0;
  const state = await view(t, { links: () => ++calls === 1 ? pending.promise : Promise.resolve({ links: [], face_coverage: { status: "compared", compared_count: 2 } }) });
  await state.activateSourceCrop("crop-A");
  state.cameraLinks.value = { links: [], face_coverage: { status: "compared", compared_count: 7 } };
  const old = state.loadLinks("crop-A");
  assert.equal(state.cameraLinks.value, null);
  await state.loadLinks("crop-A");
  pending.resolve({ links: [], face_coverage: { status: "compared", compared_count: 9 } });
  await old;
  assert.equal(state.cameraLinks.value.face_coverage.compared_count, 2);
});

test("failed links clear only their own diagnostics, not search evidence", async (t) => {
  const state = await view(t, { links: async () => { throw new Error("synthetic failure"); } });
  await state.activateSourceCrop("crop-A");
  state.resultsFaceCoverage.value = { status: "compared", compared_count: 3 };
  state.cameraLinks.value = { links: [], face_coverage: { status: "compared", compared_count: 7 } };
  await state.loadLinks("crop-A");
  assert.equal(state.cameraLinks.value, null);
  assert.equal(state.resultsFaceCoverage.value.compared_count, 3);
});

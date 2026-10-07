import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { createRequire } from "node:module";
import { fileURLToPath } from "node:url";
import { test } from "node:test";

const require = createRequire(import.meta.url);
const vue = require("vue");
const { parse, compileScript } = require("@vue/compiler-sfc");
const ts = require("typescript");
const filename = fileURLToPath(new URL("../src/components/VideoPlaybackButton.vue", import.meta.url));
const { descriptor } = parse(readFileSync(filename, "utf8"), { filename });
const { outputText } = ts.transpileModule(compileScript(descriptor, { id: "playback-test" }).content, {
  compilerOptions: { module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2022 },
});
const source = {
  available: true, source_type: "video_frame", video_url: "/data/videos/synthetic.mp4",
  offset_seconds: 8, captured_at: "2026-10-06T12:00:08Z", reason: null,
};

function mount(context, props = { cropId: "synthetic-crop" }, fetchPlayback = async () => source) {
  props = vue.reactive(props);
  const calls = [];
  const api = kind => ({ playback: (id, signal) => {
    calls.push({ kind, id, signal });
    return fetchPlayback(id, signal);
  } });
  const mocks = {
    vue, "@/api/client": { crops: api("crop"), images: api("image") },
    "@/utils/format": { fmtTime: String },
  };
  const module = { exports: {} };
  new Function("require", "module", "exports", outputText)(dependency => {
    assert.ok(dependency in mocks, `Unexpected dependency: ${dependency}`);
    return mocks[dependency];
  }, module, module.exports);
  const renderer = vue.createRenderer({
    createComment: () => ({}), insert() {}, remove() {}, parentNode() {}, nextSibling() {},
  });
  let state;
  const app = renderer.createApp({ setup() {
    state = module.exports.default.setup(props, { expose() {} });
    return () => null;
  } });
  app.mount({});
  const dialog = { open: false, showModal() { this.open = true; }, close() { this.open = false; } };
  state.dialog.value = dialog;
  const element = {
    currentTime: 0, duration: 15, seeking: true, paused: true, plays: 0, pauses: 0, loads: 0,
    async play() { this.plays++; this.paused = false; },
    pause() { this.pauses++; this.paused = true; },
    removeAttribute() {}, load() { this.loads++; },
  };
  state.video.value = element;
  const event = { currentTarget: state.video.value };
  context.after(() => app.unmount());
  return { state, calls, dialog, element, event, app, props };
}

test("cards do not fetch until clicked, and crop provenance takes precedence", async context => {
  const { state, calls, dialog } = mount(context, { cropId: "crop", imageId: "image" });
  assert.equal(calls.length, 0);
  await state.open();
  assert.equal(dialog.open, true);
  assert.equal(calls.length, 1);
  assert.equal(calls[0].kind, "crop");
  assert.equal(calls[0].id, "crop");
  assert.equal(state.position.value, "00:00:08");
});

test("image-only cards resolve the original image instead of inventing a crop", async context => {
  const { state, calls } = mount(context, { imageId: "image" });
  await state.open();
  assert.equal(calls[0].kind, "image");
  assert.equal(calls[0].id, "image");
});

test("missing provenance never calls the service", async context => {
  const { state, calls } = mount(context, {});
  await state.open();
  assert.equal(calls.length, 0);
  assert.equal(state.visible.value, false);
});

test("playback seeks first and plays once only after the seek completes", async context => {
  const { state, element, event } = mount(context);
  await state.open();
  assert.equal(element.plays, 0);
  state.seek(event);
  assert.equal(element.currentTime, 8);
  assert.equal(element.plays, 0);
  state.finishSeek(event);
  await Promise.resolve();
  assert.equal(element.plays, 1);
  state.finishSeek(event);
  assert.equal(element.plays, 1);
});

test("zero seconds is valid and can play without a separate seeked event", async context => {
  const { state, element, event } = mount(context, undefined, async () => ({ ...source, offset_seconds: 0 }));
  await state.open();
  element.seeking = false;
  state.seek(event);
  await Promise.resolve();
  assert.equal(element.currentTime, 0);
  assert.equal(element.plays, 1);
});

test("blocked autoplay provides a manual retry at the already-seeked position", async context => {
  const { state, element, event } = mount(context);
  await state.open();
  element.play = async () => { throw new Error("NotAllowedError"); };
  state.seek(event);
  state.finishSeek(event);
  await Promise.resolve();
  assert.equal(state.manualPlay.value, true);
  element.play = async () => { element.plays++; };
  await state.play();
  assert.equal(state.manualPlay.value, false);
  assert.equal(element.currentTime, 8);
});

test("closing aborts requests and stops playback; late replies cannot reopen", async context => {
  let resolve;
  const pending = new Promise(done => { resolve = done; });
  const { state, calls, dialog, element } = mount(context, undefined, () => pending);
  const opening = state.open();
  await vue.nextTick();
  state.close();
  resolve(source);
  await opening;
  assert.equal(calls[0].signal.aborted, true);
  assert.equal(state.visible.value, false);
  assert.equal(dialog.open, false);
  assert.equal(state.playback.value, null);
  assert.ok(element.pauses > 0 && element.loads > 0);
});

test("a previous request cannot overwrite a newer playback", async context => {
  let resolveOld;
  let count = 0;
  const pending = new Promise(resolve => { resolveOld = resolve; });
  const { state, calls } = mount(context, undefined, () => ++count === 1 ? pending : Promise.resolve({ ...source, offset_seconds: 4 }));
  const old = state.open();
  await vue.nextTick();
  await state.open();
  resolveOld(source);
  await old;
  assert.equal(calls[0].signal.aborted, true);
  assert.equal(state.playback.value.offset_seconds, 4);
});

test("camera frames explain missing recordings and never open a live stream", async context => {
  const { state, element } = mount(context, undefined, async () => ({
    ...source, available: false, video_url: null, reason: "recording_not_configured",
  }));
  await state.open();
  assert.match(state.error.value, /只保存抓帧图片/);
  assert.equal(state.playback.value, null);
  assert.equal(element.plays, 0);
});

test("unsafe video URLs and unknown or invalid offsets fail closed", async context => {
  for (const result of [
    { ...source, video_url: "https://external.invalid/video.mp4" },
    { ...source, video_url: "javascript:alert(1)" },
    { ...source, video_url: "/data/videos/../images/x.mp4" },
    { ...source, offset_seconds: null }, { ...source, offset_seconds: -1 },
    { ...source, offset_seconds: Number.NaN },
  ]) {
    const { state, element } = mount(context, undefined, async () => result);
    await state.open();
    assert.match(state.error.value, /无效/);
    assert.equal(state.playback.value, null);
    assert.equal(element.plays, 0);
    state.close();
  }
});

test("out-of-range timestamps and codec errors remain distinct from missing recordings", async context => {
  const { state, element, event } = mount(context);
  await state.open();
  element.duration = 4;
  state.seek(event);
  assert.match(state.error.value, /超出/);
  assert.equal(element.plays, 0);
  state.mediaError(event);
  assert.match(state.error.value, /视频编码/);
});

test("network failures offer retry instead of claiming there is no video", async context => {
  const { state } = mount(context, undefined, async () => { throw new Error("503 Service Unavailable"); });
  await state.open();
  assert.match(state.error.value, /请重试/);
  assert.equal(state.loading.value, false);
});

test("a reused result card cancels its old request when the result identity changes", async context => {
  let resolve;
  const pending = new Promise(done => { resolve = done; });
  const { state, props, calls } = mount(context, undefined, () => pending);
  const opening = state.open();
  await vue.nextTick();
  props.cropId = "another-crop";
  await vue.nextTick();
  resolve(source);
  await opening;
  assert.equal(calls[0].signal.aborted, true);
  assert.equal(state.visible.value, false);
  assert.equal(state.playback.value, null);
});

test("a reused result card stops the old video rather than showing it under a new result", async context => {
  const { state, props, element } = mount(context);
  await state.open();
  const pauses = element.pauses;
  props.cropId = "another-crop";
  await vue.nextTick();
  assert.equal(state.visible.value, false);
  assert.ok(element.pauses > pauses);
});

test("a seeked event at a different position never claims a successful hit", async context => {
  const { state, element, event } = mount(context);
  await state.open();
  state.seek(event);
  element.currentTime = 2;
  state.finishSeek(event);
  assert.equal(element.plays, 0);
  assert.match(state.error.value, /未能定位/);
});

test("unsupported codecs do not masquerade as an autoplay permission problem", async context => {
  const { state, element, event } = mount(context);
  await state.open();
  element.play = async () => {
    const error = new Error("Unsupported");
    error.name = "NotSupportedError";
    throw error;
  };
  state.seek(event);
  state.finishSeek(event);
  await Promise.resolve();
  assert.equal(state.manualPlay.value, false);
  assert.match(state.error.value, /视频编码/);
});

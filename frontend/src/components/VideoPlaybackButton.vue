<script setup lang="ts">
import { computed, nextTick, onBeforeUnmount, ref, watch } from "vue";
import { crops, images } from "@/api/client";
import type { VideoPlaybackRead } from "@/api/types";
import { fmtTime } from "@/utils/format";

const props = defineProps<{ cropId?: string | null; imageId?: string | null }>();
const dialog = ref<HTMLDialogElement | null>(null);
const video = ref<HTMLVideoElement | null>(null);
const visible = ref(false);
const loading = ref(false);
const playback = ref<VideoPlaybackRead | null>(null);
const error = ref("");
const manualPlay = ref(false);
const started = ref(false);
const locating = ref(false);
let generation = 0;
let controller: AbortController | null = null;
let previousOverflow: string | null = null;

const position = computed(() => {
  const seconds = playback.value?.offset_seconds;
  if (seconds == null || !Number.isFinite(seconds)) return "";
  const whole = Math.floor(seconds);
  return [Math.floor(whole / 3600), Math.floor(whole / 60) % 60, whole % 60]
    .map(value => String(value).padStart(2, "0")).join(":");
});
const captureTime = computed(() => {
  const value = playback.value?.captured_at;
  return value && Number.isFinite(Date.parse(value)) ? fmtTime(value) : "";
});

function unavailableMessage(reason: string | null | undefined): string {
  const messages: Record<string, string> = {
    recording_not_configured: "这个摄像头目前只保存抓帧图片，没有关联历史录像。接入录像平台或开启录像留存后才能回放。",
    source_missing: "这张历史图片没有保存原视频关联，暂时无法定位回放。新处理的上传视频会保存关联。",
    not_video: "这张图片来自图片上传，没有对应的原视频。",
    media_missing: "原视频文件已不在服务器上，无法回放。图片仍可查看。",
    offset_unknown: "没有可靠的视频时间位置，暂时无法定位回放。",
    invalid_source: "原视频关联无效，暂时无法回放。",
  };
  return messages[reason || ""] || "这张图片暂时没有可回放的录像。";
}

function stopVideo(): void {
  const element = video.value;
  if (!element) return;
  element.pause();
  element.removeAttribute("src");
  element.load();
}

function close(): void {
  if (!visible.value) return;
  generation += 1;
  controller?.abort();
  controller = null;
  stopVideo();
  dialog.value?.close();
  if (previousOverflow !== null && typeof document !== "undefined") {
    document.documentElement.style.overflow = previousOverflow;
    previousOverflow = null;
  }
  visible.value = false;
  playback.value = null;
  locating.value = false;
}

async function open(): Promise<void> {
  if (!props.cropId && !props.imageId) return;
  const request = ++generation;
  controller?.abort();
  stopVideo();
  controller = new AbortController();
  const signal = controller.signal;
  playback.value = null;
  error.value = "";
  manualPlay.value = false;
  started.value = false;
  locating.value = false;
  loading.value = true;
  visible.value = true;
  await nextTick();
  if (request !== generation || !visible.value) return;
  try {
    if (dialog.value && !dialog.value.open) dialog.value.showModal();
    if (previousOverflow === null && typeof document !== "undefined") {
      previousOverflow = document.documentElement.style.overflow;
      document.documentElement.style.overflow = "hidden";
    }
    const result = props.cropId
      ? await crops.playback(props.cropId, signal)
      : await images.playback(props.imageId!, signal);
    if (request !== generation || !visible.value) return;
    if (!result.available) {
      error.value = unavailableMessage(result.reason);
      return;
    }
    // Only same-origin passive media is accepted; never open arbitrary URL schemes.
    if (!result.video_url || !/^\/data\/videos\/[A-Za-z0-9][A-Za-z0-9_.-]*$/.test(result.video_url)
      || result.offset_seconds == null || !Number.isFinite(result.offset_seconds)
      || result.offset_seconds < 0) {
      error.value = "视频来源或时间位置无效，无法可靠定位。";
      return;
    }
    playback.value = result;
    locating.value = true;
  } catch (cause) {
    if (request !== generation || signal.aborted || !visible.value) return;
    error.value = `无法加载视频关联，请重试。${cause instanceof Error ? cause.message : ""}`;
  } finally {
    if (request === generation) loading.value = false;
  }
}

async function play(): Promise<void> {
  const element = video.value;
  const request = generation;
  if (!element || !visible.value || locating.value || error.value) return;
  manualPlay.value = false;
  started.value = false;
  try {
    await element.play();
    if (request === generation && visible.value && element === video.value) started.value = true;
  } catch (cause) {
    if (request === generation && visible.value && element === video.value) {
      if (cause instanceof Error && cause.name === "NotSupportedError") {
        error.value = "浏览器不支持此视频编码。图片仍可查看。";
      } else {
        manualPlay.value = true;
      }
    }
  }
}

function seek(event: Event): void {
  const element = event.currentTarget as HTMLVideoElement;
  const offset = playback.value?.offset_seconds;
  if (!visible.value || element !== video.value || offset == null) return;
  if (Number.isFinite(element.duration) && offset > element.duration) {
    locating.value = false;
    error.value = "记录的时间位置超出了视频时长，无法可靠定位。图片仍可查看。";
    return;
  }
  try {
    element.currentTime = offset;
    if (!element.seeking && Math.abs(element.currentTime - offset) < 0.05) finishSeek(event);
  } catch {
    locating.value = false;
    error.value = "无法定位到命中位置，请重试或使用播放器控制条。";
  }
}

function finishSeek(event: Event): void {
  if (!visible.value || event.currentTarget !== video.value || !locating.value) return;
  locating.value = false;
  const offset = playback.value?.offset_seconds;
  if (offset == null || !video.value || Math.abs(video.value.currentTime - offset) > 0.15) {
    error.value = "未能定位到记录的命中位置，请重新加载。图片仍可查看。";
    return;
  }
  void play();
}

function mediaError(event: Event): void {
  if (!visible.value || event.currentTarget !== video.value) return;
  locating.value = false;
  error.value = "视频加载失败或浏览器不支持此视频编码。请重试；图片仍可查看。";
}

watch(() => [props.cropId, props.imageId], () => { close(); });
onBeforeUnmount(() => { close(); });
</script>

<template>
  <button v-if="cropId || imageId" class="playback-button" type="button" @click.stop="open">
    回放视频
  </button>
  <Teleport to="body">
    <dialog v-if="visible" ref="dialog" class="playback-dialog" aria-label="命中视频回放" @cancel.prevent="close" @close="close">
      <header class="playback-header">
        <h2>命中视频回放</h2>
        <button class="playback-button" type="button" autofocus @click="close">关闭</button>
      </header>
      <p v-if="loading" role="status">正在查找原视频与命中位置…</p>
      <p v-else-if="error" class="playback-message" role="status">{{ error }}</p>
      <template v-if="playback">
        <p class="playback-meta">视频位置 {{ position }}<span v-if="captureTime"> · 画面时间 {{ captureTime }}</span></p>
        <video
          ref="video" :src="playback.video_url || undefined" controls muted playsinline preload="metadata"
          aria-label="原视频播放器" @loadedmetadata="seek" @seeked="finishSeek" @error="mediaError"
        />
        <p v-if="locating" role="status">正在定位到命中位置…</p>
        <p v-else-if="manualPlay" role="status">自动播放未完成，请点击下方按钮或播放器的播放键。</p>
        <p v-else-if="!error" class="playback-hint">{{ started ? "已定位到命中位置，默认静音播放。需要声音时可在播放器中取消静音。" : "已定位到命中位置，正在尝试静音播放…" }}</p>
        <button v-if="manualPlay && !error" class="playback-button" type="button" @click="play">点击播放</button>
      </template>
      <button v-if="error" class="playback-button" type="button" @click="open">重新加载</button>
    </dialog>
  </Teleport>
</template>

<style scoped>
.playback-button {
  min-height: 36px;
  padding: 6px 12px;
  border: 1px solid var(--line, #d0d5dd);
  border-radius: 8px;
  background: var(--surface, #fff);
  color: var(--text, #344054);
  font: inherit;
  font-size: 14px;
  cursor: pointer;
}
.playback-button:hover { background: var(--surface-soft, #f2f4f7); }
.playback-button:focus-visible { outline: 2px solid var(--accent, #175cd3); outline-offset: 3px; }
.playback-dialog {
  box-sizing: border-box;
  width: min(960px, calc(100vw - 32px));
  max-height: calc(100dvh - 32px);
  overflow: auto;
  padding: 20px;
  border: 0;
  border-radius: 14px;
  background: var(--surface, #fff);
  color: var(--text, #344054);
  box-shadow: 0 12px 40px rgb(0 0 0 / 25%);
  font-size: 16px;
  line-height: 1.6;
  scrollbar-color: var(--muted, #667085) var(--surface, #fff);
}
.playback-dialog::backdrop { background: rgb(16 24 40 / 65%); }
.playback-header { display: flex; align-items: center; justify-content: space-between; gap: 16px; }
.playback-header h2 { margin: 0; font-size: 20px; text-wrap: balance; }
.playback-dialog p { margin: 16px 0; overflow-wrap: anywhere; }
.playback-dialog video { display: block; width: 100%; max-height: 64dvh; background: #000; }
.playback-meta { font-variant-numeric: tabular-nums; }
.playback-hint { font-size: 14px; }
.playback-message { max-width: 70ch; }
@media (max-width: 480px) {
  .playback-dialog { padding: 16px; }
  .playback-button { min-height: 44px; }
}
</style>

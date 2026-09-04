<script setup lang="ts">
import { computed, ref, watch } from "vue";

import type { ImageRead, VideoStream } from "@/api/types";
import CountingLineCanvas, {
  formatCountingLine,
  lineToPoints,
  type LinePoint,
} from "@/components/CountingLineCanvas.vue";
import { maskUrl, shortId } from "@/utils/format";

const props = defineProps<{
  stream: VideoStream;
  expanded: boolean;
  editing: boolean;
  frame?: ImageRead | null;
  busyAction?: string | null;
  editorHint: string;
}>();

const emit = defineEmits<{
  toggle: [];
  start: [];
  stop: [];
  remove: [];
  editLine: [];
  cancelLine: [];
  saveLine: [points: LinePoint[]];
  clearLine: [];
}>();

const points = ref<LinePoint[]>([]);
const previewFailed = ref(false);
const previewEl = ref<HTMLImageElement | null>(null);
const previewNonce = ref(Date.now());

const lineText = computed(() => (props.stream.counting_line ? "线已设" : "未设线"));
const frameText = computed(() =>
  props.stream.last_frame_image_id ? `最后帧 ${shortId(props.stream.last_frame_image_id)}` : "无帧",
);
const coordinate = computed(() => formatCountingLine(props.stream.counting_line));

const fallbackUrl = computed(() => props.frame?.thumbnail_url || props.frame?.image_url || "");
const previewUrl = computed(() =>
  previewFailed.value && fallbackUrl.value
    ? fallbackUrl.value
    : `/api/streams/${props.stream.id}/mjpeg?fps=6&jpeg_quality=92&v=${previewNonce.value}`,
);

const interacted = ref(false);

const hint = computed(() => {
  if (interacted.value) {
    return points.value.length === 2 ? "计数线已设置" : "再点击一次完成计数线";
  }
  if (previewFailed.value) {
    return fallbackUrl.value ? "实时预览失败，已切到最近一帧" : "实时预览失败；仍可按比例点击两点";
  }
  if (props.editorHint) return props.editorHint;
  return points.value.length === 2 ? "可拖新线或直接保存" : "点击两点设置计数线";
});

watch(
  () => [props.editing, props.stream.counting_line] as const,
  ([editing]) => {
    if (!editing) return;
    points.value = lineToPoints(props.stream.counting_line);
    interacted.value = false;
    previewFailed.value = false;
    previewNonce.value = Date.now();
  },
  { immediate: true },
);
</script>

<template>
  <article class="stream-item" :class="{ expanded: expanded || editing }">
    <button
      class="stream-summary"
      type="button"
      :aria-expanded="expanded || editing ? 'true' : 'false'"
      @click="emit('toggle')"
    >
      <span class="stream-summary-copy">
        <span class="stream-name">{{ stream.name }}</span>
        <span class="stream-brief">
          间隔 {{ stream.frame_interval_seconds }}s / {{ lineText }} / {{ frameText }}
        </span>
      </span>
      <span class="status" :class="stream.status">{{ stream.status }}</span>
      <span class="stream-disclosure">{{ expanded || editing ? "收起" : "详情" }}</span>
    </button>

    <div v-if="expanded || editing" class="stream-details">
      <div class="stream-url">{{ maskUrl(stream.stream_url) }}</div>
      <div class="media-meta">
        <span>间隔 {{ stream.frame_interval_seconds }}s</span>
        <span>计数线 {{ stream.counting_line ? "已设置" : "未设置" }}</span>
        <span>最后帧 {{ shortId(stream.last_frame_image_id) }}</span>
        <span v-if="stream.last_error">{{ stream.last_error }}</span>
      </div>
      <div class="stream-actions">
        <button
          class="mini-button primary"
          type="button"
          :disabled="busyAction === 'start'"
          @click="emit('start')"
        >
          {{ busyAction === "start" ? "启动中" : "启动" }}
        </button>
        <button class="mini-button" type="button" :disabled="busyAction === 'stop'" @click="emit('stop')">
          {{ busyAction === "stop" ? "停止中" : "停止" }}
        </button>
        <button class="mini-button" type="button" @click="emit('editLine')">设置线</button>
        <button
          class="mini-button danger"
          type="button"
          :disabled="busyAction === 'delete'"
          @click="emit('remove')"
        >
          {{ busyAction === "delete" ? "删除中" : "删除" }}
        </button>
        <button
          v-if="stream.counting_line"
          class="mini-button"
          type="button"
          :disabled="busyAction === 'clear-line'"
          @click="emit('clearLine')"
        >
          {{ busyAction === "clear-line" ? "清除中" : "清除线" }}
        </button>
      </div>

      <div v-if="editing" class="stream-line-editor">
        <div class="line-tool-head">
          <span>{{ stream.counting_line ? "调整计数线" : "设置计数线" }}</span>
          <span class="line-coordinate">{{ coordinate }}</span>
        </div>
        <div class="line-stage stream-line-stage compact-line-stage">
          <img
            ref="previewEl"
            class="line-frame"
            :src="previewUrl"
            alt="实时预览"
            @error="previewFailed = true"
          />
          <CountingLineCanvas
            v-model="points"
            class="registered-line-canvas"
            :media="previewEl"
            @click="interacted = true"
          />
          <div class="line-hint">{{ hint }}</div>
        </div>
        <div class="stream-actions">
          <button
            class="mini-button primary"
            type="button"
            :disabled="busyAction === 'save-line'"
            @click="emit('saveLine', points)"
          >
            {{ busyAction === "save-line" ? "保存中" : "保存线" }}
          </button>
          <button class="mini-button" type="button" @click="emit('cancelLine')">取消</button>
        </div>
      </div>
    </div>
  </article>
</template>

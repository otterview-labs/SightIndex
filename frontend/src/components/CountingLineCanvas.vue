<script lang="ts">
export interface LinePoint {
  x: number;
  y: number;
}

export interface CountingLine {
  x1: string;
  y1: string;
  x2: string;
  y2: string;
}

export function lineToPoints(line: unknown): LinePoint[] {
  if (!line || typeof line !== "object") return [];
  const value = line as Record<string, unknown>;
  return [
    { x: Number(value.x1), y: Number(value.y1) },
    { x: Number(value.x2), y: Number(value.y2) },
  ];
}

export function pointsToLine(points: LinePoint[]): CountingLine | null {
  if (points.length !== 2) return null;
  const [start, end] = points;
  return {
    x1: start.x.toFixed(4),
    y1: start.y.toFixed(4),
    x2: end.x.toFixed(4),
    y2: end.y.toFixed(4),
  };
}

export function formatCountingLine(line: unknown): string {
  if (!line || typeof line !== "object") return "未设置";
  const value = line as Record<string, unknown>;
  return (
    `${Number(value.x1).toFixed(2)},${Number(value.y1).toFixed(2)} -> ` +
    `${Number(value.x2).toFixed(2)},${Number(value.y2).toFixed(2)}`
  );
}
</script>

<script setup lang="ts">
import { onBeforeUnmount, onMounted, useTemplateRef, watch } from "vue";

const props = withDefaults(
  defineProps<{
    media?: HTMLImageElement | HTMLVideoElement | null;
    readonly?: boolean;
    strokeWidth?: number;
    pointRadius?: number;
    pointColor?: string;
  }>(),
  { media: null, readonly: false, strokeWidth: 3, pointRadius: 5, pointColor: "#2477ff" },
);

const points = defineModel<LinePoint[]>({ default: () => [] });

const canvas = useTemplateRef<HTMLCanvasElement>("canvas");

function clamp(value: number, min: number, max: number) {
  return Math.max(min, Math.min(max, value));
}

function paintRect(box: DOMRect) {
  const fallback = { left: 0, top: 0, width: box.width, height: box.height };
  const media = props.media;
  if (!media) return fallback;
  const width = "videoWidth" in media ? media.videoWidth : media.naturalWidth;
  const height = "videoHeight" in media ? media.videoHeight : media.naturalHeight;
  if (!width || !height || !box.width || !box.height) return fallback;
  const mediaRatio = width / height;
  const boxRatio = box.width / box.height;
  if (mediaRatio > boxRatio) {
    const painted = box.width / mediaRatio;
    return { left: 0, top: (box.height - painted) / 2, width: box.width, height: painted };
  }
  const painted = box.height * mediaRatio;
  return { left: (box.width - painted) / 2, top: 0, width: painted, height: box.height };
}

function draw() {
  const element = canvas.value;
  if (!element) return;
  const box = element.getBoundingClientRect();
  const ratio = window.devicePixelRatio || 1;
  element.width = Math.max(1, Math.round(box.width * ratio));
  element.height = Math.max(1, Math.round(box.height * ratio));
  const context = element.getContext("2d");
  if (!context) return;
  context.scale(ratio, ratio);
  context.clearRect(0, 0, box.width, box.height);
  if (!points.value.length) return;

  const paint = paintRect(box);
  context.lineWidth = props.strokeWidth;
  context.strokeStyle = "#6ea8ff";
  context.fillStyle = props.pointColor;
  if (props.readonly) {
    context.shadowColor = "rgba(0, 0, 0, 0.42)";
    context.shadowBlur = 6;
  }

  if (points.value.length === 2) {
    const [start, end] = points.value;
    context.beginPath();
    context.moveTo(paint.left + start.x * paint.width, paint.top + start.y * paint.height);
    context.lineTo(paint.left + end.x * paint.width, paint.top + end.y * paint.height);
    context.stroke();
  }
  for (const point of points.value) {
    context.beginPath();
    context.arc(
      paint.left + point.x * paint.width,
      paint.top + point.y * paint.height,
      props.pointRadius,
      0,
      Math.PI * 2,
    );
    context.fill();
  }
}

function onClick(event: MouseEvent) {
  if (props.readonly) return;
  const element = canvas.value;
  if (!element) return;
  const box = element.getBoundingClientRect();
  if (!box.width || !box.height) return;
  const paint = paintRect(box);
  if (!paint.width || !paint.height) return;
  const point = {
    x: clamp((event.clientX - box.left - paint.left) / paint.width, 0, 1),
    y: clamp((event.clientY - box.top - paint.top) / paint.height, 0, 1),
  };
  points.value = points.value.length >= 2 ? [point] : [...points.value, point];
}

let observer: ResizeObserver | undefined;

onMounted(() => {
  draw();
  observer = new ResizeObserver(draw);
  if (canvas.value) observer.observe(canvas.value);
});

onBeforeUnmount(() => observer?.disconnect());

watch([points, () => props.media], draw, { deep: true });

defineExpose({ draw });
</script>

<template>
  <canvas ref="canvas" @click="onClick"></canvas>
</template>

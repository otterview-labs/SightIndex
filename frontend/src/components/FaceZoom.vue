<script setup lang="ts">
import { onBeforeUnmount, onMounted, ref, useTemplateRef, watch } from "vue";

import type { FaceBox } from "./FaceBoxThumb.vue";

const props = defineProps<{ src: string; faceBox: FaceBox }>();

const frame = useTemplateRef<HTMLDivElement>("frame");
const image = useTemplateRef<HTMLImageElement>("image");
const style = ref<Record<string, string>>({});

const PADDING = 1.25;

function position() {
  const frameEl = frame.value;
  const imageEl = image.value;
  if (!frameEl || !imageEl || !imageEl.naturalWidth || !imageEl.naturalHeight) return;
  const rect = frameEl.getBoundingClientRect();
  if (!rect.width || !rect.height) return;

  const faceX = Number(props.faceBox.x ?? 0);
  const faceY = Number(props.faceBox.y ?? 0);
  const faceWidth = Math.max(1, Number(props.faceBox.width ?? 1));
  const faceHeight = Math.max(1, Number(props.faceBox.height ?? 1));
  const scale = Math.min(rect.width / (faceWidth * PADDING), rect.height / (faceHeight * PADDING));

  style.value = {
    width: `${imageEl.naturalWidth * scale}px`,
    height: `${imageEl.naturalHeight * scale}px`,
    left: `${rect.width / 2 - (faceX + faceWidth / 2) * scale}px`,
    top: `${rect.height / 2 - (faceY + faceHeight / 2) * scale}px`,
  };
}

let observer: ResizeObserver | undefined;

onMounted(() => {
  position();
  observer = new ResizeObserver(position);
  if (frame.value) observer.observe(frame.value);
});

onBeforeUnmount(() => observer?.disconnect());

watch(() => [props.src, props.faceBox], position);
</script>

<template>
  <div ref="frame" class="face-zoom-frame">
    <img ref="image" :src="src" :style="style" alt="人脸放大" loading="lazy" @load="position" />
  </div>
</template>

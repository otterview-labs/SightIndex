<script setup lang="ts">
import { onBeforeUnmount, onMounted, ref, useTemplateRef, watch } from "vue";

export interface FaceBox {
  x?: number | null;
  y?: number | null;
  width?: number | null;
  height?: number | null;
}

const props = defineProps<{
  src: string;
  alt?: string;
  faceBox?: FaceBox | null;
  linkClass?: string;
}>();

const frame = useTemplateRef<HTMLAnchorElement>("frame");
const image = useTemplateRef<HTMLImageElement>("image");
const boxStyle = ref<Record<string, string>>({});

function position() {
  const frameEl = frame.value;
  const imageEl = image.value;
  const box = props.faceBox;
  if (!frameEl || !imageEl || !box) return;
  if (!imageEl.naturalWidth || !imageEl.naturalHeight) return;

  const rect = frameEl.getBoundingClientRect();
  if (!rect.width || !rect.height) return;
  const imageRatio = imageEl.naturalWidth / imageEl.naturalHeight;
  const frameRatio = rect.width / rect.height;
  const renderedWidth = frameRatio > imageRatio ? rect.height * imageRatio : rect.width;
  const renderedHeight = frameRatio > imageRatio ? rect.height : rect.width / imageRatio;
  const offsetX = (rect.width - renderedWidth) / 2;
  const offsetY = (rect.height - renderedHeight) / 2;
  const scaleX = renderedWidth / imageEl.naturalWidth;
  const scaleY = renderedHeight / imageEl.naturalHeight;

  boxStyle.value = {
    left: `${Number(box.x ?? 0) * scaleX + offsetX}px`,
    top: `${Number(box.y ?? 0) * scaleY + offsetY}px`,
    width: `${Number(box.width ?? 0) * scaleX}px`,
    height: `${Number(box.height ?? 0) * scaleY}px`,
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
  <a
    ref="frame"
    :class="linkClass ?? 'observation-thumb'"
    :href="src"
    target="_blank"
    rel="noreferrer"
  >
    <img
      v-if="src"
      ref="image"
      :src="src"
      :alt="alt ?? '裁剪图'"
      loading="lazy"
      decoding="async"
      @load="position"
    />
    <span v-else>无图</span>
    <span v-if="faceBox" class="face-bbox" :style="boxStyle"></span>
  </a>
</template>

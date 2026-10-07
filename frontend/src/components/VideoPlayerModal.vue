<script setup lang="ts">
import { nextTick, onBeforeUnmount, onMounted, ref } from "vue";

import type { VideoPlayerState } from "@/composables/useVideoPlayer";

const props = defineProps<{
  video: VideoPlayerState;
}>();

const emit = defineEmits<{ (e: "close"): void }>();

const videoElement = ref<HTMLVideoElement | null>(null);
const closeButton = ref<HTMLButtonElement | null>(null);
let trigger: HTMLElement | null = null;
let previousOverflow = "";

// Seek as soon as the duration is known, clamped so the last frame is still visible.
function onLoadedMetadata() {
  const video = videoElement.value;
  if (!video) return;
  const duration = Number.isFinite(video.duration) ? video.duration : Number.POSITIVE_INFINITY;
  const seconds = Math.min(Math.max(props.video.offsetMs / 1000, 0), Math.max(duration - 0.05, 0));
  if (seconds > 0) video.currentTime = seconds;
}

function onKeydown(event: KeyboardEvent) {
  if (event.key === "Escape") {
    event.preventDefault();
    emit("close");
  } else if (event.key === "Tab") {
    // The dialog has exactly two controls; keep focus trapped between them.
    event.preventDefault();
    if (document.activeElement === videoElement.value) closeButton.value?.focus();
    else videoElement.value?.focus();
  }
}

onMounted(() => {
  trigger = document.activeElement instanceof HTMLElement ? document.activeElement : null;
  previousOverflow = document.documentElement.style.overflow;
  document.documentElement.style.overflow = "hidden";
  window.addEventListener("keydown", onKeydown);
  // The video itself is focused so the native controls (space/arrows) work immediately.
  void nextTick(() => videoElement.value?.focus());
});

onBeforeUnmount(() => {
  window.removeEventListener("keydown", onKeydown);
  document.documentElement.style.overflow = previousOverflow;
  if (trigger?.isConnected) trigger.focus();
});
</script>

<template>
  <Teleport to="body">
    <div
      class="video-player-modal"
      role="dialog"
      aria-modal="true"
      aria-label="视频定位播放"
      @click.self="emit('close')"
    >
      <button
        ref="closeButton"
        class="video-player-close"
        type="button"
        aria-label="关闭视频播放"
        title="关闭（Esc）"
        @click="emit('close')"
      >
        ×
      </button>
      <figure>
        <!-- Muted autoplay: the seek is the point, and browsers only auto-play muted. -->
        <video
          ref="videoElement"
          :src="video.videoUrl"
          controls
          autoplay
          muted
          playsinline
          preload="metadata"
          tabindex="0"
          @loadedmetadata="onLoadedMetadata"
        />
        <figcaption>
          <span v-if="video.caption">{{ video.caption }}</span>
          <span v-if="!video.exact" class="video-player-approx">
            起播位置为近似值（历史数据，误差不超过一个抽帧间隔）
          </span>
        </figcaption>
      </figure>
    </div>
  </Teleport>
</template>

<style scoped>
.video-player-modal {
  position: fixed;
  z-index: 1000;
  inset: 0;
  display: grid;
  padding: clamp(16px, 4vw, 48px);
  background: rgb(5 10 18 / 90%);
  backdrop-filter: blur(5px);
  place-items: center;
}

.video-player-modal figure {
  display: grid;
  max-width: 100%;
  max-height: 100%;
  margin: 0;
  gap: 12px;
  place-items: center;
}

.video-player-modal video {
  display: block;
  max-width: min(94vw, 1600px);
  max-height: calc(100vh - 112px);
  border-radius: 12px;
  background: #000;
  box-shadow: 0 22px 70px rgb(0 0 0 / 45%);
}

.video-player-modal figcaption {
  display: grid;
  gap: 4px;
  max-width: min(90vw, 900px);
  color: rgb(255 255 255 / 88%);
  font-size: 13px;
  text-align: center;
}

.video-player-approx {
  color: rgb(255 255 255 / 60%);
  font-size: 12px;
}

.video-player-close {
  position: fixed;
  z-index: 1;
  top: max(14px, env(safe-area-inset-top));
  right: max(14px, env(safe-area-inset-right));
  display: grid;
  width: 42px;
  height: 42px;
  padding: 0;
  border: 1px solid rgb(255 255 255 / 28%);
  border-radius: 999px;
  background: rgb(255 255 255 / 12%);
  color: #fff;
  cursor: pointer;
  font-size: 28px;
  line-height: 1;
  place-items: center;
}

.video-player-close:hover,
.video-player-close:focus-visible {
  background: rgb(255 255 255 / 22%);
}

.video-player-close:focus-visible {
  outline: 3px solid #fff;
  outline-offset: 3px;
}
</style>

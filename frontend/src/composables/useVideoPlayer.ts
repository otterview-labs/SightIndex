import { ref } from "vue";

import { videos } from "@/api/client";

export interface VideoPlayerState {
  videoUrl: string;
  offsetMs: number;
  exact: boolean;
  caption: string;
}

/**
 * Opens a source video at the position a frame was extracted from. Rows ingested
 * after the video-position link landed carry the exact url/offset; anything older
 * falls back to GET /api/images/{id}/video-position, which reconstructs it.
 */
export function useVideoPlayer() {
  const activeVideo = ref<VideoPlayerState | null>(null);
  const resolvingImageId = ref("");

  async function openVideoAt(options: {
    imageId?: string | null;
    videoUrl?: string | null;
    videoOffsetMs?: number | null;
    caption?: string;
  }) {
    if (options.videoUrl && options.videoOffsetMs != null) {
      activeVideo.value = {
        videoUrl: options.videoUrl,
        offsetMs: options.videoOffsetMs,
        exact: true,
        caption: options.caption ?? "",
      };
      return;
    }
    if (!options.imageId) return;
    resolvingImageId.value = options.imageId;
    try {
      const position = await videos.position(options.imageId);
      activeVideo.value = {
        videoUrl: position.video_url,
        offsetMs: position.video_offset_ms,
        exact: position.exact,
        caption: options.caption ?? "",
      };
    } finally {
      resolvingImageId.value = "";
    }
  }

  function closeVideo() {
    activeVideo.value = null;
  }

  return { activeVideo, resolvingImageId, openVideoAt, closeVideo };
}

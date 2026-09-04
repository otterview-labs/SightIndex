import { computed, ref } from "vue";

import {
  crops as cropsApi,
  images as imagesApi,
  media as mediaApi,
  streams as streamsApi,
} from "@/api/client";
import type { ImageRead, PersonCropRead, VideoStream } from "@/api/types";

const streams = ref<VideoStream[]>([]);
const images = ref<ImageRead[]>([]);
const crops = ref<PersonCropRead[]>([]);
const streamFrames = ref<Record<string, ImageRead>>({});
const imageTotal = ref(0);
const cropTotal = ref(0);
const loadingMoreImages = ref(false);
const loadingMoreCrops = ref(false);

const runningCount = computed(
  () => streams.value.filter((stream) => stream.status === "running").length,
);

const latestError = computed(() => streams.value.find((stream) => stream.last_error));

const latestStatus = computed(() => {
  const failing = latestError.value;
  if (failing) return `${failing.name}: ${failing.last_error}`;
  return streams.value.length ? "视频流状态正常" : "等待注册视频流";
});

const cameraOptions = computed(() => {
  const seen = new Map<string, string>();
  for (const stream of streams.value) {
    if (stream.camera_id && !seen.has(stream.camera_id)) seen.set(stream.camera_id, stream.name);
  }
  return [...seen].map(([id, name]) => ({ id, name }));
});

async function loadStreamFrames(list: VideoStream[]): Promise<Record<string, ImageRead>> {
  const ids = [...new Set(list.map((stream) => stream.last_frame_image_id).filter(Boolean))];
  if (!ids.length) return {};
  const entries = await Promise.all(
    ids.map(async (id) => {
      try {
        return [id as string, await imagesApi.get(id as string)] as const;
      } catch {
        // A frame can be pruned between listing the stream and fetching it.
        return null;
      }
    }),
  );
  return Object.fromEntries(entries.filter((entry) => entry !== null));
}

async function refresh() {
  const [nextStreams, nextImages, nextCrops, counts] = await Promise.all([
    streamsApi.list(),
    imagesApi.listWithCrops(60),
    cropsApi.list(80),
    mediaApi.counts(),
  ]);
  streams.value = nextStreams;
  images.value = nextImages;
  crops.value = nextCrops;
  imageTotal.value = counts.image_with_crops_count;
  cropTotal.value = counts.person_crop_count;
  streamFrames.value = await loadStreamFrames(nextStreams);
}

async function loadMoreImages() {
  if (loadingMoreImages.value || images.value.length >= imageTotal.value) return;
  loadingMoreImages.value = true;
  try {
    const next = await imagesApi.listWithCrops(200, images.value.length);
    const known = new Set(images.value.map((image) => image.id));
    images.value.push(...next.filter((image) => !known.has(image.id)));
  } finally {
    loadingMoreImages.value = false;
  }
}

async function loadMoreCrops() {
  if (loadingMoreCrops.value || crops.value.length >= cropTotal.value) return;
  loadingMoreCrops.value = true;
  try {
    const next = await cropsApi.list(200, crops.value.length);
    const known = new Set(crops.value.map((crop) => crop.id));
    crops.value.push(...next.filter((crop) => !known.has(crop.id)));
  } finally {
    loadingMoreCrops.value = false;
  }
}

export function useSummary() {
  return {
    streams,
    images,
    crops,
    streamFrames,
    imageTotal,
    cropTotal,
    loadingMoreImages,
    loadingMoreCrops,
    runningCount,
    latestError,
    latestStatus,
    cameraOptions,
    refresh,
    loadMoreImages,
    loadMoreCrops,
  };
}

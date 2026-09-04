<script setup lang="ts">
import { computed, onBeforeUnmount, onMounted, ref, useTemplateRef, watch } from "vue";

import { images as imagesApi, streams as streamsApi, videos as videosApi } from "@/api/client";
import type { VideoStream } from "@/api/types";
import CountingLineCanvas, {
  type LinePoint,
  lineToPoints,
  pointsToLine,
} from "@/components/CountingLineCanvas.vue";
import FileField from "@/components/FileField.vue";
import StreamItem from "@/components/StreamItem.vue";
import { useSummary } from "@/composables/useSummary";
import { useToast } from "@/composables/useToast";
import { fmtClock, fmtTime, shortId } from "@/utils/format";

const AUTO_REFRESH_MS = 5000;
const LIVE_COUNT_MS = 5000;

const { showError, toast } = useToast();
const {
  streams,
  images,
  crops,
  streamFrames,
  runningCount,
  latestError,
  latestStatus,
  imageTotal,
  cropTotal,
  loadingMoreImages,
  loadingMoreCrops,
  refresh,
  loadMoreImages,
  loadMoreCrops,
} = useSummary();

const lastUpdated = ref("系统未刷新");
const refreshing = ref(false);
const autoRefresh = ref(false);
const processing = ref(false);
const mediaView = ref<"crops" | "frames">("crops");

const liveStreamId = ref<string>("");
const livePlaying = ref(false);
const liveSrc = ref("");
const liveHint = ref("选择一个视频流后播放");
const liveCount = ref("过线 0");
const liveClock = ref("--:--:--");
const liveImage = useTemplateRef<HTMLImageElement>("liveImage");
const livePoints = ref<LinePoint[]>([]);

const selectedLiveStream = computed(() =>
  streams.value.find((stream) => stream.id === liveStreamId.value),
);

const streamForm = ref({
  name: "",
  stream_url: "",
  location_name: "",
  protocol: "rtsp",
  frame_interval_seconds: 2,
});
const streamPoints = ref<LinePoint[]>([]);
const streamHint = ref("点击两点设置计数线");
const creatingStream = ref(false);
const configDetails = useTemplateRef<HTMLDetailsElement>("configDetails");

const videoForm = ref({ frame_interval_seconds: 1, max_frames: 120, store_empty_frames: false });
const videoPoints = ref<LinePoint[]>([]);
const videoHint = ref("选择视频后，在画面上点击两点画线");
const videoResult = ref("未处理");
const videoFile = ref<File | null>(null);
const videoPreview = useTemplateRef<HTMLVideoElement>("videoPreview");
const videoPlaying = ref(false);
const uploadingVideo = ref(false);
let videoObjectUrl: string | null = null;

const expandedIds = ref(new Set<string>());
const editingStreamId = ref<string | null>(null);
const editorHint = ref("");
const busyStream = ref<{ id: string; action: string } | null>(null);

function busyActionFor(stream: VideoStream) {
  return busyStream.value?.id === stream.id ? busyStream.value.action : null;
}

async function loadMonitor() {
  refreshing.value = true;
  try {
    await refresh();
    syncLivePoints();
    if (!streams.value.length) liveHint.value = "在右侧「视频源」添加 RTSP / HTTP 地址后即可预览";
    lastUpdated.value = `刷新 ${fmtClock(new Date())}`;
  } catch (error) {
    showError(error);
  } finally {
    refreshing.value = false;
  }
}

function syncLivePoints() {
  livePoints.value = lineToPoints(selectedLiveStream.value?.counting_line);
}

async function refreshLiveCount() {
  const stream = selectedLiveStream.value;
  if (!stream) return;
  try {
    const result = await streamsApi.counts(stream.id);
    liveCount.value = `本流 ${result.counting_event_count} / 总计 ${result.total_counting_event_count}`;
  } catch (error) {
    showError(error);
  }
}

let liveTimer: number | undefined;
let autoTimer: number | undefined;
let clockTimer: number | undefined;

function playLive() {
  const stream = selectedLiveStream.value;
  if (!stream) {
    toast("请先选择视频流");
    return;
  }
  liveSrc.value = `/api/streams/${stream.id}/mjpeg?fps=8&jpeg_quality=92&v=${Date.now()}`;
  livePlaying.value = true;
  syncLivePoints();
  void refreshLiveCount();
  window.clearInterval(liveTimer);
  liveTimer = window.setInterval(refreshLiveCount, LIVE_COUNT_MS);
  liveHint.value = `${stream.name} 实时预览中`;
}

function stopLive() {
  liveSrc.value = "";
  livePlaying.value = false;
  window.clearInterval(liveTimer);
  liveTimer = undefined;
  livePoints.value = [];
  liveHint.value = "预览已停止";
}

function onLiveError() {
  livePlaying.value = false;
  window.clearInterval(liveTimer);
  liveTimer = undefined;
  liveHint.value = "实时预览失败，请确认流已启动或重新播放";
}

function onLiveStreamChange() {
  if (livePlaying.value) {
    playLive();
    return;
  }
  syncLivePoints();
  void refreshLiveCount();
}

function toggleAutoRefresh() {
  autoRefresh.value = !autoRefresh.value;
  if (autoRefresh.value) {
    autoTimer = window.setInterval(() => {
      void loadMonitor().then(refreshLiveCount);
    }, AUTO_REFRESH_MS);
    toast("已开启自动刷新");
  } else {
    window.clearInterval(autoTimer);
    autoTimer = undefined;
    toast("已关闭自动刷新");
  }
}

async function processLatest() {
  const latest = images.value[0];
  if (!latest) {
    toast("暂无可处理帧");
    return;
  }
  processing.value = true;
  try {
    await imagesApi.process(latest.id);
    toast("最新帧已处理");
    await loadMonitor();
  } catch (error) {
    showError(error);
  } finally {
    processing.value = false;
  }
}

async function createStream() {
  creatingStream.value = true;
  try {
    await streamsApi.create({
      name: streamForm.value.name,
      stream_url: streamForm.value.stream_url,
      location_name: streamForm.value.location_name.trim() || null,
      protocol: streamForm.value.protocol,
      frame_interval_seconds: Number(streamForm.value.frame_interval_seconds || 2),
      counting_line: pointsToLine(streamPoints.value) as never,
    });
    streamForm.value = {
      name: "",
      stream_url: "",
      location_name: "",
      protocol: "rtsp",
      frame_interval_seconds: 2,
    };
    streamPoints.value = [];
    streamHint.value = "点击两点设置计数线";
    toast("视频流已新增");
    await loadMonitor();
  } catch (error) {
    showError(error);
  } finally {
    creatingStream.value = false;
  }
}

function toggleStream(stream: VideoStream) {
  const next = new Set(expandedIds.value);
  if (next.has(stream.id)) {
    next.delete(stream.id);
    if (editingStreamId.value === stream.id) editingStreamId.value = null;
  } else {
    next.add(stream.id);
  }
  expandedIds.value = next;
}

async function streamAction(stream: VideoStream, action: "start" | "stop") {
  busyStream.value = { id: stream.id, action };
  try {
    await (action === "start" ? streamsApi.start(stream.id) : streamsApi.stop(stream.id));
    toast(action === "start" ? "启动请求已发送" : "停止请求已发送");
    await loadMonitor();
  } catch (error) {
    showError(error);
  } finally {
    busyStream.value = null;
  }
}

async function removeStream(stream: VideoStream) {
  if (!window.confirm("确定删除这个视频流？历史图片和计数记录会保留。")) return;
  busyStream.value = { id: stream.id, action: "delete" };
  try {
    await streamsApi.remove(stream.id);
    if (editingStreamId.value === stream.id) editingStreamId.value = null;
    if (liveStreamId.value === stream.id) stopLive();
    toast("视频流已删除");
    await loadMonitor();
  } catch (error) {
    showError(error);
  } finally {
    busyStream.value = null;
  }
}

async function editLine(stream: VideoStream) {
  expandedIds.value = new Set(expandedIds.value).add(stream.id);
  editingStreamId.value = stream.id;
  editorHint.value = "";
  const hasFrame = stream.last_frame_image_id && streamFrames.value[stream.last_frame_image_id];
  if (hasFrame) return;
  editorHint.value = "正在抓取最近一帧...";
  try {
    const image = await streamsApi.snapshot(stream.id);
    streamFrames.value = { ...streamFrames.value, [image.id]: image };
    const current = streams.value.find((item) => item.id === stream.id);
    if (current) current.last_frame_image_id = image.id;
    editorHint.value = "";
  } catch {
    if (editingStreamId.value === stream.id) editorHint.value = "抓帧失败；也可按比例点击两点";
  }
}

async function saveLine(stream: VideoStream, points: LinePoint[]) {
  const line = pointsToLine(points);
  if (!line) {
    toast("请先点击两点画线");
    return;
  }
  busyStream.value = { id: stream.id, action: "save-line" };
  try {
    await streamsApi.saveCountingLine(stream.id, line as never);
    editingStreamId.value = null;
    toast("计数线已保存");
    await loadMonitor();
  } catch (error) {
    showError(error);
  } finally {
    busyStream.value = null;
  }
}

async function clearLine(stream: VideoStream) {
  busyStream.value = { id: stream.id, action: "clear-line" };
  try {
    await streamsApi.saveCountingLine(stream.id, null);
    if (editingStreamId.value === stream.id) editingStreamId.value = null;
    toast("计数线已清除");
    await loadMonitor();
  } catch (error) {
    showError(error);
  } finally {
    busyStream.value = null;
  }
}

function primeVideoPreview() {
  videoPoints.value = [];
  if (videoObjectUrl) {
    URL.revokeObjectURL(videoObjectUrl);
    videoObjectUrl = null;
  }
  const file = videoFile.value;
  if (!file || !videoPreview.value) {
    videoHint.value = "选择视频后，在画面上点击两点画线";
    return;
  }
  videoObjectUrl = URL.createObjectURL(file);
  videoPreview.value.src = videoObjectUrl;
  videoPreview.value.load();
  videoHint.value = "点击画面上的两个点设置计数线";
}

async function toggleVideoPreview() {
  const element = videoPreview.value;
  if (!element?.src) {
    toast("请先选择视频");
    return;
  }
  if (element.paused) await element.play();
  else element.pause();
}

async function uploadVideo() {
  const file = videoFile.value;
  if (!file) return;
  const params: Record<string, string | number | boolean> = {
    frame_interval_seconds: videoForm.value.frame_interval_seconds || 1,
    max_frames: videoForm.value.max_frames || 120,
  };
  if (videoForm.value.store_empty_frames) params.store_empty_frames = true;
  const line = pointsToLine(videoPoints.value);
  if (line) {
    params.line_x1 = line.x1;
    params.line_y1 = line.y1;
    params.line_x2 = line.x2;
    params.line_y2 = line.y2;
  }
  const body = new FormData();
  body.set("file", file);
  uploadingVideo.value = true;
  videoResult.value = "处理中";
  try {
    const result = await videosApi.upload(body, params);
    videoResult.value = `${result.counting_events_created} 计数 / ${result.crops_created} 裁剪`;
    toast(`视频处理完成：${result.counting_events_created} 次过线`);
    videoFile.value = null;
    videoPoints.value = [];
    if (videoPreview.value) videoPreview.value.removeAttribute("src");
    videoForm.value = { frame_interval_seconds: 1, max_frames: 120, store_empty_frames: false };
    await loadMonitor();
  } catch (error) {
    videoResult.value = "处理失败";
    showError(error);
  } finally {
    uploadingVideo.value = false;
  }
}

function openConfig() {
  if (!configDetails.value) return;
  configDetails.value.open = true;
  configDetails.value.querySelector("summary")?.scrollIntoView({
    behavior: "smooth",
    block: "nearest",
  });
}

watch(videoFile, primeVideoPreview);

onMounted(() => {
  clockTimer = window.setInterval(() => {
    liveClock.value = fmtClock(new Date());
  }, 1000);
  void loadMonitor();
});

onBeforeUnmount(() => {
  window.clearInterval(liveTimer);
  window.clearInterval(autoTimer);
  window.clearInterval(clockTimer);
  if (videoObjectUrl) URL.revokeObjectURL(videoObjectUrl);
});
</script>

<template>
  <main class="workspace reference-workspace">
    <Teleport to="#page-actions">
      <span class="last-updated">{{ lastUpdated }}</span>
      <button
        class="button ghost"
        :class="{ primary: autoRefresh }"
        type="button"
        :aria-pressed="autoRefresh"
        @click="toggleAutoRefresh"
      >
        {{ autoRefresh ? "自动刷新中" : "自动刷新" }}
      </button>
      <button
        class="button primary"
        type="button"
        :disabled="refreshing"
        @click="loadMonitor().then(() => toast('已刷新'))"
      >
        {{ refreshing ? "刷新中" : "刷新" }}
      </button>
    </Teleport>

    <section class="overview" aria-label="运行概览">
      <div class="metric"><span>运行视频流</span><strong>{{ runningCount }}</strong></div>
      <div class="metric"><span>全部有人帧</span><strong>{{ imageTotal }}</strong></div>
      <div class="metric"><span>全部人物裁剪</span><strong>{{ cropTotal }}</strong></div>
      <div class="metric wide" :class="{ 'has-error': latestError }">
        <span>最近状态</span><strong>{{ latestStatus }}</strong>
      </div>
    </section>

    <section class="panel media-panel" aria-labelledby="mediaTitle">
      <div class="reference-panel-head">
        <div class="panel-title-line">
          <h2 id="mediaTitle">分析画面</h2>
          <span>截取摄像头视频</span>
        </div>
        <span class="status-pill ok"><i></i>视频控制台</span>
      </div>

      <div class="live-player" aria-labelledby="livePlayerTitle">
        <div class="section-head compact">
          <div>
            <h2 id="livePlayerTitle">实时预览</h2>
          </div>
          <div class="media-actions">
            <select v-model="liveStreamId" aria-label="选择视频流" @change="onLiveStreamChange">
              <option v-if="!streams.length" value="">暂无视频流</option>
              <option v-for="stream in streams" :key="stream.id" :value="stream.id">
                {{ stream.name }} / {{ stream.status }}
              </option>
            </select>
            <button class="button primary" type="button" @click="playLive">播放实时流</button>
            <button class="button ghost" type="button" @click="stopLive">停止预览</button>
          </div>
        </div>
        <div
          class="live-stage"
          :class="{ active: livePlaying, 'is-idle': !livePlaying }"
          aria-label="实时视频预览"
        >
          <img
            v-show="liveSrc"
            ref="liveImage"
            :class="{ active: livePlaying }"
            :src="liveSrc || undefined"
            alt="实时视频流预览"
            @error="onLiveError"
          />
          <CountingLineCanvas
            v-model="livePoints"
            readonly
            :media="liveImage"
            :stroke-width="4"
            :point-radius="6"
            point-color="#6ea8ff"
          />
          <div class="viewport-badges" aria-hidden="true">
            <span class="view-badge live"><i></i>LIVE</span>
            <span class="view-badge stream"><i></i>STREAM</span>
          </div>
          <div class="live-count-badge" :class="{ active: livePlaying }" role="status" aria-live="polite">
            {{ liveCount }}
          </div>
          <div v-if="!livePlaying" class="stage-placeholder">
            <span class="stage-icon" aria-hidden="true">
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6">
                <rect x="2.5" y="6" width="13" height="12" rx="2" />
                <path d="m15.5 12 6-3.5v11l-6-3.5z" />
              </svg>
            </span>
            <strong>{{ streams.length ? "选择视频流后播放" : "还没有接入视频流" }}</strong>
            <span>{{ liveHint }}</span>
          </div>
          <div v-else class="live-hint">{{ liveHint }}</div>
          <div class="live-stage-meta">
            <span>CH-01 / Live Stream</span><span>{{ liveClock }}</span>
          </div>
        </div>
      </div>

      <div class="reference-toolbar" aria-label="视频操作">
        <button class="button ghost" type="button" :disabled="processing" @click="processLatest">
          {{ processing ? "处理中" : "处理最新帧" }}
        </button>
        <div class="segmented">
          <button
            class="segment"
            :class="{ active: mediaView === 'crops' }"
            type="button"
            @click="mediaView = 'crops'"
          >
            人物裁剪
          </button>
          <button
            class="segment"
            :class="{ active: mediaView === 'frames' }"
            type="button"
            @click="mediaView = 'frames'"
          >
            原始帧
          </button>
        </div>
        <span class="toolbar-note">实时流画面与最近识别结果</span>
      </div>

      <div class="section-head media-results-head">
        <div>
          <h2>识别结果</h2>
          <p>默认展示截取出的人物，原帧仅用于排查画面来源。</p>
        </div>
      </div>

      <div class="media-grid" :class="{ hidden: mediaView !== 'crops' }">
        <div v-if="!crops.length" class="empty">
          <strong>还没有人物裁剪</strong>
        </div>
        <a
          v-for="crop in crops"
          :key="crop.id"
          class="media-item"
          :href="crop.crop_url"
          target="_blank"
          rel="noreferrer"
        >
          <img :src="crop.crop_url" alt="裁剪" loading="lazy" />
          <div class="media-meta">
            <strong>裁剪 {{ shortId(crop.id) }}</strong>
            <span>图片 {{ shortId(crop.image_id) }}</span>
            <span>{{ fmtTime(crop.captured_at || crop.created_at) }}</span>
          </div>
        </a>
        <button
          v-if="crops.length < cropTotal"
          class="button ghost media-load-more"
          type="button"
          :disabled="loadingMoreCrops"
          @click="loadMoreCrops"
        >
          {{ loadingMoreCrops ? "加载中" : `继续加载（已显示 ${crops.length} / ${cropTotal}）` }}
        </button>
      </div>

      <div class="media-grid" :class="{ hidden: mediaView !== 'frames' }">
        <div v-if="!images.length" class="empty">
          <strong>还没有原帧</strong>
        </div>
        <a
          v-for="image in images"
          :key="image.id"
          class="media-item"
          :href="image.thumbnail_url || image.image_url"
          target="_blank"
          rel="noreferrer"
        >
          <img :src="image.thumbnail_url || image.image_url" :alt="image.thumbnail_url ? '识别框' : '原帧'" loading="lazy" />
          <div class="media-meta">
            <strong>{{ image.thumbnail_url ? "识别框" : "原帧" }} {{ shortId(image.id) }}</strong>
            <span>图片 {{ shortId(image.id) }}</span>
            <span>{{ fmtTime(image.captured_at || image.created_at) }}</span>
          </div>
        </a>
        <button
          v-if="images.length < imageTotal"
          class="button ghost media-load-more"
          type="button"
          :disabled="loadingMoreImages"
          @click="loadMoreImages"
        >
          {{ loadingMoreImages ? "加载中" : `继续加载（已显示 ${images.length} / ${imageTotal}）` }}
        </button>
      </div>
    </section>

    <section class="panel stream-panel" aria-labelledby="streamTitle">
      <div class="section-head">
        <div>
          <h2 id="streamTitle">视频源</h2>
        </div>
        <span class="source-count"><span>{{ streams.length }}</span> 个源</span>
      </div>

      <div class="source-actions">
        <button class="button ghost" type="button" @click="openConfig">选择视频流</button>
        <button class="button ghost" type="button" @click="openConfig">添加 RTSP</button>
      </div>

      <div class="list-head source-list-head">
        <span>已注册视频源</span>
        <span class="source-count-text">在线状态</span>
      </div>
      <div class="stream-list">
        <div v-if="!streams.length" class="empty">
          <strong>还没有视频流</strong>
        </div>
        <StreamItem
          v-for="stream in streams"
          :key="stream.id"
          :stream="stream"
          :expanded="expandedIds.has(stream.id)"
          :editing="editingStreamId === stream.id"
          :frame="stream.last_frame_image_id ? streamFrames[stream.last_frame_image_id] : null"
          :busy-action="busyActionFor(stream)"
          :editor-hint="editingStreamId === stream.id ? editorHint : ''"
          @toggle="toggleStream(stream)"
          @start="streamAction(stream, 'start')"
          @stop="streamAction(stream, 'stop')"
          @remove="removeStream(stream)"
          @edit-line="editLine(stream)"
          @cancel-line="editingStreamId = null"
          @save-line="(points) => saveLine(stream, points)"
          @clear-line="clearLine(stream)"
        />
      </div>

      <details ref="configDetails" class="config-disclosure">
        <summary>添加 RTSP / HTTP 视频流</summary>
        <form class="form-grid" @submit.prevent="createStream">
          <label>
            名称
            <input v-model="streamForm.name" placeholder="一楼门口" required />
          </label>
          <label>
            流地址
            <input
              v-model="streamForm.stream_url"
              placeholder="rtsp://user:password@host/stream1"
              required
            />
          </label>
          <label>
            地点
            <input v-model="streamForm.location_name" placeholder="研发中心 3F 产品部" />
          </label>
          <div class="field-row">
            <label>
              协议
              <select v-model="streamForm.protocol">
                <option value="rtsp">RTSP</option>
                <option value="http">HTTP</option>
                <option value="file">FILE</option>
              </select>
            </label>
            <label>
              间隔秒
              <input
                v-model.number="streamForm.frame_interval_seconds"
                type="number"
                min="0.2"
                max="3600"
                step="0.1"
              />
            </label>
          </div>
          <div class="line-tool">
            <div class="line-tool-head">
              <span>过线统计</span>
              <button class="mini-button" type="button" @click="streamPoints = []">清除</button>
            </div>
            <div class="line-stage stream-line-stage">
              <CountingLineCanvas v-model="streamPoints" />
              <div class="line-hint">
                {{ streamPoints.length === 2 ? "计数线已设置" : streamHint }}
              </div>
            </div>
          </div>
          <button class="button primary wide" type="submit" :disabled="creatingStream">
            {{ creatingStream ? "新增中" : "新增视频流" }}
          </button>
        </form>
      </details>

      <details class="config-disclosure">
        <summary>导入本地视频并识别</summary>
        <form class="form-grid video-form" @submit.prevent="uploadVideo">
          <div class="list-head">
            <span>视频文件</span>
            <span>{{ videoResult }}</span>
          </div>
          <FileField
            v-model="videoFile"
            label="上传视频"
            accept="video/*"
            hint="选择本地视频后可在画面上画计数线"
            required
          />
          <div class="line-tool">
            <div class="line-tool-head">
              <span>计数线</span>
              <button class="mini-button" type="button" @click="toggleVideoPreview">
                {{ videoPlaying ? "暂停" : "播放" }}
              </button>
              <button class="mini-button" type="button" @click="videoPoints = []">清除</button>
            </div>
            <div class="line-stage">
              <video
                ref="videoPreview"
                muted
                playsinline
                preload="metadata"
                @play="videoPlaying = true"
                @pause="videoPlaying = false"
                @ended="videoPlaying = false"
              ></video>
              <CountingLineCanvas v-model="videoPoints" :media="videoPreview" />
              <div class="line-hint">
                {{ videoPoints.length === 2 ? "计数线已设置" : videoHint }}
              </div>
            </div>
          </div>
          <div class="field-row">
            <label>
              抽帧间隔秒
              <input
                v-model.number="videoForm.frame_interval_seconds"
                type="number"
                min="0.1"
                max="3600"
                step="0.1"
              />
            </label>
            <label>
              最大帧数
              <input v-model.number="videoForm.max_frames" type="number" min="1" max="2000" step="1" />
            </label>
          </div>
          <label class="check-row">
            <input v-model="videoForm.store_empty_frames" type="checkbox" />
            保留无人帧
          </label>
          <button class="button ghost wide" type="submit" :disabled="uploadingVideo">
            {{ uploadingVideo ? "识别中" : "上传并识别视频" }}
          </button>
        </form>
      </details>
    </section>
  </main>
</template>

<script setup lang="ts">
import { computed, nextTick, onBeforeUnmount, onMounted, ref, watch } from "vue";
import { RouterLink, useRoute } from "vue-router";

import { crops as cropsApi, reid as reidApi } from "@/api/client";
import type {
  PersonCropRead,
  ReidCameraLink,
  ReidFaceCoverage,
  ReidFeedbackRead,
  ReidLinkResponse,
  ReidMatchItem,
  ReidStatusResponse,
} from "@/api/types";
import EmptyState from "@/components/EmptyState.vue";
import FileField from "@/components/FileField.vue";
import ReidFeedbackButtons from "@/components/ReidFeedbackButtons.vue";
import ReidEvidenceSummary from "@/components/ReidEvidenceSummary.vue";
import ReidFaceCoverageSummary from "@/components/ReidFaceCoverageSummary.vue";
import { useToast } from "@/composables/useToast";
import { fmtTime, formatScore, shortId } from "@/utils/format";

const route = useRoute();
const { showError, toast } = useToast();

const status = ref<ReidStatusResponse | null>(null);
// Set when arriving from the observation table's 找相似 link.
const sourceCropId = ref<string | null>(null);
// Judging a match needs the query in view; an id alone tells the operator nothing.
const sourceCrop = ref<PersonCropRead | null>(null);
// Where else this person most likely went. Separate from the match list because it answers a
// different question and is deliberately not threshold-gated.
const cameraLinks = ref<ReidLinkResponse | null>(null);
const showWeakCameraLinks = ref(false);
const queryFile = ref<File | null>(null);
const queryPreview = ref("");
const results = ref<ReidMatchItem[] | null>(null);
const resultsFaceCoverage = ref<ReidFaceCoverage | null>(null);
// Non-null only when the visible result list was actually queried from this stored crop. An
// uploaded file can coexist with a crop_id in the URL, but feedback must never be attached to the
// old crop in that case.
const resultsQueryCropId = ref<string | null>(null);
const queryFrameCount = ref(1);
const searching = ref(false);
const rebuilding = ref(false);
const enlargedImage = ref<{ src: string; alt: string; caption: string } | null>(null);
const feedbackByCandidate = ref<Record<string, ReidFeedbackRead>>({});
const feedbackSaving = ref<Set<string>>(new Set());
const lightboxCloseButton = ref<HTMLButtonElement | null>(null);
let lightboxTrigger: HTMLElement | null = null;
let previousPageOverflow = "";
let searchController: AbortController | null = null;
let searchSequence = 0;
let linksSequence = 0;
// A new context invalidates ALL outstanding crop/links/feedback requests, even A → B → A.
// A retry within the same context only advances searchSequence.
let queryGeneration = 0;
const activeQueryCropId = computed(() => queryFile.value ? null : sourceCropId.value);

function isCurrentQuery(cropId: string, generation: number): boolean {
  return generation === queryGeneration && activeQueryCropId.value === cropId;
}

const resultSummary = computed(() => {
  const items = results.value;
  if (!items) return "";
  const frames = items.reduce((total, item) => total + (item.frame_count ?? 1), 0);
  return frames > items.length ? `${items.length} 次出现 · ${frames} 帧` : `${items.length} 条`;
});

const coverage = computed(() => {
  const value = status.value;
  if (!value) return "";
  const total = value.indexed_crops + value.pending_crops;
  if (!total) return "还没有可索引的裁剪";
  return `${value.indexed_crops} / ${total} 已建索引`;
});

const ready = computed(() => status.value?.ready ?? false);

const credibleCameraLinks = computed(() =>
  (cameraLinks.value?.links ?? []).filter(
    (link) => link.face_match !== false && link.evidence_level !== "rejected"
      && (link.beats_chance || link.face_match === true || link.evidence_level === "reliable"),
  ),
);

const weakCameraLinks = computed(() =>
  (cameraLinks.value?.links ?? []).filter(
    (link) => link.face_match !== false && link.evidence_level !== "rejected"
      && !link.beats_chance && link.face_match !== true && link.evidence_level !== "reliable",
  ),
);

const visibleCameraLinks = computed(() =>
  showWeakCameraLinks.value
    ? [...credibleCameraLinks.value, ...weakCameraLinks.value]
    : credibleCameraLinks.value,
);

const currentFeedbackCount = computed(() => Object.keys(feedbackByCandidate.value).length);

function feedbackValue(candidateCropId: string): boolean | null {
  return feedbackByCandidate.value[candidateCropId]?.same_person ?? null;
}

function feedbackIsSaving(candidateCropId: string): boolean {
  return Boolean(
    sourceCropId.value
      && feedbackSaving.value.has(`${sourceCropId.value}:${candidateCropId}`),
  );
}

async function loadFeedback(queryCropId: string) {
  const generation = queryGeneration;
  try {
    const rows = await reidApi.feedback(queryCropId);
    if (!isCurrentQuery(queryCropId, generation)) return;
    // An initial GET can finish after a newly saved manual judgement. Keep the newer local row.
    feedbackByCandidate.value = {
      ...Object.fromEntries(rows.map((row) => [row.candidate_crop_id, row])),
      ...feedbackByCandidate.value,
    };
  } catch (error) {
    if (!isCurrentQuery(queryCropId, generation)) return;
    showError(error);
  }
}

async function saveFeedback(
  candidate: ReidMatchItem | ReidCameraLink,
  samePerson: boolean,
  source: "search" | "camera_link",
) {
  const candidateCropId = candidate.crop_id;
  const queryCropId = activeQueryCropId.value;
  if (!queryCropId) return;
  if (source === "search" && resultsQueryCropId.value !== queryCropId) return;
  if (source === "camera_link" && !cameraLinks.value?.links.some((link) => link.crop_id === candidateCropId)) return;
  const generation = queryGeneration;
  const savingKey = `${queryCropId}:${candidateCropId}`;
  if (feedbackSaving.value.has(savingKey)) return;
  feedbackSaving.value = new Set(feedbackSaving.value).add(savingKey);
  try {
    const saved = await reidApi.saveFeedback({
      query_crop_id: queryCropId,
      candidate_crop_id: candidateCropId,
      same_person: samePerson,
      source,
      body_score: candidate.score,
      face_similarity: candidate.face_similarity,
      face_reliability: candidate.face_reliability,
      face_match: candidate.face_match,
      attribute_agreement: candidate.attribute_agreement,
      attribute_comparable_count: candidate.attribute_comparable_count,
      attribute_match_count: candidate.attribute_match_count,
      attribute_conflict_count: candidate.attribute_conflict_count,
      fusion_score: candidate.fusion_score,
      evidence_level: candidate.evidence_level as
        | "reliable"
        | "similar"
        | "clue"
        | "rejected"
        | null
        | undefined,
      decision_reason: candidate.decision_reason,
    });
    if (isCurrentQuery(queryCropId, generation)) {
      feedbackByCandidate.value = {
        ...feedbackByCandidate.value,
        [candidateCropId]: saved,
      };
      toast(samePerson ? "已记录：同一个人" : "已记录：不是同一个人");
    }
  } catch (error) {
    if (isCurrentQuery(queryCropId, generation)) showError(error);
  } finally {
    const pending = new Set(feedbackSaving.value);
    pending.delete(savingKey);
    feedbackSaving.value = pending;
  }
}

// Disabled buttons explain themselves on hover; an unexplained dead button reads as a bug.
const notReadyTitle = computed(() => (ready.value ? "" : blockReason.value || "ReID 未就绪"));

const queryImage = computed(() => queryPreview.value || sourceCrop.value?.crop_url || "");

function enlargeImage(src: string | null | undefined, alt: string, caption: string) {
  if (!src) return;
  if (!enlargedImage.value) {
    previousPageOverflow = document.documentElement.style.overflow;
    lightboxTrigger = document.activeElement instanceof HTMLElement ? document.activeElement : null;
  }
  document.documentElement.style.overflow = "hidden";
  enlargedImage.value = { src, alt, caption };
  void nextTick(() => lightboxCloseButton.value?.focus());
}

function closeEnlargedImage() {
  if (!enlargedImage.value) return;
  enlargedImage.value = null;
  document.documentElement.style.overflow = previousPageOverflow;
  if (lightboxTrigger?.isConnected) lightboxTrigger.focus();
  lightboxTrigger = null;
}

function onPreviewKeydown(event: KeyboardEvent) {
  if (!enlargedImage.value) return;
  if (event.key === "Escape") {
    event.preventDefault();
    closeEnlargedImage();
  } else if (event.key === "Tab") {
    // The close button is the dialog's only interactive control.
    event.preventDefault();
    lightboxCloseButton.value?.focus();
  }
}

// Every result repeats the camera when a search stays at one door, so say it once instead.
// One camera per block. Sorted purely by score the doors interleave, and the question the page
// asks -- where has this person been -- has to be counted out of the list rather than read off it.
const resultGroups = computed(() => {
  const groups = new Map<
    string,
    { key: string; camera: string; location: string; items: ReidMatchItem[]; frames: number }
  >();
  for (const item of results.value ?? []) {
    const key = item.camera_id ?? "unknown";
    const group = groups.get(key) ?? {
      key,
      camera: item.camera_name || "未知摄像头",
      location: item.location_name || "",
      items: [],
      frames: 0,
    };
    group.items.push(item);
    group.frames += item.frame_count ?? 1;
    groups.set(key, group);
  }
  // First appearance follows the server's face-first ranking. Sorting numeric fusion_score
  // again would put a strong body-only door ahead of a reliable matching face at another door.
  return [...groups.values()];
});

function clockOf(value: string | null | undefined): string {
  return fmtTime(value).slice(-8);
}

// Every result carried the same date, wrapping each card's time onto a second line and leaving
// the grid ragged. State the day once above the grid instead, when there is only one.
const singleDay = computed(() => {
  const days = new Set(
    (results.value ?? []).map((item) => fmtTime(item.first_seen ?? item.captured_at).slice(0, 5)),
  );
  return days.size === 1 ? [...days][0] : "";
});

// One result is a visit, not a frame. Show how long it lasted rather than a single instant.
function visitWhen(item: ReidMatchItem): string {
  const stamp = item.first_seen ?? item.captured_at;
  const prefix = singleDay.value ? "" : `${fmtTime(stamp).slice(0, 5)} `;
  if ((item.frame_count ?? 1) <= 1 || !item.first_seen || !item.last_seen) {
    return `${prefix}${clockOf(stamp)}`;
  }
  return `${prefix}${clockOf(item.first_seen)}–${clockOf(item.last_seen)} ×${item.frame_count}`;
}

const blockReason = computed(() => {
  const value = status.value;
  if (!value) return "";
  if (!value.enabled) return "需要在服务端设置 REID_ENABLED、REID_SERVICE_URL 并启用 Milvus。";
  if (!value.reid_service_ok) {
    return `ReID 服务未响应${value.last_error ? `：${value.last_error}` : ""}`;
  }
  if (!value.milvus_ok) {
    return `Milvus 不可达${value.last_error ? `：${value.last_error}` : ""}`;
  }
  if (!value.ready) return value.last_error || "ReID 身份配置不匹配。";
  return "";
});

async function loadStatus() {
  try {
    status.value = await reidApi.status();
  } catch (error) {
    showError(error);
  }
}

const refreshingStatus = ref(false);

// The header 刷新 button needs its own busy flag: loadStatus also runs behind rebuild and
// mount, and reusing those flags would let a double-click fire the request twice.
async function refreshStatus() {
  if (refreshingStatus.value) return;
  refreshingStatus.value = true;
  try {
    await loadStatus();
  } finally {
    refreshingStatus.value = false;
  }
}

async function search() {
  const file = queryFile.value;
  if (!file || searching.value) return;

  const sequence = ++searchSequence;
  searchController?.abort();
  const controller = new AbortController();
  searchController = controller;
  searching.value = true;
  results.value = null;
  resultsFaceCoverage.value = null;
  resultsQueryCropId.value = null;
  try {
    const body = new FormData();
    body.set("file", file);
    const response = await reidApi.search(body, undefined, controller.signal);
    if (sequence !== searchSequence || queryFile.value !== file) return;
    results.value = response.items;
    resultsFaceCoverage.value = response.face_coverage ?? null;
    queryFrameCount.value = response.query_frame_count ?? 1;
    // Zero hits already shows as the empty state below; a toast on top would say it twice.
    if (response.items.length) toast(`找到 ${response.items.length} 次候选出现，请核对图片`);
  } catch (error) {
    if (controller.signal.aborted || sequence !== searchSequence) return;
    showError(error);
  } finally {
    if (sequence === searchSequence && searchController === controller) {
      searchController = null;
      searching.value = false;
    }
  }
}

async function rebuild() {
  if (rebuilding.value || !ready.value) return;
  rebuilding.value = true;
  try {
    const result = await reidApi.rebuild(500);
    toast(
      `重建结果：已索引 ${result.indexed} / 扫描 ${result.seen} / 跳过 ${result.skipped} / 失败 ${result.failed} / 未处理 ${result.unprocessed}`,
    );
    if (result.errors.length) {
      toast(`首条错误：${result.errors[0]}`);
      console.warn("reid backfill errors", result.errors);
    }
    await loadStatus();
  } catch (error) {
    showError(error);
  } finally {
    rebuilding.value = false;
  }
}

function onFileChange(file: File | null) {
  ++queryGeneration;
  ++searchSequence;
  searchController?.abort();
  searchController = null;
  searching.value = false;
  closeEnlargedImage();
  queryFile.value = file;
  if (queryPreview.value) URL.revokeObjectURL(queryPreview.value);
  queryPreview.value = file ? URL.createObjectURL(file) : "";
  results.value = null;
  resultsFaceCoverage.value = null;
  resultsQueryCropId.value = null;
  cameraLinks.value = null;
  showWeakCameraLinks.value = false;
  feedbackByCandidate.value = {};
  queryFrameCount.value = 1;
  // Removing an upload explicitly restores the stored crop, including its own links/feedback.
  if (!file) void activateSourceCrop(sourceCropId.value);
}

async function loadLinks(cropId: string) {
  const generation = queryGeneration;
  const sequence = ++linksSequence;
  cameraLinks.value = null;
  showWeakCameraLinks.value = false;
  try {
    const response = await reidApi.links(cropId);
    if (!isCurrentQuery(cropId, generation) || sequence !== linksSequence) return;
    cameraLinks.value = response;
    queryFrameCount.value = Math.max(
      queryFrameCount.value,
      response.query_frame_count ?? 1,
    );
  } catch {
    if (isCurrentQuery(cropId, generation) && sequence === linksSequence) {
      cameraLinks.value = null; // the match list still stands on its own
    }
  }
}

async function searchByCrop(cropId: string) {
  if (searching.value || activeQueryCropId.value !== cropId) return;
  const sequence = ++searchSequence;
  searching.value = true;
  results.value = null;
  resultsFaceCoverage.value = null;
  resultsQueryCropId.value = null;
  try {
    const response = await reidApi.similarToCrop(cropId);
    if (sequence !== searchSequence) return;
    results.value = response.items;
    resultsFaceCoverage.value = response.face_coverage ?? null;
    resultsQueryCropId.value = cropId;
    queryFrameCount.value = response.query_frame_count ?? 1;
    if (response.items.length) toast(`找到 ${response.items.length} 次候选出现，请核对图片`);
  } catch (error) {
    if (sequence !== searchSequence) return;
    showError(error);
  } finally {
    if (sequence === searchSequence) searching.value = false;
  }
}

function routeCropId(value: unknown): string | null {
  const first = Array.isArray(value) ? value[0] : value;
  return typeof first === "string" && first ? first : null;
}

async function activateSourceCrop(cropId: string | null) {
  const generation = ++queryGeneration;
  closeEnlargedImage();
  ++searchSequence;
  searchController?.abort();
  searchController = null;
  searching.value = false;
  sourceCropId.value = cropId;
  sourceCrop.value = null;
  cameraLinks.value = null;
  results.value = null;
  resultsFaceCoverage.value = null;
  resultsQueryCropId.value = null;
  queryFrameCount.value = 1;
  feedbackByCandidate.value = {};
  showWeakCameraLinks.value = false;
  if (queryPreview.value) URL.revokeObjectURL(queryPreview.value);
  queryPreview.value = "";
  queryFile.value = null;
  if (!cropId) return;

  cropsApi
    .get(cropId)
    .then((crop) => {
      if (isCurrentQuery(cropId, generation)) sourceCrop.value = crop;
    })
    .catch(() => {
      if (isCurrentQuery(cropId, generation)) sourceCrop.value = null;
    });
  const requests: Promise<unknown>[] = [loadFeedback(cropId)];
  if (ready.value) requests.push(searchByCrop(cropId), loadLinks(cropId));
  await Promise.all(requests);
}

onMounted(async () => {
  const generation = queryGeneration;
  window.addEventListener("keydown", onPreviewKeydown);
  await loadStatus();
  if (generation === queryGeneration) await activateSourceCrop(routeCropId(route.query.crop_id));
});
watch(
  () => route.query.crop_id,
  async (requested) => {
    const cropId = routeCropId(requested);
    if (cropId === sourceCropId.value) return;
    await activateSourceCrop(cropId);
  },
);
onBeforeUnmount(() => {
  ++queryGeneration;
  window.removeEventListener("keydown", onPreviewKeydown);
  if (enlargedImage.value) document.documentElement.style.overflow = previousPageOverflow;
  ++searchSequence;
  searchController?.abort();
  searchController = null;
  if (queryPreview.value) URL.revokeObjectURL(queryPreview.value);
});
</script>

<template>
  <main class="page-workspace page-shell reid-workspace">
    <Teleport to="#page-actions">
      <button
        class="button ghost"
        type="button"
        :disabled="rebuilding || !ready"
        :title="notReadyTitle || (rebuilding ? '正在重建索引，请稍候' : '为未索引的裁剪补建向量索引')"
        @click="rebuild"
      >
        {{ rebuilding ? "建索引中" : "重建索引" }}
      </button>
      <button
        class="button primary"
        type="button"
        :disabled="refreshingStatus"
        :title="refreshingStatus ? '正在刷新状态' : '重新获取 ReID 服务状态'"
        @click="refreshStatus"
      >
        {{ refreshingStatus ? "刷新中" : "刷新" }}
      </button>
    </Teleport>

    <section class="page-header" aria-labelledby="reidTitle">
      <h2 id="reidTitle">以图找人</h2>
      <div v-if="status" class="reid-status">
        <span :class="ready ? 'status-pill ok' : 'status-pill'">
          <i aria-hidden="true"></i>{{ ready ? "ReID 就绪" : status.enabled ? "已配置，未就绪" : "未启用" }}
        </span>
        <span>服务 {{ status.reid_service_ok ? "在线" : "离线" }}</span>
        <span>Milvus {{ status.milvus_ok ? "在线" : status.milvus_configured ? "不可达" : "未配置" }}</span>
        <span>{{ coverage }}</span>
        <span v-if="status.attribute_filter_enabled" class="status-pill ok">
          <i aria-hidden="true"></i>高置信标签预筛
        </span>
        <span
          v-if="status.face_priority_enabled"
          :class="status.face_priority_ready ? 'status-pill ok' : 'status-pill'"
          :title="status.face_priority_error || ''"
        >
          <i aria-hidden="true"></i>{{ status.face_priority_ready ? "人脸模型就绪" : "人脸模型不可用" }}
        </span>
        <span v-if="status.pending_crops > 0" class="reid-backlog">积压 {{ status.pending_crops }}</span>
        <!-- Fingerprints matter when something is wrong, and never otherwise; they were taking
             the most prominent line on the page. -->
        <details class="reid-fingerprint">
          <summary>模型指纹</summary>
          <dl>
            <dt>模型</dt>
            <dd>{{ status.model }} / {{ status.embedding_dim }} 维</dd>
            <dt>revision</dt>
            <dd>{{ status.checkpoint_revision }}</dd>
            <dt>namespace</dt>
            <dd>{{ status.milvus_namespace }}</dd>
            <dt>人脸</dt>
            <dd>
              {{ status.face_model || status.face_provider }} / {{ status.face_device }}
              <template v-if="status.face_priority_error"> · {{ status.face_priority_error }}</template>
            </dd>
          </dl>
        </details>
      </div>
    </section>

    <section class="panel reid-query" aria-labelledby="reidQueryTitle">
      <div class="section-head">
        <div>
          <h2 id="reidQueryTitle">查询</h2>
          <p>先看图片，再核对证据。可靠人脸优先，标签辅助判断；相似分数不是身份确认。</p>
        </div>
      </div>

      <figure v-if="queryImage" class="reid-query-figure">
        <button
          class="image-zoom-trigger"
          type="button"
          aria-label="放大查看查询图"
          title="点击放大"
          @click="
            enlargeImage(
              queryImage,
              '查询图',
              activeQueryCropId ? `查询图 · crop ${shortId(activeQueryCropId)}` : '上传的查询图',
            )
          "
        >
          <img :src="queryImage" alt="查询图" />
          <span class="image-zoom-hint" aria-hidden="true">放大</span>
        </button>
        <figcaption>
          <span v-if="activeQueryCropId">来自观察表 · crop {{ shortId(activeQueryCropId) }}</span>
          <span v-else>已选择的上传图</span>
          <span v-if="queryFrameCount > 1" class="status-pill ok">
            <i aria-hidden="true"></i>{{ queryFrameCount }} 帧联合检索
          </span>
        </figcaption>
      </figure>

      <button
        v-if="activeQueryCropId"
        class="button ghost wide"
        type="button"
        :disabled="searching || !ready"
        :title="notReadyTitle || (searching ? '检索进行中，请稍候' : '用这张裁剪重新检索')"
        @click="searchByCrop(activeQueryCropId)"
      >
        {{ searching ? "检索中" : "重新检索" }}
      </button>

      <form class="form-grid" @submit.prevent="search">
        <FileField
          v-model="queryFile"
          :label="sourceCropId ? '改用其他图' : '人体图'"
          accept="image/*"
          hint="整个人的裁剪图效果最好"
          @update:model-value="onFileChange"
        />
        <button
          class="button primary wide"
          type="submit"
          :disabled="searching || !queryFile || !ready"
          :title="
            notReadyTitle ||
            (!queryFile ? '请先选择一张人体图' : searching ? '检索进行中，请稍候' : '跨摄像头检索同一个人')
          "
        >
          {{ searching ? "检索中" : "找同一个人" }}
        </button>
        <p v-if="blockReason" class="muted-text">{{ blockReason }}</p>
      </form>
    </section>

    <section
      v-if="cameraLinks"
      class="panel reid-links"
      aria-labelledby="reidLinksTitle"
    >
      <div class="section-head">
        <div>
          <h2 id="reidLinksTitle">跨摄像头线索</h2>
          <p>
            每个其他摄像头的一条候选，不代表确认到访。人体分低于参考线
            {{ cameraLinks.chance_ceiling.toFixed(2) }} 且无更强证据的线索默认收起。
          </p>
        </div>
        <button
          v-if="weakCameraLinks.length"
          class="button ghost reid-weak-toggle"
          type="button"
          :aria-expanded="showWeakCameraLinks"
          @click="showWeakCameraLinks = !showWeakCameraLinks"
        >
          {{ showWeakCameraLinks ? "隐藏低置信线索" : `查看低置信线索（${weakCameraLinks.length}）` }}
        </button>
      </div>
      <ReidFaceCoverageSummary :coverage="cameraLinks.face_coverage" scope="links" />
      <EmptyState
        v-if="!credibleCameraLinks.length && !showWeakCameraLinks"
        class="reid-link-empty"
        title="暂无优先核对的跨摄像头候选"
        :hint="weakCameraLinks.length ? '其他摄像头仍有最佳候选，但人体分数处于巧合区间，可按需展开核对。' : '本次没有可展示的跨摄像头线索；这不等于确认没有到访。'"
      />
      <ul v-if="visibleCameraLinks.length" class="reid-link-list">
        <li v-for="link in visibleCameraLinks" :key="link.crop_id">
          <button
            v-if="link.crop_url"
            class="image-zoom-trigger reid-link-image"
            type="button"
            :aria-label="`放大查看${link.camera_name || '跨摄像头'}候选图`"
            title="点击放大"
            @click="
              enlargeImage(
                link.crop_url,
                '跨摄像头候选',
                `${link.camera_name || '未知摄像头'} · crop ${shortId(link.crop_id)}`,
              )
            "
          >
            <img :src="link.crop_url" alt="候选" loading="lazy" />
            <span class="image-zoom-hint" aria-hidden="true">放大</span>
          </button>
          <div v-else class="media-thumb-missing reid-link-image">图缺失</div>
          <div class="reid-link-meta">
            <strong>{{ link.camera_name || "未知摄像头" }}</strong>
            <span>{{ link.location_name }}</span>
            <ReidEvidenceSummary
              :item="link"
              :when="link.captured_at ? fmtTime(link.captured_at) : ''"
              :confirmed="feedbackValue(link.crop_id)"
              show-score
            />
            <RouterLink class="reid-link-query" :to="{ path: '/reid', query: { crop_id: link.crop_id } }">
              用这张图检索
            </RouterLink>
            <ReidFeedbackButtons
              v-if="activeQueryCropId"
              :value="feedbackValue(link.crop_id)"
              :saving="feedbackIsSaving(link.crop_id)"
              @choose="saveFeedback(link, $event, 'camera_link')"
            />
          </div>
        </li>
      </ul>
    </section>

    <section class="panel reid-results" aria-labelledby="reidResultsTitle">
      <div class="section-head">
        <div>
          <h2 id="reidResultsTitle">检索候选</h2>
        </div>
        <div class="reid-result-actions">
          <span v-if="results" class="source-count">{{ resultSummary }}</span>
          <a
            v-if="activeQueryCropId"
            class="button ghost reid-export"
            href="/api/reid/feedback/export.csv"
            download="reid-feedback.csv"
          >
            导出全部标注
          </a>
        </div>
      </div>

      <ReidFaceCoverageSummary v-if="results !== null" :coverage="resultsFaceCoverage" scope="search" />

      <p v-if="activeQueryCropId" class="reid-feedback-help">
        看清候选后可人工确认，本次已标注 {{ currentFeedbackCount }} 条；标注只用于后续校准，当前不会改变排序。
      </p>
      <p v-else-if="results" class="reid-feedback-help">
        上传图没有可追溯的查询 crop；从观察表进入「找相似」后即可标注候选。
      </p>

      <p v-if="singleDay" class="reid-scope">{{ singleDay }}</p>

      <!-- The live region has to exist before its content changes, or screen readers stay
           silent -- so the wrapper is unconditional and only its children switch. -->
      <div class="reid-search-state" aria-live="polite">
        <EmptyState v-if="searching" title="检索中" />
        <EmptyState v-else-if="results === null" title="还没有检索" />
        <EmptyState
          v-else-if="!results.length"
          title="没有达到筛选条件的候选"
          hint="可以试试：换一张头到脚完整、光线清晰的单人全身图；或先点右上角「重建索引」补齐覆盖后再检索。"
        />
      </div>
      <template v-if="results && results.length">
        <section v-for="group in resultGroups" :key="group.key" class="reid-camera-group">
          <h3>
            {{ group.camera }}
            <small v-if="group.location">{{ group.location }}</small>
            <em>{{ group.items.length }} 次出现 · {{ group.frames }} 帧</em>
          </h3>
          <div class="media-grid">
            <article v-for="item in group.items" :key="item.crop_id" class="media-item">
          <div class="reid-thumb">
            <button
              v-if="item.crop_url || item.image_url"
              class="image-zoom-trigger"
              type="button"
              aria-label="放大查看候选裁剪"
              title="点击放大"
              @click="
                enlargeImage(
                  item.crop_url || item.image_url,
                  '候选裁剪',
                  `${item.camera_name || '未知摄像头'} · crop ${shortId(item.crop_id)} · 相似度 ${formatScore(item.score)}`,
                )
              "
            >
              <img
                :src="item.crop_url || item.image_url || undefined"
                alt="候选裁剪"
                loading="lazy"
              />
              <span class="image-zoom-hint" aria-hidden="true">放大</span>
            </button>
            <div v-else class="media-thumb-missing">图缺失</div>
            <div class="reid-score" title="人体向量相似度，不是同人概率">
              <span>人体相似度</span>
              <b>{{ formatScore(item.score) }}</b>
            </div>
          </div>
          <div class="media-meta">
            <ReidEvidenceSummary
              :item="item"
              :when="visitWhen(item)"
              :confirmed="resultsQueryCropId === activeQueryCropId && activeQueryCropId ? feedbackValue(item.crop_id) : null"
            />
            <ReidFeedbackButtons
              v-if="resultsQueryCropId === activeQueryCropId && resultsQueryCropId"
              :value="feedbackValue(item.crop_id)"
              :saving="feedbackIsSaving(item.crop_id)"
              @choose="saveFeedback(item, $event, 'search')"
            />
          </div>
            </article>
          </div>
        </section>
      </template>
    </section>

    <Teleport to="body">
      <div
        v-if="enlargedImage"
        class="image-lightbox"
        role="dialog"
        aria-modal="true"
        aria-label="图片放大预览"
        @click.self="closeEnlargedImage"
      >
        <button
          ref="lightboxCloseButton"
          class="image-lightbox-close"
          type="button"
          aria-label="关闭图片预览"
          title="关闭（Esc）"
          @click="closeEnlargedImage"
        >
          ×
        </button>
        <figure>
          <img :src="enlargedImage.src" :alt="enlargedImage.alt" />
          <figcaption>{{ enlargedImage.caption }}</figcaption>
        </figure>
      </div>
    </Teleport>
  </main>
</template>

<style scoped>
/* 首屏留白压缩：状态行到查询面板之间的垂直间距是页面上最先被浪费的空间。 */
.reid-workspace {
  row-gap: 16px;
  align-content: start;
}

.reid-workspace .page-header {
  margin-bottom: 0;
  padding-bottom: 0;
}

.reid-query .section-head {
  margin-bottom: 12px;
}

.reid-query .section-head p {
  margin: 4px 0 0;
}

.reid-weak-toggle {
  flex: 0 0 auto;
  white-space: nowrap;
}

.reid-result-actions {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  justify-content: flex-end;
  gap: 8px;
}

.reid-export {
  min-height: 34px;
  padding: 7px 10px;
  font-size: 12px;
  text-decoration: none;
}

.reid-feedback-help {
  margin: -2px 0 14px;
  color: var(--muted, #667085);
  font-size: 12px;
}

.reid-link-empty {
  padding: 18px;
  border: 1px dashed var(--line, #dbe3ea);
  border-radius: 10px;
  background: var(--surface-soft, #f4f7fa);
}

.reid-link-empty p {
  margin: 6px 0 0;
}

/* 上传到结果的视觉连续性：查询图、跨门候选图和结果缩略图共用同一套圆角描边，
   让"这张图"到"这些匹配"读起来是同一条链路。 */
.reid-query-figure img,
.reid-link-list img,
.media-item img {
  border-radius: 10px;
  border: 1px solid rgb(0 0 0 / 8%);
}

.image-zoom-trigger {
  position: relative;
  display: block;
  margin: 0;
  padding: 0;
  overflow: hidden;
  border: 0;
  border-radius: 10px;
  background: transparent;
  color: inherit;
  cursor: zoom-in;
}

.image-zoom-trigger:focus-visible {
  outline: 3px solid color-mix(in srgb, var(--accent, #246bfd) 45%, transparent);
  outline-offset: 3px;
}

.image-zoom-hint {
  position: absolute;
  right: 7px;
  bottom: 7px;
  padding: 3px 7px;
  border-radius: 999px;
  background: rgb(13 22 34 / 72%);
  color: #fff;
  font-size: 11px;
  line-height: 1.35;
  opacity: 0;
  transform: translateY(3px);
  transition: opacity 140ms ease, transform 140ms ease;
}

.image-zoom-trigger:hover .image-zoom-hint,
.image-zoom-trigger:focus-visible .image-zoom-hint {
  opacity: 1;
  transform: translateY(0);
}

.reid-query-figure {
  margin: 0 0 12px;
}

.reid-query-figure img {
  display: block;
  width: 100%;
  height: 280px;
  max-height: none;
  object-fit: contain;
  background: var(--surface-soft, #f4f7fa);
}

.reid-query-figure .image-zoom-trigger {
  width: 100%;
}

.reid-query-figure figcaption {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  margin-top: 10px;
  color: var(--muted, #667085);
  font-size: 12px;
}

.reid-results .media-grid {
  grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
  max-height: none;
  min-height: 0;
  overflow: visible;
  align-items: start;
  gap: 14px;
  padding: 0;
}

.reid-results .media-item {
  min-width: 0;
  border-radius: 10px;
}

.reid-thumb {
  position: relative;
  height: 240px;
  background: var(--surface-soft, #f4f7fa);
}

.reid-thumb .image-zoom-trigger,
.reid-thumb img,
.reid-thumb .media-thumb-missing {
  width: 100%;
  height: 100%;
  max-height: none;
  aspect-ratio: auto;
  object-fit: contain;
}

.reid-score {
  position: absolute;
  top: 8px;
  left: 8px;
  display: flex;
  align-items: baseline;
  gap: 6px;
  padding: 4px 8px;
  border-radius: 7px;
  background: rgb(13 22 34 / 82%);
  color: #fff;
  pointer-events: none;
}

.reid-score span { font-size: 10px; }
.reid-score b { font-size: 17px; font-variant-numeric: tabular-nums; }

.reid-results .media-meta {
  display: grid;
  gap: 10px;
  padding: 12px;
}

.reid-camera-group + .reid-camera-group { margin-top: 24px; }
.reid-camera-group h3 { display: flex; flex-wrap: wrap; align-items: baseline; gap: 8px; margin: 0 0 12px; font-size: 14px; }
.reid-camera-group h3 small,
.reid-camera-group h3 em { color: var(--muted, #667085); font-size: 12px; font-weight: 400; font-style: normal; }

/* Cross-camera candidates are person crops, not full-width evidence images. Keep every card on
   the same portrait canvas so a tall source file cannot stretch the whole page. */
.reid-link-list {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(min(100%, 300px), 1fr));
  gap: 10px;
  margin: 0;
  padding: 0;
  list-style: none;
}

.reid-link-list > li {
  display: grid;
  grid-template-columns: 112px minmax(0, 1fr);
  gap: 12px;
  align-items: start;
  padding: 10px;
  border: 1px solid var(--line, #dbe3ea);
  border-radius: 10px;
  background: var(--surface, #fff);
}

.reid-link-image {
  display: block;
  width: 112px;
  height: 168px;
  overflow: hidden;
  border-radius: 10px;
  background: var(--surface-soft, #f4f7fa);
}

.reid-link-query {
  margin-top: 2px;
  color: var(--accent, #246bfd);
  font-size: 12px;
  font-weight: 600;
  text-decoration: none;
}

.reid-link-query:hover {
  text-decoration: underline;
}

.reid-link-list img,
.reid-link-list .media-thumb-missing {
  display: block;
  width: 100%;
  height: 100%;
  aspect-ratio: 3 / 4;
  object-fit: contain;
}

.reid-link-meta {
  display: flex;
  min-width: 0;
  flex-direction: column;
  align-items: flex-start;
  gap: 5px;
}

/* 状态容器常驻是给 aria-live 用的；没有子内容时不能再占一行空白。 */
.reid-search-state:empty {
  display: none;
}

.reid-search-state .empty .muted-text {
  margin: 8px 0 0;
}

.image-lightbox {
  position: fixed;
  z-index: 1000;
  inset: 0;
  display: grid;
  padding: clamp(16px, 4vw, 48px);
  background: rgb(5 10 18 / 90%);
  backdrop-filter: blur(5px);
  place-items: center;
}

.image-lightbox figure {
  display: grid;
  max-width: 100%;
  max-height: 100%;
  margin: 0;
  gap: 12px;
  place-items: center;
}

.image-lightbox img {
  display: block;
  max-width: min(94vw, 1600px);
  max-height: calc(100vh - 112px);
  border-radius: 12px;
  box-shadow: 0 22px 70px rgb(0 0 0 / 45%);
  object-fit: contain;
}

.image-lightbox figcaption {
  max-width: min(90vw, 900px);
  color: rgb(255 255 255 / 88%);
  font-size: 13px;
  text-align: center;
}

.image-lightbox-close {
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

.image-lightbox-close:hover,
.image-lightbox-close:focus-visible {
  background: rgb(255 255 255 / 22%);
}

.image-lightbox-close:focus-visible {
  outline: 3px solid #fff;
  outline-offset: 3px;
}

@media (max-width: 1024px) {
  .reid-query { position: static; }
}

@media (hover: none) {
  .image-zoom-hint { opacity: 1; transform: none; }
}

@media (max-width: 640px) {
  .reid-workspace {
    row-gap: 12px;
  }

  .reid-query-figure img {
    height: 220px;
  }

  .reid-link-list {
    grid-template-columns: minmax(0, 1fr);
  }

  .reid-results .media-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 10px; }
  .reid-thumb { height: 220px; }
}

@media (max-width: 400px) {
  .reid-results .media-grid { grid-template-columns: minmax(0, 1fr); }
  .reid-link-list > li { grid-template-columns: 88px minmax(0, 1fr); gap: 10px; }
  .reid-link-image { width: 88px; height: 132px; }
}
</style>

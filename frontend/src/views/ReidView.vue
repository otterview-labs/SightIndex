<script setup lang="ts">
import { computed, onBeforeUnmount, onMounted, ref } from "vue";
import { RouterLink, useRoute } from "vue-router";

import { crops as cropsApi, reid as reidApi } from "@/api/client";
import type {
  PersonCropRead,
  ReidLinkResponse,
  ReidMatchItem,
  ReidStatusResponse,
} from "@/api/types";
import FileField from "@/components/FileField.vue";
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
const queryFile = ref<File | null>(null);
const queryPreview = ref("");
const results = ref<ReidMatchItem[] | null>(null);
const queryFrameCount = ref(1);
const searching = ref(false);
const rebuilding = ref(false);
const enlargedImage = ref<{ src: string; alt: string; caption: string } | null>(null);
let previousPageOverflow = "";
let searchController: AbortController | null = null;
let searchSequence = 0;

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

// Disabled buttons explain themselves on hover; an unexplained dead button reads as a bug.
const notReadyTitle = computed(() => (ready.value ? "" : blockReason.value || "ReID 未就绪"));

const queryImage = computed(() => queryPreview.value || sourceCrop.value?.crop_url || "");

function enlargeImage(src: string | null | undefined, alt: string, caption: string) {
  if (!src) return;
  previousPageOverflow = document.documentElement.style.overflow;
  document.documentElement.style.overflow = "hidden";
  enlargedImage.value = { src, alt, caption };
}

function closeEnlargedImage() {
  if (!enlargedImage.value) return;
  enlargedImage.value = null;
  document.documentElement.style.overflow = previousPageOverflow;
}

function onPreviewKeydown(event: KeyboardEvent) {
  if (event.key === "Escape") closeEnlargedImage();
}

function fusedPriority(item: ReidMatchItem): number {
  return item.fusion_score ?? item.score;
}

function evidenceLabel(level: string | null | undefined): string {
  return level === "reliable" ? "可信" : level === "clue" ? "线索" : level === "rejected" ? "排除" : "相似";
}

function attributeEvidenceText(item: {
  attribute_agreement?: number | null;
  attribute_matches?: string[];
  attribute_conflicts?: string[];
  attribute_comparable_count?: number | null;
  attribute_match_count?: number | null;
}): string {
  const agreement = item.attribute_agreement;
  if (agreement === null || agreement === undefined) return "";
  const compared = item.attribute_comparable_count
    ?? (item.attribute_matches?.length ?? 0) + (item.attribute_conflicts?.length ?? 0);
  if (compared < 2) return "";
  if (!compared) return "";
  const matched = item.attribute_match_count
    ?? item.attribute_matches?.length
    ?? 0;
  const percentage = Math.round(agreement * 100);
  return `高置信标签一致 ${matched}/${compared}（${percentage}%）`;
}

function faceEvidenceText(item: {
  face_similarity?: number | null;
  face_reliability?: number | null;
  face_match?: boolean | null;
}): string {
  if (item.face_similarity === null || item.face_similarity === undefined) return "";
  const similarity = Math.round(item.face_similarity * 100);
  const reliability = Math.round((item.face_reliability ?? 0) * 100);
  if (item.face_match === true) return `人脸吻合 ${similarity}% · 质量 ${reliability}%`;
  if (item.face_match === false) return `人脸不一致 ${similarity}% · 质量 ${reliability}%`;
  return `${reliability >= 70 ? "人脸不确定" : "人脸弱证据"} ${similarity}% · 质量 ${reliability}%`;
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
  // Strongest door first: it is the likeliest answer, and the order stays stable across searches.
  return [...groups.values()].sort(
    (a, b) =>
      Math.max(...b.items.map(fusedPriority)) - Math.max(...a.items.map(fusedPriority)),
  );
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
  try {
    const body = new FormData();
    body.set("file", file);
    const response = await reidApi.search(body, undefined, controller.signal);
    if (sequence !== searchSequence || queryFile.value !== file) return;
    results.value = response.items;
    queryFrameCount.value = response.query_frame_count ?? 1;
    // Zero hits already shows as the empty state below; a toast on top would say it twice.
    if (response.items.length) toast(`命中 ${response.items.length} 次出现`);
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
  ++searchSequence;
  searchController?.abort();
  searchController = null;
  searching.value = false;
  if (queryPreview.value) URL.revokeObjectURL(queryPreview.value);
  queryPreview.value = file ? URL.createObjectURL(file) : "";
  results.value = null;
  queryFrameCount.value = 1;
}

async function loadLinks(cropId: string) {
  try {
    cameraLinks.value = await reidApi.links(cropId);
    queryFrameCount.value = Math.max(
      queryFrameCount.value,
      cameraLinks.value.query_frame_count ?? 1,
    );
  } catch {
    cameraLinks.value = null; // the match list still stands on its own
  }
}

async function searchByCrop(cropId: string) {
  if (searching.value) return;
  const sequence = ++searchSequence;
  searching.value = true;
  results.value = null;
  try {
    const response = await reidApi.similarToCrop(cropId);
    if (sequence !== searchSequence) return;
    results.value = response.items;
    queryFrameCount.value = response.query_frame_count ?? 1;
    if (response.items.length) toast(`命中 ${response.items.length} 次出现`);
  } catch (error) {
    if (sequence !== searchSequence) return;
    showError(error);
  } finally {
    if (sequence === searchSequence) searching.value = false;
  }
}

onMounted(async () => {
  window.addEventListener("keydown", onPreviewKeydown);
  await loadStatus();
  const requested = route.query.crop_id;
  const cropId = Array.isArray(requested) ? requested[0] : requested;
  if (!cropId) return;
  sourceCropId.value = cropId;
  cropsApi
    .get(cropId)
    .then((crop) => {
      sourceCrop.value = crop;
    })
    .catch(() => {
      sourceCrop.value = null; // the id still shows; only the thumbnail is lost
    });
  if (ready.value) {
    await Promise.all([searchByCrop(cropId), loadLinks(cropId)]);
  }
});
onBeforeUnmount(() => {
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
          <i aria-hidden="true"></i>{{ status.face_priority_ready ? "人脸优先" : "人脸降级" }}
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
          <p>按摄像头组织候选，高置信标签先过滤；检测到可靠人脸时优先使用人脸证据。</p>
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
              sourceCropId ? `查询图 · crop ${shortId(sourceCropId)}` : '上传的查询图',
            )
          "
        >
          <img :src="queryImage" alt="查询图" />
          <span class="image-zoom-hint" aria-hidden="true">放大</span>
        </button>
        <figcaption>
          <span v-if="sourceCropId">来自观察表 · crop {{ shortId(sourceCropId) }}</span>
          <span v-else>已选择的上传图</span>
          <span v-if="queryFrameCount > 1" class="status-pill ok">
            <i aria-hidden="true"></i>{{ queryFrameCount }} 帧联合检索
          </span>
        </figcaption>
      </figure>

      <button
        v-if="sourceCropId"
        class="button ghost wide"
        type="button"
        :disabled="searching || !ready"
        :title="notReadyTitle || (searching ? '检索进行中，请稍候' : '用这张裁剪重新检索')"
        @click="searchByCrop(sourceCropId)"
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
      v-if="cameraLinks && cameraLinks.links.length"
      class="panel reid-links"
      aria-labelledby="reidLinksTitle"
    >
      <div class="section-head">
        <div>
          <h2 id="reidLinksTitle">去过哪些门</h2>
          <p>
            每个其他摄像头最像的一次，不设阈值。真实跨门匹配约 0.43–0.48，而巧合能到
            {{ cameraLinks.chance_ceiling.toFixed(2) }}，所以低于它的只作线索。
          </p>
        </div>
      </div>
      <ul class="reid-link-list">
        <li v-for="link in cameraLinks.links" :key="link.crop_id">
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
            <span>{{ link.captured_at ? fmtTime(link.captured_at) : "无时间" }}</span>
            <span :class="link.beats_chance ? 'status-pill ok' : 'status-pill'">
              <i aria-hidden="true"></i>{{ formatScore(link.score) }}
              {{ link.beats_chance ? "高于巧合" : "巧合可解释" }}
            </span>
            <span class="status-pill" :class="{ ok: link.evidence_level === 'reliable' }"
                  :title="link.decision_reason || ''">
              <i aria-hidden="true"></i>{{ evidenceLabel(link.evidence_level) }} · {{ link.decision_reason }}
            </span>
            <!-- Shown only when both sides were measurable, so an absent pill means "not known"
                 rather than "did not match". -->
            <span v-if="link.stature_agreement !== null && link.stature_agreement !== undefined"
                  class="status-pill">
              <i aria-hidden="true"></i>身高吻合 {{ Math.round(link.stature_agreement * 100) }}%
            </span>
            <span v-if="link.attribute_agreement !== null && link.attribute_agreement !== undefined"
                  class="status-pill">
              <i aria-hidden="true"></i>{{ attributeEvidenceText(link) }}
            </span>
            <span v-if="link.face_match === true" class="status-pill ok">
              <i aria-hidden="true"></i>{{ faceEvidenceText(link) }}
            </span>
            <span v-else-if="link.face_match === false" class="status-pill">
              <i aria-hidden="true"></i>{{ faceEvidenceText(link) }}
            </span>
            <span v-else-if="link.face_similarity !== null && link.face_similarity !== undefined"
                  class="status-pill">
              <i aria-hidden="true"></i>{{ faceEvidenceText(link) }}
            </span>
            <RouterLink class="reid-link-query" :to="{ path: '/reid', query: { crop_id: link.crop_id } }">
              用这张图检索
            </RouterLink>
          </div>
        </li>
      </ul>
    </section>

    <section class="panel reid-results" aria-labelledby="reidResultsTitle">
      <div class="section-head">
        <div>
          <h2 id="reidResultsTitle">匹配结果</h2>
        </div>
        <span v-if="results" class="source-count">{{ resultSummary }}</span>
      </div>

      <p v-if="singleDay" class="reid-scope">{{ singleDay }}</p>

      <!-- The live region has to exist before its content changes, or screen readers stay
           silent -- so the wrapper is unconditional and only its children switch. -->
      <div class="reid-search-state" aria-live="polite">
        <div v-if="searching" class="empty"><strong>检索中</strong></div>
        <div v-else-if="results === null" class="empty"><strong>还没有检索</strong></div>
        <div v-else-if="!results.length" class="empty">
          <strong>没有可信匹配</strong>
          <p class="muted-text">
            可以试试：换一张头到脚完整、光线清晰的单人全身图；或先点右上角「重建索引」补齐覆盖后再检索。
          </p>
        </div>
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
          <!-- The score overlays the image, so it needs the image as its containing block, not
               the card -- anchored to the card it lands on top of the button. -->
          <div class="reid-thumb">
            <button
              v-if="item.crop_url || item.image_url"
              class="image-zoom-trigger"
              type="button"
              aria-label="放大查看匹配裁剪"
              title="点击放大"
              @click="
                enlargeImage(
                  item.crop_url || item.image_url,
                  '匹配裁剪',
                  `${item.camera_name || '未知摄像头'} · crop ${shortId(item.crop_id)} · 相似度 ${formatScore(item.score)}`,
                )
              "
            >
              <img
                :src="item.crop_url || item.image_url || undefined"
                alt="匹配裁剪"
                loading="lazy"
              />
              <span class="image-zoom-hint" aria-hidden="true">放大</span>
            </button>
            <div v-else class="media-thumb-missing">图缺失</div>
            <div class="reid-score" :title="`相似度 ${formatScore(item.score)}`">
              <b>{{ formatScore(item.score) }}</b>
              <!-- A ranked list is easier to read down than across; the bar is the ranking. -->
              <i aria-hidden="true" :style="{ '--fill': `${Math.round(item.score * 100)}%` }"></i>
            </div>
          </div>
          <div class="media-meta">
            <strong v-if="item.person_name">{{ item.person_name }}</strong>
            <span>{{ visitWhen(item) }}</span>
            <span class="mono-id">crop {{ shortId(item.crop_id) }}</span>
            <span class="status-pill" :class="{ ok: item.evidence_level === 'reliable' }"
                  :title="item.decision_reason || ''">
              <i aria-hidden="true"></i>{{ evidenceLabel(item.evidence_level) }} · {{ item.decision_reason }}
            </span>
            <span
              v-if="item.attribute_agreement !== null && item.attribute_agreement !== undefined"
              class="status-pill"
            >
              <i aria-hidden="true"></i>{{ attributeEvidenceText(item) }}
            </span>
            <span v-if="item.face_match === true" class="status-pill ok">
              <i aria-hidden="true"></i>{{ faceEvidenceText(item) }}
            </span>
            <span v-else-if="item.face_match === false" class="status-pill">
              <i aria-hidden="true"></i>{{ faceEvidenceText(item) }}
            </span>
            <span v-else-if="item.face_similarity !== null && item.face_similarity !== undefined"
                  class="status-pill">
              <i aria-hidden="true"></i>{{ faceEvidenceText(item) }}
            </span>
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
          class="image-lightbox-close"
          type="button"
          aria-label="关闭图片预览"
          title="关闭（Esc）"
          autofocus
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
  outline: 3px solid color-mix(in srgb, var(--primary, #246bfd) 45%, transparent);
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
  width: min(100%, 220px);
  height: 220px;
  max-height: 220px;
  object-fit: contain;
  background: var(--surface-soft, #f4f7fa);
}

/* Cross-camera candidates are person crops, not full-width evidence images. Keep every card on
   the same portrait canvas so a tall source file cannot stretch the whole page. */
.reid-link-list {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
  gap: 10px;
  margin: 0;
  padding: 0;
  list-style: none;
}

.reid-link-list > li {
  display: grid;
  grid-template-columns: 96px minmax(0, 1fr);
  gap: 12px;
  align-items: start;
  padding: 10px;
  border: 1px solid var(--line, #dbe3ea);
  border-radius: 10px;
  background: var(--surface, #fff);
}

.reid-link-image {
  display: block;
  width: 96px;
  height: 128px;
  overflow: hidden;
  border-radius: 10px;
  background: var(--surface-soft, #f4f7fa);
}

.reid-link-query {
  margin-top: 2px;
  color: var(--primary, #246bfd);
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
  outline: none;
}

@media (max-width: 640px) {
  .reid-workspace {
    row-gap: 12px;
  }

  .reid-query-figure img {
    width: min(100%, 160px);
    height: 160px;
    max-height: 160px;
  }

  .reid-link-list {
    grid-template-columns: minmax(0, 1fr);
  }
}
</style>

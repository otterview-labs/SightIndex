<script setup lang="ts">
import { computed, onMounted, ref } from "vue";
import { RouterLink } from "vue-router";

import { face as faceApi, persons as personsApi, search as searchApi } from "@/api/client";
import type { FaceDiagnosticItem, ObservationIndexItem } from "@/api/types";
import FaceBoxThumb from "@/components/FaceBoxThumb.vue";
import PersonSelect from "@/components/PersonSelect.vue";
import { usePersons } from "@/composables/usePersons";
import { useToast } from "@/composables/useToast";
import { fmtTime, formatScore, shortId, shortText } from "@/utils/format";

const LABEL_KEYS = [
  "上衣颜色",
  "下装颜色",
  // Read off the pose keypoints rather than a VLM, so unlike the rest of this list they carry a
  // value on most rows. Placed next to the colours because they describe the same garment.
  "袖长",
  "裤长",
  "朝向",
  "身高",
  "背包",
  "帽子",
  "眼镜",
  "拿手机",
  "抽烟",
  "跌倒",
  "打架",
] as const;

const VERDICT_TEXT: Record<string, string> = {
  known: "已命中",
  unknown: "疑似",
  no_face: "无脸",
  error: "错误",
};

const { showError, toast } = useToast();
const { persons, activePersonId, activePersonName, refresh: refreshPersons } = usePersons();

const items = ref<ObservationIndexItem[]>([]);
const diagnostics = ref(new Map<string, FaceDiagnosticItem>());
const total = ref(0);
const offset = ref(0);
const limit = ref(100);
const status = ref("未加载");
const loading = ref(false);
const rebuilding = ref(false);
const busyCrop = ref<string | null>(null);
const newPersonName = ref("");
const creatingPerson = ref(false);

const query = ref("");
const startTime = ref("");
const endTime = ref("");
const onlyNamed = ref(false);
const onlyLabeled = ref(false);
const onlyFaceVector = ref(false);
const onlyVlVector = ref(false);

const canPrev = computed(() => offset.value > 0);
const canNext = computed(() => offset.value + limit.value < total.value);

function buildParams(): URLSearchParams {
  const params = new URLSearchParams({
    limit: String(limit.value),
    offset: String(offset.value),
  });
  if (query.value.trim()) params.set("query", query.value.trim());
  if (startTime.value) params.set("start_time", new Date(startTime.value).toISOString());
  if (endTime.value) params.set("end_time", new Date(endTime.value).toISOString());
  if (onlyNamed.value) params.set("only_named", "true");
  if (onlyLabeled.value) params.set("only_labeled", "true");
  if (onlyFaceVector.value) params.set("only_face_vector", "true");
  if (onlyVlVector.value) params.set("only_vl_vector", "true");
  return params;
}

async function loadDiagnostics() {
  try {
    const result = await faceApi.diagnostics(100);
    diagnostics.value = new Map(
      (result.items ?? []).map((item) => [String(item.crop_id), item] as const),
    );
  } catch (error) {
    console.warn("observation diagnostics unavailable", error);
    diagnostics.value = new Map();
  }
}

async function load(resetOffset = false) {
  if (resetOffset) offset.value = 0;
  loading.value = true;
  status.value = "加载中";
  try {
    const data = await searchApi.observations(buildParams());
    await loadDiagnostics();
    offset.value = data.offset ?? 0;
    limit.value = data.limit ?? limit.value;
    total.value = data.total ?? 0;
    items.value = data.items ?? [];
    const start = total.value ? offset.value + 1 : 0;
    const end = Math.min(offset.value + items.value.length, total.value);
    status.value = `${start}-${end} / ${total.value}`;
  } catch (error) {
    status.value = "加载失败";
    showError(error);
  } finally {
    loading.value = false;
  }
}

function diagnosticFor(item: ObservationIndexItem) {
  return diagnostics.value.get(String(item.crop_id));
}

// Nothing on this deployment produces a VL embedding, so the column was a stack of 无 holding
// width open. Shown only when some row on the page actually has one.
const anyVectors = computed(() =>
  items.value.some((item) => item.has_face_embedding || item.has_vl_embedding),
);

function labelsOf(item: ObservationIndexItem) {
  const labels = item.labels_zh as Record<string, unknown> | null | undefined;
  if (!labels || typeof labels !== "object") return { known: [], unknown: [] };
  const present = LABEL_KEYS.filter(
    (key) => labels[key] !== undefined && labels[key] !== null && labels[key] !== "",
  ).map((key) => [key, String(labels[key])] as const);
  return {
    known: present.filter(([, value]) => value !== "未知"),
    unknown: present.filter(([, value]) => value === "未知"),
  };
}

// The unknown labels used to be reachable only by hovering for a tooltip, which is no way to
// read them and no way at all on a touchscreen.
const expandedLabels = ref(new Set<string>());

function toggleLabels(item: ObservationIndexItem) {
  const key = String(item.crop_id);
  const next = new Set(expandedLabels.value);
  next.has(key) ? next.delete(key) : next.add(key);
  expandedLabels.value = next;
}

function labelsExpanded(item: ObservationIndexItem) {
  return expandedLabels.value.has(String(item.crop_id));
}

function fileName(url: string | null | undefined) {
  // These are uuid.jpg. Printed in full they take a column's width and are still cut off, so
  // nobody could read one anyway -- the head identifies it well enough, and title has the rest.
  const name = url ? (url.split("/").pop() ?? url) : "";
  return name ? `${name.slice(0, 8)}…` : "-";
}

function vectorBadges(item: ObservationIndexItem) {
  return [
    item.has_face_embedding ? "人脸" : "",
    item.has_vl_embedding ? "图向量" : "",
    item.vl_embedding_model ? `VL ${shortText(item.vl_embedding_model, 18)}` : "",
    item.vl_embedding_dim ? `${item.vl_embedding_dim}d` : "",
  ].filter(Boolean);
}

function personParts(item: ObservationIndexItem) {
  return [
    item.person_name || "未知",
    item.employee_no ? `工号 ${item.employee_no}` : "",
    item.department || "",
  ].filter(Boolean);
}

function placeParts(item: ObservationIndexItem) {
  return [
    item.camera_name || shortId(item.camera_id),
    item.location_name || shortId(item.location_id),
  ].filter(Boolean);
}

interface Suspect {
  verdict: string;
  headline: string;
  detail: string;
  confidence: "strong" | "weak";
  personId?: string | null;
  personName: string;
  scoreText: string;
}

function suspectOf(item: ObservationIndexItem): Suspect | null {
  const diagnostic = diagnosticFor(item);
  if (!diagnostic) return null;
  const personName = diagnostic.top_person_name || "";
  if (!personName && diagnostic.verdict !== "no_face" && diagnostic.verdict !== "error") {
    return null;
  }
  const similarity =
    diagnostic.top_similarity === null || diagnostic.top_similarity === undefined
      ? null
      : Number(diagnostic.top_similarity);
  const threshold = Number(diagnostic.threshold || 0);
  const scoreText = similarity === null ? "-" : similarity.toFixed(3);
  const detText =
    diagnostic.detection_score === null || diagnostic.detection_score === undefined
      ? "-"
      : Number(diagnostic.detection_score).toFixed(3);
  return {
    verdict: VERDICT_TEXT[diagnostic.verdict] ?? "诊断",
    headline: personName || diagnostic.reason || "-",
    detail: `sim ${scoreText} / det ${detText} / 阈值 ${threshold.toFixed(2)}`,
    confidence: similarity !== null && similarity >= threshold ? "strong" : "weak",
    personId: diagnostic.top_person_id,
    personName,
    scoreText,
  };
}

async function createPerson() {
  const name = newPersonName.value.trim();
  if (!name) {
    toast("请输入新人姓名");
    return;
  }
  creatingPerson.value = true;
  try {
    const person = await personsApi.create({ name });
    activePersonId.value = person.id;
    newPersonName.value = "";
    await refreshPersons();
    toast(`已新增 ${person.name}`);
  } catch (error) {
    showError(error);
  } finally {
    creatingPerson.value = false;
  }
}

async function enrollCrop(item: ObservationIndexItem) {
  const personId = activePersonId.value;
  if (!personId) {
    toast("请先选择入库人员");
    return;
  }
  if (!item.crop_id) {
    toast("这条记录没有 crop_id");
    return;
  }
  if (!window.confirm(`把这张裁剪图补入 ${activePersonName.value} 的人脸库？`)) return;
  busyCrop.value = item.crop_id;
  try {
    await personsApi.enrollFromCrop(personId, item.crop_id);
    toast(`已补入 ${activePersonName.value} 人脸库`);
    await load(false);
  } catch (error) {
    showError(error);
  } finally {
    busyCrop.value = null;
  }
}

async function labelCrop(item: ObservationIndexItem) {
  const personId = activePersonId.value;
  if (!personId) {
    toast("请先选择人员");
    return;
  }
  if (!item.crop_id) {
    toast("这条记录没有 crop_id");
    return;
  }
  if (!window.confirm(`把这张人体裁剪标记为 ${activePersonName.value}？将作为 ReID 底图。`)) return;
  busyCrop.value = item.crop_id;
  try {
    await personsApi.labelCrop(personId, item.crop_id);
    toast(`已标记为 ${activePersonName.value}`);
    await load(false);
  } catch (error) {
    showError(error);
  } finally {
    busyCrop.value = null;
  }
}

async function unlabelCrop(item: ObservationIndexItem) {
  if (!item.crop_id || !item.person_id) return;
  if (!window.confirm(`取消 ${item.person_name || "该人员"} 在这张裁剪上的标记？`)) return;
  busyCrop.value = item.crop_id;
  try {
    await personsApi.unlabelCrop(item.person_id, item.crop_id);
    toast("已取消标记");
    await load(false);
  } catch (error) {
    showError(error);
  } finally {
    busyCrop.value = null;
  }
}

async function confirmSuspect(item: ObservationIndexItem, suspect: Suspect) {
  if (!item.crop_id || !suspect.personId) {
    toast("缺少疑似人员或 crop_id");
    return;
  }
  const label = suspect.personName || "疑似人员";
  if (!window.confirm(`确认把这条疑似 ${label}（sim ${suspect.scoreText}）的裁剪图入库并打姓名标签？`)) {
    return;
  }
  busyCrop.value = item.crop_id;
  try {
    await personsApi.enrollFromCrop(suspect.personId, item.crop_id);
    activePersonId.value = suspect.personId;
    toast(`已确认入库 ${label}`);
    await load(false);
  } catch (error) {
    showError(error);
  } finally {
    busyCrop.value = null;
  }
}

async function rebuild() {
  rebuilding.value = true;
  status.value = "重建中";
  try {
    const result = await searchApi.rebuildObservations(500);
    toast(`已重建 ${result.indexed ?? 0}/${result.seen ?? 0}`);
    await load(true);
  } catch (error) {
    showError(error);
  } finally {
    rebuilding.value = false;
  }
}

function prevPage() {
  offset.value = Math.max(0, offset.value - limit.value);
  void load();
}

function nextPage() {
  if (!canNext.value) return;
  offset.value += limit.value;
  void load();
}

onMounted(async () => {
  await refreshPersons().catch(showError);
  await load(true);
});
</script>

<template>
  <main class="page-workspace page-shell observation-workspace">
    <Teleport to="#page-actions">
      <button class="button primary" type="button" :disabled="loading" @click="load(false)">
        {{ loading ? "刷新中" : "刷新" }}
      </button>
    </Teleport>

    <section class="observation-shell" aria-labelledby="observationTitle">
      <div class="observation-head page-header">
        <h2 id="observationTitle">观察大宽表</h2>
        <div class="observation-actions">
          <label class="observation-person-select">
            入库人员
            <PersonSelect v-model="activePersonId" :persons="persons" />
          </label>
          <form class="observation-person-create" @submit.prevent="createPerson">
            <label>
              新人姓名
              <input v-model="newPersonName" placeholder="输入姓名" autocomplete="off" required />
            </label>
            <button class="mini-button primary" type="submit" :disabled="creatingPerson">
              {{ creatingPerson ? "新增中" : "新增" }}
            </button>
          </form>
          <button class="button ghost" type="button" :disabled="rebuilding" @click="rebuild">
            {{ rebuilding ? "重建中" : "重建最近 500" }}
          </button>
          <div class="table-status">{{ status }}</div>
        </div>
      </div>

      <form class="observation-filters" @submit.prevent="load(true)">
        <label>
          关键词
          <input v-model="query" placeholder="姓名 / 黑衣 / 背包 / 摄像头" />
        </label>
        <label>
          开始时间
          <input v-model="startTime" type="datetime-local" />
        </label>
        <label>
          结束时间
          <input v-model="endTime" type="datetime-local" />
        </label>
        <label>
          每页
          <select v-model.number="limit">
            <option :value="50">50</option>
            <option :value="100">100</option>
            <option :value="200">200</option>
            <option :value="500">500</option>
          </select>
        </label>
        <div class="filter-toggles">
          <label class="filter-toggle">
            <input v-model="onlyNamed" type="checkbox" />仅有姓名
          </label>
          <label class="filter-toggle">
            <input v-model="onlyLabeled" type="checkbox" />仅有标签
          </label>
          <label class="filter-toggle">
            <input v-model="onlyFaceVector" type="checkbox" />人脸向量
          </label>
          <label class="filter-toggle">
            <input v-model="onlyVlVector" type="checkbox" />图向量
          </label>
        </div>
        <button class="button primary" type="submit">查询</button>
      </form>

      <div class="observation-summary" aria-label="观察表统计">
        <div><span>总数</span><strong>{{ total }}</strong></div>
        <div><span>本页</span><strong>{{ items.length }}</strong></div>
      </div>

      <div class="observation-table-wrap">
        <table class="observation-table">
          <thead>
            <tr>
              <th class="col-crop">裁剪</th>
              <th class="col-time">时间</th>
              <th class="col-person">人员</th>
              <th class="col-labels">标签</th>
              <th v-if="anyVectors" class="col-vector">向量</th>
              <th class="col-place">位置</th>
              <th class="col-source">来源</th>
              <th class="col-actions">操作</th>
            </tr>
          </thead>
          <tbody aria-live="polite">
            <tr v-if="loading">
              <td :colspan="anyVectors ? 8 : 7" class="table-empty">加载中</td>
            </tr>
            <tr v-else-if="!items.length">
              <td :colspan="anyVectors ? 8 : 7" class="table-empty">暂无观察记录</td>
            </tr>
            <tr v-for="item in items" v-else :key="item.crop_id ?? item.image_id ?? ''">
              <td>
                <FaceBoxThumb
                  :src="item.crop_url || item.thumbnail_url || item.image_url || ''"
                  :face-box="diagnosticFor(item)?.face_bbox"
                />
              </td>
              <td>
                <div class="table-main">{{ fmtTime(item.captured_at) }}</div>
                <!-- updated_at is bookkeeping, and it reads eight hours earlier than the capture
                     beside it: created_at/updated_at default to SQLite's func.now() (UTC) while
                     captured_at is stored as local wall clock. Showing both invites the reader to
                     believe a row was written before the moment it records. -->
              </td>
              <td>
                <div class="table-main">{{ personParts(item)[0] || "未知" }}</div>
                <div class="table-sub">{{ personParts(item).slice(1).join(" / ") || "-" }}</div>
                <div class="badge-row">
                  <span v-if="item.recognition_result_type">
                    {{ item.recognition_result_type }}
                  </span>
                  <span v-if="item.face_similarity !== null && item.face_similarity !== undefined">
                    face {{ formatScore(item.face_similarity) }}
                  </span>
                </div>
                <div
                  v-if="suspectOf(item)"
                  class="observation-suspect"
                  :class="suspectOf(item)!.confidence"
                >
                  <span>{{ suspectOf(item)!.verdict }}</span>
                  <strong>{{ suspectOf(item)!.headline }}</strong>
                  <small>{{ suspectOf(item)!.detail }}</small>
                  <button
                    v-if="suspectOf(item)!.personId && !item.person_id"
                    class="mini-button"
                    type="button"
                    :disabled="busyCrop === item.crop_id"
                    @click="confirmSuspect(item, suspectOf(item)!)"
                  >
                    {{ busyCrop === item.crop_id ? "入库中" : "确认入库" }}
                  </button>
                </div>
              </td>
              <td>
                <div
                  v-if="labelsOf(item).known.length || labelsOf(item).unknown.length"
                  class="observation-labels"
                >
                  <span v-for="[key, value] in labelsOf(item).known" :key="key">
                    <small>{{ key }}</small>{{ value }}
                  </span>
                  <template v-if="labelsExpanded(item)">
                    <span
                      v-for="[key, value] in labelsOf(item).unknown"
                      :key="key"
                      class="label-unknown"
                    >
                      <small>{{ key }}</small>{{ value }}
                    </span>
                  </template>
                  <button
                    v-if="labelsOf(item).unknown.length"
                    class="label-more"
                    type="button"
                    :title="labelsExpanded(item) ? '收起' : '这些项需要 VLM 才能识别'"
                    @click="toggleLabels(item)"
                  >
                    {{
                      labelsExpanded(item)
                        ? "收起"
                        : `+${labelsOf(item).unknown.length} 未知`
                    }}
                  </button>
                </div>
                <span v-else class="muted-text">无标签</span>
              </td>
              <td v-if="anyVectors">
                <div class="badge-row">
                  <template v-if="vectorBadges(item).length">
                    <span v-for="badge in vectorBadges(item)" :key="badge">{{ badge }}</span>
                  </template>
                  <span v-else>无</span>
                </div>
                <div class="table-sub">{{ shortText(item.milvus_collection || "-", 28) }}</div>
              </td>
              <td>
                <div class="table-main">{{ placeParts(item)[0] || "-" }}</div>
                <div class="table-sub">{{ placeParts(item)[1] || "-" }}</div>
              </td>
              <td>
                <!-- Every identifier for the row in one place. They used to be split between
                     here and under the action buttons, which put debug text in the column a
                     person clicks. -->
                <div class="source-cell">
                  <code :title="item.crop_id || ''">crop {{ shortId(item.crop_id) }}</code>
                  <code v-if="item.person_id" :title="item.person_id">
                    person {{ shortId(item.person_id) }}
                  </code>
                  <code :title="item.image_url || ''">{{ fileName(item.image_url) }}</code>
                </div>
              </td>
              <td>
                <div class="observation-row-actions">
                  <button
                    class="mini-button primary"
                    type="button"
                    :title="activePersonId ? '' : '请先选择入库人员'"
                    :disabled="busyCrop === item.crop_id || !activePersonId"
                    @click="enrollCrop(item)"
                  >
                    {{ busyCrop === item.crop_id ? "入库中" : "入库" }}
                  </button>
                  <RouterLink
                    v-if="item.person_id"
                    class="mini-link"
                    :to="{ path: '/faces', query: { person_id: item.person_id } }"
                  >
                    看轨迹
                  </RouterLink>
                  <RouterLink
                    v-if="item.crop_id"
                    class="mini-link"
                    :to="{ path: '/reid', query: { crop_id: item.crop_id } }"
                  >
                    找相似
                  </RouterLink>
                  <button
                    v-if="item.crop_id && !item.person_id"
                    class="mini-button"
                    type="button"
                    :disabled="busyCrop === item.crop_id || !activePersonId"
                    @click="labelCrop(item)"
                  >
                    标记人员
                  </button>
                  <button
                    v-if="item.crop_id && item.person_id"
                    class="mini-button"
                    type="button"
                    :disabled="busyCrop === item.crop_id"
                    @click="unlabelCrop(item)"
                  >
                    取消标记
                  </button>
                </div>
              </td>
            </tr>
          </tbody>
        </table>
      </div>

      <div class="observation-pager">
        <button class="button ghost" type="button" :disabled="!canPrev" @click="prevPage">
          上一页
        </button>
        <button class="button ghost" type="button" :disabled="!canNext" @click="nextPage">
          下一页
        </button>
      </div>
    </section>
  </main>
</template>

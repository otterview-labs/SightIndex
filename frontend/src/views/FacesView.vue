<script setup lang="ts">
import { computed, onMounted, ref, watch } from "vue";
import { useRoute } from "vue-router";

import { face as faceApi, persons as personsApi } from "@/api/client";
import type {
  FaceDiagnosticItem,
  FaceRecognitionResponse,
  PersonTrajectoryPoint,
} from "@/api/types";
import FaceBoxThumb from "@/components/FaceBoxThumb.vue";
import FaceZoom from "@/components/FaceZoom.vue";
import FileField from "@/components/FileField.vue";
import PersonSelect from "@/components/PersonSelect.vue";
import { usePersons } from "@/composables/usePersons";
import { useToast } from "@/composables/useToast";
import { fmtTime, shortId } from "@/utils/format";

const VERDICT_LABELS: Record<string, string> = {
  known: "命中",
  unknown: "未命中",
  no_face: "无脸",
  error: "错误",
};

const MODE_TEXT: Record<string, string> = {
  all: "观察表人脸",
  face: "人脸",
  vector: "外观向量",
  reid: "ReID 疑似出现点",
};

const route = useRoute();
const { showError, toast } = useToast();
const { persons, activePersonId, refresh: refreshPersons } = usePersons();

const enrollPersonId = ref<string | null>(null);
const newPerson = ref({ name: "", employee_no: "", department: "", phone: "" });
const creating = ref(false);
const enrolling = ref(false);
const recognizing = ref(false);
const rebuilding = ref(false);
const diagnosing = ref(false);
const busyDiagnosticCrop = ref<string | null>(null);

const enrollFile = ref<File | null>(null);
const recognizeFile = ref<File | null>(null);
const threshold = ref("0.45");

const faceResult = ref<FaceRecognitionResponse | null>(null);
const diagnostics = ref<FaceDiagnosticItem[] | null>(null);
const diagnosticsRunning = ref(false);

const trajectoryMode = ref("face");
const trajectory = ref<PersonTrajectoryPoint[]>([]);
const trajectoryWarnings = ref<string[]>([]);
const trajectoryLoading = ref(false);

const bestMatch = computed(() => faceResult.value?.matches?.[0]);

const trajectoryLoadingText = computed(() => {
  const label = MODE_TEXT[trajectoryMode.value] ?? "轨迹";
  const hint =
    trajectoryMode.value === "face"
      ? "正在检索历史记录。"
      : trajectoryMode.value === "reid"
        ? "正在按 SapiensID 身份向量召回跨摄像头逐 crop 候选，耗时更长。"
        : "正在检索历史记录；外观向量会调用视觉向量模型，耗时会更长。";
  return { label: `${label}轨迹加载中`, hint };
});

function personInitial(name: string | null | undefined): string {
  const text = String(name ?? "").trim();
  return text ? [...text][0].toUpperCase() : "人";
}

function personSubtitle(person: { employee_no?: string | null; department?: string | null }) {
  return [person.employee_no, person.department].filter(Boolean).join(" / ") || "无档案编号";
}

let trajectoryController: AbortController | null = null;
let trajectorySeq = 0;

async function loadTrajectory(personId: string) {
  const seq = ++trajectorySeq;
  trajectoryController?.abort();
  const controller = new AbortController();
  trajectoryController = controller;
  trajectory.value = [];
  trajectoryWarnings.value = [];
  trajectoryLoading.value = true;
  try {
    const result = await personsApi.trajectory(personId, trajectoryMode.value, controller.signal);
    if (seq !== trajectorySeq) return;
    if (result.person.id !== personId) {
      throw new Error("轨迹响应人员与当前选择不一致，请重试");
    }
    trajectory.value = result.items ?? [];
    trajectoryWarnings.value = result.warnings ?? [];
  } catch (error) {
    if ((error as Error)?.name === "AbortError") return;
    if (seq === trajectorySeq) trajectoryWarnings.value = [];
    showError(error);
  } finally {
    if (trajectoryController === controller) trajectoryController = null;
    if (seq === trajectorySeq) trajectoryLoading.value = false;
  }
}

async function loadLibrary() {
  await refreshPersons();
  if (activePersonId.value && !persons.value.some((p) => p.id === activePersonId.value)) {
    activePersonId.value = persons.value[0]?.id ?? null;
  }
  if (!enrollPersonId.value) enrollPersonId.value = activePersonId.value;
  if (activePersonId.value) {
    await loadTrajectory(activePersonId.value);
  } else {
    trajectory.value = [];
    trajectoryWarnings.value = [];
  }
}

async function createPerson() {
  if (!newPerson.value.name.trim()) return;
  creating.value = true;
  try {
    const person = await personsApi.create({
      name: newPerson.value.name.trim(),
      employee_no: newPerson.value.employee_no.trim() || null,
      department: newPerson.value.department.trim() || null,
      phone: newPerson.value.phone.trim() || null,
    });
    activePersonId.value = person.id;
    enrollPersonId.value = person.id;
    newPerson.value = { name: "", employee_no: "", department: "", phone: "" };
    toast("人员已创建");
    await loadLibrary();
  } catch (error) {
    showError(error);
  } finally {
    creating.value = false;
  }
}

async function enrollFace() {
  const personId = enrollPersonId.value;
  if (!personId) {
    toast("请先选择人员");
    return;
  }
  const file = enrollFile.value;
  if (!file) return;
  enrolling.value = true;
  try {
    const body = new FormData();
    body.set("file", file);
    await personsApi.addFace(personId, body);
    activePersonId.value = personId;
    enrollFile.value = null;
    toast("人脸已入库");
    await loadLibrary();
  } catch (error) {
    showError(error);
  } finally {
    enrolling.value = false;
  }
}

async function recognize() {
  const file = recognizeFile.value;
  if (!file) return;
  recognizing.value = true;
  try {
    const body = new FormData();
    body.set("file", file);
    const result = await faceApi.recognize(body, Number(threshold.value || "0.45"));
    faceResult.value = result;
    if (result.person?.id) {
      activePersonId.value = result.person.id;
      await loadTrajectory(result.person.id);
    }
    toast(result.result_type === "known" ? "识别成功" : "未命中人员");
  } catch (error) {
    showError(error);
  } finally {
    recognizing.value = false;
  }
}

async function rebuildRecognition() {
  rebuilding.value = true;
  try {
    const result = await faceApi.rebuildRecognition(1000);
    const updated = result.events_updated ? ` / ${result.events_updated} 更新` : "";
    toast(`重建完成：${result.matched} 命中 / ${result.events_created} 新事件${updated}`);
    if (activePersonId.value) await loadTrajectory(activePersonId.value);
  } catch (error) {
    showError(error);
  } finally {
    rebuilding.value = false;
  }
}

async function runDiagnostics() {
  diagnosing.value = true;
  diagnosticsRunning.value = true;
  try {
    const result = await faceApi.diagnostics(24);
    diagnostics.value = result.items ?? [];
    toast(`诊断完成：${result.items?.length ?? 0} 条`);
  } catch (error) {
    showError(error);
  } finally {
    diagnosing.value = false;
    diagnosticsRunning.value = false;
  }
}

async function enrollDiagnostic(item: FaceDiagnosticItem) {
  if (!activePersonId.value) {
    toast("请先选择人员");
    return;
  }
  if (!item.crop_id) return;
  busyDiagnosticCrop.value = item.crop_id;
  try {
    await personsApi.enrollFromCrop(activePersonId.value, item.crop_id);
    toast("已补入人脸库，请重建历史轨迹");
    await loadLibrary();
  } catch (error) {
    showError(error);
  } finally {
    busyDiagnosticCrop.value = null;
  }
}

function selectPerson(personId: string) {
  activePersonId.value = personId;
  void loadTrajectory(personId);
}

function diagnosticFaceSize(item: FaceDiagnosticItem): string {
  if (!item.face_bbox) return "-";
  const box = item.face_bbox as { width?: number | null; height?: number | null };
  return `${Math.round(Number(box.width ?? 0))}x${Math.round(Number(box.height ?? 0))}`;
}

function trajectoryMeta(item: PersonTrajectoryPoint): string {
  const parts = [item.match_source || "face"];
  if (item.vector_score !== null && item.vector_score !== undefined) {
    parts.push(`vector ${Number(item.vector_score).toFixed(3)}`);
  }
  // Prefer the name; a bare uuid tells an operator nothing about where this was.
  if (item.camera_name || item.location_name) {
    parts.push([item.camera_name, item.location_name].filter(Boolean).join(" / "));
  } else if (item.camera_id) {
    parts.push(`camera ${shortId(item.camera_id)}`);
  }
  if (item.event_id) parts.push(`event ${shortId(item.event_id)}`);
  return parts.join(" / ");
}

function trajectoryFaceSize(item: PersonTrajectoryPoint): string {
  if (!item.face_bbox) return "";
  const box = item.face_bbox as { width?: number | null; height?: number | null };
  return `face ${Math.round(Number(box.width ?? 0))}x${Math.round(Number(box.height ?? 0))}`;
}

function numberText(value: unknown, digits: number): string {
  return value === null || value === undefined ? "-" : Number(value).toFixed(digits);
}

watch(trajectoryMode, () => {
  if (activePersonId.value) void loadTrajectory(activePersonId.value);
});

onMounted(async () => {
  try {
    await refreshPersons();
    const requested = route.query.person_id;
    const requestedId = Array.isArray(requested) ? requested[0] : requested;
    if (requestedId && persons.value.some((person) => person.id === requestedId)) {
      activePersonId.value = requestedId;
    }
    await loadLibrary();
  } catch (error) {
    showError(error);
  }
});
</script>

<template>
  <main class="page-workspace page-shell face-workspace">
    <Teleport to="#page-actions">
      <button class="button primary" type="button" @click="loadLibrary().then(() => toast('已刷新'))">
        刷新
      </button>
    </Teleport>

    <section class="panel face-context" aria-labelledby="faceTitle">
      <div class="section-head">
        <div>
          <h2 id="faceTitle">人脸库</h2>
        </div>
      </div>

      <form class="form-grid person-form" @submit.prevent="createPerson">
        <div class="field-row">
          <label>
            姓名
            <input v-model="newPerson.name" placeholder="张三" required />
          </label>
          <label>
            工号
            <input v-model="newPerson.employee_no" placeholder="E001" />
          </label>
        </div>
        <div class="field-row">
          <label>
            部门
            <input v-model="newPerson.department" placeholder="研发部" />
          </label>
          <label>
            手机
            <input v-model="newPerson.phone" placeholder="138..." />
          </label>
        </div>
        <button class="button primary wide" type="submit" :disabled="creating">
          {{ creating ? "新增中" : "新增人员" }}
        </button>
      </form>

      <form class="form-grid face-form" @submit.prevent="enrollFace">
        <div class="list-head">
          <span>人脸入库</span>
          <span>{{ persons.length }} 人</span>
        </div>
        <label>
          人员
          <PersonSelect v-model="enrollPersonId" :persons="persons" />
        </label>
        <FileField v-model="enrollFile" label="人脸图片" accept="image/*" required />
        <button class="button ghost wide" type="submit" :disabled="enrolling">
          {{ enrolling ? "入库中" : "上传人脸" }}
        </button>
        <button
          class="button ghost wide"
          type="button"
          :disabled="rebuilding"
          @click="rebuildRecognition"
        >
          {{ rebuilding ? "重建中" : "重建历史轨迹" }}
        </button>
      </form>

      <div class="person-list">
        <div v-if="!persons.length" class="empty">
          <strong>还没有人员</strong>
        </div>
        <article
          v-for="person in persons"
          :key="person.id"
          class="person-item"
          :class="{ active: person.id === activePersonId }"
        >
          <img v-if="person.avatar_url" :src="person.avatar_url" :alt="person.name" />
          <div v-else class="person-avatar" aria-hidden="true">{{ personInitial(person.name) }}</div>
          <div>
            <strong>{{ person.name }}</strong>
            <span>{{ personSubtitle(person) }}</span>
          </div>
          <button class="mini-button" type="button" @click="selectPerson(person.id)">轨迹</button>
        </article>
      </div>
    </section>

    <section class="panel face-main" aria-labelledby="recognizeTitle">
      <div class="section-head">
        <div>
          <h2 id="recognizeTitle">相似度识别</h2>
        </div>
      </div>

      <form class="form-grid face-form" @submit.prevent="recognize">
        <FileField v-model="recognizeFile" label="识别图片" accept="image/*" required />
        <label>
          阈值
          <input v-model="threshold" type="number" min="0" max="1" step="0.01" />
        </label>
        <button class="button primary wide" type="submit" :disabled="recognizing">
          {{ recognizing ? "识别中" : "识别人脸" }}
        </button>
      </form>

      <div class="face-result">
        <div v-if="faceResult" class="face-result-card" :class="faceResult.result_type ?? ''">
          <strong>{{ faceResult.person ? faceResult.person.name : "未命中人员" }}</strong>
          <span>
            结果 {{ faceResult.result_type }} / 阈值 {{ numberText(faceResult.threshold, 2) }}
          </span>
          <span>相似度 {{ numberText(faceResult.similarity, 4) }}</span>
          <span v-if="bestMatch">
            Top1 {{ bestMatch.person_name }} {{ numberText(bestMatch.similarity, 4) }}
          </span>
        </div>
      </div>

      <div class="list-head">
        <span>人脸诊断</span>
        <button class="mini-button" type="button" :disabled="diagnosing" @click="runDiagnostics">
          {{ diagnosing ? "诊断中" : "诊断最近 crop" }}
        </button>
      </div>
      <div class="diagnostic-list">
        <div v-if="diagnosticsRunning" class="empty">
          <strong>诊断中</strong>正在分析最近人体 crop。
        </div>
        <div v-else-if="diagnostics === null" class="empty">
          <strong>未运行诊断</strong>
        </div>
        <div v-else-if="!diagnostics.length" class="empty">
          <strong>暂无诊断数据</strong>最近没有人体 crop。
        </div>
        <article
          v-for="item in diagnostics ?? []"
          v-else
          :key="item.crop_id ?? ''"
          class="diagnostic-item"
          :class="item.verdict ?? ''"
        >
          <div class="trajectory-image-frame">
            <img :src="item.crop_url || item.image_url || ''" alt="诊断截图" loading="lazy" />
          </div>
          <div class="media-meta">
            <strong>
              {{ VERDICT_LABELS[item.verdict] ?? item.verdict }} / Top1
              {{
                item.top_person_name
                  ? `${item.top_person_name} ${Number(item.top_similarity || 0).toFixed(4)}`
                  : "-"
              }}
            </strong>
            <span>
              det {{ numberText(item.detection_score, 3) }} / face {{ diagnosticFaceSize(item) }} /
              阈值 {{ Number(item.threshold || 0).toFixed(2) }}
            </span>
            <span>{{ item.reason || "" }}</span>
            <span>{{ fmtTime(item.captured_at) }} / crop {{ shortId(item.crop_id) }}</span>
            <button
              v-if="item.can_enroll && activePersonId"
              class="mini-button"
              type="button"
              :disabled="busyDiagnosticCrop === item.crop_id"
              @click="enrollDiagnostic(item)"
            >
              {{ busyDiagnosticCrop === item.crop_id ? "补入中" : "补入当前人员" }}
            </button>
          </div>
        </article>
      </div>

      <div class="list-head">
        <span>人员轨迹</span>
        <div class="trajectory-controls">
          <select v-model="trajectoryMode" aria-label="轨迹模式">
            <option value="face">人脸轨迹</option>
            <option value="all">大表人脸</option>
            <option value="reid">ReID 疑似出现点（逐 crop）</option>
            <option value="vector">外观向量轨迹（非身份）</option>
          </select>
          <span>{{ trajectoryLoading ? "…" : trajectory.length }}</span>
        </div>
      </div>
      <p v-if="trajectoryMode === 'reid'" class="muted-text">
        这是依据已知人员身体 gallery 检索出的概率性摄像头出现点，不是盲区内的连续路径，也尚未形成持久 tracklet / global identity。
      </p>
      <div v-if="trajectoryWarnings.length" class="trajectory-warning" role="status">
        <strong>ReID 检索未完成</strong>
        <span v-for="warning in trajectoryWarnings" :key="warning">{{ warning }}</span>
      </div>
      <div class="trajectory-list">
        <div v-if="trajectoryLoading" class="empty">
          <strong>{{ trajectoryLoadingText.label }}</strong>{{ trajectoryLoadingText.hint }}
        </div>
        <div v-else-if="!trajectory.length" class="empty">
          <strong>暂无轨迹</strong>
        </div>
        <article
          v-for="(item, index) in trajectory"
          v-else
          :key="item.event_id ?? item.crop_id ?? index"
          class="trajectory-item"
        >
          <div class="trajectory-visuals">
            <FaceBoxThumb
              link-class="trajectory-image-frame"
              :src="item.crop_url || item.image_url || ''"
              alt="轨迹截图"
              :face-box="item.face_bbox"
            />
            <FaceZoom
              v-if="item.face_bbox"
              :src="item.crop_url || item.image_url || ''"
              :face-box="item.face_bbox"
            />
          </div>
          <div class="media-meta">
            <strong>
              {{ item.person_name }}
              {{ item.similarity === null || item.similarity === undefined ? "" : Number(item.similarity).toFixed(3) }}
            </strong>
            <span>{{ fmtTime(item.recognized_at) }}</span>
            <span>{{ trajectoryMeta(item) }}</span>
            <span v-if="trajectoryFaceSize(item)">{{ trajectoryFaceSize(item) }}</span>
          </div>
        </article>
      </div>
    </section>
  </main>
</template>

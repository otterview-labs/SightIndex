<script setup lang="ts">
import { computed, onMounted, ref } from "vue";

import { attributes as attributesApi, search as searchApi } from "@/api/client";
import type { SearchFilters, SearchResultItem, SemanticSearchStatus } from "@/api/types";
import EmptyState from "@/components/EmptyState.vue";
import SearchResultCard from "@/components/SearchResultCard.vue";
import { useSummary } from "@/composables/useSummary";
import { useToast } from "@/composables/useToast";
import { DISPLAY_TIME_ZONE } from "@/utils/format";

const QUERY_CHIPS = [
  { query: "红衣戴帽的人", label: "红衣戴帽" },
  { query: "戴眼镜的人", label: "戴眼镜" },
  { query: "背包的人", label: "背包" },
  { query: "黑衣背包的人", label: "黑衣背包" },
  { query: "白衣戴眼镜的人", label: "白衣眼镜" },
  { query: "玩手机的人", label: "玩手机" },
  { query: "抽烟的人", label: "抽烟" },
  { query: "跌倒的人", label: "跌倒" },
  { query: "打架的人", label: "打架" },
];

const ATTRIBUTE_HINT = "衣服 / 背包 / 眼镜 / 抽烟 / 手机 / 跌倒 / 打架";

const { showError, toast } = useToast();
const { crops, imageTotal, cropTotal, runningCount, cameraOptions, refresh } = useSummary();

const query = ref("");
const cameraId = ref("");
const startTime = ref("");
const endTime = ref("");
const mode = ref<"semantic" | "structured">("structured");
const resultMode = ref<"recent" | "semantic" | "structured">("recent");
const capabilities = ref<SemanticSearchStatus | null>(null);
const searchNotice = ref("");
const capabilityError = ref("");
const semanticEnabled = computed(() =>
  capabilities.value?.enabled && capabilities.value?.configured,
);

const results = ref<SearchResultItem[]>([]);
const hint = ref("按时间分组展示候选裁剪");
const status = ref<"idle" | "loading" | "empty" | "error">("loading");
const loadingLabel = ref("加载最近裁剪...");
const searching = ref(false);
const backfilling = ref(false);
const attributeStatus = ref(ATTRIBUTE_HINT);

const dayFormatter = new Intl.DateTimeFormat("zh-CN", {
  timeZone: DISPLAY_TIME_ZONE,
  month: "2-digit",
  day: "2-digit",
});

const groups = computed(() => {
  if (resultMode.value === "semantic") {
    return results.value.length
      ? [{ title: "按语义相似度排序 · 待人工核验", items: results.value }]
      : [];
  }
  const today = dayFormatter.format(new Date());
  const buckets = new Map<string, { title: string; items: SearchResultItem[] }>();
  for (const item of results.value) {
    const date = item.captured_at ? new Date(item.captured_at) : null;
    const key = date && !Number.isNaN(date.valueOf()) ? dayFormatter.format(date) : "未记录时间";
    if (!buckets.has(key)) buckets.set(key, { title: key === today ? "今天" : key, items: [] });
    buckets.get(key)!.items.push(item);
  }
  return [...buckets.values()];
});

function filters(): SearchFilters {
  const value: SearchFilters = {};
  if (cameraId.value) value.camera_id = cameraId.value;
  if (startTime.value) value.start_time = new Date(startTime.value).toISOString();
  if (endTime.value) value.end_time = new Date(endTime.value).toISOString();
  return value;
}

function showRecentCrops() {
  resultMode.value = "recent";
  searchNotice.value = "";
  const fallback: SearchResultItem[] = crops.value.map((crop) => ({
    crop_id: crop.id,
    image_id: crop.image_id,
    crop_url: crop.crop_url,
    captured_at: crop.captured_at ?? crop.created_at,
    attributes: crop.attributes,
    person_id: crop.person_id,
    camera_id: crop.camera_id,
    location_id: crop.location_id,
    score: null as unknown as number,
  }));
  if (!fallback.length) {
    results.value = [];
    status.value = "empty";
    hint.value = "暂无候选";
    return;
  }
  results.value = fallback;
  status.value = "idle";
  hint.value = `最近 ${fallback.length} 个裁剪（尚未执行检索）`;
}

function changeMode() {
  results.value = [];
  resultMode.value = mode.value;
  searchNotice.value = "";
  status.value = "idle";
  hint.value = mode.value === "semantic"
    ? "语义候选仅表示相似，不代表全部条件成立；请输入描述后检索"
    : "严格标签匹配仅返回已解析的标签；请输入标签后检索";
}

async function runSearch(text: string) {
  const trimmed = text.trim();
  if (!trimmed || searching.value) return;
  searching.value = true;
  const requestedMode = mode.value;
  resultMode.value = requestedMode;
  results.value = [];
  searchNotice.value = "";
  status.value = "loading";
  loadingLabel.value = requestedMode === "semantic" ? "正在检索 Qwen 语义候选..." : "正在匹配结构化标签...";
  hint.value = requestedMode === "semantic"
    ? "按视觉语义相似度召回，结果需要人工核验"
    : "仅返回同时满足全部已解析标签和筛选条件的结果";
  try {
    if (startTime.value && endTime.value && startTime.value > endTime.value) {
      throw new Error("开始时间不能晚于结束时间");
    }
    const payload = {
      query: trimmed,
      top_k: 20,
      filters: filters(),
    };
    const response = requestedMode === "semantic"
      ? await searchApi.semanticPersonCrops(payload)
      : await searchApi.personCrops({ ...payload, rerank: false });
    if ("notice" in response) searchNotice.value = String(response.notice);
    const items = response.items ?? [];
    if (!items.length) {
      results.value = [];
      status.value = "empty";
      hint.value = requestedMode === "semantic"
        ? `“${trimmed}”暂无达到门槛且可读取的语义候选`
        : `没有匹配“${trimmed}”的结构化标签结果`;
      return;
    }
    results.value = items;
    status.value = "idle";
    hint.value = requestedMode === "semantic"
      ? `返回 ${items.length} 个语义候选（非标签命中，非身份确认）`
      : `返回 ${items.length} 个标签命中结果`;
  } catch (error) {
    results.value = [];
    status.value = "error";
    hint.value = error instanceof Error ? error.message : "检索服务暂不可用，请稍后重试";
    showError(error);
  } finally {
    searching.value = false;
  }
}

function useChip(chip: (typeof QUERY_CHIPS)[number]) {
  query.value = chip.query;
  void runSearch(chip.query);
}

async function backfillAttributes() {
  if (backfilling.value) return;
  backfilling.value = true;
  attributeStatus.value = "正在解析最近 50 个未解析裁剪";
  try {
    const result = await attributesApi.backfillCrops(50, true);
    const message = `已解析 ${result.updated ?? 0}/${result.seen ?? 0}`;
    attributeStatus.value = message;
    toast(message);
    await refresh();
    showRecentCrops();
  } catch (error) {
    attributeStatus.value = ATTRIBUTE_HINT;
    showError(error);
  } finally {
    backfilling.value = false;
  }
}

onMounted(async () => {
  try {
    await Promise.all([
      refresh(),
      searchApi.semanticStatus().then((value) => {
        capabilities.value = value;
        if (value.enabled && value.configured) mode.value = "semantic";
      }).catch(() => {
        capabilityError.value = "无法读取语义检索状态，暂用严格标签匹配";
      }),
    ]);
    showRecentCrops();
  } catch (error) {
    status.value = "empty";
    showError(error);
  }
});
</script>

<template>
  <main class="page-workspace page-shell sentinel-search-workspace">
    <section class="search-main sentinel-question-page" aria-labelledby="searchTitle">
      <div class="sentinel-page-header page-header">
        <h2 id="searchTitle">问图检索</h2>
        <div class="entity-segment" aria-label="检索对象">
          <button class="active" type="button">人员</button>
          <button type="button" disabled>车辆</button>
        </div>
      </div>

      <form class="sentinel-search-form" @submit.prevent="runSearch(query)">
        <div class="semantic-input">
          <span class="semantic-prefix" aria-hidden="true">⌕</span>
          <input
            v-model="query"
            name="query"
            placeholder="例如：黑色上衣并且背包、戴眼镜、看手机"
            required
          />
          <button class="button primary" type="submit" :disabled="searching">检索</button>
        </div>

        <div class="question-layout">
          <aside class="question-filter-panel" aria-label="检索筛选">
            <div class="filter-panel-head">
              <strong>筛选条件</strong>
              <span>模式 / 相机 / 时间</span>
            </div>
            <label>
              检索模式
              <select v-model="mode" name="search_mode" :disabled="searching" @change="changeMode">
                <option value="semantic" :disabled="!semanticEnabled">Qwen 语义候选（需核验）</option>
                <option value="structured">严格标签匹配</option>
              </select>
            </label>
            <p v-if="capabilities" class="search-coverage">
              语义索引 {{ capabilities.indexed_crops }}/{{ capabilities.total_crops }}；
              已有属性 {{ capabilities.labeled_crops }}/{{ capabilities.total_crops }}
              <span v-if="!capabilities.auto_index_on_ingest"> · 新增数据自动索引未开启</span>
            </p>
            <p v-if="capabilityError" role="status">{{ capabilityError }}</p>
            <label>
              摄像头
              <select v-model="cameraId" name="camera_id">
                <option value="">全部摄像头</option>
                <option v-for="camera in cameraOptions" :key="camera.id" :value="camera.id">
                  {{ camera.name }}
                </option>
              </select>
            </label>
            <label>
              开始时间
              <input v-model="startTime" name="start_time" type="datetime-local" />
            </label>
            <label>
              结束时间
              <input v-model="endTime" name="end_time" type="datetime-local" />
            </label>
            <div class="query-chips" aria-label="常用检索">
              <button
                v-for="chip in QUERY_CHIPS"
                :key="chip.query"
                type="button"
                @click="useChip(chip)"
              >
                {{ chip.label }}
              </button>
            </div>
            <div class="attribute-panel">
              <div>
                <strong>结构化解析</strong>
                <span>{{ attributeStatus }}</span>
              </div>
              <button
                class="button ghost wide"
                type="button"
                :disabled="backfilling || !capabilities?.attributes_enabled"
                @click="backfillAttributes"
              >
                {{ backfilling ? "解析中" : capabilities?.attributes_enabled ? "解析最近裁剪" : "属性解析模型未启用" }}
              </button>
            </div>
            <div class="metric-list search-metrics">
              <div><span>全部人物裁剪</span><strong>{{ cropTotal }}</strong></div>
              <div><span>全部有人帧</span><strong>{{ imageTotal }}</strong></div>
              <div><span>运行视频流</span><strong>{{ runningCount }}</strong></div>
            </div>
          </aside>

          <section class="question-content-panel" aria-label="检索结果">
            <div class="results-toolbar">
              <div>
                <strong>检索结果</strong>
                <span>{{ hint }}</span>
              </div>
              <span class="result-mode">{{ resultMode === "semantic" ? "语义候选 · 非身份确认" : resultMode === "structured" ? "严格标签" : "最近裁剪" }}</span>
            </div>
            <p v-if="searchNotice" class="semantic-notice" role="status">{{ searchNotice }}</p>
            <div class="question-results" aria-live="polite">
              <EmptyState v-if="status === 'loading'">{{ loadingLabel }}</EmptyState>
              <EmptyState
                v-else-if="status === 'error'"
                title="检索失败"
                hint="服务错误不代表没有目标；请检查索引或服务状态后重试，没有自动切换到其他检索方式。"
              />
              <EmptyState
                v-else-if="status === 'empty'"
                :title="resultMode === 'semantic' ? '暂无语义候选' : '没有标签命中'"
                :hint="resultMode === 'semantic'
                  ? '请调整描述或筛选条件；未召回不代表目标一定不存在。'
                  : '严格标签匹配依赖已解析的属性；属性为空时不会命中，可切换到语义候选。'"
              />
              <section v-for="group in groups" v-else :key="group.title" class="result-day-group">
                <div class="result-day-head">
                  <strong>{{ group.title }}</strong>
                  <span>{{ group.items.length }} 个结果</span>
                </div>
                <div class="result-card-grid">
                  <SearchResultCard
                    v-for="(item, index) in group.items"
                    :key="item.crop_id ?? index"
                    :item="item"
                    :semantic="resultMode === 'semantic'"
                  />
                </div>
              </section>
            </div>
          </section>
        </div>
      </form>
    </section>
  </main>
</template>

<style scoped>
.search-coverage {
  color: #64748b;
  font-size: 12px;
  line-height: 1.6;
}

.semantic-notice {
  margin: 12px;
  padding: 12px 16px;
  border: 1px solid #bfdbfe;
  border-radius: 8px;
  background: #eff6ff;
  color: #1e40af;
  font-size: 13px;
  line-height: 1.6;
}
</style>

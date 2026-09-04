<script setup lang="ts">
import { computed, onMounted, ref } from "vue";

import { attributes as attributesApi, search as searchApi } from "@/api/client";
import type { SearchFilters, SearchResultItem } from "@/api/types";
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

const results = ref<SearchResultItem[]>([]);
const hint = ref("按时间分组展示候选裁剪");
const status = ref<"idle" | "loading" | "empty">("loading");
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
  hint.value = `最近 ${fallback.length} 个裁剪（尚未执行标签检索）`;
}

async function runSearch(text: string) {
  const trimmed = text.trim();
  if (!trimmed || searching.value) return;
  searching.value = true;
  status.value = "loading";
  loadingLabel.value = "正在匹配结构化标签...";
  hint.value = "仅返回同时满足全部标签和筛选条件的结果";
  try {
    const response = await searchApi.personCrops({
      query: trimmed,
      top_k: 20,
      filters: filters(),
      rerank: false,
    });
    const items = response.items ?? [];
    if (!items.length) {
      results.value = [];
      status.value = "empty";
      hint.value = `没有匹配“${trimmed}”的结构化标签结果`;
      return;
    }
    results.value = items;
    status.value = "idle";
    hint.value = `返回 ${items.length} 个标签命中结果`;
  } catch (error) {
    status.value = "empty";
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
    await refresh();
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
              <span>相机 / 时间 / 标签</span>
            </div>
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
                :disabled="backfilling"
                @click="backfillAttributes"
              >
                {{ backfilling ? "解析中" : "解析最近裁剪" }}
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
              <span class="result-mode">PERSON CROP</span>
            </div>
            <div class="question-results" aria-live="polite">
              <div v-if="status === 'loading'" class="empty">{{ loadingLabel }}</div>
              <div v-else-if="status === 'empty'" class="empty">
                <strong>没有标签命中</strong>
                请使用衣服颜色、帽子、眼镜、背包、手机、抽烟、跌倒或打架等明确标签。
              </div>
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

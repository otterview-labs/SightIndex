<script setup lang="ts">
import { computed } from "vue";

import type { SearchResultItem } from "@/api/types";
import { structuredAttributeChips } from "@/utils/attributes";
import { fmtTime, formatScore, shortId, shortText } from "@/utils/format";

const props = defineProps<{
  item: SearchResultItem & { duplicate_crop_ids?: string[] };
  semantic?: boolean;
}>();

const emit = defineEmits<{
  (e: "locate-video", item: SearchResultItem & { duplicate_crop_ids?: string[] }): void;
}>();

const imageUrl = computed(() => props.item.crop_url || props.item.image_url || "");

const title = computed(() =>
  props.item.person_name
    ? `${props.item.person_name} · crop ${shortId(props.item.crop_id)}`
    : `crop ${shortId(props.item.crop_id)}`,
);

const place = computed(() => {
  const parts = [
    props.item.camera_name || props.item.camera_id,
    props.item.location_name || props.item.location_id,
  ]
    .filter(Boolean)
    .map((value) => shortText(String(value), 18));
  return parts.length ? parts.join(" / ") : "";
});

const chips = computed(() => structuredAttributeChips(props.item.attributes));

const hasScore = computed(() => props.item.score !== null && props.item.score !== undefined);

const hasEmbeddingScore = computed(
  () => props.item.embedding_rerank_score !== null && props.item.embedding_rerank_score !== undefined,
);
const hasRerankScore = computed(
  () => props.item.rerank_score !== null && props.item.rerank_score !== undefined,
);
</script>

<template>
  <article class="question-result-card">
    <a class="question-card-cover" :href="imageUrl" target="_blank" rel="noreferrer">
      <img :src="imageUrl" alt="检索结果" loading="lazy" decoding="async" />
      <span v-if="hasScore" class="score-badge">{{ semantic ? "相似度" : "score" }} {{ semantic ? item.score.toFixed(3) : formatScore(item.score) }}</span>
      <button
        v-if="item.image_id"
        class="card-locate-video"
        type="button"
        title="在源视频中定位此画面"
        @click.prevent="emit('locate-video', item)"
      >
        ▶ 定位视频
      </button>
    </a>
    <div class="question-card-body">
      <strong>{{ title }}</strong>
      <p v-if="semantic">语义候选 · 标签未核验 · 非身份确认</p>
      <p v-if="item.duplicate_crop_ids?.length">
        合并 {{ item.duplicate_crop_ids.length }} 条同内容记录
      </p>
      <div class="score-row">
        <span v-if="hasEmbeddingScore" class="score-pill">
          向量 {{ formatScore(item.embedding_rerank_score) }}
        </span>
        <span v-if="hasRerankScore" class="score-pill">
          Rerank {{ formatScore(item.rerank_score) }}
        </span>
      </div>
      <div class="question-card-meta">
        <span>图片 {{ shortId(item.image_id) }}</span>
        <span>{{ fmtTime(item.captured_at) }}</span>
        <span v-if="place">{{ place }}</span>
      </div>
      <div v-if="chips.length" class="attribute-chip-row">
        <span v-for="chip in chips" :key="chip.label + chip.value" class="attribute-chip">
          <small>{{ chip.label }}</small>{{ chip.value }}
        </span>
      </div>
      <p v-if="item.rerank_reason">{{ item.rerank_reason }}</p>
    </div>
  </article>
</template>

<style scoped>
/* Inside the cover link overlaying the image; the click is stopped so the link
   navigation does not fire. */
.card-locate-video {
  position: absolute;
  z-index: 1;
  top: 8px;
  right: 8px;
  padding: 4px 10px;
  border: 1px solid rgba(255, 255, 255, 0.55);
  border-radius: 999px;
  background: rgba(15, 23, 42, 0.72);
  color: #fff;
  cursor: pointer;
  font-size: 12px;
  line-height: 1.5;
}

.card-locate-video:hover,
.card-locate-video:focus-visible {
  background: rgba(15, 23, 42, 0.9);
}

.card-locate-video:focus-visible {
  outline: 2px solid #fff;
  outline-offset: 1px;
}
</style>

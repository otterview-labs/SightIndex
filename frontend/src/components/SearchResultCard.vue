<script setup lang="ts">
import { computed } from "vue";

import type { SearchResultItem } from "@/api/types";
import { structuredAttributeChips } from "@/utils/attributes";
import { fmtTime, formatScore, shortId, shortText } from "@/utils/format";

const props = defineProps<{ item: SearchResultItem }>();

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
      <span v-if="hasScore" class="score-badge">score {{ formatScore(item.score) }}</span>
    </a>
    <div class="question-card-body">
      <strong>{{ title }}</strong>
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

<script lang="ts">
export interface ChatResultItem {
  title?: string | null;
  name?: string | null;
  status?: string | null;
  result_type?: string | null;
  face_url?: string | null;
  crop_url?: string | null;
  image_url?: string | null;
  avatar_url?: string | null;
  crop_id?: string | null;
  image_id?: string | null;
  stream_id?: string | null;
  person_id?: string | null;
  person_name?: string | null;
  score?: number | null;
  similarity?: number | null;
  captured_at?: string | null;
  created_at?: string | null;
  recognized_at?: string | null;
  updated_at?: string | null;
}
</script>

<script setup lang="ts">
import { computed } from "vue";

import { fmtTime, formatScore, shortId } from "@/utils/format";

const props = defineProps<{ items: ChatResultItem[] }>();

const MAX_CARDS = 8;

const cards = computed(() =>
  props.items.slice(0, MAX_CARDS).map((item) => {
    const url = item.face_url || item.crop_url || item.image_url || item.avatar_url || "";
    return {
      item,
      url,
      alt: item.title || item.person_name || "检索候选",
      placeholder: item.status || item.person_name || item.title || "数据",
      label: label(item),
      identifier: identifier(item),
      time: fmtTime(item.captured_at || item.created_at || item.recognized_at || item.updated_at),
    };
  }),
);

function label(item: ChatResultItem): string {
  if (item.person_name && item.similarity !== null && item.similarity !== undefined) {
    return `${item.person_name} ${formatScore(item.similarity)}`;
  }
  if (item.score === null || item.score === undefined) {
    return item.title || item.name || item.person_name || "候选";
  }
  return `score ${formatScore(item.score)}`;
}

function identifier(item: ChatResultItem): string {
  if (item.crop_id) return `crop ${shortId(item.crop_id)}`;
  if (item.image_id) return `image ${shortId(item.image_id)}`;
  if (item.stream_id) return `stream ${shortId(item.stream_id)}`;
  if (item.person_id) return `person ${shortId(item.person_id)}`;
  return item.status || item.result_type || "";
}
</script>

<template>
  <div class="chat-results">
    <component
      :is="card.url ? 'a' : 'article'"
      v-for="(card, index) in cards"
      :key="index"
      class="chat-result-card"
      v-bind="card.url ? { href: card.url, target: '_blank', rel: 'noreferrer' } : {}"
    >
      <img
        v-if="card.url"
        :src="card.url"
        :alt="card.alt"
        loading="lazy"
        decoding="async"
      />
      <div v-else class="chat-result-placeholder">{{ card.placeholder }}</div>
      <div class="media-meta">
        <strong>{{ card.label }}</strong>
        <span>{{ card.identifier }}</span>
        <span>{{ card.time }}</span>
      </div>
    </component>
  </div>
</template>

<script setup lang="ts">
import { computed } from "vue";

// Shared presentation-only subset of a search result and a per-camera candidate.
interface ReidEvidence {
  crop_id: string;
  score: number;
  person_name?: string | null;
  face_similarity?: number | null;
  face_reliability?: number | null;
  face_match?: boolean | null;
  face_query_identity_verified?: boolean | null;
  face_candidate_identity_verified?: boolean | null;
  face_candidate_source_crop_id?: string | null;
  attribute_agreement?: number | null;
  attribute_matches?: string[];
  attribute_conflicts?: string[];
  attribute_comparable_count?: number | null;
  attribute_match_count?: number | null;
  stature_agreement?: number | null;
  fusion_score?: number | null;
  evidence_level?: string | null;
  decision_reason?: string | null;
  beats_chance?: boolean | null;
}

const props = defineProps<{
  item: ReidEvidence;
  when: string;
  confirmed: boolean | null;
  showScore?: boolean;
}>();

function measured(value: unknown): value is number {
  return typeof value === "number" && Number.isFinite(value);
}

function score(value: unknown): string {
  return measured(value) ? value.toFixed(2) : "未提供";
}

const unverifiedFace = computed(() => measured(props.item.face_similarity)
  && (props.item.face_query_identity_verified === false
    || props.item.face_candidate_identity_verified === false));

const verdict = computed(() => {
  if (props.confirmed === true) return "人工已确认：同一个人";
  if (props.confirmed === false) return "人工已确认：不是同一个人";
  if (unverifiedFace.value) return "人脸来源身份未验证 · 仅作辅助线索";
  if (props.item.face_match === false || props.item.evidence_level === "rejected") {
    return "证据冲突 · 建议排除";
  }
  if (props.item.face_match === true) return "人脸证据支持 · 待核对";
  if (props.item.beats_chance === false || props.item.evidence_level === "clue") {
    return "低置信线索 · 可能只是相似";
  }
  if (props.item.evidence_level === "reliable") return "人体高度相似 · 待核对";
  return "相似候选 · 待核对";
});

const attributeText = computed(() => {
  const item = props.item;
  const compared = item.attribute_comparable_count
    ?? (item.attribute_matches?.length ?? 0) + (item.attribute_conflicts?.length ?? 0);
  if (!measured(item.attribute_agreement) || compared < 2) {
    return "可比高置信标签不足，保持中性";
  }
  const matched = item.attribute_match_count ?? item.attribute_matches?.length ?? 0;
  return `高置信标签一致 ${matched}/${compared} · 加权一致 ${Math.round(item.attribute_agreement * 100)}%`;
});

const faceText = computed(() => {
  const item = props.item;
  if (!measured(item.face_similarity)) return "未提供可用人脸比较，不代表不一致";
  if (unverifiedFace.value) {
    return `补充帧人脸的身份关联未验证，仅作软证据，不能据此确认或排除 · 相似度 ${score(item.face_similarity)} · 质量评分 ${score(item.face_reliability)}`;
  }
  const decision = item.face_match === true ? "支持匹配"
    : item.face_match === false ? "明确冲突" : "未形成明确结论";
  return `${decision} · 相似度 ${score(item.face_similarity)} · 质量评分 ${score(item.face_reliability)}`;
});
</script>

<template>
  <div class="reid-evidence">
    <p v-if="showScore" class="reid-evidence-score">人体相似度 <b>{{ score(item.score) }}</b></p>
    <p class="reid-evidence-time">{{ when || "无时间" }}</p>
    <p class="reid-evidence-verdict" :class="{ confirmed: confirmed === true, denied: confirmed === false }">
      {{ verdict }}
    </p>
    <details class="reid-evidence-details">
      <summary>证据明细</summary>
      <dl>
        <dt>人体</dt>
        <dd>相似度 {{ score(item.score) }}，不是同人概率</dd>
        <template v-if="item.decision_reason">
          <dt>依据</dt><dd>{{ item.decision_reason }}</dd>
        </template>
        <dt>人脸</dt><dd>{{ faceText }}</dd>
        <template v-if="item.face_candidate_source_crop_id && item.face_candidate_source_crop_id !== item.crop_id">
          <dt>取脸来源</dt><dd>同次出现的补充帧 · crop {{ item.face_candidate_source_crop_id }}；卡片仍展示候选代表图</dd>
        </template>
        <dt>标签</dt><dd>{{ attributeText }}</dd>
        <template v-if="measured(item.stature_agreement)">
          <dt>身高</dt><dd>几何一致评分 {{ score(item.stature_agreement) }}，仅辅助参考</dd>
        </template>
        <template v-if="item.beats_chance != null">
          <dt>巧合线</dt>
          <dd>{{ item.beats_chance ? "人体分高于参考线，仍需核对" : "人体分处于巧合区间，不能据此确认到访" }}</dd>
        </template>
        <template v-if="measured(item.fusion_score)">
          <dt>融合分</dt><dd>{{ score(item.fusion_score) }}，非概率；可靠人脸结论优先于数值分</dd>
        </template>
        <template v-if="item.person_name">
          <dt>关联姓名</dt><dd>{{ item.person_name }}（需核对身份）</dd>
        </template>
        <dt>裁剪</dt><dd class="reid-evidence-id">{{ item.crop_id }}</dd>
      </dl>
    </details>
  </div>
</template>

<style scoped>
.reid-evidence { display: grid; min-width: 0; width: 100%; gap: 7px; }
.reid-evidence p { margin: 0; }
.reid-evidence-time { color: var(--muted, #667085); font-size: 12px; font-variant-numeric: tabular-nums; }
.reid-evidence-score { display: flex; align-items: baseline; gap: 8px; font-size: 11px; color: var(--muted, #667085); }
.reid-evidence-score b { font-size: 18px; color: var(--ink, #182230); font-variant-numeric: tabular-nums; }
.reid-evidence-verdict { color: var(--ink, #182230); font-size: 12px; font-weight: 650; line-height: 1.6; }
.reid-evidence-verdict.confirmed { color: #116538; }
.reid-evidence-verdict.denied { color: #9c2630; }
.reid-evidence-details { min-width: 0; font-size: 12px; }
.reid-evidence-details summary { width: fit-content; min-height: 32px; padding: 5px 0; cursor: pointer; color: var(--accent, #246bfd); }
.reid-evidence-details summary:focus-visible { outline: 2px solid var(--accent, #246bfd); outline-offset: 2px; border-radius: 3px; }
.reid-evidence-details dl { display: grid; grid-template-columns: 1fr; gap: 3px; margin: 5px 0 0; padding: 10px; border: 1px solid var(--line, #dbe3ea); border-radius: 8px; background: var(--surface-soft, #f4f7fa); }
.reid-evidence-details dt { color: var(--muted, #667085); margin-top: 5px; font-size: 11px; }
.reid-evidence-details dt:first-child { margin-top: 0; }
.reid-evidence-details dd { margin: 0; color: var(--ink, #182230); overflow-wrap: anywhere; line-height: 1.6; }
.reid-evidence-id { font-family: ui-monospace, monospace; font-size: 11px; }
</style>

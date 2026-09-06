<script setup lang="ts">
import { computed } from "vue";
import type { ReidFaceCoverage } from "@/api/types";

// Partial also handles an older API or a rolling upgrade without inventing zero counts.
const props = defineProps<{
  coverage?: Partial<ReidFaceCoverage> | null;
  scope: "search" | "links";
}>();

const reasonLabels: Record<string, string> = {
  no_face: "检测／提取阶段没有可用人脸候选",
  multiple_faces: "检测到多张人脸，已弃权",
  face_outside_head: "人脸位置不符合人体头部区域",
  invalid_bbox: "人体框无效",
  source_missing: "原图暂不可用",
  candidate_row_missing: "候选裁剪记录不存在或已移除",
  source_unreadable: "原图读取失败",
  temp_write_failed: "临时裁剪生成失败",
  unsupported_url: "不支持的图片路径",
  low_quality: "人脸质量不足",
  model_incompatible: "人脸模型不兼容",
  invalid_embedding: "人脸向量无效",
  extraction_unexplained: "旧提取记录未包含原因，待重试",
  inference_error: "人脸推理异常",
  query_identity_unverified: "补充查询帧与源图的身份关联未验证",
};

function measured(value: unknown): value is number {
  return typeof value === "number" && Number.isFinite(value);
}

function count(value: unknown): string {
  return measured(value) && value >= 0 ? String(value) : "未提供";
}

function reasons(values: Record<string, number> | undefined): string[] {
  return Object.entries(values ?? {}).filter(([, value]) => measured(value) && value > 0)
    .map(([reason, value]) => `${reasonLabels[reason] ?? "其他未分类原因"} ×${value}`);
}

const summary = computed(() => {
  switch (props.coverage?.status) {
    case "disabled": return "本次未启用人脸优先，使用人体与标签证据。";
    case "unavailable": return "本次人脸服务不可用，使用人体与标签证据。";
    case "no_candidates": return "本次没有进入人脸比较阶段的候选。";
    case "query_unavailable": return "本次查询未提取到可用人脸，使用人体与标签证据；不代表候选图片都没有人脸。";
    case "candidate_unavailable": return props.coverage.query_face_found === true
      ? "查询侧有可用人脸，但本次尝试的候选未形成可用比较；不能据此判断为不同人。"
      : "候选资料不可用或未形成可用比较，未形成身份结论；不能据此判断为不同人。";
    case "compared": return props.coverage.query_identity_verified === false
      ? "已进行人脸比较，但查询人脸与源图的身份关联未验证，仅作软证据。"
      : "本次已进行人脸比较；可靠且身份关联已验证的人脸证据优先，仍需人工核对。";
    case "error": return "本次人脸比较发生异常，不能将缺失的人脸结果理解为不一致。";
    default: return "服务端未提供本次人脸参与诊断，不能仅凭模型就绪判断人脸已参与排序。";
  }
});

const queryReasons = computed(() => reasons(props.coverage?.query_absence_reasons));
const candidateReasons = computed(() => reasons(props.coverage?.candidate_absence_reasons));
const queryFace = computed(() => props.coverage?.query_face_found === true ? "有可用人脸"
  : props.coverage?.query_attempted_count === 0 ? "本次未尝试提取查询人脸"
    : props.coverage?.query_face_found === false ? "未获得可用人脸" : "未提供");
</script>

<template>
  <div class="reid-face-coverage" aria-live="polite">
    <p class="reid-face-coverage-title">{{ scope === "search" ? "本次检索的人脸参与" : "跨摄像头线索的人脸参与" }}</p>
    <p>{{ summary }}</p>
    <template v-if="coverage">
      <p v-if="measured(coverage.compared_count) || measured(coverage.shortlist_count)" class="reid-face-coverage-count">
        已完成人脸比较 {{ count(coverage.compared_count) }} / {{ count(coverage.shortlist_count) }} 个入选候选（最终筛选前）
      </p>
      <details>
        <summary>查看人脸诊断</summary>
        <dl>
          <dt>查询侧</dt><dd>{{ queryFace }}；尝试 {{ count(coverage.query_attempted_count) }} 帧<template v-if="measured(coverage.query_face_quality)">；质量评分 {{ coverage.query_face_quality.toFixed(2) }}</template></dd>
          <dt>候选侧</dt><dd>尝试 {{ count(coverage.candidate_attempted_count) }} 帧；其中 {{ count(coverage.borrowed_candidate_count) }} 个候选使用补充帧人脸</dd>
          <dt>明确证据</dt><dd>支持匹配 {{ count(coverage.hard_match_count) }} 个；明确冲突 {{ count(coverage.hard_conflict_count) }} 个（不是人工确认）</dd>
          <template v-if="queryReasons.length"><dt>查询侧弃权原因</dt><dd>{{ queryReasons.join("；") }}</dd></template>
          <template v-if="candidateReasons.length"><dt>候选侧弃权原因</dt><dd>{{ candidateReasons.join("；") }}</dd></template>
        </dl>
        <p class="reid-face-coverage-note">统计针对人脸候选池，不等于下方最终显示数量；尝试帧数可能包含同次出现的补充帧。没有比较不代表人脸冲突。</p>
      </details>
    </template>
  </div>
</template>

<style scoped>
.reid-face-coverage { display: grid; gap: 6px; margin: 12px 0; padding: 12px; border: 1px solid var(--line, #dbe3ea); border-radius: 8px; background: var(--surface-soft, #f4f7fa); color: var(--ink, #182230); font-size: 12px; line-height: 1.7; overflow-wrap: anywhere; }
.reid-face-coverage p { margin: 0; }
.reid-face-coverage-title { font-weight: 650; }
.reid-face-coverage-count { font-variant-numeric: tabular-nums; }
.reid-face-coverage summary { width: fit-content; cursor: pointer; min-height: 32px; padding: 5px 0; color: var(--accent, #246bfd); }
.reid-face-coverage summary:focus-visible { outline: 2px solid var(--accent, #246bfd); outline-offset: 2px; border-radius: 3px; }
.reid-face-coverage dl { display: grid; grid-template-columns: max-content 1fr; gap: 6px 12px; margin: 4px 0 10px; }
.reid-face-coverage dt, .reid-face-coverage-note { color: var(--muted, #667085); }
.reid-face-coverage dd { margin: 0; }
</style>

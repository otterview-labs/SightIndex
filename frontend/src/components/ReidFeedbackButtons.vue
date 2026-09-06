<script setup lang="ts">
defineProps<{
  value: boolean | null;
  saving?: boolean;
}>();

const emit = defineEmits<{
  choose: [samePerson: boolean];
}>();
</script>

<template>
  <div class="reid-feedback" aria-label="人工确认匹配结果">
    <span class="reid-feedback-label">
      {{ value === null ? "人工确认" : value ? "已确认同一个人" : "已确认不是同一个人" }}
    </span>
    <div class="reid-feedback-actions">
      <button
        class="reid-feedback-button"
        :class="{ selected: value === true }"
        type="button"
        :disabled="saving"
        :aria-pressed="value === true"
        title="保存为同一个人，仅用于后续阈值校准"
        @click="emit('choose', true)"
      >
        同一个人
      </button>
      <button
        class="reid-feedback-button negative"
        :class="{ selected: value === false }"
        type="button"
        :disabled="saving"
        :aria-pressed="value === false"
        title="保存为不是同一个人，仅用于后续阈值校准"
        @click="emit('choose', false)"
      >
        不是同一个人
      </button>
    </div>
  </div>
</template>

<style scoped>
.reid-feedback {
  display: grid;
  width: 100%;
  gap: 5px;
}

.reid-feedback-label {
  color: var(--muted, #667085);
  font-size: 11px;
  font-weight: 600;
}

.reid-feedback-actions {
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
}

.reid-feedback-button {
  min-height: 28px;
  padding: 4px 9px;
  border: 1px solid var(--line, #dbe3ea);
  border-radius: 7px;
  background: var(--surface, #fff);
  color: var(--text, #182230);
  cursor: pointer;
  font: inherit;
  font-size: 11px;
  line-height: 1.2;
}

.reid-feedback-button:hover:not(:disabled),
.reid-feedback-button:focus-visible {
  border-color: color-mix(in srgb, var(--accent, #246bfd) 60%, var(--line, #dbe3ea));
  outline: none;
}

.reid-feedback-button.selected {
  border-color: color-mix(in srgb, #18864b 72%, transparent);
  background: color-mix(in srgb, #18864b 11%, var(--surface, #fff));
  color: #116538;
  font-weight: 700;
}

.reid-feedback-button.negative.selected {
  border-color: color-mix(in srgb, #c43f48 72%, transparent);
  background: color-mix(in srgb, #c43f48 10%, var(--surface, #fff));
  color: #9c2630;
}

.reid-feedback-button:disabled {
  cursor: wait;
  opacity: 0.58;
}
</style>

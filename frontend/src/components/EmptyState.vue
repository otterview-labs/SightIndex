<script setup lang="ts">
defineProps<{
  title?: string;
  /** Rendered immediately after the title, no separating space (a short inline remark). */
  inline?: string;
  /** Rendered as its own paragraph below the title (a longer hint or suggestion). */
  hint?: string;
}>();
</script>

<template>
  <div class="empty">
    <strong v-if="title">{{ title }}</strong><template v-if="inline">{{ inline }}</template>
    <p v-if="hint" class="muted-text">{{ hint }}</p>
    <slot />
  </div>
</template>

<script setup lang="ts">
/* A datetime-local input plus a Chinese placeholder.
 *
 * Chromium renders the native "mm/dd/yyyy, --:-- --" hint in the browser's own UI language, so on
 * an English-locale browser it ignores html[lang="zh-CN"] and shows English in a Chinese console.
 * The hint below is drawn over the input's own box while the field is empty, and steps aside on
 * focus so the real editor stays usable. */

withDefaults(
  defineProps<{
    label: string;
    name?: string;
    /** Replaces the native format hint. Say what the field is for, not the format. */
    placeholder?: string;
  }>(),
  { name: undefined, placeholder: "选择日期与时间" },
);

const model = defineModel<string>({ default: "" });
</script>

<template>
  <label class="datetime-field">
    {{ label }}
    <span class="datetime-field-control">
      <input v-model="model" :name="name" type="datetime-local" />
      <span v-if="!model" class="datetime-field-hint" aria-hidden="true">{{ placeholder }}</span>
    </span>
  </label>
</template>

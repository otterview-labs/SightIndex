<script setup lang="ts">
import { computed, ref, useTemplateRef, watch } from "vue";

const props = withDefaults(
  defineProps<{
    label: string;
    accept?: string;
    hint?: string;
    required?: boolean;
  }>(),
  { accept: "image/*", hint: "", required: false },
);

const file = defineModel<File | null>({ default: null });

const input = useTemplateRef<HTMLInputElement>("input");
const dragging = ref(false);
const previewUrl = ref("");

const isImage = computed(() => (props.accept ?? "").startsWith("image"));

const sizeText = computed(() => {
  if (!file.value) return "";
  const kb = file.value.size / 1024;
  return kb < 1024 ? `${kb.toFixed(0)} KB` : `${(kb / 1024).toFixed(1)} MB`;
});

function setFile(next: File | null) {
  file.value = next;
  if (previewUrl.value) URL.revokeObjectURL(previewUrl.value);
  previewUrl.value = next && isImage.value ? URL.createObjectURL(next) : "";
}

function onChange() {
  setFile(input.value?.files?.[0] ?? null);
}

function onDrop(event: DragEvent) {
  dragging.value = false;
  const dropped = event.dataTransfer?.files?.[0];
  if (!dropped) return;
  if (input.value) input.value.files = event.dataTransfer!.files;
  setFile(dropped);
}

function clear() {
  if (input.value) input.value.value = "";
  setFile(null);
}

defineExpose({ clear });

watch(file, (next) => {
  if (!next && input.value) input.value.value = "";
});
</script>

<template>
  <div class="file-field">
    <span class="file-field-label">{{ label }}</span>
    <div
      class="file-drop"
      :class="{ 'is-dragging': dragging, 'has-file': !!file }"
      @click="input?.click()"
      @dragover.prevent="dragging = true"
      @dragleave="dragging = false"
      @drop.prevent="onDrop"
    >
      <input
        ref="input"
        type="file"
        :accept="accept"
        :required="required && !file"
        @change="onChange"
      />

      <template v-if="file">
        <img v-if="previewUrl" class="file-thumb" :src="previewUrl" alt="" />
        <span v-else class="file-thumb file-thumb-generic" aria-hidden="true">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6">
            <path d="M14 3v5h5M14 3H6a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" />
          </svg>
        </span>
        <span class="file-meta">
          <strong>{{ file.name }}</strong>
          <small>{{ sizeText }}</small>
        </span>
        <button class="file-clear" type="button" aria-label="移除文件" @click.stop="clear">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8">
            <path d="M18 6 6 18M6 6l12 12" />
          </svg>
        </button>
      </template>

      <template v-else>
        <span class="file-icon" aria-hidden="true">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6">
            <path d="M12 16V4m0 0L7.5 8.5M12 4l4.5 4.5" />
            <path d="M4 15v3a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2v-3" />
          </svg>
        </span>
        <span class="file-meta">
          <strong>拖入文件或点击选择</strong>
          <small>{{ hint || (isImage ? "支持 JPG / PNG" : "选择本地文件") }}</small>
        </span>
      </template>
    </div>
  </div>
</template>

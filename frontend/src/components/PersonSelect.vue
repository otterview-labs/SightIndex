<script setup lang="ts">
import { computed } from "vue";

import type { Person } from "@/api/types";

const props = defineProps<{ persons: Person[] }>();
const model = defineModel<string | null>();

const options = computed(() =>
  props.persons.map((person) => ({
    id: person.id,
    label: [person.name, person.employee_no, person.department].filter(Boolean).join(" / "),
  })),
);
</script>

<template>
  <select v-model="model">
    <option v-if="!options.length" :value="null">暂无人员</option>
    <option v-for="option in options" :key="option.id" :value="option.id">
      {{ option.label }}
    </option>
  </select>
</template>

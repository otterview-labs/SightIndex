import { computed, ref } from "vue";

import { persons as personsApi } from "@/api/client";
import type { Person } from "@/api/types";

const persons = ref<Person[]>([]);
const activePersonId = ref<string | null>(null);

const activePerson = computed(() =>
  persons.value.find((person) => person.id === activePersonId.value),
);

const activePersonName = computed(() => activePerson.value?.name ?? "当前人员");

async function refresh(limit = 200) {
  persons.value = await personsApi.list(limit);
  if (!activePersonId.value && persons.value.length) {
    activePersonId.value = persons.value[0].id;
  }
}

export function usePersons() {
  return { persons, activePersonId, activePerson, activePersonName, refresh };
}

<script setup lang="ts">
import { RouterLink, RouterView } from "vue-router";

import { SECTIONS } from "@/router";
import { useToast } from "@/composables/useToast";

const { toasts } = useToast();
</script>

<template>
  <header class="topbar">
    <div class="brand-lockup">
      <h1>SightIndex 视频分析台</h1>
      <p>智能视频检索 · CAMERA CONSOLE</p>
    </div>

    <!-- Defined once for the whole console. Adding a section is one entry in SECTIONS. -->
    <nav class="top-nav">
      <RouterLink
        v-for="section in SECTIONS"
        :key="section.name"
        class="nav-link"
        exact-active-class="active"
        :to="section.path"
      >
        {{ section.label }}
      </RouterLink>
    </nav>

    <div class="top-actions">
      <!-- Views teleport their controls here, so page buttons can never shift the nav.
           display:contents keeps this wrapper out of the flex layout. -->
      <div id="page-actions" class="page-actions"></div>
      <a class="button ghost" href="/docs" target="_blank" rel="noreferrer">API</a>
    </div>
  </header>

  <RouterView />

  <div class="toast-host" aria-live="polite">
    <div v-for="item in toasts" :key="item.id" class="toast" :class="item.tone">
      {{ item.message }}
    </div>
  </div>
</template>

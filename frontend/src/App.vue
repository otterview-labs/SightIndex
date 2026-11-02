<script setup lang="ts">
import { RouterLink, RouterView } from "vue-router";

import { SECTIONS } from "@/router";
import { PUBLIC_DEMO } from "@/demo";
const visibleSections = PUBLIC_DEMO ? SECTIONS.filter(s => ["search", "observations"].includes(s.name)) : SECTIONS;
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
        v-for="section in visibleSections"
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
      <a v-if="!PUBLIC_DEMO" class="button ghost" href="/docs" target="_blank" rel="noreferrer">API</a>
    </div>
  </header>

  <section v-if="PUBLIC_DEMO" class="demo-intro" aria-label="在线演示说明">
    <div>
      <strong>用一张公开照片，试试标签检索。</strong>
      <p>无需登录。点击「深色」检索预处理示例中的已有标签，再到观察表核对裁剪和来源。</p>
      <details>
        <summary>示例来源与演示范围</summary>
        <p>示例为 NASA 公开照片（2011 年）。示例经 CPU HOG 检测并预先提取衣着标签，得到 2 个裁剪，原图有 4 人，存在漏检和裁剪重叠，不能用作准确率评估。未启用语义模型、人脸识别、ReID 或实时摄像头；上传和数据修改请自行部署。</p>
        <a href="https://github.com/otterview-labs/SightIndex/blob/main/samples/README.md" target="_blank" rel="noreferrer">素材说明与署名</a>
      </details>
    </div>
    <div class="demo-links">
      <a class="button ghost" href="/data/uploads/nasa-walk.jpg" target="_blank" rel="noreferrer">查看原图</a>
      <a class="button primary" href="https://github.com/otterview-labs/SightIndex" target="_blank" rel="noreferrer">GitHub / 自行部署</a>
    </div>
  </section>
  <RouterView />

  <div class="toast-host" aria-live="polite">
    <div v-for="item in toasts" :key="item.id" class="toast" :class="item.tone">
      {{ item.message }}
    </div>
  </div>
</template>

<style scoped>
.demo-intro {
  margin: 20px 28px 0;
  padding: 20px 24px;
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: 24px;
  background: #eef4fc;
  color: #253e60;
  border: 1px solid #c4d3e8;
  border-radius: 12px;
}
.demo-intro strong { font-size: 18px; }
.demo-intro p { margin: 8px 0; max-width: 72ch; line-height: 1.7; }
.demo-intro details { font-size: 13px; }
.demo-intro summary { cursor: pointer; padding: 4px 0; }
.demo-intro details a { color: #164d9c; text-decoration: underline; text-underline-offset: 3px; }
.demo-links { display: flex; flex-wrap: wrap; gap: 8px; flex-shrink: 0; }
.demo-intro :focus-visible { outline: 2px solid #164d9c; outline-offset: 3px; }
@media (max-width: 760px) {
  .demo-intro { margin: 12px 12px 0; padding: 16px; flex-direction: column; gap: 12px; }
  .demo-links a { min-height: 44px; }
}
</style>

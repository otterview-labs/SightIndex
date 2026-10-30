<script setup lang="ts">
import { nextTick, ref, useTemplateRef } from "vue";

import { chat as chatApi, images } from "@/api/client";
import type { ChatResponse, ImageRead } from "@/api/types";
import ChatResults, { type ChatResultItem } from "@/components/ChatResults.vue";
import FileField from "@/components/FileField.vue";
import { useToast } from "@/composables/useToast";

interface ChatMessage {
  id: number;
  role: "system" | "user" | "assistant";
  text: string;
  results?: ChatResultItem[];
}

const PROMPTS = [
  "今天有多少人？",
  "人脸库有多少人？",
  "最近识别到谁？",
  "找黑色上衣并且背包的人",
  "这张图片里的人脸是谁？",
  "现在有几个视频流在运行？",
  "张三今天在哪？",
];

const { showError } = useToast();

const logEl = useTemplateRef<HTMLDivElement>("logEl");
const messages = ref<ChatMessage[]>([
  {
    id: 0,
    role: "system",
    text: "当前支持统计问答、人员查询和结构化标签检索；人体 ReID 与人脸向量用于身份关联。",
  },
]);
const draft = ref("");
const sending = ref(false);
const uploading = ref(false);
const uploadResult = ref("");
const uploadFile = ref<File | null>(null);
// Set only when the last upload produced zero crops, so the retry button cannot re-run a
// successful image (re-detection is opt-in and only ever acts on a zero-crop result).
const zeroCropImageId = ref<string | null>(null);
const messageInput = useTemplateRef<HTMLInputElement>("messageInput");

const isPrimer = ref(true);
const lastImageId = ref<string | null>(null);

let nextMessageId = 1;

async function append(role: ChatMessage["role"], text: string, results?: ChatResultItem[]) {
  if (role !== "system") isPrimer.value = false;
  messages.value.push({ id: nextMessageId++, role, text, results });
  await nextTick();
  if (logEl.value) logEl.value.scrollTop = logEl.value.scrollHeight;
}

async function send() {
  const message = draft.value.trim();
  if (!message || sending.value) return;
  await append("user", message);
  draft.value = "";
  sending.value = true;
  try {
    const response: ChatResponse = await chatApi.ask({
      message,
      context: { last_image_id: lastImageId.value },
    });
    const data = response.data as { items?: ChatResultItem[] } | null | undefined;
    await append("assistant", response.answer, Array.isArray(data?.items) ? data.items : []);
  } catch (error) {
    await append("assistant", "请求失败，请检查后端服务。");
    showError(error);
  } finally {
    sending.value = false;
  }
}

function explainZeroCrops(image: ImageRead): string {
  const reason = image.crop_reason;
  if (!reason) return "未生成裁剪候选。";
  if (reason.reason === "no_raw_detections") {
    return "图中未检出人体，所以没有裁剪候选。";
  }
  if (reason.reason === "quality_filtered") {
    const rejected = reason.rejected ?? {};
    const parts: string[] = [];
    if (rejected.touches_frame_edge) {
      parts.push(`${rejected.touches_frame_edge} 人的人体框贴到画面边缘`);
    }
    if (rejected.below_min_confidence) {
      parts.push(`${rejected.below_min_confidence} 人置信度低于阈值`);
    }
    if (rejected.too_small) parts.push(`${rejected.too_small} 人框小于最小尺寸`);
    const total = reason.raw_total ?? 0;
    return parts.length
      ? `检出 ${total} 人，但均未通过裁剪规则（${parts.join("；")}）。`
      : `检出 ${total} 人，但均未通过裁剪规则。`;
  }
  if (reason.reason === "source_missing") {
    return "原图文件已不存在，无法生成裁剪。";
  }
  if (reason.reason === "already_processed") {
    return "这张图此前处理过且没有裁剪候选，可以重新检测。";
  }
  return "未生成裁剪候选。";
}

async function upload() {
  const file = uploadFile.value;
  if (!file || uploading.value) return;
  await append("user", `上传图片：${file.name}`);
  uploading.value = true;
  uploadResult.value = "上传中...";
  try {
    const body = new FormData();
    body.set("file", file);
    const image = await images.upload(body);
    lastImageId.value = image.id;
    uploadResult.value = "检测中...";
    const crops = await images.process(image.id);
    const original: ChatResultItem = {
      image_id: image.id,
      image_url: image.thumbnail_url || image.image_url,
      title: "原图",
      created_at: image.created_at,
    };
    const cropItems: ChatResultItem[] = crops.map((crop) => ({
      crop_id: crop.id,
      image_id: crop.image_id,
      crop_url: crop.crop_url,
      title: "裁剪",
      captured_at: crop.captured_at || crop.created_at,
    }));
    if (crops.length === 0) {
      const refreshed = await images.get(image.id);
      const explanation = explainZeroCrops(refreshed);
      zeroCropImageId.value = image.id;
      await append("assistant", `图片已上传并处理，但${explanation}`, [original]);
      uploadResult.value = `0 个裁剪候选 · ${explanation}`;
      uploadFile.value = null;
      return;
    }
    zeroCropImageId.value = null;
    await append(
      "assistant",
      `图片已上传并处理，生成 ${crops.length} 个裁剪候选。可以继续问“找和刚才图片相似的人”或“最近识别到谁？”。`,
      [original, ...cropItems],
    );
    uploadResult.value = `已入库 ${crops.length} 个裁剪候选`;
    uploadFile.value = null;
  } catch (error) {
    await append("assistant", "图片上传或处理失败，请检查图片格式和后端服务。");
    uploadResult.value = "处理失败";
    showError(error);
  } finally {
    uploading.value = false;
  }
}

async function reprocessZeroCropImage() {
  const id = zeroCropImageId.value;
  if (!id || uploading.value) return;
  uploading.value = true;
  uploadResult.value = "重新检测中...";
  try {
    const crops = await images.process(id, true);
    if (crops.length > 0) {
      zeroCropImageId.value = null;
      await append(
        "assistant",
        `重新检测完成，生成 ${crops.length} 个裁剪候选。`,
        crops.map((crop) => ({
          crop_id: crop.id,
          image_id: crop.image_id,
          crop_url: crop.crop_url,
          title: "裁剪",
          captured_at: crop.captured_at || crop.created_at,
        })),
      );
      uploadResult.value = `已入库 ${crops.length} 个裁剪候选`;
    } else {
      const refreshed = await images.get(id);
      uploadResult.value = `重新检测后仍为 0 个裁剪候选 · ${explainZeroCrops(refreshed)}`;
    }
  } catch (error) {
    uploadResult.value = "重新检测失败";
    showError(error);
  } finally {
    uploading.value = false;
  }
}

function usePrompt(prompt: string) {
  draft.value = prompt;
  messageInput.value?.focus();
}
</script>

<template>
  <main class="page-workspace page-shell chat-workspace">
    <section class="panel chat-main" aria-labelledby="chatTitle">
      <div class="section-head">
        <div>
          <h2 id="chatTitle">对话记录</h2>
        </div>
      </div>

      <div ref="logEl" class="chat-log chat-log-large" :class="{ 'is-primer': isPrimer }">
        <div
          v-for="message in messages"
          :key="message.id"
          class="message"
          :class="[message.role, { 'has-results': message.results?.length }]"
        >
          <div class="message-text">{{ message.text }}</div>
          <ChatResults v-if="message.results?.length" :items="message.results" />
        </div>
      </div>

      <form class="chat-form-large" @submit.prevent="send">
        <input
          ref="messageInput"
          v-model="draft"
          name="message"
          placeholder="找和刚才图片相似的人；张三今天在哪？"
          required
        />
        <button class="button primary" type="submit" :disabled="sending">
          {{ sending ? "发送中" : "发送" }}
        </button>
      </form>
    </section>

    <aside class="panel chat-context">
      <form class="form-grid chat-upload-form" @submit.prevent="upload">
        <div class="section-head compact">
          <div>
            <h2>图片入库</h2>
            <p>检测人形并生成识别事件</p>
          </div>
        </div>
        <FileField v-model="uploadFile" label="图片" accept="image/*" required />
        <button class="button primary wide" type="submit" :disabled="uploading">
          {{ uploading ? "处理中" : "上传并处理" }}
        </button>
        <button
          v-if="zeroCropImageId"
          class="button wide"
          type="button"
          :disabled="uploading"
          @click="reprocessZeroCropImage"
        >
          重新检测
        </button>
        <div class="upload-result">{{ uploadResult }}</div>
      </form>

      <div class="section-head compact">
        <div>
          <h2>快捷问题</h2>
        </div>
      </div>
      <div class="prompt-list">
        <button v-for="prompt in PROMPTS" :key="prompt" type="button" @click="usePrompt(prompt)">
          {{ prompt === "找和刚才图片相似的人" ? "找相似图片" : prompt }}
        </button>
      </div>
    </aside>
  </main>
</template>

<style scoped>
.chat-log-large {
  scroll-behavior: smooth;
}
</style>

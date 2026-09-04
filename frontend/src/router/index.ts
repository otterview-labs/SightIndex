import { createRouter, createWebHistory, type RouteRecordRaw } from "vue-router";

export const SECTIONS = [
  { path: "/", name: "monitor", label: "监控", title: "视频分析台" },
  { path: "/search", name: "search", label: "检索", title: "检索" },
  { path: "/observations", name: "observations", label: "观察表", title: "观察表" },
  { path: "/faces", name: "faces", label: "人脸库", title: "人脸库" },
  { path: "/reid", name: "reid", label: "找人", title: "以图找人" },
  { path: "/chat-ui", name: "chat", label: "Chat", title: "Chat" },
] as const;

const routes: RouteRecordRaw[] = [
  { path: "/", name: "monitor", component: () => import("@/views/MonitorView.vue") },
  { path: "/search", name: "search", component: () => import("@/views/SearchView.vue") },
  {
    path: "/observations",
    name: "observations",
    component: () => import("@/views/ObservationsView.vue"),
  },
  { path: "/faces", name: "faces", component: () => import("@/views/FacesView.vue") },
  { path: "/reid", name: "reid", component: () => import("@/views/ReidView.vue") },
  { path: "/chat-ui", name: "chat", component: () => import("@/views/ChatView.vue") },
  { path: "/chat", redirect: "/chat-ui" },
  { path: "/:pathMatch(.*)*", redirect: "/" },
];

export const router = createRouter({
  history: createWebHistory(),
  routes,
  scrollBehavior: () => ({ top: 0 }),
});

router.afterEach((to) => {
  const section = SECTIONS.find((item) => item.name === to.name);
  document.title = section ? `SightIndex ${section.title}` : "SightIndex 视频分析台";
});

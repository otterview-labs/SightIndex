import { ref } from "vue";

export type ToastTone = "info" | "error";

export interface Toast {
  id: number;
  message: string;
  tone: ToastTone;
}

const VISIBLE_MS = 2200;

const toasts = ref<Toast[]>([]);
let nextId = 0;

function push(message: string, tone: ToastTone) {
  const id = ++nextId;
  toasts.value.push({ id, message, tone });
  window.setTimeout(() => {
    toasts.value = toasts.value.filter((item) => item.id !== id);
  }, VISIBLE_MS);
}

export function useToast() {
  return {
    toasts,
    toast: (message: string) => push(message, "info"),
    showError: (error: unknown) => {
      console.error(error);
      const detail = error instanceof Error ? error.message : "";
      push(detail ? "操作失败，查看控制台" : "操作失败", "error");
    },
  };
}

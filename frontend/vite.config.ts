import { fileURLToPath, URL } from "node:url";

import vue from "@vitejs/plugin-vue";
import { defineConfig } from "vite";

// Dev only: FastAPI serves dist/ itself in production.
const BACKEND = process.env.SIGHTINDEX_API ?? "http://127.0.0.1:8000";
const BACKEND_PATHS = ["/api", "/v1", "/data", "/health", "/docs", "/openapi.json"];

export default defineConfig({
  plugins: [vue()],
  resolve: {
    alias: { "@": fileURLToPath(new URL("./src", import.meta.url)) },
  },
  build: {
    outDir: "dist",
    emptyOutDir: true,
    sourcemap: true,
  },
  server: {
    port: 5173,
    strictPort: true,
    proxy: Object.fromEntries(
      BACKEND_PATHS.map((path) => [path, { target: BACKEND, changeOrigin: true }]),
    ),
  },
});

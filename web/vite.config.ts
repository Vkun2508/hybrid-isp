import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Dev: the FastAPI server runs on :7860; the Vite dev server proxies /api to it.
// Build: static files in web/dist, served by FastAPI (server/main.py) in production.
export default defineConfig({
  plugins: [react()],
  server: { port: 5173, proxy: { "/api": "http://localhost:7860" } },
  build: { outDir: "dist", assetsDir: "assets", sourcemap: false, target: "es2020" },
});

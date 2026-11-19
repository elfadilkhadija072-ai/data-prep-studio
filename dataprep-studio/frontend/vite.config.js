import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// En dev (npm run dev → :5173) les appels /api sont relayés vers FastAPI (:8501).
export default defineConfig({
  plugins: [react()],
  server: { port: 5173, proxy: { '/api': 'http://127.0.0.1:8501' } },
  build: { chunkSizeWarningLimit: 1200 },
})

import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'

export default defineConfig({
  plugins: [vue()],
  server: {
    host: '127.0.0.1',
    port: 5173,
    proxy: {
      '/api/query': {
        target: 'http://127.0.0.1:9080',
        changeOrigin: true,
        rewrite: () => '/api/ai/query',
      },
    },
  },
})

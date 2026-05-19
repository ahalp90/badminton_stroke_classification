import { defineConfig, loadEnv } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), '')
  const apiTarget = env.VITE_API_TARGET || 'http://localhost:24082'
  const allowedHosts = (env.VITE_ALLOWED_HOSTS || 'localhost').split(',').map(s => s.trim())

  return {
    plugins: [react()],
    server: {
      host: true,
      allowedHosts,
      proxy: {
        '/api': {
          target: apiTarget,
          changeOrigin: true,
        },
      },
    },
  }
})

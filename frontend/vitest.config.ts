import {fileURLToPath, URL} from 'node:url'
import {defineConfig} from 'vitest/config'

// 单测只覆盖纯 TS 状态层（Pinia store），不需要浏览器环境；
// 组件级测试若要挂载 DOM，再单独把 environment 改成 jsdom 并排除对应文件。
export default defineConfig({
  resolve: {
    alias: {'@': fileURLToPath(new URL('./src', import.meta.url))},
  },
  test: {
    environment: 'node',
    include: ['src/**/*.test.ts'],
    clearMocks: true,
    restoreMocks: true,
  },
})

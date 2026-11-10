/** chat store 单测：SSE 归并与切换会话竞态。
 *
 * 用 vite 的 `@` 别名 + 真实 Pinia（不挂组件），SSE 与 HTTP 全部 mock，
 * 不依赖后端、不产生网络请求。
 */
import {beforeEach, describe, expect, it, vi} from 'vitest'
import {createPinia, setActivePinia} from 'pinia'

vi.mock('@/api/sessions', () => ({
  listSessions: vi.fn(),
  createSession: vi.fn(),
  listMessages: vi.fn(),
  updateSession: vi.fn(),
  deleteSession: vi.fn(),
}))

vi.mock('@/api/chat', () => ({
  streamChat: vi.fn(),
  fetchSuggestions: vi.fn(),
}))

vi.mock('@/api/http', () => ({
  errorText: (err: unknown) => (err instanceof Error ? err.message : String(err)),
}))

import * as sessionApi from '@/api/sessions'
import type {ChatMessage, Session} from '@/api/sessions'
import {fetchSuggestions, streamChat} from '@/api/chat'
import type {SSEEvent} from '@/api/chat'
import {useChatStore} from '@/stores/chat'

const sessionA: Session = {
  id: 'A',
  title: '会话 A',
  kb_ids: [],
  model: 'deepseek-v4.1-flash',
  temperature: 0.2,
  top_p: 0.9,
  max_tokens: 2048,
  history_rounds: 5,
  message_count: 0,
  created_at: null,
  updated_at: null,
}

const sessionB: Session = {...sessionA, id: 'B', title: '会话 B'}

const metaBase = {
  message_id: 'msg-1',
  intent: '法条定位',
  route: 'dense',
  rewrite_type: 'none',
  complexity: 'simple',
  cached: false,
  degraded: false,
}

const source = {
  law: '中华人民共和国劳动合同法',
  chapter_path: '第四章 劳动合同的解除和终止',
  article_no: '第四十七条',
  score: 0.91,
  snippet: '经济补偿按劳动者在本单位工作的年限…',
  kind: 'law' as const,
}

const historyB: ChatMessage = {
  id: 7,
  seq: 1,
  role: 'user',
  content: 'B 的历史提问',
  sources: null,
  intent: null,
  rewrite_type: null,
  route: null,
  cached: false,
  latency_ms: null,
  ttft_ms: null,
  token_usage: null,
  error: null,
  created_at: null,
}

function event<T extends SSEEvent['event']>(name: T, data: Extract<SSEEvent, {event: T}>['data']): SSEEvent {
  return {event: name, data} as SSEEvent
}

beforeEach(() => {
  setActivePinia(createPinia())
  vi.clearAllMocks()
  vi.mocked(sessionApi.listSessions).mockResolvedValue([sessionA, sessionB])
  vi.mocked(sessionApi.listMessages).mockResolvedValue({count: 0, items: []})
  vi.mocked(sessionApi.createSession).mockResolvedValue(sessionA)
  vi.mocked(fetchSuggestions).mockResolvedValue([])
})

describe('SSE 归并', () => {
  it('二次 meta 合并而非覆盖，stage 不被 meta 拉回，delta 累积为一条消息', async () => {
    const store = useChatStore()
    await store.loadSessions()
    expect(store.currentId).toBe('A')

    let stageAfterSecondMeta = ''
    let mergedMetaCached: boolean | undefined
    let mergedMetaCitation: unknown = null

    vi.mocked(streamChat).mockImplementation(async (_payload, onEvent) => {
      onEvent(event('meta', {...metaBase}))
      onEvent(event('stage', {stage: 'retrieval'}))
      onEvent(event('stage', {stage: 'rerank'}))
      onEvent(event('stage', {stage: 'generate'}))
      onEvent(event('delta', {text: '根据《劳动合同法》'}))
      onEvent(event('sources', {sources: [source]}))
      onEvent(event('delta', {text: '第四十七条，应支付经济补偿。'}))
      // 生成结束后后端会再补一条带 citation_check 的 meta
      onEvent(
        event('meta', {
          ...metaBase,
          citation_check: {
            ok: true,
            cited: ['第四十七条'],
            verified: ['第四十七条'],
            unverified: [],
            sources_uncited: [],
            coverage: 1,
            source_coverage: 1,
          },
        }),
      )
      stageAfterSecondMeta = store.draft?.stage ?? ''
      mergedMetaCached = store.draft?.meta?.cached
      mergedMetaCitation = store.draft?.meta?.citation_check ?? null
      onEvent(
        event('done', {
          ttft_ms: 120,
          latency_ms: 800,
          token_usage: {total_tokens: 42},
          sources_count: 1,
          cached: false,
        }),
      )
    })

    await store.send('公司违法解除劳动合同怎么赔偿？')

    expect(stageAfterSecondMeta).toBe('generate')
    expect(mergedMetaCached).toBe(false)
    expect(mergedMetaCitation).toEqual(expect.objectContaining({ok: true}))

    expect(store.draft).toBeNull()
    expect(store.messages).toHaveLength(2)
    const assistant = store.messages[1]
    expect(assistant.role).toBe('assistant')
    expect(assistant.content).toBe('根据《劳动合同法》第四十七条，应支付经济补偿。')
    expect(assistant.sources).toHaveLength(1)
    expect(assistant.sources?.[0].article_no).toBe('第四十七条')
    expect(assistant.intent).toBe('法条定位')
    expect(assistant.route).toBe('dense')
    expect(assistant.ttft_ms).toBe(120)
    expect(assistant.latency_ms).toBe(800)
    expect(assistant.error).toBeNull()
  })

  it('error 帧保留已生成内容并标记错误', async () => {
    const store = useChatStore()
    await store.loadSessions()

    vi.mocked(streamChat).mockImplementation(async (_payload, onEvent) => {
      onEvent(event('delta', {text: '已生成一半'}))
      onEvent(event('error', {code: 'llm_error', message: '模型超时', retriable: true}))
    })

    await store.send('测试问题')

    const assistant = store.messages[1]
    expect(assistant.content).toBe('已生成一半')
    expect(assistant.error).toBe('模型超时')
  })
})

describe('切换会话竞态', () => {
  it('切走后旧流的迟到事件不重建草稿，也不写入新会话', async () => {
    const store = useChatStore()
    await store.loadSessions()
    expect(store.currentId).toBe('A')

    let emit!: (evt: SSEEvent) => void
    let finish!: () => void
    vi.mocked(streamChat).mockImplementation((_payload, onEvent) => {
      emit = onEvent
      return new Promise<void>((resolve) => {
        finish = resolve
      })
    })
    vi.mocked(sessionApi.listMessages).mockImplementation(async (id: string) =>
      id === 'B' ? {count: 1, items: [historyB]} : {count: 0, items: []},
    )

    const sending = store.send('旧会话的问题')
    emit(event('delta', {text: '旧会话已生成一半'}))
    emit(event('stage', {stage: 'generate'}))
    expect(store.draft?.content).toBe('旧会话已生成一半')

    await store.selectSession('B')
    expect(store.currentId).toBe('B')
    expect(store.draft).toBeNull()

    // 模拟 fetch 缓冲里的迟到帧：当前实现必须整帧丢弃
    emit(event('delta', {text: '切换后才到的事件'}))
    emit(event('stage', {stage: 'generate'}))
    emit(event('done', {ttft_ms: 1, latency_ms: 2, token_usage: {}, sources_count: 0, cached: false}))
    expect(store.draft).toBeNull()

    finish()
    await sending

    expect(store.currentId).toBe('B')
    expect(store.messages.map((m) => m.content)).toEqual(['B 的历史提问'])
    expect(store.draft).toBeNull()
  })

  it('快速切换时先发出的消息请求不得覆盖当前会话', async () => {
    const store = useChatStore()
    const pending: Record<string, (page: {count: number; items: ChatMessage[]}) => void> = {}
    vi.mocked(sessionApi.listMessages).mockImplementation(
      (id: string) =>
        new Promise<{count: number; items: ChatMessage[]}>((resolve) => {
          pending[id] = resolve
        }),
    )

    const toB = store.selectSession('B')
    const toA = store.selectSession('A')
    // A 先返回并落地，B 的旧响应后到，必须被丢弃
    pending.A({count: 1, items: [{...historyB, content: 'A 的历史提问'}]})
    await toA
    pending.B({count: 1, items: [{...historyB, content: 'B 的历史提问'}]})
    await toB

    expect(store.currentId).toBe('A')
    expect(store.messages.map((m) => m.content)).toEqual(['A 的历史提问'])
    expect(store.loadingMessages).toBe(false)
  })
})

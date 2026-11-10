/** 对话状态：会话列表 + 当前会话 + 消息流 + SSE 事件归并。
 *
 * 设计取舍：
 * * 会话列表与消息都放 store —— 去知识库页再回来不丢上下文；
 * * SSE 事件在这里归并为"一条 assistant 消息"，视图只负责渲染；
 * * 参数（模型/温度/topP/max_tokens/轮数）与会话绑定，切换会话自动回填。
 */
import {computed, ref} from 'vue'
import {defineStore} from 'pinia'

import * as sessionApi from '@/api/sessions'
import type {ChatMessage, Session, SessionParams, SourceRef} from '@/api/sessions'
import {fetchSuggestions} from '@/api/chat'
import type {ChatMeta, SSEEvent} from '@/api/chat'
import {errorText} from '@/api/http'
import {useSSE} from '@/composables/useSSE'

export type ChatStage = 'intent' | 'retrieval' | 'rerank' | 'generate' | ''

/** 界面用消息：在落库字段之外挂「追问预测」结果。
 *
 * 追问只存在于内存（不写库）：它是一次性的辅助 UI，刷新后重建成本极低，
 * 存库反而要加字段、加迁移，收益不匹配。 */
export interface UiMessage extends ChatMessage {
  /** 预测出的 3 条追问；为空/缺省时不渲染推荐区 */
  suggestions?: string[]
  /** 预测请求进行中（用于「猜你想问」占位） */
  suggesting?: boolean
}

interface AssistantDraft {
  content: string
  sources: SourceRef[]
  meta: ChatMeta | null
  stage: ChatStage
  error: string | null
  ttft_ms: number
  latency_ms: number
}

function emptyDraft(): AssistantDraft {
  return {content: '', sources: [], meta: null, stage: '', error: null, ttft_ms: 0, latency_ms: 0}
}

export const useChatStore = defineStore('chat', () => {
  const sessions = ref<Session[]>([])
  const currentId = ref<string>('')
  const messages = ref<UiMessage[]>([])
  const draft = ref<AssistantDraft | null>(null)
  const loadingSessions = ref(false)
  const loadingMessages = ref(false)
  const lastQuery = ref('')

  const sse = useSSE()
  const streaming = computed(() => sse.streaming.value)

  const currentSession = computed(() => sessions.value.find((s) => s.id === currentId.value) || null)
  const kbIds = computed(() => currentSession.value?.kb_ids || [])

  async function loadSessions() {
    loadingSessions.value = true
    try {
      sessions.value = await sessionApi.listSessions()
      if (!currentId.value && sessions.value.length) {
        await selectSession(sessions.value[0].id)
      }
    } finally {
      loadingSessions.value = false
    }
  }

  async function createSession(params: SessionParams = {}): Promise<Session> {
    const session = await sessionApi.createSession(params)
    sessions.value.unshift(session)
    currentId.value = session.id
    messages.value = []
    return session
  }

  async function selectSession(id: string) {
    if (sse.streaming.value) sse.abort()
    currentId.value = id
    draft.value = null
    loadingMessages.value = true
    try {
      const page = await sessionApi.listMessages(id)
      // 快速连续切换时，先发出的请求可能后返回；只让当前会话的结果落地
      if (currentId.value === id) messages.value = page.items
    } catch (err) {
      if (currentId.value === id) messages.value = []
      throw err
    } finally {
      if (currentId.value === id) loadingMessages.value = false
    }
  }

  async function renameSession(id: string, title: string) {
    const updated = await sessionApi.updateSession(id, {title})
    const idx = sessions.value.findIndex((s) => s.id === id)
    if (idx >= 0) sessions.value[idx] = updated
  }

  async function updateParams(id: string, params: SessionParams) {
    const updated = await sessionApi.updateSession(id, params)
    const idx = sessions.value.findIndex((s) => s.id === id)
    if (idx >= 0) sessions.value[idx] = updated
    return updated
  }

  /** 乐观更新：先移除，失败再回滚（§4.2 P3 删除二次确认 → 乐观更新）。 */
  async function removeSession(id: string) {
    const snapshot = [...sessions.value]
    sessions.value = sessions.value.filter((s) => s.id !== id)
    try {
      await sessionApi.deleteSession(id)
      if (currentId.value === id) {
        currentId.value = sessions.value[0]?.id || ''
        messages.value = []
        if (currentId.value) await selectSession(currentId.value)
      }
    } catch (err) {
      sessions.value = snapshot
      throw err
    }
  }

  function appendUser(text: string) {
    messages.value.push({
      id: Date.now(),
      seq: messages.value.length + 1,
      role: 'user',
      content: text,
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
    })
  }

  /** 归并一帧 SSE。
   *
   * ``sid`` 是发起本次流时的会话 id：用户切走后旧流可能还有缓冲事件，
   * 必须整帧丢弃，既不能重建草稿（会在新会话里凭空出现回答），也不能
   * 把旧会话的来源/正文写进当前界面。draft 为 null 同样意味着流已被
   * 切换/结束清理，不允许用事件把它恢复。
   */
  function handleEvent(evt: SSEEvent, sid: string) {
    if (currentId.value !== sid || !draft.value) return
    const d = draft.value
    switch (evt.event) {
      case 'meta':
        // 生成结束后还会再发一条带 citation_check 的 meta，这里必须合并而非覆盖；
        // 也不能把 stage 拉回 intent（那会把已完成的生成进度条打回去）。
        d.meta = d.meta ? {...d.meta, ...evt.data} : evt.data
        if (!d.stage) d.stage = 'intent'
        break
      case 'stage':
        d.stage = evt.data.stage
        break
      case 'sources':
        d.sources = evt.data.sources
        break
      case 'delta':
        d.content += evt.data.text
        break
      case 'done':
        d.ttft_ms = evt.data.ttft_ms
        d.latency_ms = evt.data.latency_ms
        break
      case 'error':
        d.error = evt.data.message
        break
    }
  }

  /** 追问预测：回答落屏后异步取 3 条推荐问题，失败静默（只是锦上添花）。
   *
   * 三个细节：
   * * 取回后必须校验会话没被切走、消息还在数组里，否则会把 A 会话的推荐
   *   写到 B 会话的界面上（用户切得快时确实会发生）；
   * * 存的是 **reactive 数组里的那个代理对象**，不是 push 进去的原始对象——
   *   改原始对象不会触发视图更新；
   * * 只对本次新生成的回答请求，历史消息不补算（切回旧会话不重复烧 token）。
   */
  async function loadSuggestions(message: UiMessage, query: string) {
    const answer = (message.content || '').trim()
    if (!answer) return
    const sid = currentId.value
    const lawNames = Array.from(
      new Set((message.sources || []).filter((s) => s.kind === 'law').map((s) => s.law)),
    ).slice(0, 3)

    message.suggesting = true
    try {
      const items = await fetchSuggestions({query, answer, law_names: lawNames})
      if (currentId.value !== sid) return
      if (!messages.value.some((m) => m.id === message.id)) return
      message.suggestions = items
    } catch {
      /* 静默失败：推荐位干脆不出现，用户不会察觉 */
    } finally {
      message.suggesting = false
    }
  }

  /** 发送一轮问答：立即插入用户气泡 + 流式草稿，结束转正式消息。 */
  async function send(query: string, kb_override?: string[] | null) {
    const text = query.trim()
    if (!text || sse.streaming.value) return
    lastQuery.value = text

    if (!currentId.value) {
      await createSession()
    }
    // 本次流归属的会话：后续事件与落库都以它为准，切走即作废
    const sid = currentId.value
    appendUser(text)
    draft.value = emptyDraft()

    const history = messages.value
      .filter((m) => m.role !== 'system')
      .slice(0, -1)
      .map((m) => ({role: m.role, content: m.content}))

    try {
      await sse.start(
        {
          query: text,
          history,
          kb_ids: kb_override ?? (kbIds.value.length ? kbIds.value : null),
          session_id: sid || null,
        },
        (evt) => handleEvent(evt, sid),
      )
    } catch (err) {
      if (sse.aborted.value) {
        if (draft.value) draft.value.error = draft.value.error || '已中断（已生成内容已保存）'
      } else {
        if (draft.value) draft.value.error = errorText(err)
      }
    } finally {
      // 切走会话后旧流的残留草稿不得落到新会话的消息列表里
      const d = currentId.value === sid ? draft.value : null
      let created: UiMessage | null = null
      if (d && (d.content || d.error)) {
        messages.value.push({
          id: Date.now() + 1,
          seq: messages.value.length + 1,
          role: 'assistant',
          content: d.content,
          sources: d.sources,
          intent: d.meta?.intent ?? null,
          rewrite_type: d.meta?.rewrite_type ?? null,
          route: d.meta?.route ?? null,
          cached: d.meta?.cached ?? false,
          latency_ms: d.latency_ms || null,
          ttft_ms: d.ttft_ms || null,
          token_usage: null,
          error: d.error,
          created_at: null,
        })
        // 取回数组里的响应式代理：直接改 push 进去的原始对象不会触发视图更新
        created = messages.value[messages.value.length - 1] as UiMessage
      }
      draft.value = null
      if (created && !created.error) void loadSuggestions(created, text)
      // 会话的 updated_at / message_count 变了，静默刷新列表顺序
      sessionApi.listSessions().then((list) => {
        sessions.value = list
      }).catch(() => void 0)
    }
  }

  /** 重发上一轮：摘掉带错误标记的助手气泡，再把同一个问题发一次。
   *
   * 收敛到 store 的原因：视图里直接 `messages.pop()` 会让"消息数组的增删"
   * 散落在 store 与组件两处，后续任何挂在 push 上的副作用（滚动定位、追问
   * 预测、会话计数刷新）都容易漏掉一条路径。视图只该说"重试"，不该知道
   * 消息怎么增删。
   */
  async function retryLast(kb_override?: string[] | null) {
    if (sse.streaming.value) return
    const last = messages.value[messages.value.length - 1]
    if (last && last.role === 'assistant' && last.error) messages.value.pop()
    const q = lastQuery.value
    if (!q) return
    await send(q, kb_override)
  }

  function stop() {
    sse.abort()
  }

  return {
    sessions,
    currentId,
    currentSession,
    kbIds,
    messages,
    draft,
    loadingSessions,
    loadingMessages,
    streaming,
    lastQuery,
    loadSessions,
    createSession,
    selectSession,
    renameSession,
    updateParams,
    removeSession,
    send,
    retryLast,
    stop,
  }
})

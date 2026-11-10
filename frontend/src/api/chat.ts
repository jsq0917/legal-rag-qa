/** SSE 流式问答（P3 核心）。
 *
 * 为什么不用 EventSource：需要带 Authorization 头、POST body、可中断
 * （AbortController），EventSource 三者都不支持（设计方案 §4.3）。
 * 这里用 fetch + ReadableStream 手动按 "\n\n" 切帧并解析 event/data。
 */
import {TOKEN_KEY, ApiError, ApiErrorBody, http} from './http'
import type {SourceRef} from './sessions'

/** 生成后引用校验结果（后端 meta.citation_check 原样透出） */
export interface CitationCheck {
  ok: boolean
  cited: string[]
  verified: string[]
  unverified: string[]
  sources_uncited: string[]
  coverage: number | null
  source_coverage: number | null
}

export interface ChatMeta {
  message_id: string
  intent: string
  route: string
  rewrite_type: string
  complexity: string
  cached: boolean
  degraded: boolean | string
  /** 仅在生成完成后的第二条 meta 上出现 */
  citation_check?: CitationCheck
}

export interface ChatDone {
  ttft_ms: number
  latency_ms: number
  token_usage: Record<string, number>
  sources_count: number
  cached: boolean
}

export interface ChatError {
  code: string
  message: string
  retriable: boolean
}

export type SSEEvent =
  | {event: 'meta'; data: ChatMeta}
  | {event: 'stage'; data: {stage: 'intent' | 'retrieval' | 'rerank' | 'generate'}}
  | {event: 'sources'; data: {sources: SourceRef[]}}
  | {event: 'delta'; data: {text: string}}
  | {event: 'done'; data: ChatDone}
  | {event: 'error'; data: ChatError}

export interface StreamPayload {
  query: string
  history?: Array<{role: string; content: string}>
  kb_ids?: string[] | null
  session_id?: string | null
}

export interface SuggestionsPayload {
  query: string
  answer: string
  law_names?: string[]
}

/** 追问预测：一次问答结束后，让模型猜用户接下来最可能问的 3 个问题。
 *
 * 刻意**不在 SSE 流里**做：预测慢几秒不能拖住 done 事件（否则输入框一直被锁着），
 * 且失败只返回空数组，前端不展示推荐区即可，属于纯加分项。
 */
export async function fetchSuggestions(payload: SuggestionsPayload): Promise<string[]> {
  const {data} = await http.post<{suggestions: string[]}>('/chat/suggestions', payload)
  return Array.isArray(data?.suggestions) ? data.suggestions : []
}

function parseFrame(raw: string): SSEEvent | null {
  let event = ''
  const dataLines: string[] = []
  for (const line of raw.split('\n')) {
    if (line.startsWith('event: ')) event = line.slice(7).trim()
    else if (line.startsWith('data: ')) dataLines.push(line.slice(6))
  }
  if (!event) return null
  try {
    return {event, data: JSON.parse(dataLines.join('\n'))} as SSEEvent
  } catch {
    return null
  }
}

export async function streamChat(
  payload: StreamPayload,
  onEvent: (evt: SSEEvent) => void,
  signal?: AbortSignal,
): Promise<void> {
  const token = localStorage.getItem(TOKEN_KEY)
  const resp = await fetch('/api/chat/stream', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      ...(token ? {Authorization: `Bearer ${token}`} : {}),
    },
    body: JSON.stringify(payload),
    signal,
  })

  if (!resp.ok) {
    let body: ApiErrorBody | null = null
    try {
      body = (await resp.json()) as ApiErrorBody
    } catch {
      /* 非 JSON 错误体 */
    }
    throw new ApiError(body?.message || `请求失败（${resp.status}）`, body?.code, resp.status, body?.detail)
  }
  if (!resp.body) throw new ApiError('浏览器不支持流式读取')

  const reader = resp.body.getReader()
  const decoder = new TextDecoder('utf-8')
  let buffer = ''
  while (true) {
    const {done, value} = await reader.read()
    if (done) break
    buffer += decoder.decode(value, {stream: true})
    let idx = buffer.indexOf('\n\n')
    while (idx >= 0) {
      const raw = buffer.slice(0, idx)
      buffer = buffer.slice(idx + 2)
      const evt = parseFrame(raw)
      if (evt) onEvent(evt)
      idx = buffer.indexOf('\n\n')
    }
  }
  // 收尾：最后没有空行结尾的残帧也解析掉
  const tail = parseFrame(buffer)
  if (tail) onEvent(tail)
}

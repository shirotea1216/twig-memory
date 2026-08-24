/**
 * Node 端 LLM 传输层：直连 Moonshot API，密钥从环境变量读取。
 * 通过 setChatTransport 注入后，src/engine/llm.ts 里的全部判定函数
 * （adjudicateFree / adjudicateClosure / adjudicateCounter）即自动走直连，
 * 前端浏览器路径（vite 代理）不受影响。
 */
import { setChatTransport } from '../src/engine/llm'
import { readFileSync } from 'node:fs'

/** 零依赖读取项目根目录 .env.local（与前端 vite 共用同一份密钥文件） */
export function loadEnvLocal(): void {
  try {
    const path = new URL('../.env.local', import.meta.url)
    const text = readFileSync(path, 'utf8')
    for (const line of text.split(/\r?\n/)) {
      const m = line.match(/^\s*([A-Z0-9_]+)\s*=\s*(.*)\s*$/)
      if (m && !process.env[m[1]]) process.env[m[1]] = m[2]
    }
  } catch {
    // 文件不存在则忽略
  }
}

/** 批量作答 prompt 较长（k×碎片 × 5 题），且异源模型可能是慢思考型，超时给足 */
const TIMEOUT_MS = 60000
/** 429 指数退避：5s / 10s / 20s / 40s（免费档 RPM 低，评测或反刍连发时会被限流） */
const RETRY_BASE_MS = 5000
const MAX_RETRIES = 4

/** 注入直连传输层。无可用 key 时返回 false，调用方应回退到规则判定。
 *  key 读取顺序：MUNINN_API_KEY（任意 OpenAI 兼容方）→ KIMI_API_KEY（历史命名） */
export function registerNodeTransport(): boolean {
  loadEnvLocal()
  const apiKey = process.env.MUNINN_API_KEY || process.env.KIMI_API_KEY
  if (!apiKey) return false
  const model = process.env.MUNINN_MODEL || 'moonshot-v1-8k'
  // 归一化：容忍用户把 base 写成 .../v1（代码会自行拼接 /v1/chat/completions）
  const baseUrl = (process.env.MUNINN_BASE_URL || 'https://api.moonshot.cn').replace(/\/+$/, '').replace(/\/v1$/, '')

  setChatTransport(async (messages, opts) => {
    let lastErr: unknown = null
    for (let attempt = 0; attempt <= MAX_RETRIES; attempt++) {
      if (attempt > 0) await new Promise((r) => setTimeout(r, RETRY_BASE_MS * 2 ** (attempt - 1)))
      const ctrl = new AbortController()
      const timer = setTimeout(() => ctrl.abort(), TIMEOUT_MS)
      try {
        const resp = await fetch(`${baseUrl}/v1/chat/completions`, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            Authorization: `Bearer ${apiKey}`,
          },
          body: JSON.stringify({
            // 按调用覆盖（异源反证生成的第二模型），缺省回落默认模型
            model: opts?.model || model,
            temperature: opts?.temperature ?? 0.3,
            max_tokens: opts?.maxTokens ?? 700,
            reasoning_effort: 'none',
            messages,
          }),
          signal: ctrl.signal,
        })
        if (resp.status === 429 && attempt < MAX_RETRIES) {
          lastErr = new Error('HTTP 429（限流）')
          continue
        }
        if (!resp.ok) throw new Error(`HTTP ${resp.status}`)
        const data = await resp.json()
        const raw = data?.choices?.[0]?.message?.content
        if (typeof raw !== 'string' || !raw.trim()) throw new Error('empty response')
        // 思考型模型（如 MiniMax-M3）会在 content 内联 <think>…</think>，且思考 token 计入
        // max_tokens——预算不足时输出只剩半截思考。剥离后再返回，截断残留的空内容按失败重试。
        const text = raw.replace(/<think>[\s\S]*?(<\/think>|$)/g, '').trim()
        if (!text) throw new Error('empty response after <think> strip（大概率 max_tokens 被思考耗尽）')
        return text
      } catch (err) {
        lastErr = err
        // 4xx（非 429）是请求本身有问题，重试无意义，立即抛；网络/超时/思考截断按瞬态退避重试
        if (err instanceof Error && /^HTTP 4\d\d/.test(err.message)) throw err
        if (attempt >= MAX_RETRIES) throw err
      } finally {
        clearTimeout(timer)
      }
    }
    throw lastErr ?? new Error('LLM 调用失败')
  })
  return true
}

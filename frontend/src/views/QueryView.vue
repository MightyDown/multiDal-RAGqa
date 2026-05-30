<template>
  <div class="page active">
    <div class="page-header">
      <div>
        <h1>智能问答</h1>
        <p class="subtitle">基于知识库文档的 RAG 检索增强问答</p>
      </div>
    </div>
    <div class="qa-layout">
      <div class="chat-container" style="flex:0 0 auto;width:100%;max-width:780px">
        <!-- KB & Session selector -->
        <div class="card" style="margin-bottom:12px">
          <div class="card-body" style="padding:14px 20px">
            <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:6px">
              <span style="font-size:.72rem;font-weight:650;color:var(--text-tertiary);letter-spacing:.04em;text-transform:uppercase">知识库</span>
            </div>
            <div class="chip-group">
              <span v-if="kbLoading" style="color:var(--text-tertiary);font-size:.8rem">加载中...</span>
              <span v-else-if="kbs.length === 0" style="color:var(--text-tertiary);font-size:.8rem">暂无知识库，请先创建</span>
              <button
                v-for="kb in kbs" :key="kb.kb_id"
                class="chip"
                :class="{ selected: selectedKb === kb.kb_id }"
                @click="selectedKb = selectedKb === kb.kb_id ? '' : kb.kb_id"
              >
                {{ kb.name }}
                <span style="opacity:.6;font-size:.68rem">{{ kb.doc_count ?? 0 }} 文档</span>
              </button>
            </div>

            <div style="display:flex;align-items:center;justify-content:space-between;margin:10px 0 4px">
              <span style="font-size:.72rem;font-weight:650;color:var(--text-tertiary);letter-spacing:.04em;text-transform:uppercase">会话历史</span>
              <button class="btn btn-ghost btn-sm" @click="newSession" style="font-size:.7rem;padding:4px 12px">+ 新对话</button>
            </div>
            <SessionChips
              :sessions="sessions"
              :currentId="sessionId"
              :loading="sessionLoading"
              :error="sessionError"
              @switch="switchSession"
              @delete="deleteSession"
            />

            <div class="toggle-group" style="margin-top:10px">
              <label class="toggle-item">
                <input type="checkbox" v-model="optRewrite" checked>
                <span>自动改写查询</span>
              </label>
              <label class="toggle-item">
                <input type="checkbox" v-model="optRetrieval" checked>
                <span>允许检索</span>
              </label>
            </div>
          </div>
        </div>

        <!-- Chat messages -->
        <div class="chat-messages" ref="chatMessages">
          <div v-if="messages.length === 0" class="empty-state" style="padding:40px 0">
            <div class="empty-icon">💬</div>
            <div class="empty-title">开始对话</div>
            <div class="empty-desc">输入问题，AI 将检索知识库并生成回答</div>
          </div>
          <ChatMessage v-for="(m, i) in messages" :key="i" :role="m.role" :content="m.content" :sources="m.sources" @show-sources="s => { currentSources = s; sourcePanelRef.value?.activateSources(s); showSources = true }" />
          <!-- Streaming AI message -->
          <div v-if="streaming" class="chat-msg ai">
            <div class="chat-avatar">AI</div>
            <div class="chat-bubble">
              <div class="ai-content" v-html="streamHtml"></div>
              <div v-if="currentSources.length" class="sources-trigger" @click="showSources = true">
                引用来源 ({{ currentSources.length }})
              </div>
            </div>
          </div>
        </div>

        <!-- Typing indicator -->
        <div class="chat-typing" :class="{ show: typing }">
          <span>检索中</span>
          <span class="typing-dot"></span>
          <span class="typing-dot"></span>
          <span class="typing-dot"></span>
        </div>

        <!-- Input row -->
        <div class="chat-input-row">
          <input
            class="form-input"
            v-model="question"
            placeholder="输入问题，如：「Q1营收同比增长多少？」"
            style="flex:1;font-size:.88rem;padding:13px 20px;border-radius:var(--radius-full)"
            @keydown.enter.exact.prevent="doQuery"
          >
          <button v-if="streaming" class="btn btn-stop" @click="doQuery" style="min-width:90px">停止</button>
          <button v-else class="btn btn-primary" :disabled="querying" @click="doQuery" style="min-width:90px">发送</button>
        </div>
      </div>

      <!-- Source Panel (右侧引用来源面板) -->
      <SourcePanel ref="sourcePanelRef" :visible="showSources" :sources="currentSources" @close="showSources = false" />
    </div>
  </div>
</template>

<script setup>
import { ref, nextTick, onMounted, watch } from 'vue'
import { api } from '../api.js'
import { marked } from 'marked'
import katex from 'katex'
import mermaid from 'mermaid'
import ChatMessage from '../components/ChatMessage.vue'
import SourcePanel from '../components/SourcePanel.vue'
import SessionChips from '../components/SessionChips.vue'

mermaid.initialize({ startOnLoad: false, theme: 'neutral' })

// KB selector
const kbs = ref([])
const kbLoading = ref(true)
const selectedKb = ref('')

// Toggles
const optRewrite = ref(true)
const optRetrieval = ref(true)

// Chat
const question = ref('')
const messages = ref([])
const typing = ref(false)
const querying = ref(false)
const streaming = ref(false)
const streamHtml = ref('')
const currentSources = ref([])
const showSources = ref(false)
const chatMessages = ref(null)
const stopController = ref(null)
const sourcePanelRef = ref(null)

// Watch streamHtml changes to trigger mermaid rendering
// Handles streaming: multiple markdown fragments processed incrementally
// style/classDef/click lines may be split across fragments — merge them
watch(streamHtml, () => {
  nextTick(() => {
    document.querySelectorAll('.ai-content .mermaid:not([data-processed])').forEach(el => {
      const raw = el.textContent || ''
      // Merge lines that start with style/classDef/click into previous line
      // Also handles lines that start with spaces+style (streaming split)
      const merged = raw
        .replace(/\n\s+(style|classDef|click)/g, '\n    $1')
        .replace(/([\]\)])\n\s+(style|classDef|click)/g, '$1\n    $2')
      el.textContent = merged
      el.setAttribute('data-processed', 'true')
      mermaid.run({ nodes: [el] }).catch(() => {})
    })
  })
})

// Session
const SESSION_KEY = 'multidal_chat_session_id'
const sessionId = ref(getSessionId())
const sessions = ref([])
const sessionLoading = ref(true)
const sessionError = ref(false)

function getSessionId() {
  let id = localStorage.getItem(SESSION_KEY)
  if (!id) {
    id = 'chat_' + Date.now().toString(36) + '_' + Math.random().toString(36).slice(2, 8)
    localStorage.setItem(SESSION_KEY, id)
  }
  return id
}

function newSession() {
  messages.value = []
  currentSources.value = []
  showSources.value = false
  const id = 'chat_' + Date.now().toString(36) + '_' + Math.random().toString(36).slice(2, 8)
  localStorage.setItem(SESSION_KEY, id)
  sessionId.value = id
  loadSessions()
}

async function loadSessions() {
  sessionLoading.value = true
  sessionError.value = false
  try {
    const data = await api('/sessions')
    const list = data.sessions || []
    const currentId = sessionId.value
    if (!list.some(s => s.session_id === currentId)) {
      list.unshift({ session_id: currentId, session_name: '新对话', msg_count: 0, created_at: new Date().toISOString() })
    }
    sessions.value = list
    // Refresh again soon if current not yet in DB
    if (!data.sessions.length || !data.sessions.some(s => s.session_id === currentId)) {
      setTimeout(loadSessions, 2000)
    }
  } catch {
    sessionError.value = true
  }
  sessionLoading.value = false
}

async function switchSession(sid) {
  if (sid === sessionId.value) return
  localStorage.setItem(SESSION_KEY, sid)
  sessionId.value = sid
  messages.value = []
  currentSources.value = []
  showSources.value = false
  await loadChatHistory(sid)
  await loadSessions()
}

async function loadChatHistory(sid) {
  try {
    const data = await api('/sessions/' + sid + '/messages')
    messages.value = (data.messages || []).map((m, i) => {
      const role = m.role === 'assistant' ? 'ai' : 'user'
      if (role === 'user') {
        const match = m.content.match(/用户问题:\s*(.+?)(?=\n请基于以上文档内容回答)/s)
        return { role, content: match ? match[1].trim() : m.content }
      } else {
        try {
          const arr = typeof m.content === 'string' ? JSON.parse(m.content) : m.content
          if (Array.isArray(arr) && arr[0]?.text) {
            return { role, content: arr.map(item => item.text).join(''), sources: m.sources || [] }
          }
        } catch {}
        return { role, content: typeof m.content === 'string' ? m.content : JSON.stringify(m.content), sources: m.sources || [] }
      }
    })
  } catch {
    messages.value = []
  }
  await nextTick()
  if (chatMessages.value) chatMessages.value.scrollTop = chatMessages.value.scrollHeight
}

async function deleteSession(sid) {
  if (!confirm('确认删除此会话？')) return
  try {
    await api('/sessions/' + sid, { method: 'DELETE' })
    if (sid === sessionId.value) {
      newSession()
    } else {
      loadSessions()
    }
  } catch {}
}

function renderMarkdown(text) {
  if (!text) return ''
  const mathBlocks = []
  let safe = text
    .replace(/\$\$([\s\S]*?)\$\$/g, (_, m) => { mathBlocks.push({ type: 'block', tex: m.trim() }); return `@@MATH${mathBlocks.length - 1}@@` })
    .replace(/\$(.*?)\$/g, (_, m) => { mathBlocks.push({ type: 'inline', tex: m.trim() }); return `@@MINLINE${mathBlocks.length - 1}@@` })
  let html = marked.parse(safe)
  html = html.replace(/@@MATH(\d+)@@/g, (_, i) => {
    try { return katex.renderToString(mathBlocks[+i].tex, { displayMode: true, throwOnError: false }) } catch { return mathBlocks[+i].tex }
  })
  html = html.replace(/@@MINLINE(\d+)@@/g, (_, i) => {
    try { return katex.renderToString(mathBlocks[+i].tex, { displayMode: false, throwOnError: false }) } catch { return mathBlocks[+i].tex }
  })
  // mermaid: render code blocks that look like mermaid diagrams
  // marked escapes > as &gt; inside code, so decode first
  // Also handles indented code blocks (4-space indent → <pre><code> without class)
  html = html.replace(/<pre><code(?: class="language-mermaid")?>([\s\S]*?)<\/code><\/pre>/gi, (_, code) => {
    const trimmed = code.replace(/&lt;/g, '<').replace(/&gt;/g, '>').replace(/&amp;/g, '&').trim()
    if (!trimmed) return ''
    // Accept both fenced (```mermaid) and indented (4-space) mermaid
    if (!code.includes('language-mermaid') && !/^flowchart|^graph|^pie|^sequence|^class|^state|^er|^gantt|^requirement/i.test(trimmed)) {
      return `<pre><code>${trimmed}</code></pre>`
    }
    const id = 'mermaid-' + Math.random().toString(36).slice(2, 9)
    return `<div class="mermaid" id="${id}">${trimmed}</div>`
  })
  // Also catch p-tagged mermaid (no blank line → single <p> block with all content)
  html = html.replace(/<p>(flowchart[\s\S]*?)<\/p>/gi, (_, content) => {
    const trimmed = content.replace(/&lt;/g, '<').replace(/&gt;/g, '>').replace(/&amp;/g, '&').trim()
    if (!/^flowchart|^graph|^pie|^sequence|^class|^state|^er|^gantt|^requirement/i.test(trimmed)) {
      return `<p>${content}</p>`
    }
    const id = 'mermaid-' + Math.random().toString(36).slice(2, 9)
    return `<div class="mermaid" id="${id}">${trimmed}</div>`
  })
  return html
}

async function doQuery() {
  const q = question.value.trim()
  if (!q) return
  if (streaming.value && stopController.value) {
    stopController.value.abort()
    streaming.value = false
    streamHtml.value = ''
    typing.value = false
    querying.value = false
    return
  }
  if (streaming.value || querying.value) return
  question.value = ''
  querying.value = true
  typing.value = true
  streaming.value = false
  streamHtml.value = ''
  currentSources.value = []
  showSources.value = false

  // 记录本轮 query 的 session_id，防止切会话后本轮结果写到新会话
  const activeSid = sessionId.value
  messages.value.push({ role: 'user', content: q })

  try {
    stopController.value = new AbortController()
    const resp = await fetch('/api/query/stream', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        question: q,
        kb_ids: selectedKb.value ? [selectedKb.value] : [],
        retrieval: optRetrieval.value,
        rewrite_query: optRewrite.value,
        session_id: activeSid,
      }),
      signal: stopController.value.signal,
    })
    typing.value = false
    streaming.value = true

    const reader = resp.body.getReader()
    const decoder = new TextDecoder()
    let buffer = ''
    let fullAnswer = ''
    let savedSources = []

    while (true) {
      const { done, value } = await reader.read()
      if (done) break
      buffer += decoder.decode(value, { stream: true })
      const lines = buffer.split('\n')
      buffer = lines.pop()

      for (const line of lines) {
        if (!line.startsWith('data: ')) continue
        try {
          const evt = JSON.parse(line.slice(6))
          if (evt.type === 'sources') {
            savedSources = evt.sources || []
          } else if (evt.type === 'delta') {
            fullAnswer += evt.content
            streamHtml.value = renderMarkdown(fullAnswer)
          }
        } catch {}
      }
    }

    // 只有当前会话匹配时才写入消息（防止切会话后旧流结果写进来）
    // sources 保留在 currentSources，显示在新 ai 消息的 sources 区
    if (sessionId.value === activeSid) {
      messages.value.push({ role: 'ai', content: fullAnswer, sources: savedSources })
      currentSources.value = savedSources
      // 从会话历史中修正刚才那条 user 消息：RAG prompt 很冗长，用原始问题替换
      const lastUserIdx = messages.value.length - 2
      if (lastUserIdx >= 0 && messages.value[lastUserIdx].role === 'user') {
        messages.value[lastUserIdx] = { role: 'user', content: q }
      }
    }

    streaming.value = false
    streamHtml.value = ''

    // Refresh sessions for message count + LLM naming
    setTimeout(loadSessions, 600)
  } catch (e) {
    typing.value = false
    streaming.value = false
    if (sessionId.value === activeSid) {
      messages.value.push({ role: 'ai', content: '抱歉，查询失败了：' + e.message })
    }
  }

  querying.value = false
  await nextTick()
  if (chatMessages.value) chatMessages.value.scrollTop = chatMessages.value.scrollHeight
}

function showSourceDetail(src) {
  showSources.value = true
}

async function loadKbs() {
  kbLoading.value = true
  try {
    const data = await api('/kb/list')
    kbs.value = data.kbs || []
  } catch {}
  kbLoading.value = false
}

onMounted(() => {
  loadKbs()
  loadSessions()
  loadChatHistory(sessionId.value)
})
</script>

<style scoped>
.sources-trigger {
  font-size: .75rem;
  color: var(--text-tertiary);
  cursor: pointer;
  padding: 6px 0 2px;
  display: inline-flex;
  align-items: center;
  gap: 4px;
  opacity: 0.7;
  transition: opacity .2s;
}
.sources-trigger:hover {
  opacity: 1;
  color: var(--accent-primary);
}
.ai-content :deep(.mermaid-diagram) {
  margin: 12px 0;
  overflow-x: auto;
}
.ai-content :deep(.mermaid-diagram svg) {
  max-width: 100%;
  height: auto;
}
.ai-content :deep(.mermaid-error) {
  background: var(--stone);
  padding: 8px;
  border-radius: var(--radius-sm);
  font-size: 0.75rem;
  color: var(--text-error);
}
</style>

<template>
  <div class="page active">
    <Breadcrumb
      :items="[
        { label: '知识库', onClick: () => router.push('/') },
        { label: '文档列表', onClick: () => router.push({ name: 'kb-detail', params: { id: kbId } }) }
      ]"
      :current="filename"
    />

    <div class="page-header">
      <div>
        <h1>{{ filename || '文档内容' }}</h1>
        <p class="subtitle">{{ meta }}</p>
      </div>
      <div class="page-header-actions">
        <button class="btn btn-outline btn-sm" @click="router.push({ name: 'kb-detail', params: { id: kbId } })">返回</button>
      </div>
    </div>

    <div v-if="loading" class="spin-wrap"><span class="spinner"></span></div>
    <div v-else-if="error" class="empty-state">
      <div class="empty-title" style="color:var(--rose)">加载失败</div>
      <div class="empty-desc">{{ error }}</div>
    </div>
    <div v-else-if="!text.trim()" class="empty-state">
      <div class="empty-icon">🔍</div>
      <div class="empty-title">无文本内容</div>
      <div class="empty-desc">文档已入库但暂无完整文本</div>
    </div>
    <div v-else class="doc-content" v-html="renderedHtml" style="padding:20px 24px;line-height:1.85;font-size:.85rem;color:var(--text-secondary);min-height:200px"></div>
  </div>
</template>

<script setup>
import { ref, onMounted } from 'vue'
import { useRouter, useRoute } from 'vue-router'
import { api } from '../api.js'
import Breadcrumb from '../components/Breadcrumb.vue'
import { marked } from 'marked'
import katex from 'katex'

const router = useRouter()
const route = useRoute()
const kbId = route.params.kbId
const taskId = route.params.taskId
const filename = ref(route.query.filename || '')

const text = ref('')
const renderedHtml = ref('')
const meta = ref('加载中...')
const loading = ref(true)
const error = ref('')

async function loadDoc() {
  loading.value = true
  error.value = ''
  try {
    const data = await api('/docs/' + taskId)
    filename.value = data.filename || filename.value
    meta.value = `${data.page_count || 0} 页 · ${data.chunk_count || 0} 个分块 · KB: ${data.kb_id}`
    text.value = data.full_text || ''
    renderedHtml.value = renderMarkdown(text.value)
  } catch (e) {
    error.value = e.message
  }
  loading.value = false
}

function renderMarkdown(t) {
  if (!t) return ''
  const mathBlocks = []
  let safe = t
    .replace(/\$\$([\s\S]*?)\$\$/g, (_, m) => { mathBlocks.push({ type: 'block', tex: m.trim() }); return `@@MATH${mathBlocks.length - 1}@@` })
    .replace(/\$(.*?)\$/g, (_, m) => { mathBlocks.push({ type: 'inline', tex: m.trim() }); return `@@MINLINE${mathBlocks.length - 1}@@` })

  let html = marked.parse(safe)
  html = html.replace(/@@MATH(\d+)@@/g, (_, i) => {
    try { return katex.renderToString(mathBlocks[+i].tex, { displayMode: true, throwOnError: false }) } catch { return mathBlocks[+i].tex }
  })
  html = html.replace(/@@MINLINE(\d+)@@/g, (_, i) => {
    try { return katex.renderToString(mathBlocks[+i].tex, { displayMode: false, throwOnError: false }) } catch { return mathBlocks[+i].tex }
  })
  return html
}

onMounted(loadDoc)
</script>

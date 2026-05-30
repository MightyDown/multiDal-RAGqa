<template>
  <div class="page active">
    <Breadcrumb :items="[{ label: '知识库', onClick: () => router.push('/') }]" :current="kbName" />

    <div class="page-header">
      <div>
        <h1>{{ kbName || '知识库详情' }}</h1>
        <p class="subtitle">{{ kbDesc }}</p>
      </div>
      <div class="page-header-actions">
        <button class="btn btn-ghost btn-sm" @click="loadDocs">刷新</button>
        <button class="btn btn-outline btn-sm" @click="router.push('/')">返回</button>
      </div>
    </div>

    <div v-if="loading" class="spin-wrap"><span class="spinner"></span></div>
    <div v-else-if="error" class="empty-state">
      <div class="empty-title" style="color:var(--rose)">加载失败</div>
      <div class="empty-desc">{{ error }}</div>
    </div>
    <div v-else-if="docs.length === 0" class="empty-state">
      <div class="empty-icon">📃</div>
      <div class="empty-title">暂无文档</div>
      <div class="empty-desc">此知识库中还未上传任何文档<br>请在上传页面选择此知识库并上传 PDF</div>
    </div>
    <div v-else class="card-grid">
      <div v-for="d in docs" :key="d.task_id" class="card doc-card" @click="goDoc(d.task_id, d.filename)">
        <div class="card-body">
          <div style="display:flex;align-items:center;gap:14px">
            <div class="doc-icon">📎</div>
            <div style="flex:1;min-width:0">
              <div class="doc-name">{{ d.filename }}</div>
              <div class="doc-meta">
                {{ d.page_count ? d.page_count + ' 页 · ' : '' }}{{ d.status }}
                · {{ formatTime(d.created_at) }}
              </div>
            </div>
            <span class="tag" :class="d.status === 'completed' ? 'tag-moss' : 'tag-default'">{{ d.status }}</span>
            <button
              class="btn btn-sm"
              style="color:var(--rose);border:1px solid rgba(203,139,136,.3);padding:4px 12px;font-size:.7rem;border-radius:100px;z-index:2"
              @click.stop="deleteDoc(d.task_id)"
            >删除</button>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, onMounted } from 'vue'
import { useRouter, useRoute } from 'vue-router'
import { api } from '../api.js'
import Breadcrumb from '../components/Breadcrumb.vue'

const router = useRouter()
const route = useRoute()
const kbId = route.params.id

const kbName = ref('')
const kbDesc = ref('')
const docs = ref([])
const loading = ref(true)
const error = ref('')

async function loadDocs() {
  loading.value = true
  error.value = ''
  try {
    // Also fetch KB info
    const [kbData, docsData] = await Promise.all([
      api('/kb/list').catch(() => ({ kbs: [] })),
      api('/kb/' + kbId + '/docs')
    ])
    const kb = (kbData.kbs || []).find(k => k.kb_id === kbId) || {}
    kbName.value = kb.name || kbId
    kbDesc.value = kb.description || ''
    docs.value = docsData.docs || []
  } catch (e) {
    error.value = e.message
  }
  loading.value = false
}

function goDoc(taskId, filename) {
  router.push({ name: 'doc-detail', params: { kbId, taskId }, query: { filename } })
}

async function deleteDoc(taskId) {
  if (!confirm('确认删除此文档？将清除本地文件、向量数据和解析记录。')) return
  try {
    await api('/docs/' + taskId, { method: 'DELETE' })
    await loadDocs()
  } catch (e) { alert(e.message) }
}

function formatTime(iso) {
  if (!iso) return '—'
  return new Date(iso).toLocaleString('zh-CN', { month: '2-digit', day: '2-digit', hour: '2-digit', minute: '2-digit' })
}

onMounted(loadDocs)
</script>

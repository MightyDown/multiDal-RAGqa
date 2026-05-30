<template>
  <div class="page active">
    <div class="page-header">
      <div>
        <h1>知识库</h1>
        <p class="subtitle">每个知识库是独立的文档集合与向量存储空间</p>
      </div>
      <div class="page-header-actions">
        <button class="btn btn-ghost btn-sm" @click="loadKbs">刷新</button>
        <button class="btn btn-primary" @click="showModal = true">+ 新建知识库</button>
      </div>
    </div>
    <div v-if="loading" class="spin-wrap"><span class="spinner"></span></div>
    <div v-else-if="error" class="empty-state">
      <div class="empty-title" style="color:var(--rose)">加载失败</div>
      <div class="empty-desc">{{ error }}</div>
    </div>
    <div v-else-if="kbs.length === 0" class="empty-state">
      <div class="empty-icon">📭</div>
      <div class="empty-title">暂无知识库</div>
      <div class="empty-desc">点击「新建知识库」创建第一个知识库，然后上传 PDF 文档开始构建向量索引</div>
    </div>
    <div v-else class="card-grid">
      <div v-for="(kb, i) in kbs" :key="kb.kb_id" class="card kb-card" @click="goDetail(kb.kb_id)">
        <div class="card-body">
          <div style="display:flex;align-items:flex-start;gap:14px">
            <div class="kb-icon" v-html="icons[i % 3]"></div>
            <div style="flex:1;min-width:0">
              <div class="kb-title">{{ kb.name }}</div>
              <div class="kb-desc">{{ kb.description || '暂无描述' }}</div>
              <div class="kb-meta">
                <span>{{ kb.doc_count ?? 0 }} 份文档</span>
                <span>{{ kb.kb_id }}</span>
              </div>
            </div>
          </div>
        </div>
        <div class="card-footer">
          <button class="btn btn-danger btn-sm" @click.stop="deleteKb(kb.kb_id)">删除</button>
        </div>
      </div>
    </div>

    <CreateKbModal :visible="showModal" @close="showModal = false" @created="onCreated" />
  </div>
</template>

<script setup>
import { ref, onMounted } from 'vue'
import { useRouter } from 'vue-router'
import { api } from '../api.js'
import CreateKbModal from '../components/CreateKbModal.vue'

const router = useRouter()
const kbs = ref([])
const loading = ref(true)
const error = ref('')
const showModal = ref(false)
const icons = ['📚', '📗', '📕']

async function loadKbs() {
  loading.value = true
  error.value = ''
  try {
    const data = await api('/kb/list')
    kbs.value = data.kbs || []
  } catch (e) {
    error.value = e.message
  }
  loading.value = false
}

function goDetail(kbId) {
  router.push({ name: 'kb-detail', params: { id: kbId } })
}

async function deleteKb(kbId) {
  if (!confirm(`确认删除知识库「${kbId}」及其所有文档？此操作不可恢复。`)) return
  try {
    await api('/kb/' + kbId, { method: 'DELETE' })
    await loadKbs()
  } catch (e) {
    alert(e.message)
  }
}

function onCreated() {
  showModal.value = false
  loadKbs()
}

onMounted(loadKbs)
</script>

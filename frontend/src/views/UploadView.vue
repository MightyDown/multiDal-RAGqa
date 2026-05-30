<template>
  <div class="page active">
    <div class="page-header">
      <div>
        <h1>文档上传</h1>
        <p class="subtitle">上传 PDF 到知识库，系统将异步解析并构建向量索引</p>
      </div>
    </div>
    <div style="max-width:660px">
      <div class="card" style="margin-bottom:16px">
        <div class="card-header">目标知识库</div>
        <div class="card-body">
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
        </div>
      </div>

      <div class="upload-dragger" :class="{ dragover }" @click="fileInput?.click()"
        @dragover.prevent="dragover = true" @dragleave="dragover = false" @drop.prevent="onDrop">
        <div class="upload-icon">📅</div>
        <div class="upload-text">拖拽 PDF 文件到此处，或点击选择</div>
        <div class="upload-hint">仅支持 .pdf 格式，单文件最大 100MB</div>
        <input type="file" ref="fileInput" accept=".pdf" multiple @change="onInputChange" style="display:none">
      </div>

      <div class="file-list">
        <div v-for="(f, i) in files" :key="i" class="file-item">
          <span style="font-size:1.1rem">📎</span>
          <span class="file-name">{{ f.name }}</span>
          <span class="file-size">{{ formatSize(f.size) }}</span>
          <button class="file-remove" @click="files.splice(i, 1)">&times;</button>
        </div>
      </div>

      <div style="margin-top:14px; display:flex; gap:10px; align-items:center">
        <button class="btn btn-primary btn-lg" :disabled="!selectedKb || files.length === 0 || uploading" @click="startUpload">
          {{ uploading ? '上传中...' : '开始上传' }}
        </button>
        <span style="font-size:.76rem;color:var(--text-tertiary)">{{ files.length ? '已选择 ' + files.length + ' 个文件' : '' }}</span>
      </div>

      <div style="margin-top:16px">
        <div v-for="(r, i) in results" :key="i"
          :style="{ padding:'10px 16px', marginBottom:'6px', fontSize:'.8rem', display:'flex', alignItems:'center', gap:'8px',
            background: r.ok ? 'rgba(93,112,82,.06)' : 'var(--rose-dim)',
            border: '1px solid ' + (r.ok ? 'rgba(93,112,82,.2)' : 'rgba(168,84,72,.2)'),
            borderRadius: 'var(--radius)', color: r.ok ? 'var(--moss)' : 'var(--rose)' }">
          <span :style="{ fontWeight:700 }">{{ r.ok ? '✓' : '✗' }}</span>
          <span style="flex:1">{{ r.filename }}</span>
          <span v-if="r.taskId" class="tag tag-moss">{{ r.taskId }}</span>
          <span v-if="r.ok" style="font-size:.72rem;color:var(--moss)">已提交</span>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, onMounted } from 'vue'
import { api, formatSize } from '../api.js'

const kbs = ref([])
const kbLoading = ref(true)
const selectedKb = ref('')
const files = ref([])
const dragover = ref(false)
const uploading = ref(false)
const results = ref([])
const fileInput = ref(null)

async function loadKbs() {
  kbLoading.value = true
  try {
    const data = await api('/kb/list')
    kbs.value = data.kbs || []
  } catch {}
  kbLoading.value = false
}

function onDrop(e) {
  dragover.value = false
  addFiles(e.dataTransfer.files)
}

function onInputChange() {
  if (fileInput.value) addFiles(fileInput.value.files)
}

function addFiles(fl) {
  const pdfs = Array.from(fl).filter(f => f.name.toLowerCase().endsWith('.pdf'))
  if (pdfs.length === 0) return
  const existing = new Set(files.value.map(f => f.name + f.size))
  files.value = [...files.value, ...pdfs.filter(f => !existing.has(f.name + f.size))]
}

async function startUpload() {
  if (!selectedKb.value || files.value.length === 0) return
  uploading.value = true
  results.value = []

  for (const file of files.value) {
    const form = new FormData()
    form.append('file', file)
    form.append('kb_id', selectedKb.value)
    try {
      const data = await api('/ingest', { method: 'POST', body: form })
      trackTask(data.task_id, file.name)
      results.value.push({ ok: true, filename: file.name, taskId: data.task_id })
    } catch (e) {
      results.value.push({ ok: false, filename: file.name + ' — ' + e.message })
    }
  }

  files.value = []
  uploading.value = false
}

function trackTask(taskId, filename) {
  try {
    const ids = JSON.parse(localStorage.getItem('multidal_task_ids') || '[]')
    ids.unshift({ taskId, filename, kbId: selectedKb.value, time: new Date().toISOString() })
    if (ids.length > 50) ids.length = 50
    localStorage.setItem('multidal_task_ids', JSON.stringify(ids))
  } catch {}
}

onMounted(loadKbs)
</script>

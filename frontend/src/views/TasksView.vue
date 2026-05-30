<template>
  <div class="page active">
    <div class="page-header">
      <div>
        <h1>任务监控</h1>
        <p class="subtitle">追踪 PDF 解析与向量化处理进度</p>
      </div>
      <div class="page-header-actions">
        <button class="btn btn-ghost btn-sm" @click="loadTasks">刷新</button>
      </div>
    </div>

    <div v-if="tasks.length === 0" class="table-wrap">
      <table class="table">
        <thead>
          <tr><th>任务 ID</th><th>文件名</th><th>知识库</th><th>状态</th><th>阶段</th><th>重试</th><th>创建时间</th></tr>
        </thead>
        <tbody>
          <tr><td colspan="7">
            <div class="empty-state">
              <div class="empty-icon">📭</div>
              <div class="empty-title">暂无任务</div>
              <div class="empty-desc">上传 PDF 文档后，任务会自动出现在这里</div>
            </div>
          </td></tr>
        </tbody>
      </table>
    </div>

    <div v-else class="table-wrap">
      <table class="table">
        <thead>
          <tr><th>任务 ID</th><th>文件名</th><th>知识库</th><th>状态</th><th>阶段</th><th>重试</th><th>创建时间</th></tr>
        </thead>
        <tbody>
          <tr v-for="t in tasks" :key="t.taskId">
            <td class="mono">{{ t.taskId }}</td>
            <td>{{ t.filename || '(追踪中)' }}</td>
            <td><span class="tag tag-default">{{ t.kbId || '?' }}</span></td>
            <td>
              <span v-if="!t._status" class="badge badge-failed"><span class="badge-dot"></span>获取失败</span>
              <span v-else class="badge" :class="'badge-' + (t._status.status || 'pending')"><span class="badge-dot"></span>{{ t._status.status }}</span>
            </td>
            <td>
              <span v-if="t._status && t._status.stage" class="tag tag-teal">{{ t._status.stage }}</span>
              <span v-else style="color:var(--text-tertiary)">—</span>
            </td>
            <td style="font-size:.7rem">{{ t._status ? (t._status.retry_count ?? 0) + '/' + (t._status.max_retries ?? 3) : '—' }}</td>
            <td style="font-size:.68rem;color:var(--text-tertiary)">{{ formatTime(t.time) }}</td>
          </tr>
        </tbody>
      </table>
    </div>
  </div>
</template>

<script setup>
import { ref, onMounted } from 'vue'
import { api } from '../api.js'

const tasks = ref([])

async function loadTasks() {
  try {
    const ids = JSON.parse(localStorage.getItem('multidal_task_ids') || '[]')
    // Fetch status for each task
    for (const t of ids) {
      try {
        t._status = await api('/ingest/' + t.taskId)
      } catch { t._status = null }
    }
    tasks.value = ids
  } catch {}
}

function formatTime(iso) {
  if (!iso) return '—'
  return new Date(iso).toLocaleString('zh-CN', { month: '2-digit', day: '2-digit', hour: '2-digit', minute: '2-digit' })
}

onMounted(loadTasks)
</script>

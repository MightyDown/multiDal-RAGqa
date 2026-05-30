<template>
  <aside class="sidebar">
    <div class="sidebar-header">
      <div class="sidebar-logo">Mc<span>Dawn</span></div>
      <div class="sidebar-version">v0.1.0 · Document Intelligence</div>
    </div>
    <nav class="sidebar-nav">
      <button
        v-for="item in navItems"
        :key="item.page"
        class="nav-item"
        :class="{ active: currentPage === item.page || (item.page === 'kb' && currentPage.startsWith('kb')) }"
        @click="navigate(item.page)"
      >
        <span class="nav-icon" v-html="item.icon"></span> {{ item.label }}
      </button>
    </nav>
    <div class="sidebar-footer">
      <span class="status-dot" :class="{ off: !healthy }"></span>
      <span>{{ healthy ? '服务正常' : '服务离线' }}</span>
      <button class="btn-logout" @click="logout" title="退出登录">
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M9 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h4"/><polyline points="16 17 21 12 16 7"/><line x1="21" y1="12" x2="9" y2="12"/></svg>
      </button>
    </div>
  </aside>
</template>

<script setup>
import { ref, computed, onMounted, onUnmounted } from 'vue'
import { useRouter, useRoute } from 'vue-router'

const router = useRouter()
const route = useRoute()

const navItems = [
  { page: 'kb', label: '知识库', icon: '📚' },
  { page: 'upload', label: '文档上传', icon: '📄' },
  { page: 'query', label: '智能问答', icon: '🔎' },
  { page: 'tasks', label: '任务监控', icon: '📋' },
]

const healthy = ref(false)
let _timer = null

const API = import.meta.env.PROD ? '' : '/api'

const pageToRoute = {
  kb: '/',
  upload: '/upload',
  query: '/query',
  tasks: '/tasks',
}

const currentPage = computed(() => {
  const name = route.name
  if (name === 'kb' || name === 'kb-detail' || name === 'doc-detail') return 'kb'
  return name || 'kb'
})

function navigate(page) {
  router.push(pageToRoute[page] || '/')
}

function logout() {
  localStorage.removeItem('multidal_chat_session_id')
  window.location.href = '/'
}

async function check() {
  try {
    const r = await fetch(API + '/health')
    healthy.value = r.ok
  } catch {
    healthy.value = false
  }
}

onMounted(() => {
  check()
  _timer = setInterval(check, 30000)
})

onUnmounted(() => clearInterval(_timer))
</script>

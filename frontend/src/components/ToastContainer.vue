<template>
  <div class="toast-container">
    <div
      v-for="(t, i) in toasts"
      :key="t.id"
      class="toast"
      :class="'toast-' + t.type"
      :style="{ opacity: t.leaving ? 0 : 1, transform: t.leaving ? 'translateX(28px)' : '', transition: 'all .3s ease' }"
    >
      <span class="toast-icon">{{ icons[t.type] }}</span>
      <span v-html="t.message"></span>
    </div>
  </div>
</template>

<script setup>
import { ref } from 'vue'

const icons = { success: '✓', error: '✗', info: 'ℹ' }
const toasts = ref([])
let _id = 0

function toast(msg, type = 'success') {
  const id = ++_id
  toasts.value.push({ id, message: escHtml(msg), type, leaving: false })
  setTimeout(() => {
    const t = toasts.value.find(t => t.id === id)
    if (t) t.leaving = true
    setTimeout(() => { toasts.value = toasts.value.filter(t => t.id !== id) }, 300)
  }, 3400)
}

function escHtml(s) {
  if (!s) return ''
  const d = document.createElement('div')
  d.textContent = String(s)
  return d.innerHTML
}

defineExpose({ toast })
</script>

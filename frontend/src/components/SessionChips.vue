<template>
  <div class="chip-group" style="max-height:100px;overflow-y:auto">
    <span v-if="loading" style="color:var(--text-tertiary);font-size:.78rem">加载中...</span>
    <span v-else-if="error" style="color:var(--text-tertiary);font-size:.78rem">加载失败</span>
    <template v-else>
      <span v-for="s in sessions" :key="s.session_id" style="display:inline-flex;align-items:center;gap:2px">
        <button
          class="chip"
          :class="{ selected: s.session_id === currentId }"
          style="max-width:180px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap"
          :title="(s.session_name || '新对话') + ' · ' + formatTime(s.created_at) + ' · ' + (s.msg_count || 0) + ' 条消息'"
          @click="$emit('switch', s.session_id)"
        >
          {{ (s.session_name || '').trim() || '新对话' }}
          <span style="opacity:.5;font-size:.65rem;margin-left:2px">{{ s.msg_count || 0 }}条</span>
        </button>
        <span
          style="cursor:pointer;color:var(--text-tertiary);font-size:.85rem;padding:2px 4px;border-radius:50%;transition:all .15s"
          @click.stop="$emit('delete', s.session_id)"
          @mouseenter="hoverStyle($event, true)"
          @mouseleave="hoverStyle($event, false)"
          title="删除此会话"
        >&times;</span>
      </span>
    </template>
  </div>
</template>

<script setup>
defineProps({
  sessions: { type: Array, default: () => [] },
  currentId: { type: String, default: '' },
  loading: { type: Boolean, default: false },
  error: { type: Boolean, default: false }
})

defineEmits(['switch', 'delete'])

function hoverStyle(e, enter) {
  if (enter) {
    e.target.style.color = 'var(--rose)'
    e.target.style.background = 'var(--rose-dim)'
  } else {
    e.target.style.color = 'var(--text-tertiary)'
    e.target.style.background = 'none'
  }
}

function formatTime(iso) {
  if (!iso) return '—'
  return new Date(iso).toLocaleString('zh-CN', { month: '2-digit', day: '2-digit', hour: '2-digit', minute: '2-digit' })
}
</script>

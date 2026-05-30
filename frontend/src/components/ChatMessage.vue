<template>
  <div class="chat-msg" :class="role">
    <div class="chat-avatar">{{ role === 'user' ? '我' : 'AI' }}</div>
    <div class="chat-bubble">
      <div v-html="renderedContent"></div>
      <div v-if="role === 'ai' && sources?.length" class="sources-trigger" @click="$emit('show-sources', sources)">
        引用来源 ({{ sources.length }})
      </div>
    </div>
  </div>
</template>

<script setup>
import { computed } from 'vue'
import { marked } from 'marked'
import katex from 'katex'

const props = defineProps({
  role: { type: String, required: true },
  content: { type: String, default: '' },
  sources: { type: Array, default: null }
})

defineEmits(['show-sources'])

function parseAIResponse(content) {
  try {
    const arr = JSON.parse(content)
    if (Array.isArray(arr) && arr[0]?.text) {
      return arr.map(item => item.text).join('')
    }
  } catch {}
  return content
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
  return html
}

const renderedContent = computed(() => {
  const raw = props.role === 'ai' ? parseAIResponse(props.content) : props.content
  return renderMarkdown(raw)
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
</style>

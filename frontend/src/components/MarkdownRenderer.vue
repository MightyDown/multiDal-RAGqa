<template>
  <div ref="el" class="ai-content" v-html="html"></div>
</template>

<script setup>
import { computed, watch, ref, onMounted } from 'vue'
import { marked } from 'marked'
import katex from 'katex'

const props = defineProps({
  text: { type: String, default: '' },
  className: { type: String, default: 'ai-content' }
})

const el = ref(null)

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

const html = computed(() => renderMarkdown(props.text))
</script>

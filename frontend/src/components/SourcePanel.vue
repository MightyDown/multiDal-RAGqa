<template>
  <div v-if="visible" class="source-panel" id="sourcePanel">
    <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:16px">
      <span style="font-family:var(--font-display);font-size:1rem;font-weight:650;color:var(--text-primary)">引用来源详情</span>
      <button @click="handleClose" style="background:none;border:none;cursor:pointer;font-size:1.4rem;color:var(--text-tertiary);line-height:1;padding:4px 8px;border-radius:50%;transition:all .2s">&times;</button>
    </div>
    <div style="font-size:.82rem;color:var(--text-secondary);line-height:1.75">
      <div v-for="(src, i) in displaySources" :key="i" class="chat-src-item" :class="{ active: selectedSrc === src }" @click="showDetail(src)" style="cursor:pointer">
        <span class="chat-src-score">{{ (src.score * 100).toFixed(0) }}%</span>
        <div style="flex:1;min-width:0">
          <div style="font-size:.68rem;color:var(--text-tertiary);margin-bottom:2px">
            {{ src.kb_id || '?' }} · {{ src.doc_id || '?' }} · p{{ src.page ?? '?' }}
          </div>
          <div style="font-size:.74rem;color:var(--text-secondary);display:-webkit-box;-webkit-line-clamp:2;-webkit-box-orient:vertical;overflow:hidden">
            {{ (src.content || '').substring(0, 150) }}
          </div>
        </div>
      </div>
    </div>
    <div v-if="selectedSrc" style="margin-top:16px;padding-top:16px;border-top:1px solid rgba(222,216,207,.4)">
      <div style="margin-bottom:8px;display:flex;gap:8px">
        <span class="tag tag-moss">p{{ selectedSrc.page }}</span>
        <span class="tag tag-teal">相关度 {{ (selectedSrc.score * 100).toFixed(0) }}%</span>
        <span v-if="selectedSrc.image_path" class="tag tag-clay">图片</span>
      </div>
      <div v-if="selectedSrc.image_path" class="detail-image-wrapper" @click="showLightbox(imageUrl(selectedSrc.image_path))">
        <img :src="imageUrl(selectedSrc.image_path)" :alt="'p' + selectedSrc.page" />
      </div>
      <div class="detail-content" v-html="renderDetail(selectedSrc.content)"></div>
    </div>

    <!-- Lightbox: click to enlarge image -->
    <div v-if="lightboxUrl" class="lightbox" @click.self="closeLightbox">
      <div class="lightbox-inner">
        <button class="lightbox-close" @click="closeLightbox">&times;</button>
        <img :src="lightboxUrl" class="lightbox-img" />
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, computed } from 'vue'
import { marked } from 'marked'
import katex from 'katex'
import mermaid from 'mermaid'

mermaid.initialize({ startOnLoad: false, theme: 'neutral' })

const props = defineProps({
  visible: { type: Boolean, default: false },
  sources: { type: Array, default: () => [] }
})

const emit = defineEmits(['close'])

const selectedSrc = ref(null)
const activeSources = ref(null)
const lightboxUrl = ref(null)

const displaySources = computed(() => activeSources.value || props.sources)

function showLightbox(url) {
  lightboxUrl.value = url
}

function closeLightbox() {
  lightboxUrl.value = null
}

function showDetail(src) {
  if (selectedSrc.value === src) {
    selectedSrc.value = null
  } else {
    selectedSrc.value = src
  }
}

function imageUrl(imagePath) {
  // image_path stored as /app/docs/{task_id}/images/... -> /raw/{task_id}/images/...
  const stripped = imagePath.replace(/^\/app\/docs\//, '')
  return '/raw/' + stripped
}

function activateSources(sources) {
  if (activeSources.value === sources) {
    activeSources.value = null
    selectedSrc.value = null
    return
  }
  selectedSrc.value = null
  activeSources.value = sources
}

function handleClose() {
  activeSources.value = null
  selectedSrc.value = null
  emit('close')
}

defineExpose({ activateSources })

function renderDetail(text) {
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
  // mermaid: render code blocks that look like mermaid diagrams
  html = html.replace(/<pre><code(?: class="language-mermaid")?>([\s\S]*?)<\/code><\/pre>/g, (_, code) => {
    const trimmed = code.trim()
    if (!trimmed) return ''
    const id = 'mermaid-' + Math.random().toString(36).slice(2, 9)
    try {
      const svg = mermaid.render(id, trimmed)
      return `<div class="mermaid-diagram">${svg}</div>`
    } catch (e) {
      return `<pre class="mermaid-error">${trimmed}</pre>`
    }
  })
  return html
}
</script>

<style scoped>
.chat-src-item.active {
  background: var(--stone);
}
.detail-image-wrapper {
  margin-bottom: 12px;
  border-radius: var(--radius-sm);
  overflow: hidden;
  max-height: 300px;
  display: flex;
  align-items: center;
  justify-content: center;
  background: var(--bg-root);
  cursor: zoom-in;
}
.detail-image-wrapper img {
  max-width: 100%;
  max-height: 300px;
  object-fit: contain;
  display: block;
}
.detail-content :deep(img) {
  max-width: 100%;
  border-radius: var(--radius-sm);
  margin: 8px 0;
}
.detail-content :deep(.katex-display) {
  margin: 12px 0;
  overflow-x: auto;
}
.detail-content :deep(.mermaid-diagram) {
  margin: 12px 0;
  overflow-x: auto;
}
.detail-content :deep(.mermaid-diagram svg) {
  max-width: 100%;
  height: auto;
}
.detail-content :deep(.mermaid-error) {
  background: var(--stone);
  padding: 8px;
  border-radius: var(--radius-sm);
  font-size: 0.75rem;
  color: var(--text-error);
}
.lightbox {
  position: fixed;
  inset: 0;
  z-index: 9999;
  background: rgba(0, 0, 0, 0.85);
  display: flex;
  align-items: center;
  justify-content: center;
  cursor: pointer;
}
.lightbox-inner {
  position: relative;
  max-width: 90vw;
  max-height: 90vh;
}
.lightbox-close {
  position: absolute;
  top: -40px;
  right: 0;
  background: none;
  border: none;
  color: white;
  font-size: 2rem;
  cursor: pointer;
  line-height: 1;
}
.lightbox-img {
  max-width: 90vw;
  max-height: 85vh;
  object-fit: contain;
  border-radius: var(--radius-sm);
  display: block;
}
</style>
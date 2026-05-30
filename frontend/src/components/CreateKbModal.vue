<template>
  <div class="modal-mask" :class="{ show: visible }" @click.self="$emit('close')">
    <div class="modal">
      <div class="modal-header">新建知识库</div>
      <div class="modal-body">
        <div class="form-group">
          <label class="form-label">名称 <span class="req">*</span></label>
          <input class="form-input" ref="nameInput" v-model="name" placeholder="例如：财务报告、技术文档" maxlength="128" @keydown.enter="create">
          <div class="form-hint">1-128 字符，用于标识知识库</div>
        </div>
        <div class="form-group" style="margin-bottom:0">
          <label class="form-label">描述</label>
          <textarea class="form-textarea" v-model="desc" placeholder="知识库用途说明（可选）" maxlength="512"></textarea>
          <div class="form-hint">最多 512 字符</div>
        </div>
      </div>
      <div class="modal-footer">
        <button class="btn btn-outline" @click="$emit('close')">取消</button>
        <button class="btn btn-primary" :disabled="creating" @click="create">{{ creating ? '创建中...' : '创建' }}</button>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, watch, nextTick } from 'vue'
import { api } from '../api.js'

const props = defineProps({ visible: Boolean })
const emit = defineEmits(['close', 'created'])

const name = ref('')
const desc = ref('')
const creating = ref(false)
const nameInput = ref(null)

watch(() => props.visible, async (v) => {
  if (v) {
    name.value = ''
    desc.value = ''
    await nextTick()
    nameInput.value?.focus()
  }
})

async function create() {
  const n = name.value.trim()
  if (!n) return
  creating.value = true
  try {
    await api('/kb/create', { method: 'POST', body: { name: n, description: desc.value.trim() } })
    emit('created')
  } finally {
    creating.value = false
  }
}
</script>

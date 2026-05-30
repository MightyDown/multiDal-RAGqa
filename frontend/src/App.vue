<template>
  <div class="blob-layer">
    <div class="blob blob-1"></div>
    <div class="blob blob-2"></div>
    <div class="blob blob-3"></div>
  </div>

  <Sidebar />
  <main class="main">
    <router-view v-slot="{ Component }">
      <component :is="Component" />
    </router-view>
  </main>

  <ToastContainer ref="toastRef" />
</template>

<script setup>
import { ref, provide } from 'vue'
import Sidebar from './components/Sidebar.vue'
import ToastContainer from './components/ToastContainer.vue'

const toastRef = ref(null)

// Provide toast function to all child components
function toast(msg, type) {
  toastRef.value?.toast(msg, type)
}
provide('toast', toast)
</script>

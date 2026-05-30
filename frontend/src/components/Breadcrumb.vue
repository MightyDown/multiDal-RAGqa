<template>
  <div class="breadcrumb">
    <a v-for="(item, i) in items" :key="i" @click="item.onClick ? item.onClick() : null">
      {{ item.label }}
    </a>
    <span v-if="items.length > 0">/</span>
    <span>{{ current }}</span>
  </div>
</template>

<script setup>
defineProps({
  items: { type: Array, default: () => [] },
  current: { type: String, default: '' }
})
</script>

<script setup lang="ts">
defineProps<{
  title?: string
}>()
</script>

<template>
  <div class="pipeline-box pipeline-box--unified">
    <div v-if="title" class="pipeline-box__title">
      {{ title }}
    </div>
    <div class="pipeline-box__equation">
      <slot />
    </div>
  </div>
</template>

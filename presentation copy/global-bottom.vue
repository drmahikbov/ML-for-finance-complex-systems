<script setup lang="ts">
import { useNav } from '@slidev/client'

const { currentPage } = useNav()
</script>

<template>
  <span class="deck-slide-number" aria-hidden="true">{{ currentPage }}</span>
</template>

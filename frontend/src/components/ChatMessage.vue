<script setup>
import { computed, ref } from 'vue'
import SourceCard from './SourceCard.vue'
import { groupSources } from '../utils/groupSources'

const props = defineProps({
  message: {
    type: Object,
    required: true,
  },
})

const showAllSources = ref(false)
const sourceGroups = computed(() => groupSources(props.message.sources))
const visibleSourceGroups = computed(() => (
  showAllSources.value ? sourceGroups.value : sourceGroups.value.slice(0, 2)
))
const hiddenSourceCount = computed(() => Math.max(sourceGroups.value.length - 2, 0))
</script>

<template>
  <div class="message-row" :class="`message-${message.role}`">
    <div v-if="message.role === 'assistant'" class="assistant-avatar" aria-hidden="true">
      M
    </div>

    <div class="message-column">
      <span v-if="message.role === 'assistant'" class="message-label">MAX</span>

      <div
        class="message-bubble"
        :class="{
          'error-bubble': message.error,
          'loading-bubble': message.loading,
        }"
      >
        <div v-if="message.loading" class="thinking" role="status" aria-live="polite">
          <span>MAX is thinking</span>
          <span class="thinking-dots" aria-hidden="true">
            <i></i><i></i><i></i>
          </span>
        </div>
        <p v-else>{{ message.content }}</p>
      </div>

      <section
        v-if="message.role === 'assistant' && message.sources?.length"
        class="sources-section"
        aria-label="Answer sources"
      >
        <h2>Sources</h2>
        <div class="sources-grid">
          <SourceCard
            v-for="group in visibleSourceGroups"
            :key="group.key"
            :group="group"
          />
        </div>
        <button
          v-if="hiddenSourceCount"
          class="sources-toggle"
          type="button"
          :aria-expanded="showAllSources"
          @click="showAllSources = !showAllSources"
        >
          {{ showAllSources ? 'Hide extra sources ↑' : `View ${hiddenSourceCount} more sources ↓` }}
        </button>
      </section>
    </div>
  </div>
</template>

<style scoped>
.message-row {
  display: flex;
  gap: 11px;
  animation: message-in 220ms ease-out both;
}

.message-user {
  justify-content: flex-end;
}

.message-column {
  min-width: 0;
  max-width: min(calc(100% - 42px), 880px);
}

.message-user .message-column {
  max-width: min(74%, 680px);
  display: flex;
  justify-content: flex-end;
}

.assistant-avatar {
  width: 30px;
  height: 30px;
  flex: 0 0 auto;
  margin-top: 19px;
  border-radius: 10px;
  display: grid;
  place-items: center;
  background: var(--blue-soft);
  color: var(--blue-primary-dark);
  font-size: 12px;
  font-weight: 760;
}

.message-label {
  display: block;
  margin: 0 0 6px 2px;
  color: var(--text-secondary);
  font-size: 11px;
  font-weight: 700;
  letter-spacing: 0.05em;
}

.message-bubble {
  padding: 15px 17px;
  border: 1px solid var(--blue-border);
  border-radius: 8px 18px 18px 18px;
  background: var(--bg-panel);
  color: var(--text-primary);
  box-shadow: 0 3px 13px rgba(49, 88, 128, 0.05);
}

.message-user .message-bubble {
  border-color: #c9e2fa;
  border-radius: 18px 18px 8px 18px;
  background: var(--blue-user);
  color: #174a7c;
  box-shadow: 0 4px 14px rgba(55, 125, 197, 0.09);
}

.message-bubble p {
  margin: 0;
  white-space: pre-wrap;
  overflow-wrap: anywhere;
  font-size: 14px;
  line-height: 1.7;
}

.error-bubble {
  border-color: #f2d4d2;
  background: #fffafa;
  color: #8b3b38;
}

.loading-bubble {
  min-width: 176px;
}

.thinking {
  display: flex;
  align-items: center;
  gap: 11px;
  color: var(--text-secondary);
  font-size: 13px;
}

.thinking-dots {
  display: inline-flex;
  gap: 4px;
}

.thinking-dots i {
  width: 5px;
  height: 5px;
  border-radius: 50%;
  background: var(--blue-primary);
  animation: dot-pulse 1.2s infinite ease-in-out;
}

.thinking-dots i:nth-child(2) {
  animation-delay: 140ms;
}

.thinking-dots i:nth-child(3) {
  animation-delay: 280ms;
}

.sources-section {
  margin-top: 13px;
}

.sources-section h2 {
  margin: 0 0 8px 2px;
  color: var(--text-secondary);
  font-size: 11px;
  font-weight: 700;
  letter-spacing: 0.06em;
  text-transform: uppercase;
}

.sources-grid {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 8px;
}

.sources-toggle {
  margin: 9px 0 0 2px;
  padding: 3px 0;
  border: 0;
  background: transparent;
  color: var(--blue-primary-dark);
  font: inherit;
  font-size: 11px;
  font-weight: 650;
  cursor: pointer;
}

.sources-toggle:hover {
  color: #1f6fbe;
  text-decoration: underline;
  text-underline-offset: 3px;
}

.sources-toggle:focus-visible {
  border-radius: 4px;
  outline: 3px solid var(--focus-ring);
  outline-offset: 2px;
}

@keyframes dot-pulse {
  0%, 70%, 100% { opacity: 0.28; transform: translateY(0); }
  35% { opacity: 1; transform: translateY(-3px); }
}

@keyframes message-in {
  from { opacity: 0; transform: translateY(7px); }
  to { opacity: 1; transform: translateY(0); }
}

@media (max-width: 720px) {
  .message-column {
    max-width: calc(100% - 42px);
  }

  .message-user .message-column {
    max-width: 84%;
  }

  .sources-grid {
    grid-template-columns: 1fr;
  }
}

@media (prefers-reduced-motion: reduce) {
  .message-row,
  .thinking-dots i {
    animation: none;
  }
}
</style>

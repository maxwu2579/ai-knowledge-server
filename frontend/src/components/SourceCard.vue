<script setup>
import { ref } from 'vue'

defineProps({
  group: {
    type: Object,
    required: true,
  },
})

const expanded = ref(false)
</script>

<template>
  <article class="source-card">
    <div class="source-heading">
      <svg viewBox="0 0 24 24" aria-hidden="true">
        <path d="M7 3h7l4 4v14H7z" />
        <path d="M14 3v5h5M10 12h5M10 16h5" />
      </svg>
      <span class="source-name">{{ group.source }}</span>
    </div>
    <div class="source-summary">
      <span>Page {{ group.page }}</span>
      <span aria-hidden="true">·</span>
      <span>
        {{ group.passages.length }} relevant
        {{ group.passages.length === 1 ? 'passage' : 'passages' }}
      </span>
    </div>

    <button
      class="passages-toggle"
      type="button"
      :aria-expanded="expanded"
      @click="expanded = !expanded"
    >
      {{ expanded ? 'Hide passages ↑' : 'View passages ↓' }}
    </button>

    <div v-if="expanded" class="passage-list">
      <div
        v-for="(passage, index) in group.passages"
        :key="`${group.key}-${index}`"
        class="passage"
      >
        <div class="passage-label">
          <strong>Passage {{ index + 1 }}</strong>
          <span>distance {{ Number(passage.distance).toFixed(3) }}</span>
        </div>
        <p :title="passage.text">{{ passage.text }}</p>
      </div>
    </div>
  </article>
</template>

<style scoped>
.source-card {
  min-width: 0;
  padding: 10px 12px;
  border: 1px solid var(--blue-border);
  border-radius: 13px;
  background: #fbfdff;
}

.source-heading {
  display: flex;
  align-items: center;
  gap: 7px;
  min-width: 0;
  color: var(--text-primary);
  font-size: 12px;
  font-weight: 650;
}

.source-heading svg {
  width: 15px;
  height: 15px;
  flex: 0 0 auto;
  fill: none;
  stroke: var(--blue-primary);
  stroke-width: 1.8;
  stroke-linecap: round;
  stroke-linejoin: round;
}

.source-name {
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.source-summary {
  margin: 7px 0 0 22px;
  display: flex;
  flex-wrap: wrap;
  gap: 5px;
  color: var(--text-secondary);
  font-size: 11px;
  font-weight: 500;
}

.passages-toggle {
  margin: 8px 0 0 22px;
  padding: 2px 0;
  border: 0;
  background: transparent;
  color: var(--blue-primary-dark);
  font: inherit;
  font-size: 11px;
  font-weight: 650;
  cursor: pointer;
}

.passages-toggle:hover {
  color: #1f6fbe;
  text-decoration: underline;
  text-underline-offset: 3px;
}

.passages-toggle:focus-visible {
  border-radius: 4px;
  outline: 3px solid var(--focus-ring);
  outline-offset: 2px;
}

.passage-list {
  margin-top: 10px;
  padding-top: 2px;
  border-top: 1px solid #e5f0fb;
}

.passage {
  padding: 10px 0 7px;
}

.passage + .passage {
  border-top: 1px solid #edf4fb;
}

.passage-label {
  display: flex;
  justify-content: space-between;
  gap: 10px;
  color: var(--text-secondary);
  font-size: 10px;
}

.passage-label strong {
  color: #46627f;
  font-weight: 700;
}

.passage p {
  margin: 6px 0 0;
  overflow: hidden;
  display: -webkit-box;
  -webkit-box-orient: vertical;
  -webkit-line-clamp: 3;
  color: var(--text-secondary);
  font-size: 12px;
  line-height: 1.5;
}

@media (max-width: 520px) {
  .source-summary,
  .passages-toggle {
    margin-left: 0;
  }
}
</style>

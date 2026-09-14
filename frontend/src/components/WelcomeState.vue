<script setup>
defineProps({
  examples: {
    type: Array,
    required: true,
  },
})

defineEmits(['select'])
</script>

<template>
  <section class="welcome-state" aria-labelledby="welcome-title">
    <div class="welcome-mark" aria-hidden="true">M</div>
    <p class="eyebrow">Your knowledge, made accessible</p>
    <h1 id="welcome-title">Hi, I'm MAX</h1>
    <p class="welcome-copy">Ask me anything about the knowledge base.</p>

    <div class="example-list" aria-label="Example questions">
      <button
        v-for="example in examples"
        :key="example"
        class="example-chip"
        type="button"
        @click="$emit('select', example)"
      >
        <span>{{ example }}</span>
        <span class="chip-arrow" aria-hidden="true">→</span>
      </button>
    </div>
  </section>
</template>

<style scoped>
.welcome-state {
  width: min(100%, 680px);
  margin: auto;
  padding: 48px 0 92px;
  text-align: center;
}

.welcome-mark {
  width: 62px;
  height: 62px;
  margin: 0 auto 22px;
  border: 1px solid var(--blue-border);
  border-radius: 21px;
  display: grid;
  place-items: center;
  background: var(--bg-panel);
  color: var(--blue-primary);
  font-size: 22px;
  font-weight: 760;
  box-shadow: var(--shadow-soft);
}

.eyebrow {
  margin: 0 0 10px;
  color: var(--blue-primary-dark);
  font-size: 12px;
  font-weight: 700;
  letter-spacing: 0.1em;
  text-transform: uppercase;
}

h1 {
  margin: 0;
  color: var(--text-primary);
  font-size: clamp(34px, 6vw, 48px);
  line-height: 1.12;
  letter-spacing: -0.035em;
}

.welcome-copy {
  margin: 14px 0 32px;
  color: var(--text-secondary);
  font-size: 17px;
}

.example-list {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 12px;
}

.example-chip {
  min-height: 82px;
  padding: 16px;
  border: 1px solid var(--blue-border);
  border-radius: 16px;
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: 12px;
  background: rgba(255, 255, 255, 0.78);
  color: var(--text-primary);
  font: inherit;
  font-size: 13px;
  line-height: 1.45;
  text-align: left;
  cursor: pointer;
  transition: border-color 160ms ease, box-shadow 160ms ease, transform 160ms ease;
}

.example-chip:hover {
  border-color: #a9d0f7;
  background: #fff;
  box-shadow: 0 12px 28px rgba(41, 82, 124, 0.1);
  transform: translateY(-3px);
}

.example-chip:focus-visible {
  outline: 3px solid var(--focus-ring);
  outline-offset: 2px;
}

.chip-arrow {
  color: var(--blue-primary);
  font-size: 16px;
}

@media (max-width: 720px) {
  .welcome-state {
    padding-top: 34px;
  }

  .example-list {
    grid-template-columns: 1fr;
  }

  .example-chip {
    min-height: 58px;
  }
}
</style>

<script setup>
import { nextTick, ref } from 'vue'

defineProps({
  disabled: {
    type: Boolean,
    default: false,
  },
})

const emit = defineEmits(['send'])
const message = ref('')
const textarea = ref(null)

function resizeTextarea() {
  const element = textarea.value
  if (!element) return

  element.style.height = 'auto'
  element.style.height = `${Math.min(element.scrollHeight, 160)}px`
}

function submit() {
  const question = message.value.trim()
  if (!question) return

  emit('send', question)
  message.value = ''
  nextTick(resizeTextarea)
}

function handleKeydown(event) {
  if (event.key === 'Enter' && !event.shiftKey) {
    event.preventDefault()
    if (!event.isComposing) submit()
  }
}
</script>

<template>
  <div class="input-shell">
    <div class="input-card" :class="{ 'is-disabled': disabled }">
      <textarea
        ref="textarea"
        v-model="message"
        rows="1"
        maxlength="4000"
        placeholder="Ask MAX anything..."
        aria-label="Ask MAX a question"
        :disabled="disabled"
        @input="resizeTextarea"
        @keydown="handleKeydown"
      ></textarea>

      <button
        type="button"
        aria-label="Send question"
        :disabled="disabled || !message.trim()"
        @click="submit"
      >
        <svg viewBox="0 0 24 24" aria-hidden="true">
          <path d="M12 19V5" />
          <path d="m6.5 10.5 5.5-5.5 5.5 5.5" />
        </svg>
      </button>
    </div>
    <p class="input-hint">Enter to send · Shift + Enter for a new line</p>
  </div>
</template>

<style scoped>
.input-shell {
  padding: 12px 0 18px;
  background: linear-gradient(to top, var(--bg-main) 72%, rgba(245, 249, 255, 0));
}

.input-card {
  min-height: 66px;
  padding: 10px 10px 10px 19px;
  border: 1px solid var(--blue-border);
  border-radius: 20px;
  display: flex;
  align-items: flex-end;
  gap: 10px;
  background: var(--bg-panel);
  box-shadow: var(--shadow-input);
  transition: border-color 160ms ease, box-shadow 160ms ease;
}

.input-card:focus-within {
  border-color: #a9d0f7;
  box-shadow: 0 10px 32px rgba(54, 101, 151, 0.13), 0 0 0 3px var(--focus-ring);
}

.input-card.is-disabled {
  opacity: 0.75;
}

textarea {
  width: 100%;
  min-height: 44px;
  max-height: 160px;
  padding: 10px 0 8px;
  border: 0;
  outline: 0;
  resize: none;
  overflow-y: auto;
  background: transparent;
  color: var(--text-primary);
  font: inherit;
  font-size: 14px;
  line-height: 1.5;
}

textarea::placeholder {
  color: #98a7ba;
}

button {
  width: 44px;
  height: 44px;
  flex: 0 0 auto;
  padding: 0;
  border: 0;
  border-radius: 50%;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  background: var(--blue-primary);
  color: #fff;
  font: inherit;
  cursor: pointer;
  transition: background 160ms ease, transform 160ms ease, opacity 160ms ease;
}

button:hover:not(:disabled) {
  background: var(--blue-primary-dark);
  transform: translateY(-1px);
}

button:focus-visible {
  outline: 3px solid var(--focus-ring);
  outline-offset: 2px;
}

button:disabled {
  cursor: not-allowed;
  opacity: 0.45;
}

button svg {
  width: 18px;
  height: 18px;
  fill: none;
  stroke: currentColor;
  stroke-width: 1.8;
  stroke-linecap: round;
  stroke-linejoin: round;
}

.input-hint {
  margin: 8px 0 0;
  color: #91a0b3;
  font-size: 10px;
  text-align: center;
}

@media (max-width: 560px) {
  .input-card {
    min-height: 58px;
    padding-left: 14px;
    border-radius: 17px;
  }

  button {
    width: 42px;
    height: 42px;
  }
}
</style>

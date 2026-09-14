<script setup>
defineProps({
  newChatDisabled: {
    type: Boolean,
    default: false,
  },
})

defineEmits(['new-chat'])
</script>

<template>
  <header class="chat-header">
    <div class="header-content">
      <div class="brand">
        <div class="brand-mark" aria-hidden="true">M</div>
        <div class="brand-copy">
          <strong>MAX</strong>
          <span>AI Knowledge Assistant</span>
        </div>
      </div>

      <button
        class="new-chat-button"
        type="button"
        :disabled="newChatDisabled"
        @click="$emit('new-chat')"
      >
        <span aria-hidden="true">+</span>
        New Chat
      </button>
    </div>
  </header>
</template>

<style scoped>
.chat-header {
  position: sticky;
  top: 0;
  z-index: 20;
  border-bottom: 1px solid rgba(215, 233, 251, 0.9);
  background: rgba(255, 255, 255, 0.94);
  backdrop-filter: blur(14px);
}

.header-content {
  width: min(100% - 40px, var(--content-width));
  height: 72px;
  margin: 0 auto;
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 24px;
}

.brand {
  min-width: 0;
  display: flex;
  align-items: center;
  gap: 12px;
}

.brand-mark {
  width: 38px;
  height: 38px;
  border-radius: 50%;
  display: grid;
  place-items: center;
  background: var(--blue-primary);
  color: #fff;
  font-size: 15px;
  font-weight: 750;
  box-shadow: 0 6px 16px rgba(55, 140, 237, 0.22);
}

.brand-copy {
  display: flex;
  flex-direction: column;
  line-height: 1.2;
}

.brand-copy strong {
  color: var(--text-primary);
  font-size: 17px;
  letter-spacing: 0.04em;
}

.brand-copy span {
  margin-top: 3px;
  color: var(--text-secondary);
  font-size: 12px;
}

.new-chat-button {
  min-height: 38px;
  padding: 0 14px;
  border: 1px solid var(--blue-border);
  border-radius: 12px;
  display: inline-flex;
  align-items: center;
  gap: 7px;
  background: #fff;
  color: #285b8c;
  font: inherit;
  font-size: 13px;
  font-weight: 650;
  cursor: pointer;
  box-shadow: 0 3px 12px rgba(54, 101, 151, 0.06);
  transition: border-color 160ms ease, background 160ms ease, transform 160ms ease;
}

.new-chat-button span {
  font-size: 18px;
  font-weight: 400;
  line-height: 1;
}

.new-chat-button:hover:not(:disabled) {
  border-color: #afd3f7;
  background: #f7fbff;
  transform: translateY(-1px);
}

.new-chat-button:focus-visible {
  outline: 3px solid var(--focus-ring);
  outline-offset: 2px;
}

.new-chat-button:disabled {
  cursor: not-allowed;
  opacity: 0.5;
}

@media (max-width: 640px) {
  .header-content {
    width: calc(100% - 28px);
    height: 64px;
  }

  .brand-copy span {
    display: none;
  }

  .new-chat-button {
    min-height: 36px;
    padding-inline: 12px;
  }
}
</style>

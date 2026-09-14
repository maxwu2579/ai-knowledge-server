<script setup>
import { nextTick, ref } from 'vue'
import ChatHeader from './components/ChatHeader.vue'
import ChatInput from './components/ChatInput.vue'
import ChatMessage from './components/ChatMessage.vue'
import WelcomeState from './components/WelcomeState.vue'
import { askQuestion } from './services/api'

const examples = [
  'What information is available?',
  'What does the document say?',
  'Help me find something',
]

const messages = ref([])
const isLoading = ref(false)
const messageList = ref(null)
let messageId = 0

function startNewChat() {
  messages.value = []
}

function createMessage(role, content, extra = {}) {
  return {
    id: ++messageId,
    role,
    content,
    ...extra,
  }
}

async function scrollToLatest() {
  await nextTick()
  const element = messageList.value
  if (element) element.scrollTop = element.scrollHeight
}

async function sendQuestion(question) {
  const cleanedQuestion = question.trim()
  if (!cleanedQuestion || isLoading.value) return

  messages.value.push(createMessage('user', cleanedQuestion))
  const loadingMessage = createMessage('assistant', '', { loading: true })
  messages.value.push(loadingMessage)
  isLoading.value = true
  await scrollToLatest()

  try {
    const result = await askQuestion(cleanedQuestion)
    const index = messages.value.findIndex((message) => message.id === loadingMessage.id)
    messages.value[index] = createMessage('assistant', result.answer, {
      sources: result.sources,
    })
  } catch {
    const index = messages.value.findIndex((message) => message.id === loadingMessage.id)
    messages.value[index] = createMessage(
      'assistant',
      "MAX couldn't complete the request. Please try again.",
      { error: true },
    )
  } finally {
    isLoading.value = false
    await scrollToLatest()
  }
}
</script>

<template>
  <div class="app-shell">
    <ChatHeader
      :new-chat-disabled="isLoading"
      @new-chat="startNewChat"
    />

    <main class="chat-main">
      <div ref="messageList" class="message-scroll" aria-live="polite">
        <div class="chat-content">
          <WelcomeState
            v-if="messages.length === 0"
            :examples="examples"
            @select="sendQuestion"
          />

          <div v-else class="message-list">
            <ChatMessage
              v-for="message in messages"
              :key="message.id"
              :message="message"
            />
          </div>
        </div>
      </div>

      <div class="input-wrap">
        <ChatInput :disabled="isLoading" @send="sendQuestion" />
      </div>
    </main>
  </div>
</template>

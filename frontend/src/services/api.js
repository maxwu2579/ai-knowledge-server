export async function askQuestion(question) {
  const response = await fetch('/api/query', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ question }),
  })

  if (!response.ok) {
    throw new Error(`Request failed with status ${response.status}`)
  }

  const data = await response.json()

  if (typeof data.answer !== 'string' || !Array.isArray(data.sources)) {
    throw new Error('Unexpected response from the knowledge server')
  }

  return data
}

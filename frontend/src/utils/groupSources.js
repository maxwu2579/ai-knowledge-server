export function groupSources(sources = []) {
  const groups = new Map()

  for (const source of sources) {
    const key = JSON.stringify([source.source, source.page])
    const existing = groups.get(key)
    if (existing) {
      existing.passages.push(source)
    } else {
      groups.set(key, {
        key,
        source: source.source,
        page: source.page,
        passages: [source],
      })
    }
  }

  return [...groups.values()]
}

import test from 'node:test'
import assert from 'node:assert/strict'

import { groupSources } from '../src/utils/groupSources.js'

test('groups the same filename and page while preserving passage objects', () => {
  const first = { source: 'offer.pdf', page: 1, text: 'one', distance: 0.2 }
  const second = { source: 'offer.pdf', page: 1, text: 'two', distance: 0.3 }

  const groups = groupSources([first, second])

  assert.equal(groups.length, 1)
  assert.equal(groups[0].source, 'offer.pdf')
  assert.equal(groups[0].page, 1)
  assert.deepEqual(groups[0].passages, [first, second])
  assert.equal(groups[0].passages[0], first)
  assert.equal(groups[0].passages[1], second)
})

test('keeps different pages and filenames as separate source groups', () => {
  const groups = groupSources([
    { source: 'offer.pdf', page: 1, text: 'one', distance: 0.2 },
    { source: 'offer.pdf', page: 2, text: 'two', distance: 0.3 },
    { source: 'university.pdf', page: 1, text: 'three', distance: 0.4 },
  ])

  assert.deepEqual(
    groups.map((group) => [group.source, group.page, group.passages.length]),
    [
      ['offer.pdf', 1, 1],
      ['offer.pdf', 2, 1],
      ['university.pdf', 1, 1],
    ],
  )
})

test('returns an empty array for missing or empty sources', () => {
  assert.deepEqual(groupSources(), [])
  assert.deepEqual(groupSources([]), [])
})

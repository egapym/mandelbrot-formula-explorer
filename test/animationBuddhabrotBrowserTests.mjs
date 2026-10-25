const result = document.getElementById('result')
const delay = (ms) => new Promise((resolve) => setTimeout(resolve, ms))
const assert = (condition, message) => { if (!condition) throw new Error(message) }
async function waitFor(check, message) {
  const deadline = Date.now() + 30000
  while (!check()) {
    if (Date.now() > deadline) throw new Error(message)
    await delay(25)
  }
}
function change(doc, id, value) {
  const input = doc.getElementById(id)
  if (typeof value === 'boolean') input.checked = value
  else input.value = value
  input.dispatchEvent(new doc.defaultView.Event('input', { bubbles: true }))
  input.dispatchEvent(new doc.defaultView.Event('change', { bubbles: true }))
}
function pixels(doc) {
  const canvas = doc.getElementById('mandelbrot-canvas')
  return canvas.getContext('2d').getImageData(0, 0, canvas.width, canvas.height).data
}
try {
  for (const gpu of [new URLSearchParams(location.search).get('gpu') === 'true']) {
    result.textContent = `Checking ${gpu ? 'GPU' : 'CPU'}…`
    const frame = document.createElement('iframe')
    frame.style.cssText = 'width:500px;height:500px'
    frame.src = '../index.html'
    const loaded = new Promise((resolve) => frame.addEventListener('load', resolve, { once: true }))
    document.body.append(frame)
    await loaded
    const doc = frame.contentDocument
    await waitFor(() => doc.getElementById('renderTimeValue').textContent.includes('ms'), 'Initial render missing')
    change(doc, 'max-iterations', '64')
    change(doc, 'buddha-iterations', '4096')
    change(doc, 'buddha-gpu', gpu)
    doc.getElementById('buddha-render').click()
    await waitFor(() => doc.getElementById('buddha-toggle').checked && !doc.getElementById('buddha-toggle').disabled,
      'Buddhabrot render missing')
    change(doc, 'buddha-fixed-samples', true)
    await waitFor(() => !doc.getElementById('buddha-toggle').disabled, 'Fixed Samples render missing')
    // Exact fixed-seed endpoint parity proves that preparation used the same
    // Buddhabrot kernel, sampling rectangle, palette, bands and exposure.
    const baseline = pixels(doc)
    change(doc, 'anim-enable', true)
    change(doc, 'anim-fps', '15')
    doc.getElementById('anim-prepare').click()
    await waitFor(() => doc.getElementById('anim-status').textContent === 'Ready to play from the beginning.',
      `Prepare failed: ${doc.getElementById('anim-status').textContent}`)
    doc.getElementById('anim-play').click()
    await waitFor(() => doc.getElementById('anim-play').textContent === 'Stop', 'Playback did not start')
    await waitFor(() => doc.getElementById('anim-play').textContent === 'Play', 'Playback did not finish')
    const actual = pixels(doc)
    assert(baseline.every((v, i) => v === actual[i]), `${gpu}: Buddhabrot endpoint pixels changed`)
    assert(actual.some((v, i) => i % 4 !== 3 && v > 0), 'Blank animation')
    assert(doc.getElementById('buddha-toggle').checked, 'Playback lost Buddhabrot View')
    doc.getElementById('anim-play').click()
    await waitFor(() => doc.getElementById('anim-play').textContent === 'Stop', 'Replay did not start')
    doc.getElementById('anim-play').click()
    await waitFor(() => doc.getElementById('anim-play').textContent === 'Play', 'Stop failed')
    change(doc, 'buddha-gamma', '2')
    await waitFor(() => doc.getElementById('anim-prepare').disabled === false, 'Settings did not invalidate animation')
    doc.getElementById('anim-prepare').click()
    await waitFor(() => doc.getElementById('anim-play').textContent === 'Stop', 'Second prepare did not start')
    doc.getElementById('anim-play').click()
    await waitFor(() => doc.getElementById('anim-play').textContent === 'Play', 'Preparation Stop failed')
    frame.remove()
  }
  result.textContent = `PASS: ${new URLSearchParams(location.search).get('gpu') === 'true' ? 'GPU' : 'CPU'} Buddhabrot animation, fixed-seed endpoint pixel parity, replay, playback Stop, setting invalidation, preparation Stop`
  document.documentElement.dataset.result = 'passed'
} catch (error) {
  result.textContent = error.stack
  document.documentElement.dataset.result = 'failed'
}

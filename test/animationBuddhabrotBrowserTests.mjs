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
async function editExposure(doc, phase) {
  // Allow the normal runner to finish rebuilding editable density for this
  // small fixture, independently of the animation's already enabled controls.
  await delay(500)
  for (const [id, value] of [['buddha-brightness', '0.3'], ['buddha-gamma', '3']]) {
    const input = doc.getElementById(id)
    assert(!input.disabled, `${id} disabled after ${phase}`)
    const original = input.value
    const before = pixels(doc)
    change(doc, id, value)
    await waitFor(() => pixels(doc).some((v, i) => v !== before[i]), `${id} did not update pixels after ${phase}`)
    change(doc, id, original)
    await waitFor(() => pixels(doc).every((v, i) => v === before[i]), `${id} reset did not restore density colors after ${phase}`)
  }
}

async function switchView(doc, phase) {
  await delay(500)
  assert(!doc.getElementById('anim-play').disabled, `Rebuilt density discarded prepared replay after ${phase}`)
  const canvas = doc.getElementById('mandelbrot-canvas')
  const toggle = doc.getElementById('buddha-toggle')
  const held = pixels(doc)
  const prototype = doc.defaultView.CanvasRenderingContext2D.prototype
  const originalDraw = prototype.drawImage
  let normalDraws = 0
  prototype.drawImage = function (...args) {
    if (this.canvas === canvas) normalDraws++
    return originalDraw.apply(this, args)
  }
  try {
    assert(!toggle.disabled, `View disabled after ${phase}`)
    toggle.checked = false
    toggle.dispatchEvent(new doc.defaultView.Event('input', { bubbles: true }))
    // Native checkbox events and saved-bitmap waits can yield to cleanup UI
    // updates before change runs. Reproduce that ordering explicitly.
    await Promise.resolve()
    toggle.dispatchEvent(new doc.defaultView.Event('change', { bubbles: true }))
    // Exclude Stop's synchronous density flush; require a later ordinary frame.
    normalDraws = 0
    const switching = pixels(doc)
    assert(switching.some((v, i) => i % 4 === 3 && v > 0), `View OFF exposed a blank/white canvas after ${phase}`)
    await waitFor(() => normalDraws > 0, `View OFF did not render the ordinary fractal after ${phase}`)
    await delay(200)
    const ordinary = pixels(doc)
    assert(ordinary.some((v, i) => i % 4 !== 3 && v > 0), 'View OFF left a blank canvas')
    assert(ordinary.some((v, i) => v !== held[i]), `View OFF retained Buddhabrot after ${phase}`)
    await waitFor(() => !toggle.disabled, 'View remained disabled after cleanup')
    change(doc, 'buddha-toggle', true)
    await waitFor(() => toggle.checked && !toggle.disabled, `View ON failed after ${phase}`)
    const restored = pixels(doc)
    assert(held.every((v, i) => restored[i] === v), `View ON did not restore the held frame after ${phase}`)
  } finally { prototype.drawImage = originalDraw }
}

async function navigate(doc, gesture, phase) {
  assert(!doc.getElementById('buddha-toggle').disabled, `View toggle disabled after ${phase}`)
  const canvas = doc.getElementById('mandelbrot-canvas')
  const win = doc.defaultView
  const rect = canvas.getBoundingClientRect()
  const x = rect.x + rect.width * 0.6, y = rect.y + rect.height / 2
  const coordinate = doc.getElementById(gesture === 'wheel' ? 'coordZoom' : 'coordX')
  const before = coordinate.value
  if (gesture === 'wheel') {
    canvas.dispatchEvent(new win.WheelEvent('wheel', {
      bubbles: true, cancelable: true, clientX: x, clientY: y, deltaY: 100,
    }))
  } else {
    // Synthetic pointer events have no native pointer capture.
    const capture = canvas.setPointerCapture
    canvas.setPointerCapture = () => {}
    try {
      for (const [type, dx, buttons] of [['pointerdown', 0, 1], ['pointermove', 40, 1], ['pointerup', 40, 0]]) {
        canvas.dispatchEvent(new win.PointerEvent(type, {
          bubbles: true, cancelable: true, pointerId: 1, button: 0, buttons,
          clientX: x + dx, clientY: y + dx / 2,
        }))
      }
    } finally { canvas.setPointerCapture = capture }
  }
  await waitFor(() => coordinate.value !== before, `${gesture} blocked after ${phase}`)
  await delay(350)
  await waitFor(() => doc.getElementById('buddha-toggle').checked && !doc.getElementById('buddha-toggle').disabled,
    `Buddhabrot navigation did not finish after ${phase}`)
  await waitFor(() => !doc.getElementById('anim-prepare').disabled, `Navigation did not invalidate animation after ${phase}`)
  assert(pixels(doc).some((v, i) => i % 4 !== 3 && v > 0), `Navigation left a blank canvas after ${phase}`)
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
    await switchView(doc, 'completion')
    await editExposure(doc, 'completion')
    await navigate(doc, 'wheel', 'completion')
    await navigate(doc, 'drag', 'completion')
    doc.getElementById('anim-prepare').click()
    await waitFor(() => doc.getElementById('anim-status').textContent === 'Ready to play from the beginning.',
      'Could not prepare again after completed animation navigation')
    doc.getElementById('anim-play').click()
    await waitFor(() => doc.getElementById('anim-play').textContent === 'Stop', 'Replay did not start')
    await waitFor(() => doc.getElementById('buddha-toggle').checked, 'Replay did not present a Buddhabrot frame')
    doc.getElementById('anim-play').click()
    await waitFor(() => doc.getElementById('anim-play').textContent === 'Play', 'Stop failed')
    await switchView(doc, 'playback Stop')
    await editExposure(doc, 'playback Stop')
    await navigate(doc, 'wheel', 'playback Stop')
    await navigate(doc, 'drag', 'playback Stop')
    change(doc, 'buddha-gamma', '2')
    await waitFor(() => doc.getElementById('anim-prepare').disabled === false, 'Settings did not invalidate animation')
    doc.getElementById('anim-prepare').click()
    await waitFor(() => doc.getElementById('anim-play').textContent === 'Stop', 'Second prepare did not start')
    doc.getElementById('anim-play').click()
    await waitFor(() => doc.getElementById('anim-play').textContent === 'Play', 'Preparation Stop failed')
    await navigate(doc, 'wheel', 'preparation Stop')
    await navigate(doc, 'drag', 'preparation Stop')
    frame.remove()
  }
  result.textContent = `PASS: ${new URLSearchParams(location.search).get('gpu') === 'true' ? 'GPU' : 'CPU'} Buddhabrot animation, fixed-seed endpoint pixel parity, replay, playback Stop, setting invalidation, preparation Stop, Brightness/Gamma, View OFF/ON and wheel/drag after completion and Stop`
  document.documentElement.dataset.result = 'passed'
} catch (error) {
  result.textContent = error.stack
  document.documentElement.dataset.result = 'failed'
}

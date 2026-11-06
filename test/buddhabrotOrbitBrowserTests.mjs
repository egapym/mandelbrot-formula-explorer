const result = document.getElementById('result')
const delay = (ms) => new Promise((resolve) => setTimeout(resolve, ms))
async function waitFor(check, message) {
  const deadline = performance.now() + 20000
  while (!check()) {
    if (performance.now() > deadline) throw new Error(message)
    await delay(25)
  }
}
function toggle(doc, id, checked) {
  const input = doc.getElementById(id)
  input.checked = checked
  input.dispatchEvent(new doc.defaultView.Event('change', { bubbles: true }))
}
function snapshot(canvas) {
  return canvas.getContext('2d').getImageData(0, 0, canvas.width, canvas.height).data
}
function assertOrbit(canvas, previous, label) {
  const pixels = snapshot(canvas)
  if (!pixels.some((value, i) => i % 4 === 3 && value > 0)) throw new Error(`${label}: orbit disappeared`)
  if (!pixels.some((value, i) => value !== previous[i])) throw new Error(`${label}: orbit did not follow navigation`)
}
try {
  const gpu = new URLSearchParams(location.search).get('gpu') === 'true'
  const passed = []
  for (const julia of [false, true]) {
    const frame = document.createElement('iframe')
    frame.style.cssText = 'width:1000px;height:800px;border:0;display:block'
    frame.src = '../index.html'
    const loaded = new Promise((resolve) => frame.addEventListener('load', resolve, { once: true }))
    document.body.append(frame)
    await loaded
    const doc = frame.contentDocument
    const win = doc.defaultView
    await waitFor(() => doc.getElementById('renderTimeValue').textContent.includes('ms'), 'Initial render missing')
    toggle(doc, 'gpu', gpu)
    toggle(doc, 'buddha-gpu', gpu)
    if (gpu && !await win.navigator.gpu?.requestAdapter()) throw new Error('GPU adapter unavailable')
    if (julia) toggle(doc, 'julia-toggle', true)
    doc.getElementById('buddha-iterations').value = '2048'
    doc.getElementById('buddha-render').click()
    const view = doc.getElementById('buddha-toggle')
    await waitFor(() => view.checked && !view.disabled, 'Buddhabrot render did not complete')
    toggle(doc, 'orbit-toggle', true)
    const canvas = doc.getElementById(julia ? 'julia-canvas' : 'mandelbrot-canvas')
    const overlay = doc.getElementById(julia ? 'julia-orbit-canvas' : 'orbit-canvas')
    // Synthetic pointers cannot acquire native capture.
    canvas.setPointerCapture = () => {}
    for (const pinned of [false, true]) {
      const rect = canvas.getBoundingClientRect()
      const x = rect.x + rect.width * 0.55
      const y = rect.y + rect.height * 0.55
      canvas.dispatchEvent(new win.MouseEvent('mousemove', { clientX: x, clientY: y, bubbles: true }))
      const pointer = (type, dx, dy, pointerId = 1) => canvas.dispatchEvent(new win.PointerEvent(type, {
        bubbles: true, cancelable: true, pointerId, button: 0, buttons: type === 'pointerup' ? 0 : 1,
        clientX: x + dx, clientY: y + dy,
      }))
      if (pinned) {
        pointer('pointerdown', 0, 0)
        pointer('pointerup', 0, 0)
      }
      for (const gesture of ['pan', 'zoom', 'pinch']) {
        const label = `GPU ${gpu}, ${julia ? 'Julia' : 'main'}, ${pinned ? 'pinned' : 'hover'}, ${gesture}`
        const before = snapshot(overlay)
        if (gesture === 'zoom') {
          canvas.dispatchEvent(new win.WheelEvent('wheel', {
            bubbles: true, cancelable: true, clientX: x, clientY: y, deltaY: 30,
          }))
        } else {
          pointer('pointerdown', 0, 0)
          if (gesture === 'pinch') pointer('pointerdown', 60, 0, 2)
          pointer('pointermove', 15, 10)
        }
        assertOrbit(overlay, before, `${label} during gesture`)
        if (gesture !== 'zoom') {
          pointer('pointerup', 15, 10)
          if (gesture === 'pinch') pointer('pointerup', 60, 0, 2)
        }
        await delay(600)
        await waitFor(() => view.checked && !view.disabled, `${label}: final render did not complete`)
        assertOrbit(overlay, before, `${label} after render`)
        passed.push(`PASS: ${label}`)
        result.textContent = passed.join('\n')
      }
    }
    toggle(doc, 'orbit-toggle', false)
    if (snapshot(overlay).some((value, i) => i % 4 === 3 && value > 0)) throw new Error('Orbit OFF did not clear overlay')
    frame.remove()
  }
  result.textContent += '\nALL PASSED'
} catch (error) {
  result.textContent += `\nFAIL: ${error.stack || error}`
  console.error(error)
}

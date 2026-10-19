// Integration regression: run against the real controls, resize observers and renderers.
const result = document.getElementById('result')
const delay = (ms) => new Promise((resolve) => setTimeout(resolve, ms))
async function waitFor(check, message) {
  const deadline = performance.now() + 15000
  while (!check()) {
    if (performance.now() > deadline) throw new Error(message)
    await delay(25)
  }
}
function setToggle(doc, id, checked) {
  const toggle = doc.getElementById(id)
  toggle.checked = checked
  toggle.dispatchEvent(new doc.defaultView.Event('change', { bubbles: true }))
}
function hasRenderedPixels(doc, id) {
  const canvas = doc.getElementById(id)
  const pixels = canvas.getContext('2d').getImageData(0, 0, canvas.width, canvas.height).data
  let opaque = 0
  let colored = 0
  for (let i = 0; i < pixels.length; i += 4) {
    if (pixels[i + 3] === 255) opaque++
    if (pixels[i] !== pixels[i + 1] || pixels[i + 1] !== pixels[i + 2]) colored++
  }
  return opaque === canvas.width * canvas.height && colored > 0
}

try {
  const passed = []
  for (const gpu of [false, true]) {
    for (const width of [1500, 500]) {
      const frame = document.createElement('iframe')
      frame.style.cssText = `width:${width}px;height:900px;border:0;display:block`
      frame.src = '../index.html'
      const loaded = new Promise((resolve) => frame.addEventListener('load', resolve, { once: true }))
      document.body.append(frame)
      try {
        await loaded
        const doc = frame.contentDocument
        await waitFor(() => doc.getElementById('mandelbrot-canvas').width > 1, 'App did not initialize')
        setToggle(doc, 'gpu', gpu)
        setToggle(doc, 'buddha-gpu', gpu)
        setToggle(doc, 'julia-toggle', true)
        await waitFor(() => hasRenderedPixels(doc, 'mandelbrot-canvas') && hasRenderedPixels(doc, 'julia-canvas'), 'Initial Julia render missing')
        await delay(750)
        // Separate frames exercise the original 500ms suppression bug; same-frame
        // changes exercise queued toggle callbacks acting on the latest layout.
        for (const interval of [75, 0]) {
          doc.getElementById('buddha-render').click()
          await waitFor(() => doc.getElementById('julia-progress-canvas').style.display === 'block', 'Buddhabrot did not start')
          setToggle(doc, 'julia-toggle', false)
          if (interval) await delay(interval)
          setToggle(doc, 'julia-toggle', true)
          await delay(750)
          await waitFor(() => hasRenderedPixels(doc, 'mandelbrot-canvas') && hasRenderedPixels(doc, 'julia-canvas'), 'Blank view after rapid OFF → ON')
          const viewToggle = doc.getElementById('buddha-toggle')
          if (viewToggle.checked || !viewToggle.disabled) throw new Error('Buddhabrot view was not cleared')
        }
        setToggle(doc, 'julia-toggle', false)
        await delay(750)
        await waitFor(() => hasRenderedPixels(doc, 'mandelbrot-canvas'), 'Main view missing after Julia OFF')
        passed.push(`PASS: ${width}px, GPU ${gpu ? 'ON (fallback allowed)' : 'OFF'}, rapid toggles and final OFF`)
        result.textContent = passed.join('\n') + '\nRunning…'
      } finally {
        frame.remove()
      }
    }
  }
  result.textContent = passed.join('\n')
} catch (error) {
  result.textContent = `FAIL: ${error.stack || error}`
  throw error
}

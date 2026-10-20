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
  const params = new URLSearchParams(location.search)
  const gpuModes = params.has('gpu') ? [params.get('gpu') === 'true'] : [false, true]
  const widths = params.has('width') ? [Number(params.get('width'))] : [1500, 500]
  for (const gpu of gpuModes) {
    for (const width of widths) {
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
        // Source navigation updates Julia c while keeping Buddhabrot visible.
        doc.getElementById('buddha-iterations').value = '512'
        const canvas = doc.getElementById('mandelbrot-canvas')
        const win = doc.defaultView
        const mouse = (type, x, y, buttons = 0) => canvas.dispatchEvent(new win.MouseEvent(type, {
          bubbles: true, clientX: x, clientY: y, buttons,
        }))
        const touch = (type, points) => canvas.dispatchEvent(new win.TouchEvent(type, {
          bubbles: true, cancelable: true,
          touches: points.map(([clientX, clientY], identifier) => new win.Touch({
            identifier, target: canvas, clientX, clientY, pageX: clientX, pageY: clientY,
          })),
        }))
        for (const gesture of ['wheel', 'drag', 'touch pan', 'pinch']) {
          doc.getElementById('buddha-render').click()
          const viewToggle = doc.getElementById('buddha-toggle')
          await waitFor(() => viewToggle.checked && !viewToggle.disabled, 'Julia Buddhabrot did not complete')
          const rect = canvas.getBoundingClientRect()
          const x = rect.x + rect.width * 0.6
          const y = rect.y + rect.height / 2
          const juliaCanvas = doc.getElementById('julia-canvas')
          const previousImage = juliaCanvas.toDataURL()
          mouse('mousemove', x, y)
          if (!viewToggle.checked || viewToggle.disabled) throw new Error('Hover cleared Buddhabrot')
          if (gesture === 'wheel') {
            canvas.dispatchEvent(new win.WheelEvent('wheel', { bubbles: true, cancelable: true, clientX: x, clientY: y, deltaY: -100 }))
          } else if (gesture === 'drag') {
            mouse('mousedown', x, y, 1)
            mouse('mousemove', x + 20, y + 10, 1)
            mouse('mouseup', x + 20, y + 10)
          } else if (gesture === 'touch pan') {
            touch('touchstart', [[x, y]])
            touch('touchmove', [[x + 20, y + 10]])
            touch('touchend', [])
          } else {
            touch('touchstart', [[x - 20, y], [x + 20, y]])
            touch('touchmove', [[x - 30, y], [x + 30, y]])
            touch('touchend', [])
          }
          if (!viewToggle.checked) throw new Error(`${gesture}: Buddhabrot View was turned OFF`)
          await delay(750)
          await waitFor(() => viewToggle.checked && !viewToggle.disabled && juliaCanvas.toDataURL() !== previousImage, `${gesture}: Julia Buddhabrot was not updated`)
          await waitFor(() => hasRenderedPixels(doc, 'mandelbrot-canvas'), `${gesture}: source fractal render missing`)
          if (!viewToggle.checked) throw new Error(`${gesture}: Buddhabrot View was not preserved`)
        }
        // Keep moving more frequently than the old 150ms debounce. Julia must
        // update repeatedly before mouseup, even while source renders restart.
        {
          // A full sample budget must also present partial density during drag.
          doc.getElementById('buddha-iterations').value = '3000000'
          const rect = canvas.getBoundingClientRect()
          const x = rect.x + rect.width / 2
          const y = rect.y + rect.height / 2
          const juliaCanvas = doc.getElementById('julia-canvas')
          let previousImage = juliaCanvas.toDataURL()
          let updatesDuringDrag = 0
          mouse('mousedown', x, y, 1)
          try {
            for (let step = 1; step <= 20; step++) {
              mouse('mousemove', x + step * 2, y + step, 1)
              await delay(60)
              const image = juliaCanvas.toDataURL()
              if (image !== previousImage) updatesDuringDrag++
              previousImage = image
            }
          } finally {
            mouse('mouseup', x + 40, y + 20)
          }
          if (updatesDuringDrag < 2) throw new Error(`Continuous drag: only ${updatesDuringDrag} Julia updates before mouseup`)
          passed.push(`PASS: ${width}px, GPU ${gpu ? 'ON' : 'OFF'}, ${updatesDuringDrag} updates during sustained drag`)
        }
        setToggle(doc, 'julia-toggle', false)
        await delay(750)
        await waitFor(() => hasRenderedPixels(doc, 'mandelbrot-canvas'), 'Main view missing after Julia OFF')
        passed.push(`PASS: ${width}px, GPU ${gpu ? 'ON (fallback allowed)' : 'OFF'}, rapid toggles, source gestures, and final OFF`)
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

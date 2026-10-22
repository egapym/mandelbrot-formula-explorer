// Exercise the real toggle and GPU callbacks, including the synchronous frame
// that used to repaint ordinary offscreens from before Buddhabrot navigation.
import { decodeSettings, encodeSettings } from '../settingsState.mjs'

const result = document.getElementById('result')
const delay = (ms) => new Promise((resolve) => setTimeout(resolve, ms))
async function waitFor(check, message) {
  const deadline = performance.now() + 15000
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
function hasColoredPixels(canvas) {
  const pixels = canvas.getContext('2d').getImageData(0, 0, canvas.width, canvas.height).data
  for (let i = 0; i < pixels.length; i += 4) {
    if (pixels[i + 3] && (pixels[i] !== pixels[i + 1] || pixels[i + 1] !== pixels[i + 2])) return true
  }
  return false
}
try {
  const passed = []
  const params = new URLSearchParams(location.search)
  // Run ?gpu=false separately so each renderer mode starts in a fresh page.
  const gpuModes = [params.get('gpu') !== 'false']
  for (const gpu of gpuModes) {
    const frame = document.createElement('iframe')
    frame.style.cssText = `width:${params.get('width') || 1000}px;height:800px;border:0;display:block`
    frame.src = '../index.html'
    const loaded = new Promise((resolve) => frame.addEventListener('load', resolve, { once: true }))
    document.body.append(frame)
    await loaded
    const doc = frame.contentDocument
    const win = doc.defaultView
    const canvas = doc.getElementById('mandelbrot-canvas')
    await waitFor(() => canvas.width > 1 && doc.getElementById('renderTimeValue').textContent.includes('ms'), 'Initial render missing')
    toggle(doc, 'gpu', gpu)
    toggle(doc, 'buddha-gpu', gpu)
    doc.getElementById('buddha-iterations').value = '8192'
    await delay(500)
    if (gpu && !await win.navigator.gpu?.requestAdapter()) throw new Error('GPU adapter unavailable')
    // A fresh permalink starts Buddhabrot before ordinary worker/GPU jobs
    // finish. View OFF must render the fractal instead of restoring that
    // unfinished (possibly entirely transparent) initial canvas.
    const settings = decodeSettings(new URL(win.location.href).searchParams.get('params'))
    for (const julia of [false, true]) {
      const linkedFrame = document.createElement('iframe')
      linkedFrame.style.cssText = frame.style.cssText
      const url = new URL('../index.html', location.href)
      url.searchParams.set('params', encodeSettings({
        ...settings,
        buddhaView: true,
        controls: { ...settings.controls, 'julia-toggle': julia },
      }))
      linkedFrame.src = url.href
      const linkedLoaded = new Promise((resolve) => linkedFrame.addEventListener('load', resolve, { once: true }))
      document.body.append(linkedFrame)
      await linkedLoaded
      const linkedDoc = linkedFrame.contentDocument
      const view = linkedDoc.getElementById('buddha-toggle')
      const targetCanvas = linkedDoc.getElementById(julia ? 'julia-canvas' : 'mandelbrot-canvas')
      await waitFor(() => view.checked && !view.disabled, 'Permalink Buddhabrot did not complete')
      for (let cycle = 0; cycle < 2; cycle++) {
        toggle(linkedDoc, 'buddha-toggle', false)
        await waitFor(() => !view.checked && hasColoredPixels(targetCanvas), `Permalink View OFF left ${julia ? 'Julia' : 'main'} canvas blank`)
        if (cycle === 0) {
          toggle(linkedDoc, 'buddha-toggle', true)
          await waitFor(() => view.checked && hasColoredPixels(targetCanvas), 'View ON did not restore Buddhabrot')
        }
      }
      passed.push(`PASS: GPU ${gpu ? 'ON' : 'OFF'}, ${julia ? 'Julia' : 'main'} permalink: View OFF restores colored fractal`)
      result.textContent = passed.join('\n')
      linkedFrame.remove()
    }
    let draws = 0
    let cachedDraws = 0
    const prototype = win.CanvasRenderingContext2D.prototype
    const original = prototype.drawImage
    prototype.drawImage = function (...args) {
      if (this.canvas === canvas) {
        draws++
        if (new Error().stack.includes('Offscreen.render')) cachedDraws++
      }
      return original.apply(this, args)
    }
    for (const gesture of ['zoom', 'pan']) {
      doc.getElementById('buddha-render').click()
      const view = doc.getElementById('buddha-toggle')
      await waitFor(() => view.checked && !view.disabled, 'Buddhabrot did not complete')
      await delay(250)
      const rect = canvas.getBoundingClientRect()
      const x = rect.x + rect.width * 0.6
      const y = rect.y + rect.height / 2
      const previousZoom = doc.getElementById('coordZoom').value
      const previousX = doc.getElementById('coordX').value
      if (gesture === 'zoom') {
        canvas.dispatchEvent(new win.WheelEvent('wheel', {
          bubbles: true, cancelable: true, clientX: x, clientY: y, deltaY: 100,
        }))
      } else {
        // Synthetic pointer events cannot acquire native capture.
        canvas.setPointerCapture = () => {}
        for (const [type, dx, buttons] of [['pointerdown', 0, 1], ['pointermove', 40, 1], ['pointerup', 40, 0]]) {
          canvas.dispatchEvent(new win.PointerEvent(type, {
            bubbles: true, cancelable: true, pointerId: 1, button: 0, buttons,
            clientX: x + dx, clientY: y + dx / 2,
          }))
        }
      }
      await delay(600)
      await waitFor(() => view.checked && !view.disabled, 'Navigation did not complete')
      if (gesture === 'zoom' ? doc.getElementById('coordZoom').value === previousZoom : doc.getElementById('coordX').value === previousX) {
        throw new Error(`${gesture} did not change coordinates`)
      }
      draws = 0
      cachedDraws = 0
      toggle(doc, 'buddha-toggle', false)
      if (cachedDraws !== 0) throw new Error(`Stale ordinary frame was drawn synchronously after ${gesture}: ${cachedDraws} draws`)
      // Stop flushes the current Buddhabrot density synchronously; count only
      // subsequent frames when waiting for the ordinary renderer.
      draws = 0
      await waitFor(() => draws > 0, 'Ordinary renderer did not present a fresh frame')
      await delay(500)
      const pixels = canvas.getContext('2d').getImageData(0, 0, canvas.width, canvas.height).data
      let colored = 0
      for (let i = 0; i < pixels.length; i += 4) {
        if (pixels[i] !== pixels[i + 1] || pixels[i + 1] !== pixels[i + 2]) colored++
      }
      if (!colored) throw new Error('Restored fractal has no colored pixels')
      passed.push(`PASS: GPU ${gpu ? 'ON' : 'OFF'}, ${gesture}: no stale frame; fresh colored fractal rendered`)
      result.textContent = passed.join('\n')
    }
    prototype.drawImage = original
    frame.remove()
  }
  result.textContent += '\nPASS: all View restoration checks'
} catch (error) {
  result.textContent += `\nFAIL: ${error.stack || error}`
  console.error(error)
}

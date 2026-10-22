import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import { runInNewContext } from 'node:vm'
import { BUDDHA_DEFAULT_GAMMA, BUDDHA_TARGET_BRIGHTNESS } from '../buddhabrotExposure.mjs'
import { CpuDensityColorMap } from '../buddhabrotCpuColorMap.mjs'

function render(values, gamma = 4.8, brightness = 1.2, background = 0) {
  const r = Float32Array.from([...values, ...new Array(background).fill(0)])
  const g = Float32Array.from(r, v => v * 0.5)
  const b = Float32Array.from(r, v => v * 0.1)
  const pixels = new Uint8ClampedArray(r.length * 4)
  new CpuDensityColorMap(r, g, b, r.length, 1, pixels).update(brightness, gamma)
  return pixels
}
const distributions = [
  new Array(1000).fill(1),
  Array.from({ length: 1000 }, (_, i) => 2 ** (i / 50 - 10)),
  [...new Array(999).fill(1), 1e12],
]
for (const gamma of [0.1, 0.8, 4.8, 20]) {
  for (const values of distributions) {
    const base = render(values, gamma)
    const mean = base.filter((_, i) => i % 4 === 0).reduce((a, b) => a + b, 0) / values.length / 255
    if (gamma === 4.8) assert.ok(Math.abs(mean - BUDDHA_TARGET_BRIGHTNESS) < 0.006, `Metered brightness drift: ${mean}`)
    for (const factor of [1e-12, 0.01, 100, 1e12]) {
      const scaled = render(values.map(v => v * factor), gamma)
      assert.ok(base.every((v, i) => Math.abs(v - scaled[i]) <= 3), `Density scaling changed brightness: ${factor}, gamma=${gamma}`)
    }
    const padded = render(values, gamma, 1.2, 5000)
    assert.deepEqual(padded.slice(0, base.length), base, 'Black background changed exposure')
    for (let i = base.length; i < padded.length; i += 4) assert.deepEqual([...padded.slice(i, i + 4)], [0, 0, 0, 255])
  }
}
const dark = render(distributions[1], 4.8, 0.6)
const normal = render(distributions[1])
const bright = render(distributions[1], 4.8, 2.4)
assert.ok(normal.some((v, i) => i % 4 < 3 && v > dark[i]))
assert.ok(normal.some((v, i) => i % 4 < 3 && v < bright[i]))
assert.ok(normal.every((v, i) => i % 4 === 3 || dark[i] <= v && v <= bright[i]))
assert.deepEqual([...render([0, 0])], [0, 0, 0, 255, 0, 0, 0, 255])
console.log('PASS: density scale invariance, sparse/dense/outlier metering, gamma range, background, manual brightness and empty frame')

const defaultGammaPixels = render(distributions[1], BUDDHA_DEFAULT_GAMMA)
assert.ok(defaultGammaPixels.some((v, i) => i % 4 < 3 && v > normal[i]), 'Lower default Gamma did not brighten the image')
assert.ok(defaultGammaPixels.every((v, i) => i % 4 === 3 || v >= normal[i]))
const gammaLow = render(distributions[1], 1)
const gammaHigh = render(distributions[1], 10)
assert.ok(gammaLow.some((v, i) => i % 4 < 3 && v > normal[i] && normal[i] > gammaHigh[i]), 'Gamma was canceled by auto exposure')
assert.ok(gammaLow.every((v, i) => i % 4 === 3 || v >= normal[i] && normal[i] >= gammaHigh[i]))
const channels = Float32Array.from(distributions[1])
const mapper = new CpuDensityColorMap(channels, channels, channels, channels.length, 1, new Uint8ClampedArray(channels.length * 4))
mapper.update(1.2, 0.1)
const scale = mapper.scale
mapper.update(1.2, 20)
assert.equal(mapper.scale, scale, 'Manual gamma changed metering')
console.log('PASS: dark default midpoint, independent Gamma and smooth Brightness exposure')

// Exercise the real UI scheduling/reset paths with frozen density after Stop.
const source = readFileSync(new URL('../index.js', import.meta.url), 'utf8')
const toggle = { checked: false }, control = { value: '3' }, output = { textContent: '3' }
let drawCalls = 0
const callbacks = []
const context = {
  document: { getElementById: id => id === 'buddha-toggle' ? toggle : id.endsWith('-value') ? output : control },
  buddhaRedrawScheduled: false, buddhaRunnerGeneration: 1,
  buddhaPreservedDisplay: true, buddhaRunner: { brightness: 1.2, gamma: 4.8 },
  BuddhabrotState: { active: false, isViewEnabled: () => false },
  requestAnimationFrame: callback => callbacks.push(callback),
  drawBuddhaDensityChannels: () => drawCalls++,
  redraw: () => assert.fail('Stopped density was replaced by ordinary fractal'),
  console,
}
const schedulingStart = source.indexOf('function scheduleBuddhaRedraw()')
runInNewContext(source.slice(schedulingStart, source.indexOf('let redrawTimeout', schedulingStart)), context)
context.scheduleBuddhaRedraw()
context.scheduleBuddhaRedraw()
assert.equal(callbacks.length, 1, 'Slider changes were not coalesced')
callbacks.shift()()
assert.equal(drawCalls, 1, 'View OFF prevented updating Stop-preserved density')
context.buddhaPreservedDisplay = false
context.scheduleBuddhaRedraw()
callbacks.shift()()
assert.equal(drawCalls, 1, 'Hidden Buddhabrot was repainted')
context.buddhaPreservedDisplay = true
context.scheduleBuddhaRedraw()
context.buddhaRunnerGeneration++
callbacks.shift()()
assert.equal(drawCalls, 1, 'A stale scheduled draw repainted a newer run')
const resetStart = source.indexOf('function resetControlToDefault(')
runInNewContext(source.slice(resetStart, source.indexOf('// ============================================================================', resetStart)), context)
context.applyDefaultAndRefresh('buddha-brightness', 1.2)
callbacks.shift()()
assert.equal(drawCalls, 2)
assert.equal(control.value, '1.2')
assert.equal(output.textContent, '1.2')
console.log('PASS: stopped-frame controls/reset, hidden-view guard, coalescing, stale draws and displayed values')

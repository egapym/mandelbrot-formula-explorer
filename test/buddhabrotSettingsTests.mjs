import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import { runInNewContext } from 'node:vm'

// Run the actual UI handler without canvas/GPU initialization.
const source = readFileSync(new URL('../index.js', import.meta.url), 'utf8')
const start = source.indexOf('function getBuddhaRenderSettingsSignature()')
const end = source.indexOf('/**\n * Buddhabrot の描画ジョブを開始する', start)
let renders = 0
let shown = false
const context = {
  DOM: { buddha: {
    iterations: { value: '1000' }, palette: { value: 'Preset-01' },
    mode: { value: 'buddha' }, bandMode: { value: 'perTrajectory' },
    samplingRange: { value: 'current' }, gpu: { checked: false },
  } },
  buddhaActive: false,
  BuddhabrotState: { isViewEnabled: () => shown },
  buddhaRenderSettingsSignature: null,
  buddhaRenderRequestGeneration: 0,
  buddhaRenderPending: false,
  startBuddhaRender: () => {
    renders++
    context.buddhaRenderRequestGeneration++
    context.buddhaRenderSettingsSignature = context.getBuddhaRenderSettingsSignature()
    return Promise.resolve()
  },
  console: { warn() {} },
}
runInNewContext(source.slice(start, end), context)
context.buddhaRenderSettingsSignature = context.getBuddhaRenderSettingsSignature()
const change = context.redrawBuddhaForRenderSettingChange
context.DOM.buddha.iterations.value = '2000'
change()
assert.equal(renders, 0, 'Ordinary fractal view must not start Buddhabrot')
context.buddhaActive = true
for (const [key, value] of [
  ['palette', 'Preset-02'], ['mode', 'antibuddha'],
  ['bandMode', 'perPoint'], ['samplingRange', 'initial'],
]) {
  const before = renders
  context.DOM.buddha[key].value = value
  change()
  assert.equal(renders, before + 1, `${key} must immediately redraw a visible Buddhabrot`)
  change()
  assert.equal(renders, before + 1, `${key} must not redraw again on duplicate change/blur`)
}
context.buddhaActive = false
shown = true
context.DOM.buddha.iterations.value = '8192'
change()
assert.equal(renders, 4, 'Samples must not redraw a stopped or running Buddhabrot')
context.DOM.buddha.palette.value = 'Preset-04'
change()
assert.equal(renders, 5, 'A stopped but displayed Buddhabrot must redraw on palette changes')
assert.ok(!source.includes("iterations?.addEventListener('input', redrawBuddhaForRenderSettingChange)"))

context.DOM.buddha.continuous = { checked: true }
context.redrawBuddhaForExecutionSettingChange()
assert.equal(renders, 6, 'Continuous ON must immediately restart a visible Buddhabrot')
context.DOM.buddha.continuous.checked = false
context.redrawBuddhaForExecutionSettingChange()
assert.equal(renders, 7, 'Continuous OFF must immediately restart a visible Buddhabrot')
context.DOM.buddha.continuous.checked = true
context.DOM.buddha.iterations.value = '1000000'
change()
assert.equal(renders, 7, 'Computed sample totals must not trigger continuous rerenders')
context.DOM.buddha.mode.value = 'buddha'
change()
assert.equal(renders, 8, 'Other render settings must still restart continuous mode')

context.DOM.buddha.gpu.checked = true
context.redrawBuddhaForExecutionSettingChange()
assert.equal(renders, 9, 'GPU ON must immediately restart a visible Buddhabrot')
context.DOM.buddha.gpu.checked = false
context.redrawBuddhaForExecutionSettingChange()
assert.equal(renders, 10, 'GPU OFF must immediately restart a visible Buddhabrot')

context.buddhaRenderPending = true
context.startBuddhaRender = () => Promise.reject(new Error('Older request failed'))
context.DOM.buddha.palette.value = 'Preset-03'
change()
context.buddhaRenderRequestGeneration++
await Promise.resolve()
assert.equal(context.buddhaRenderPending, true, 'An old request must not clear a newer request\'s pending state')
console.log('PASS: four render settings, Samples exclusion, view gating, duplicate events and stale failures')

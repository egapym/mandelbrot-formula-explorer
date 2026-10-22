import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import { runInNewContext } from 'node:vm'
import { FixedBuddhabrotSamples, getBuddhabrotSampleSeed } from '../buddhabrotSampling.mjs'

let seeds = 10
const fixed = new FixedBuddhabrotSamples(() => seeds++)
const first = { center: { x: -0.5, y: 0 }, zoom: 1, width: 800, height: 600 }
const original = fixed.apply(first, 'main')
const moved = fixed.apply({ ...first, center: { x: 5, y: 6 }, zoom: 20, height: 400 }, 'main')
assert.equal(original.sampleSeed, moved.sampleSeed)
assert.deepEqual(moved.samplingCenter, first.center)
assert.equal(moved.samplingZoom, 1)
assert.equal(moved.samplingAspect, 0.75)
assert.equal(moved.center.x, 5, 'Display projection must still move')
assert.equal(moved.zoom, 20)
assert.deepEqual(moved.convergenceView, { center: first.center, zoom: 1, width: 800, height: 600 },
  'Convergence must not change when the locked view is panned, zoomed or resized')
assert.equal(seeds, 11, 'Navigation must not request fresh entropy')
const julia = fixed.apply(first, 'julia')
assert.notEqual(julia.sampleSeed, original.sampleSeed)
const streamSeeds = Array.from({ length: 20 }, (_, i) => getBuddhabrotSampleSeed(original.sampleSeed, i))
assert.equal(new Set(streamSeeds).size, 20, 'Workers/batches must sample different coordinates')
assert.deepEqual(streamSeeds, Array.from({ length: 20 }, (_, i) => getBuddhabrotSampleSeed(moved.sampleSeed, i)))
fixed.clear()
assert.notEqual(fixed.apply(first, 'main').sampleSeed, original.sampleSeed)

// Enabling a lock adopts the last render's seed without requesting entropy.
const inherited = new FixedBuddhabrotSamples(() => seeds++)
const beforeAdopt = seeds
const adopted = inherited.apply({ ...first, sampleSeed: 123456 }, 'main')
assert.equal(adopted.sampleSeed, 123456)
assert.equal(seeds, beforeAdopt)
inherited.reseed('main')
const refreshed = inherited.apply({ ...first, center: { x: 8, y: 9 }, zoom: 10 }, 'main')
assert.notEqual(refreshed.sampleSeed, adopted.sampleSeed)
assert.deepEqual(refreshed.samplingCenter, first.center, 'Manual reseeding must keep the fixed sampling range')
assert.equal(refreshed.samplingZoom, first.zoom)
assert.equal(inherited.apply(first, 'main').sampleSeed, refreshed.sampleSeed)

// Exercise actual toggle synchronization, including restoring incompatible settings.
const source = readFileSync(new URL('../index.js', import.meta.url), 'utf8')
const context = {
  DOM: { buddha: {
    iterations: { value: '1000', readOnly: false },
    regenerateSeed: { disabled: false },
    continuous: { checked: true }, fixedSamples: { checked: true, disabled: false },
  } },
  performance: { now: () => 0 }, fixedBuddhaSamples: fixed,
  syncBuddhaPointBatchSize() {},
}
runInNewContext(source.slice(source.indexOf('let buddhaFiniteSampleValue = null'),
  source.indexOf('function wrapContinuousBuddhaRunner(')), context)
context.syncBuddhaContinuousModeUi()
assert.equal(context.DOM.buddha.fixedSamples.checked, false)
assert.equal(context.DOM.buddha.fixedSamples.disabled, true)
assert.equal(fixed.views.size, 0)
assert.equal(context.DOM.buddha.regenerateSeed.disabled, true)
context.DOM.buddha.continuous.checked = false
context.syncBuddhaContinuousModeUi()
assert.equal(context.DOM.buddha.fixedSamples.disabled, false)
assert.equal(context.DOM.buddha.fixedSamples.checked, false, 'Disabling Continuous must not restore the old lock')
context.DOM.buddha.fixedSamples.checked = true
context.syncBuddhaFixedSamplesUi()
assert.equal(context.DOM.buddha.regenerateSeed.disabled, false)
console.log('PASS: fixed sample ranges, deterministic streams, independent targets, reset and Continuous exclusion')

// Sampling Range must release both targets before requesting a render, even when hidden.
let rangeChange
let rangeRedraws = 0
const rangeContext = {
  DOM: { buddha: {
    fixedSamples: { checked: true },
    samplingRange: { addEventListener(event, handler) {
      assert.equal(event, 'change')
      rangeChange = handler
    } },
  } },
  fixedBuddhaSamples: fixed,
  syncBuddhaFixedSamplesUi() {},
  redrawBuddhaForRenderSettingChange() {
    assert.equal(rangeContext.DOM.buddha.fixedSamples.checked, false)
    assert.equal(fixed.views.size, 0, 'Both main and Julia locks must clear before redraw')
    rangeRedraws++
  },
}
fixed.apply(first, 'main')
fixed.apply(first, 'julia')
const rangeStart = source.indexOf("  DOM.buddha.samplingRange?.addEventListener('change', () => {")
assert.ok(rangeStart >= 0)
runInNewContext(source.slice(rangeStart, source.indexOf('// Buddhabrot 実行ボタン', rangeStart)), rangeContext)
rangeChange()
assert.equal(rangeRedraws, 1)
assert.deepEqual(fixed.apply({ ...first, zoom: 5 }, 'main').samplingZoom, 5,
  'A later lock must adopt the newly selected range')
console.log('PASS: Sampling Range releases Fixed Samples and clears both target ranges before redraw')

// Render replays the locked seed; only the explicit seed button changes it.
const callbacks = {}
const renderFixed = new FixedBuddhabrotSamples(() => seeds++)
const locked = renderFixed.apply(first, 'main')
const otherTarget = renderFixed.apply(first, 'julia')
let rendered
let redraws = 0
const buttonContext = {
  DOM: { buddha: {
    fixedSamples: { checked: true }, continuous: { checked: false },
    regenerateSeed: { addEventListener(event, callback) { callbacks.reseed = callback } },
  } },
  document: { getElementById() {
    return { addEventListener(event, callback) { callbacks.render = callback } }
  } },
  fixedBuddhaSamples: renderFixed,
  getBuddhabrotTargetSnapshot: () => ({ kind: 'main' }),
  async startBuddhaRender() { rendered = renderFixed.apply(first, 'main') },
  redrawBuddhaForExecutionSettingChange() { redraws++ },
  console,
}
const buttonStart = source.indexOf("  DOM.buddha.regenerateSeed?.addEventListener('click'")
runInNewContext(source.slice(buttonStart, source.indexOf('// Buddhabrot 停止ボタン', buttonStart)), buttonContext)
await callbacks.render()
await callbacks.render()
assert.equal(rendered.sampleSeed, locked.sampleSeed, 'Repeated Render must keep the locked seed')
callbacks.reseed()
assert.equal(redraws, 1, 'Explicit reseeding requests a visible-view refresh')
await callbacks.render()
assert.notEqual(rendered.sampleSeed, locked.sampleSeed)
assert.deepEqual(rendered.samplingCenter, locked.samplingCenter)
assert.equal(renderFixed.apply(first, 'julia').sampleSeed, otherTarget.sampleSeed)
const newSeed = rendered.sampleSeed
await callbacks.render()
assert.equal(rendered.sampleSeed, newSeed, 'Render must keep the manually regenerated seed')
buttonContext.DOM.buddha.continuous.checked = true
callbacks.reseed()
buttonContext.DOM.buddha.continuous.checked = false
buttonContext.DOM.buddha.fixedSamples.checked = false
callbacks.reseed()
assert.equal(redraws, 1, 'Reseeding requires Fixed Samples ON and Continuous OFF')
console.log('PASS: Render retains locked seeds; explicit reseeding preserves range and the other target')

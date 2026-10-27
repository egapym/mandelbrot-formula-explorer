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

// Exercise actual toggle synchronization, including preserving the fixed sample lock.
const source = readFileSync(new URL('../index.js', import.meta.url), 'utf8')
// Exercise the range selection used by both normal and realtime rendering.
const samplingContext = {
  DOM: { buddha: { samplingRange: { value: 'initial' } }, fractalTypeSelect: { value: 'preset:0' } },
  functionPresets: [{ coordX: -0.5, coordY: 0 }],
  getInitialFractalZoom: () => ({ toNumber: () => 1 }),
}
const samplingStart = source.indexOf('function getBuddhaSamplingRange(')
runInNewContext(source.slice(samplingStart, source.indexOf('function applyFixedBuddhaSamples(', samplingStart)), samplingContext)
function resolveRange(zoom, kind = 'main') {
  const renderer = { precision: 64, zoom: { toNumber: () => zoom } }
  const range = samplingContext.getBuddhaSamplingRange(renderer, kind)
  return JSON.parse(JSON.stringify(range))
}
for (const kind of ['main', 'julia']) {
  for (const zoom of [0.01, 0.9566303671497955, 1, 1.0231724767554273, 10]) {
    assert.deepEqual(resolveRange(zoom, kind), {
      samplingCenter: { x: 0, y: 0 }, samplingZoom: 0.5, samplingRadius: 4,
    }, 'Initial View must always sample the origin-centered radius-4 disk')
  }
}
samplingContext.DOM.fractalTypeSelect.value = 'preset:-1'
samplingContext.getInitialFractalZoom = () => ({ toNumber: () => 1.5 })
assert.equal(resolveRange(0.5).samplingRadius, 4, 'Custom presets and embedded zoom must not change the disk')
samplingContext.DOM.buddha.samplingRange.value = 'current'
for (const zoom of [0.5, 1, 10]) assert.deepEqual(resolveRange(zoom), {})
const diskLock = new FixedBuddhabrotSamples(() => 12345)
const diskParams = { ...first, samplingCenter: { x: 0, y: 0 }, samplingZoom: 0.5, samplingRadius: 4 }
diskLock.apply(diskParams, 'main')
assert.equal(diskLock.apply({ ...first, zoom: 10 }, 'main').samplingRadius, 4,
  'Fixed Samples must retain the disk when display/settings change')
const rectangleLock = new FixedBuddhabrotSamples(() => 12345)
rectangleLock.apply(first, 'main')
assert.equal(rectangleLock.apply(diskParams, 'main').samplingRadius, undefined,
  'A fixed rectangular range must not inherit a later disk setting')
console.log('PASS: Initial View always selects the radius-4 disk; Current View and Fixed Samples retain their ranges')

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
const lockedViews = new Map(fixed.views)
context.syncBuddhaContinuousModeUi()
assert.equal(context.DOM.buddha.fixedSamples.checked, true)
assert.equal(context.DOM.buddha.fixedSamples.disabled, true)
assert.deepEqual(fixed.views, lockedViews, 'Continuous must retain the fixed seeds and sampling ranges')
assert.equal(context.DOM.buddha.regenerateSeed.disabled, true)
context.DOM.buddha.continuous.checked = false
context.syncBuddhaContinuousModeUi()
assert.equal(context.DOM.buddha.fixedSamples.disabled, false)
assert.equal(context.DOM.buddha.fixedSamples.checked, true, 'Disabling Continuous must keep Fixed Samples ON')
assert.deepEqual(fixed.views, lockedViews)
assert.equal(context.DOM.buddha.regenerateSeed.disabled, false)
context.DOM.buddha.fixedSamples.checked = false
context.DOM.buddha.continuous.checked = true
context.syncBuddhaContinuousModeUi()
context.DOM.buddha.continuous.checked = false
context.syncBuddhaContinuousModeUi()
assert.equal(context.DOM.buddha.fixedSamples.checked, false, 'Continuous must also preserve Fixed Samples OFF')
assert.equal(context.DOM.buddha.regenerateSeed.disabled, true)
console.log('PASS: fixed sample ranges, deterministic streams, independent targets, reset and lock preservation across Continuous')

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

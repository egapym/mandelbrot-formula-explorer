import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import vm from 'node:vm'
import { BUDDHA_PALETTES } from '../buddhaPalettes.mjs'
import { DEFAULT_RENDER_POINT_BATCH_SIZE, normalizeRenderPointBatchSize } from '../buddhabrotRenderConfig.mjs'
import { compileIterationFunction } from '../customFunctionParser.mjs'
import { getBuddhabrotProjection } from '../buddhabrotProjection.mjs'

const source = readFileSync(
  process.env.BUDDHA_WORKER_SOURCE || new URL('../buddhabrotWorker.mjs', import.meta.url),
  'utf8',
).replace(/^import .*$/gm, '')
// Observe the worker's actual contributions before Float32 buffering, so batching
// must preserve every hit and band color exactly even when rounded sums differ.
function observe(workerSource) {
  const instrumented = workerSource.replace(
    /local([RGB])\[idx\] \+= ([^\n]+)/g,
    (_, channel, value) => `local${channel}[idx] += recordContribution('${channel}', idx, ${value})`,
  ).replace(/(    const sampleIm = [^\n]+)/, '$1\n    recordSample(sampleRe, sampleIm)')
  return workerSource.includes('function* drawTrajectory')
    ? instrumented.replace('    const sampleRe =', '    const observedSampleId = nextSample - 1\n    const sampleRe =')
      .replace('      const pr = points[k * 2]', '      recordStep(observedSampleId, k)\n      const pr = points[k * 2]')
    : instrumented
}
const observedSource = observe(source)
const referenceSource = process.env.BUDDHA_REFERENCE_WORKER
  ? observe(readFileSync(process.env.BUDDHA_REFERENCE_WORKER, 'utf8').replace(/^import .*$/gm, ''))
  : null
const base = {
  width: 160,
  height: 90,
  samples: 512,
  maxIter: 1000,
  center: { x: -0.5, y: 0 },
  zoom: 1,
  escapeRadius: 4,
  z0Real: 1,
  z0Imag: 0,
  mode: 'antibuddha',
  iterationFunction: 'c*(z+1/(z^2))',
}

async function sample(options = {}, delay = 0, onYield = null, pointBatchSize = DEFAULT_RENDER_POINT_BATCH_SIZE, onPresentation = null) {
  const opts = { ...base, ...options }
  const density = Array.from({ length: 3 }, () => new Float64Array(opts.width * opts.height))
  const contributions = Object.fromEntries(
    ['R', 'G', 'B'].map((key) => [key, new Float64Array(opts.width * opts.height)]),
  )
  const hits = new Map()
  const stepFrames = []
  const sampledCoordinates = []
  let pendingSteps = []
  const progress = []
  const waits = []
  let chunks = 0
  let clock = 0
  const context = vm.createContext({
    self: {},
    console,
    compileIterationFunction,
    getBuddhabrotProjection,
    DEFAULT_RENDER_POINT_BATCH_SIZE,
    normalizeRenderPointBatchSize,
    recordSample(re, im) {
      sampledCoordinates.push([re, im])
    },
    recordContribution(channel, index, value) {
      contributions[channel][index] += value
      const key = `${channel}:${index}:${value}`
      hits.set(key, (hits.get(key) || 0) + 1)
      return value
    },
    recordStep(sample, step) {
      pendingSteps.push([sample, step])
    },
    Math: Object.assign(Object.create(Math), { random: () => 0.25 }),
    Date: { now: () => (clock += opts.clockStep ?? 8) },
    postMessage(message) {
      if (message.type === 'progress') progress.push(message.done)
      if (message.type !== 'chunk') return
      chunks++
      stepFrames.push(pendingSteps)
      pendingSteps = []
      const chunk = message.chunk
      for (const [channel, key] of ['r', 'g', 'b'].entries()) {
        for (let i = 0; i < chunk[key].length; i++) density[channel][chunk.indices[i]] += chunk[key][i]
      }
      if (chunk.presentationId !== undefined) {
        const chunkCount = chunks
        queueMicrotask(() => {
          assert.equal(chunks, chunkCount, 'Worker produced another batch before presentation')
          if (onPresentation) onPresentation(context, message)
          else context.self.onmessage({ data: {
            cmd: 'presented', jobId: message.jobId, presentationId: chunk.presentationId,
          } })
        })
      }
    },
    setTimeout(callback, ms) {
      waits.push(ms)
      queueMicrotask(() => {
        onYield?.(context, waits.length, ms)
        callback()
      })
    },
  })
  vm.runInContext(opts.workerSource || observedSource, context)
  vm.runInContext(`running = true; currentJobId = 1; renderDelay = ${delay}`, context)
  context.self.onmessage({ data: { cmd: 'setPointBatchSize', renderPointBatchSize: pointBatchSize } })
  await context.runSampling({
    ...opts,
    iterationFunctionCompiled: opts.iterationFunction ? compileIterationFunction(opts.iterationFunction) : null,
  })
  return { density, contributions, hits, progress, waits, chunks, stepFrames, sampledCoordinates }
}

function equalDensity(expected, actual, exact = false) {
  assert.deepEqual(actual.hits, expected.hits, 'Orbit hits or band colors changed')
  if (exact) assert.deepEqual(actual.density, expected.density, 'Immediate Float32 density changed')
  for (let c = 0; c < 3; c++) {
    for (let i = 0; i < expected.density[c].length; i++) {
      // Batching changes Float32 addition grouping, but must retain all hits and colors.
      const tolerance = Math.max(0.0001, Math.abs(expected.density[c][i]) * 0.0001)
      assert.ok(
        Math.abs(expected.density[c][i] - actual.density[c][i]) <= tolerance,
        `density channel ${c}, pixel ${i}: ${expected.density[c][i]} versus ${actual.density[c][i]}`,
      )
    }
  }
}

// Observe actual sampled coordinates: the same seed and saved sampling bounds
// must survive display pan/zoom, even when the canvas aspect ratio changes.
const seeded = await sample({ sampleSeed: 12345 })
const replayed = await sample({ sampleSeed: 12345 })
assert.deepEqual(seeded.sampledCoordinates, replayed.sampledCoordinates)
const movedSeeded = await sample({ sampleSeed: 12345, center: { x: 10, y: 20 }, zoom: 8,
  height: 80, samplingCenter: base.center, samplingZoom: base.zoom, samplingAspect: base.height / base.width })
assert.deepEqual(seeded.sampledCoordinates, movedSeeded.sampledCoordinates)
assert.notDeepEqual(seeded.sampledCoordinates, (await sample({ sampleSeed: 54321 })).sampledCoordinates)

// A long orbit may enter a deep zoom only after its first 200 points.
// Offscreen points in the current view must not truncate the locked orbit.
const lateEntry = { samples: 1, maxIter: 600, mode: 'antibuddha',
  iterationFunction: 'z+0.005', z0Real: 0, center: { x: 1.5, y: 0 }, zoom: 1000,
  sampleSeed: 12345, samplingCenter: { x: 0, y: 0 }, samplingZoom: 1,
  convergenceView: { center: { x: 0, y: 0 }, zoom: 1, width: 160, height: 90 } }
assert.ok((await sample(lateEntry)).hits.size > 0, 'Deep zoom discarded an orbit before it entered the view')
equalDensity(await sample({ ...lateEntry, convergenceView: undefined }), await sample(lateEntry))

// Preset-05's yellow/red bands occur after the first 200 orbit steps.
// Zooming OUT from a small locked grid must still color its offscreen steps.
const zoomOutPalette = { samples: 1, maxIter: 600, mode: 'antibuddha',
  iterationFunction: 'z+0.005', z0Real: 0, center: { x: 1.5, y: 0 }, zoom: 1,
  sampleSeed: 12345, samplingCenter: { x: 0, y: 0 }, samplingZoom: 1,
  paletteStops: BUDDHA_PALETTES[4], buddhaBandMode: 'perPoint' }
const widePalette = await sample(zoomOutPalette)
const lockedZoomOutPalette = await sample({ ...zoomOutPalette,
  convergenceView: { center: { x: 1.5, y: 0 }, zoom: 1000, width: 160, height: 90 } })
equalDensity(widePalette, lockedZoomOutPalette)
equalDensity(widePalette, await sample({ ...zoomOutPalette,
  convergenceView: { center: { x: 1.5, y: 0 }, zoom: 1000, width: 160, height: 90 } }, 10))
assert.ok(lockedZoomOutPalette.contributions.R.some(value => value > 0), 'Preset-05 lost its yellow/red bands outside the locked view')

const immediate = await sample()
const interactiveImmediate = await sample({ interactive: true })
equalDensity(immediate, interactiveImmediate)
assert.ok(interactiveImmediate.chunks > immediate.chunks, 'Interactive jobs did not present partial density sooner')
const continuousImmediate = await sample({ continuous: true })
equalDensity(immediate, continuousImmediate, true)
assert.equal(immediate.chunks, continuousImmediate.chunks, 'Finite jobs must share the continuous display cadence')
assert.deepEqual(immediate.waits, continuousImmediate.waits, 'Finite jobs must share the continuous yield cadence')
equalDensity(immediate, await sample({ continuous: true }, 10))
// Fixed random seeds prove that sampling stays at the initial view after pan/zoom,
// while drawing still projects orbit hits into the current display bounds.
for (const fractalType of ['mandelbrot', 'julia']) {
  for (const mode of ['buddha', 'antibuddha']) {
    const options = { fractalType, mode, iterationFunction: 'z*z+c', z0Real: 0, juliaRe: -0.7, juliaIm: 0.2 }
    const initial = await sample(options)
    assert.ok(initial.hits.size > 0, 'Initial view must have orbit hits for the projection check')
    const fixed = await sample({ ...options, center: { x: 100, y: 100 }, zoom: 8,
      samplingCenter: base.center, samplingZoom: base.zoom })
    assert.deepEqual(fixed.sampledCoordinates, initial.sampledCoordinates, 'Pan/zoom changed initial-view sampling')
    assert.equal(fixed.hits.size, 0, 'Orbit hits were projected into sampling bounds instead of display bounds')
    const current = await sample({ ...options, center: { x: 100, y: 100 }, zoom: 8 })
    assert.notDeepEqual(current.sampledCoordinates, initial.sampledCoordinates, 'Current-view sampling ignored pan/zoom')
    for (const [re, im] of current.sampledCoordinates) {
      assert.ok(re >= 99.75 && re <= 100.25 && im >= 99.859375 && im <= 100.140625)
    }
  }
}
const explicitCurrent = await sample({ samplingCenter: base.center, samplingZoom: base.zoom })
equalDensity(immediate, explicitCurrent, true)
console.log('PASS: initial-view sampling is independent of pan/zoom; orbit hits use current display bounds')
// Fast samples should not incur a timer on every sample-count checkpoint.
// Slow slices still need to accept Stop, even when they have visible density.
const fastSlices = await sample({ clockStep: 0 })
assert.equal(fastSlices.waits.length, 0, 'Cheap immediate samples paid timer roundtrips')
assert.equal(fastSlices.chunks, 1, 'Jobs shorter than the display interval must publish their final density')
equalDensity(immediate, fastSlices)
const stoppedImmediate = await sample({}, 0,
  (context) => context.self.onmessage({ data: { cmd: 'stop' } }))
assert.equal(stoppedImmediate.waits.length, 1, 'Immediate rendering ignored Stop at a slice boundary')
assert.ok(!stoppedImmediate.progress.includes(base.samples))
for (const delay of [0.01, 1, 10, 100]) {
  const delayed = await sample({}, delay)
  assert.ok(delayed.waits.length < 2000, `Long anti-buddhabrot orbits waited ${delayed.waits.length} times`)
  assert.ok(delayed.chunks > 0, 'Missing visible density')
  assert.equal(delayed.progress.at(-1), base.samples)
  equalDensity(immediate, delayed)
}

let cases = 4
for (const expression of [null, 'z*z+c', 'z*z+c*i', 'z*z+c+i', 'c*(z+1/(z^2))', 'z*z+c+0.1*zDelay(2)']) {
  for (const mode of ['buddha', 'antibuddha']) {
    for (const buddhaBandMode of ['perPoint', 'perTrajectory']) {
      for (const paletteStops of [null, ...BUDDHA_PALETTES]) {
        const opts = { iterationFunction: expression, mode, buddhaBandMode, paletteStops, samples: 128, maxIter: 240 }
        equalDensity(await sample(opts), await sample(opts, 10))
        if (referenceSource) equalDensity(await sample({ ...opts, workerSource: referenceSource }), await sample(opts), true)
        cases++
      }
    }
  }
}

for (const fractalType of ['julia', 'julia-custom']) {
  for (const mode of ['buddha', 'antibuddha']) {
    for (const maxIter of [1, 2, 240]) {
      const opts = {
        fractalType,
        mode,
        maxIter,
        samples: 128,
        juliaRe: -0.7,
        juliaIm: 0.2,
        iterationFunction: fractalType === 'julia-custom' ? 'z*z+c+0.1*zAt(0)' : null,
      }
      equalDensity(await sample(opts), await sample(opts, 10))
      if (referenceSource) equalDensity(await sample({ ...opts, workerSource: referenceSource }), await sample(opts), true)
      cases++
    }
  }
}

// Exercise Float32/Float64 trajectory storage, convergence boundaries and
// palette rounding at zero/negative ratios with non-default initial states.
for (const maxIter of [0, 199, 200, 201, 5000, 5001]) {
  for (const buddhaBandMode of ['perPoint', 'perTrajectory']) {
    const opts = {
      samples: 16, maxIter, buddhaBandMode, iterationFunction: 'z*z+c+0.1*zDelay(2)',
      z0Real: 0.1, z0Imag: -0.2, escapeRadius: 2,
      paletteStops: { bands: [
        { color: [201, 50, 255], ratio: 0.04 },
        { color: [0, 0, 0], ratio: 0 },
        { color: [17, 83, 129], ratio: -0.01 },
        { color: [230, 58, 15], ratio: 0.97 },
      ] },
    }
    equalDensity(await sample(opts), await sample(opts, 10))
    if (referenceSource) equalDensity(await sample({ ...opts, workerSource: referenceSource }), await sample(opts), true)
    cases++
  }
}

for (const options of [
  { iterationFunction: '-1', center: { x: 1, y: 0 }, zoom: 100 },
  { paletteStops: { bands: [{ color: [0, 0, 0], ratio: 1 }] } },
]) {
  const empty = await sample(options, 10)
  assert.equal(empty.chunks, 0)
  assert.ok(empty.waits.length > 0, 'Invisible trajectories never yielded')
  assert.ok(
    empty.waits.every((ms) => ms === 0),
    'Empty density incurred render delay',
  )
  assert.ok(
    empty.progress.some((done) => done > 0 && done < base.samples),
    'Missing intermediate progress',
  )
  const stopped = await sample(options, 10, (context) => context.self.onmessage({ data: { cmd: 'stop' } }))
  assert.equal(stopped.waits.length, 1, 'Stop did not interrupt invisible trajectories')
  assert.ok(!stopped.progress.includes(base.samples), 'Stopped job reported completion')
}

let speedChanged = false
const changedSpeed = await sample({}, 10, (context, _count, ms) => {
  if (ms > 0 && !speedChanged) {
    speedChanged = true
    context.self.onmessage({ data: { cmd: 'setSpeed', renderDelay: 0 } })
  }
})
assert.equal(changedSpeed.waits.filter((ms) => ms > 0).length, 1)
equalDensity(immediate, changedSpeed)

const batchOptions = { iterationFunction: 'z+0.005', z0Real: 0, samples: 16, maxIter: 128 }
const defaultBatch = await sample(batchOptions, 10)
const pointwise = await sample(batchOptions, 10, null, 1)
const maximumBatch = await sample(batchOptions, 10, null, 1024)
equalDensity(defaultBatch, pointwise)
equalDensity(defaultBatch, maximumBatch)
assert.ok(pointwise.chunks > defaultBatch.chunks, 'Smaller batches did not produce more updates')
assert.equal(defaultBatch.chunks, maximumBatch.chunks, 'Batch size above the sample count changed orbit speed')
for (const count of [1, 4, 16]) {
  const concurrent = await sample(batchOptions, 10, null, count)
  for (const [frame, steps] of concurrent.stepFrames.entries()) {
    assert.equal(steps.length, count, 'Unexpected number of concurrent trajectory steps')
    const firstOrbitStep = steps.find(([sample]) => sample === 0)?.[1]
    if (firstOrbitStep !== undefined) assert.equal(firstOrbitStep, frame, 'Batch size accelerated one orbit')
    assert.equal(new Set(steps.map(([sample]) => sample)).size, steps.length, 'One trajectory advanced twice in a frame')
  }
  assert.equal(concurrent.stepFrames.filter((steps) => steps.some(([sample]) => sample === 0)).length, 128)
  equalDensity(pointwise, concurrent)
}
assert.deepEqual((await sample(batchOptions, 10, null, 64)).waits, defaultBatch.waits)

// Exercise the actual ceiling with enough drawable orbits to fill every slot.
const boundaryOptions = { ...batchOptions, samples: 1025, maxIter: 4 }
const boundaryImmediate = await sample(boundaryOptions)
for (const requested of [1023, 1024, 1025]) {
  const concurrent = await sample(boundaryOptions, 10, null, requested)
  const effective = Math.min(requested, 1024)
  assert.equal(concurrent.stepFrames[0].length, effective, 'Worker did not draw the effective maximum batch')
  assert.equal(new Set(concurrent.stepFrames[0].map(([sample]) => sample)).size, effective)
  for (let frame = 0; frame < boundaryOptions.maxIter; frame++) {
    assert.equal(concurrent.stepFrames[frame].length, effective)
    assert.ok(concurrent.stepFrames[frame].every(([, step]) => step === frame),
      'Maximum batch did not advance each trajectory exactly one step per pass')
  }
  equalDensity(boundaryImmediate, concurrent)
}

const changedBatch = await sample(batchOptions, 10, (context, count) => {
  if (count === 1) context.self.onmessage({ data: { cmd: 'setPointBatchSize', renderPointBatchSize: 1024 } })
}, 1)
equalDensity(defaultBatch, changedBatch)
assert.ok(changedBatch.chunks < pointwise.chunks, 'Live batch-size change was not applied')
const resized = await sample(batchOptions, 10, (context, count, ms) => {
  if (ms > 0 && count === 1) context.self.onmessage({ data: { cmd: 'setPointBatchSize', renderPointBatchSize: 1 } })
  if (ms > 0 && count === 4) context.self.onmessage({ data: { cmd: 'setPointBatchSize', renderPointBatchSize: 8 } })
}, 4)
assert.deepEqual(resized.stepFrames.slice(0, 6).map((steps) => steps.length), [4, 1, 1, 1, 8, 8])
assert.deepEqual(resized.stepFrames.slice(0, 6).map((steps) => steps.find(([sample]) => sample === 0)[1]), [0, 1, 2, 3, 4, 5])
equalDensity(pointwise, resized)
for (const size of [1, 1024]) {
  equalDensity(immediate, await sample({}, 0, null, size))
  const stopped = await sample(batchOptions, 10, (context) => context.self.onmessage({ data: { cmd: 'stop' } }), size)
  assert.equal(stopped.waits.length, 1, 'Stop did not interrupt a point batch')
  assert.ok(!stopped.progress.includes(batchOptions.samples), 'Stopped batch reported completion')
}

const presented = await sample({ ...batchOptions, waitForPresentation: true }, 10, null, 1)
equalDensity(pointwise, presented)
assert.equal(presented.waits.length, 0, 'Worker duplicated the presentation delay')
const stoppedAwaitingPresentation = await sample({ ...batchOptions, waitForPresentation: true }, 10, null, 1,
  (context) => context.self.onmessage({ data: { cmd: 'stop' } }))
assert.equal(stoppedAwaitingPresentation.chunks, 1, 'Stop failed while awaiting presentation')
assert.ok(!stoppedAwaitingPresentation.progress.includes(batchOptions.samples))

let staleAcks = 0
const acknowledged = await sample({ ...batchOptions, waitForPresentation: true }, 10, null, 1, (context, message) => {
  const ack = { cmd: 'presented', jobId: message.jobId, presentationId: message.chunk.presentationId }
  context.self.onmessage({ data: { ...ack, jobId: -1 } })
  assert.equal(vm.runInContext('pendingPresentations.size', context), 1, 'Stale acknowledgement released this job')
  staleAcks++
  context.self.onmessage({ data: ack })
})
equalDensity(pointwise, acknowledged)
assert.equal(staleAcks, acknowledged.chunks)
console.log(`PASS: ${cases} density comparisons; concurrent trajectories advance one step per Delay, live resize, ACK and Stop`)

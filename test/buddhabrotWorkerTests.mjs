import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import vm from 'node:vm'
import { BUDDHA_PALETTES } from '../buddhaPalettes.mjs'
import { DEFAULT_RENDER_POINT_BATCH_SIZE, normalizeRenderPointBatchSize } from '../buddhabrotRenderConfig.mjs'
import { compileIterationFunction } from '../customFunctionParser.mjs'

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
  )
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
  let pendingSteps = []
  const progress = []
  const waits = []
  let chunks = 0
  const context = vm.createContext({
    self: {},
    console,
    compileIterationFunction,
    DEFAULT_RENDER_POINT_BATCH_SIZE,
    normalizeRenderPointBatchSize,
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
  return { density, contributions, hits, progress, waits, chunks, stepFrames }
}

function equalDensity(expected, actual) {
  assert.deepEqual(actual.hits, expected.hits, 'Orbit hits or band colors changed')
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

const immediate = await sample()
for (const delay of [0.01, 1, 10, 100]) {
  const delayed = await sample({}, delay)
  assert.ok(delayed.waits.length < 2000, `Long anti-buddhabrot orbits waited ${delayed.waits.length} times`)
  assert.ok(delayed.chunks > 0, 'Missing visible density')
  assert.equal(delayed.progress.at(-1), base.samples)
  equalDensity(immediate, delayed)
}

let cases = 4
for (const expression of [null, 'z*z+c', 'c*(z+1/(z^2))', 'z*z+c+0.1*zDelay(2)']) {
  for (const mode of ['buddha', 'antibuddha']) {
    for (const buddhaBandMode of ['perPoint', 'perTrajectory']) {
      for (const paletteStops of [null, ...BUDDHA_PALETTES]) {
        const opts = { iterationFunction: expression, mode, buddhaBandMode, paletteStops, samples: 128, maxIter: 240 }
        equalDensity(await sample(opts), await sample(opts, 10))
        if (referenceSource) equalDensity(await sample({ ...opts, workerSource: referenceSource }), await sample(opts))
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
      if (referenceSource) equalDensity(await sample({ ...opts, workerSource: referenceSource }), await sample(opts))
      cases++
    }
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

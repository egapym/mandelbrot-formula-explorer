import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import vm from 'node:vm'
import { compileIterationFunction } from '../customFunctionParser.mjs'
import { getBuddhabrotProjection } from '../buddhabrotProjection.mjs'
import { DEFAULT_RENDER_POINT_BATCH_SIZE, normalizeRenderPointBatchSize } from '../buddhabrotRenderConfig.mjs'

const source = readFileSync(new URL('../buddhabrotWorker.mjs', import.meta.url), 'utf8').replace(/^import .*$/gm, '')
function worker() {
  let calls = 0, clock = 0, channels, chunks
  const context = vm.createContext({
    self: {}, console, Math,
    Date: { now: () => clock },
    setTimeout: callback => { queueMicrotask(callback) },
    getBuddhabrotProjection, compileIterationFunction,
    DEFAULT_RENDER_POINT_BATCH_SIZE, normalizeRenderPointBatchSize,
    postMessage(message) {
      if (message.type !== 'chunk') return
      chunks++
      const chunk = message.chunk
      for (const [channel, data] of ['r', 'g', 'b'].entries()) {
        if (chunk.indices) for (let i = 0; i < chunk.indices.length; i++) channels[channel][chunk.indices[i]] += chunk[data][i]
        else for (let i = 0; i < channels[channel].length; i++) channels[channel][i] += chunk[data][i]
      }
    },
  })
  vm.runInContext(source, context)
  return {
    async run(options) {
      channels = Array.from({ length: 3 }, () => new Float32Array(options.width * options.height))
      chunks = 0
      const previous = calls
      const compiled = compileIterationFunction(options.iterationFunction)
      vm.runInContext('running = true; currentJobId++', context)
      await context.runSampling({ ...options, iterationFunctionCompiled: (...args) => {
        calls++; clock += options.iterationCost || 0
        return compiled(...args)
      } })
      return { channels, calls: calls - previous, chunks }
    },
    stop: () => context.self.onmessage({ data: { cmd: 'stop' } }),
  }
}
const base = {
  width: 64, height: 48, maxIter: 128, samples: 512, sampleSeed: 4321,
  samplingCenter: { x: -0.5, y: 0 }, samplingZoom: 1, samplingAspect: 0.75,
  center: { x: -0.5, y: 0 }, zoom: 1, z0Real: 0, z0Imag: 0,
  escapeRadius: 4, fractalType: 'mandelbrot', buddhaBandMode: 'perPoint',
  iterationFunction: 'z*z+c', orbitCacheBytes: 1024 ** 2,
}
let checks = 0
for (const fractalType of ['mandelbrot', 'julia']) {
  for (const mode of ['buddha', 'antibuddha']) {
    for (const iterationFunction of ['z*z+c', 'z*z+c+0.01*zDelay(2)']) {
      const cache = worker()
      const options = { ...base, mode, fractalType, iterationFunction, juliaRe: -0.7, juliaIm: 0.2 }
      const initial = await cache.run(options)
      assert.ok(initial.calls > 0)
      const next = { ...options, center: { x: -0.75, y: 0.1 }, zoom: 2 }
      const replay = await cache.run(next)
      const reference = await worker().run({ ...next, orbitCacheBytes: 0 })
      assert.deepEqual(replay.channels, reference.channels, 'Cached full orbits changed density after zoom/pan')
      assert.equal(replay.calls, 0, 'Cached sampling recomputed iteration functions')
      for (const override of [{ sampleSeed: 99 }, { maxIter: 129 }, { samplingZoom: 2 }, { z0Real: 0.1 }, { escapeRadius: 3 }]) {
        const changed = { ...next, ...override }
        const actual = await cache.run(changed)
        const expected = await worker().run({ ...changed, orbitCacheBytes: 0 })
        assert.deepEqual(actual.channels, expected.channels)
        assert.ok(actual.calls > 0, 'A changed sampling/iteration setting reused stale orbits')
        checks++
      }
      cache.stop()
      assert.ok((await cache.run(next)).calls > 0, 'Stop retained another session cache')
      checks++
    }
  }
}
// Bounded storage keeps rejected classifications and recomputes trajectories
// whose full point buffers do not fit. It never truncates accepted orbits.
const bounded = worker()
await bounded.run({ ...base, mode: 'antibuddha', orbitCacheBytes: 2048 })
const small = await bounded.run({ ...base, mode: 'antibuddha', orbitCacheBytes: 2048, zoom: 2 })
assert.deepEqual(small.channels, (await worker().run({ ...base, mode: 'antibuddha', orbitCacheBytes: 0, zoom: 2 })).channels)
assert.ok(small.calls > 0, 'The bounded cache did not fall back to recomputing full orbits')
// The first frame's timed flush boundaries must survive faster replay, retaining
// Float32 summation groups even when replay consumes no simulated compute time.
const timed = worker()
const timedOptions = { ...base, samples: 4096, mode: 'antibuddha', orbitCacheBytes: 8 * 1024 ** 2, iterationCost: 1 }
const first = await timed.run(timedOptions)
const second = await timed.run({ ...timedOptions, zoom: 2 })
const expected = await worker().run({ ...timedOptions, zoom: 2, orbitCacheBytes: 0 })
assert.ok(first.chunks > 1, 'Timed fixture did not produce intermediate flushes')
assert.equal(second.chunks, first.chunks)
assert.equal(second.calls, 0)
assert.deepEqual(second.channels, expected.channels)
console.log(`PASS: ${checks} orbit-cache checks, exact full-orbit projection, history/Julia/anti modes, invalidation, Stop, bounded fallback and Float32 flush replay`)

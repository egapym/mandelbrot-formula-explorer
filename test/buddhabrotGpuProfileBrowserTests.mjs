// Native GPU timestamps cover every dispatch, not only the last sample batch.
// ?samples=100000000&rounds=5&cases=provided&candidate=./fixtures/prototype.mjs
import { BuddhabrotWebGPU, colorMapDensity } from '../buddhabrotWebGPU.mjs'
import { getBuddhaPalette } from '../buddhaPalettes.mjs'
const median = values => values.toSorted((a, b) => a - b)[Math.floor(values.length / 2)]
const assert = (ok, message) => { if (!ok) throw new Error(message) }
const query = new URLSearchParams(location.search)
const samples = Number(query.get('samples') || 100000000)
const rounds = Number(query.get('rounds') || 5)
const cases = {
  provided: {},
  interactive: { interactive: true },
  anti: { mode: 'antibuddha' },
  julia: { fractalType: 'julia', juliaRe: -0.7, juliaIm: 0.2 },
  custom: { fractalType: 'custom', iterationFunction: 'z*z+c*i' },
  history: { fractalType: 'custom', iterationFunction: 'z*z+c+0.1*zDelay(2)' },
  lowEscape: { escapeRadius: 1 },
  nonzeroZ0: { z0Real: 0.3, z0Imag: -0.1 },
  zoom: { center: { x: -0.7435, y: 0.1314 }, zoom: 100 },
}
async function run() {
  assert(Number.isSafeInteger(samples) && samples > 0 && rounds > 0, 'Invalid benchmark size')
  const adapter = await navigator.gpu?.requestAdapter()
  assert(adapter, 'WebGPU unavailable')
  const timestamps = adapter.features.has('timestamp-query')
  const device = await adapter.requestDevice({ requiredFeatures: timestamps ? ['timestamp-query'] : [] })
  const queryCount = 2048
  const querySet = timestamps ? device.createQuerySet({ type: 'timestamp', count: queryCount }) : null
  const resolve = timestamps ? device.createBuffer({ size: queryCount * 8, usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC }) : null
  const read = timestamps ? device.createBuffer({ size: queryCount * 8, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ }) : null
  let stats, active = false
  const queue = new Proxy(device.queue, { get(target, key) {
    if (key === 'submit') return buffers => { if (active) stats.submits++; return target.submit(buffers) }
    if (key === 'onSubmittedWorkDone') return () => {
      if (active) stats.waits++
      return target.onSubmittedWorkDone()
    }
    const value = Reflect.get(target, key, target)
    return typeof value === 'function' ? value.bind(target) : value
  } })
  const probe = new Proxy(device, { get(target, key) {
    if (key === 'queue') return queue
    if (key === 'createComputePipeline') return descriptor => {
      const start = performance.now(), pipeline = target.createComputePipeline(descriptor)
      if (active) { stats.pipelines++; stats.pipelineMs += performance.now() - start }
      return pipeline
    }
    if (key === 'createBuffer') return descriptor => {
      const buffer = target.createBuffer(descriptor)
      if (descriptor.usage & GPUBufferUsage.MAP_READ) {
        const map = buffer.mapAsync.bind(buffer), unmap = buffer.unmap.bind(buffer)
        let mappedAt, measured = false
        buffer.mapAsync = async (...args) => {
          measured = active
          const start = performance.now()
          await map(...args)
          if (measured) { stats.readWaitMs += performance.now() - start; mappedAt = performance.now() }
        }
        buffer.unmap = () => {
          if (measured) stats.densityCopyMs += performance.now() - mappedAt
          return unmap()
        }
      }
      return buffer
    }
    if (key === 'createCommandEncoder' && timestamps) return (...args) => {
      const encoder = target.createCommandEncoder(...args)
      return new Proxy(encoder, { get(object, name) {
        if (name === 'beginComputePass') return (descriptor = {}) => {
          if (!active) return object.beginComputePass(descriptor)
          const index = stats.dispatches++ * 2
          assert(index + 1 < queryCount, 'Increase queryCount or reduce samples')
          return object.beginComputePass({ ...descriptor, timestampWrites: {
            querySet, beginningOfPassWriteIndex: index, endOfPassWriteIndex: index + 1,
          } })
        }
        const value = Reflect.get(object, name, object)
        return typeof value === 'function' ? value.bind(object) : value
      } })
    }
    const value = Reflect.get(target, key, target)
    return typeof value === 'function' ? value.bind(target) : value
  } })
  const variants = [['current', BuddhabrotWebGPU]]
  for (const [index, path] of (query.get('candidate') || '').split(',').filter(Boolean).entries()) {
    const url = new URL(path, location.href)
    assert(url.origin === location.origin, 'Candidate must be a local module')
    variants.push([`candidate-${index + 1}`, (await import(url.href)).BuddhabrotWebGPU])
  }
  const rows = []
  try {
    for (const name of (query.get('cases') || 'provided').split(',')) {
      assert(cases[name], `Unknown case ${name}`)
      const runners = variants.map(([label, C]) => ({ label, runner: new C({ devicePromise: Promise.resolve(probe) }), runs: [] }))
      let expected
      for (let round = 0; round <= rounds; round++) {
        for (const { label, runner, runs } of round % 2 ? runners.toReversed() : runners) {
          stats = { submits: 0, waits: 0, dispatches: 0, pipelines: 0, pipelineMs: 0, readWaitMs: 0, densityCopyMs: 0 }
          let count = 0, error
          runner.onProgress = data => { count += data.delta }
          runner.onComplete = result => { if (result.error) error = result.error }
          device.pushErrorScope('validation')
          active = true
          const start = performance.now()
          await runner.start({ width: 800, height: 600, samples, maxIter: 1000,
            center: { x: -0.5, y: 0 }, zoom: 1, escapeRadius: 4, z0Real: 0, z0Imag: 0,
            fractalType: 'mandelbrot', iterationFunction: 'z*z + c', gpu: true,
            sampleSeed: Number(query.get('seed') || 0x1badb002), mode: 'buddha', buddhaBandMode: 'perTrajectory',
            palette: getBuddhaPalette('Preset-01'), brightness: 1.2, gamma: 4.8, ...cases[name] })
          const samplerMs = performance.now() - start
          active = false
          const validation = await device.popErrorScope()
          assert(!error && !validation && !runner._cpuRunner, String(error || validation?.message || 'CPU fallback'))
          assert(count === samples, `${label}: sample count ${count} != ${samples}`)
          let gpuMs = null
          if (timestamps) {
            const encoder = device.createCommandEncoder(), bytes = stats.dispatches * 16
            encoder.resolveQuerySet(querySet, 0, stats.dispatches * 2, resolve, 0)
            encoder.copyBufferToBuffer(resolve, 0, read, 0, bytes)
            device.queue.submit([encoder.finish()])
            await read.mapAsync(GPUMapMode.READ)
            const values = new BigUint64Array(read.getMappedRange(0, bytes))
            let sum = 0n
            for (let i = 0; i < values.length; i += 2) sum += values[i + 1] - values[i]
            gpuMs = Number(sum) / 1e6
            read.unmap()
          }
          const channels = [runner.densityR, runner.densityG, runner.densityB]
          if (!expected) expected = channels.map(channel => channel.slice())
          let changed = 0, absolute = 0, mass = 0
          for (let ch = 0; ch < 3; ch++) for (let i = 0; i < channels[ch].length; i++) {
            const difference = Math.abs(channels[ch][i] - expected[ch][i])
            if (difference) changed++
            absolute += difference; mass += expected[ch][i]
          }
          assert(changed === 0, `${name}/${label}: ${changed} density values changed, relative L1=${absolute / mass}`)
          const colorStart = performance.now()
          const bitmap = await colorMapDensity({ rBuf: channels[0], gBuf: channels[1], bBuf: channels[2],
            width: 800, height: 600, brightness: 1.2, gamma: 4.8 })
          bitmap.close()
          const colorMs = performance.now() - colorStart
          if (round > 0) runs.push({ ...stats, samplerMs, gpuMs, colorMs })
        }
      }
      for (const { label, runs } of runners) rows.push({ case: name, variant: label,
        ...Object.fromEntries(Object.keys(runs[0]).map(key => [key, runs[0][key] === null ? null : +median(runs.map(run => run[key])).toFixed(3)])) })
    }
    return { status: 'PASS: exact RGB density parity and sample counts', samples, rounds,
      adapter: Object.fromEntries(['vendor', 'architecture', 'device', 'description'].map(key => [key, adapter.info[key]])),
      userAgent: navigator.userAgent, timestamps, rows }
  } finally { querySet?.destroy(); resolve?.destroy(); read?.destroy(); device.destroy() }
}
try { document.querySelector('#result').textContent = JSON.stringify(await run(), null, 2) }
catch (error) { document.querySelector('#result').textContent = error.stack; console.error(error) }

import { BuddhabrotWebGPU } from '../buddhabrotWebGPU.mjs'
import { getBuddhaPalette } from '../buddhaPalettes.mjs'

const assert = (condition, message) => {
  if (!condition) throw new Error(message)
}
const median = (values) => values.toSorted((a, b) => a - b)[Math.floor(values.length / 2)]
const query = new URLSearchParams(location.search)
const measuredRounds = Number(query.get('rounds') || 7)
const benchmarkSize = {
  width: Number(query.get('width') || 640), height: Number(query.get('height') || 640),
  samples: Number(query.get('samples') || 250000), maxIter: Number(query.get('maxIter') || 1000),
  ...(query.has('palette') ? { palette: getBuddhaPalette(query.get('palette')), buddhaBandMode: 'perPoint' } : {}),
}

async function run() {
  const adapter = await navigator.gpu?.requestAdapter()
  assert(adapter, 'WebGPU unavailable')
  const timestamps = adapter.features.has('timestamp-query')
  const device = await adapter.requestDevice({ requiredFeatures: timestamps ? ['timestamp-query'] : [] })
  // Full orbits may require many overflow-safe histogram dispatches per run.
  const timestampCapacity = 1024
  const querySet = timestamps ? device.createQuerySet({ type: 'timestamp', count: timestampCapacity }) : null
  const resolveBuffer = timestamps
    ? device.createBuffer({ size: timestampCapacity * 8, usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC })
    : null
  const readBuffer = timestamps
    ? device.createBuffer({ size: timestampCapacity * 8, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ })
    : null
  const shaderKinds = new WeakMap()
  let usedSampler = false
  let timestampCount = 0
  let captureTimestamps = false
  let densityMaps = 0
  const queue = new Proxy(device.queue, {
    get(target, key) {
      if (key === 'writeBuffer')
        return (buffer, offset, data, dataOffset = 0, size) => {
          // Fix the actual sampler seed without replacing its random generator.
          if (offset === 0 && data instanceof ArrayBuffer && size === 44) {
            const copy = new Uint32Array(data.slice(dataOffset, dataOffset + size))
            copy[8] = 0x1badb002
            return target.writeBuffer(buffer, offset, copy)
          }
          return target.writeBuffer(buffer, offset, data, dataOffset, size)
        }
      const value = Reflect.get(target, key, target)
      return typeof value === 'function' ? value.bind(target) : value
    },
  })
  const probe = new Proxy(device, {
    get(target, key) {
      if (key === 'queue') return queue
      if (key === 'createBuffer') return descriptor => {
        const buffer = target.createBuffer(descriptor)
        if (descriptor.usage & GPUBufferUsage.MAP_READ) {
          const map = buffer.mapAsync.bind(buffer)
          buffer.mapAsync = (...args) => { if (captureTimestamps) densityMaps++; return map(...args) }
        }
        return buffer
      }
      if (key === 'createShaderModule')
        return (descriptor) => {
          const module = target.createShaderModule(descriptor)
          shaderKinds.set(module, descriptor.code.includes('fn coordToIndex('))
          return module
        }
      if (key === 'createComputePipeline')
        return (descriptor) => {
          usedSampler = shaderKinds.get(descriptor.compute.module) === true
          return target.createComputePipeline(descriptor)
        }
      if (key === 'createCommandEncoder' && timestamps)
        return (...args) => {
          const encoder = target.createCommandEncoder(...args)
          return new Proxy(encoder, {
            get(object, name) {
              if (name === 'beginComputePass')
                return (descriptor = {}) => {
                  if (!captureTimestamps) return object.beginComputePass(descriptor)
                  const index = timestampCount++ * 2
                  return object.beginComputePass({ ...descriptor,
                    timestampWrites: { querySet, beginningOfPassWriteIndex: index, endOfPassWriteIndex: index + 1 },
                  })
                }
              const value = Reflect.get(object, name, object)
              return typeof value === 'function' ? value.bind(object) : value
            },
          })
        }
      const value = Reflect.get(target, key, target)
      return typeof value === 'function' ? value.bind(target) : value
    },
  })
  const types = [['current', BuddhabrotWebGPU]]
  const reference = new URL(location.href).searchParams.get('reference')
  const compareDensity = new URL(location.href).searchParams.get('compareDensity') !== 'false'
  if (reference) {
    const url = new URL(reference, location.href)
    assert(url.origin === location.origin, 'Reference must be a local module')
    types.unshift(['before', (await import(url.href)).BuddhabrotWebGPU])
  }
  const runners = types.map(([name, Type]) => {
    class Injected extends Type {
      initGpu() {
        return Promise.resolve(probe)
      }
    }
    return { name, runner: new Injected({ width: 640, height: 640 }) }
  })
  const rows = []
  device.pushErrorScope('validation')
  try {
    for (const [label, options] of [
      ['Mandelbrot', {}],
      ['Rotated', { iterationFunction: 'z*z+c*i', fractalType: 'custom' }],
      ['Translated', { iterationFunction: 'z*z+c+i', fractalType: 'custom' }],
      ['Anti Mandelbrot', { mode: 'antibuddha' }],
      ['History Anti', { mode: 'antibuddha', fractalType: 'custom', iterationFunction: 'z*z+c+0.1*zDelay(2)' }],
      ['Julia', { fractalType: 'julia-custom', iterationFunction: 'z*z+c', juliaRe: -0.7, juliaIm: 0.2 }],
    ]) {
      if (query.has('cases') && !query.get('cases').split(',').includes(label)) continue
      const timings = runners.map(() => ({ wall: [], gpu: [], maps: [] }))
      let expected
      // Alternate versions on every round; exclude compilation/warmup rounds.
      for (let round = 0; round < measuredRounds + 2; round++) {
        for (let i = 0; i < runners.length; i++) {
          const { name, runner } = runners[i]
          let failure
          runner.onComplete = (result) => {
            if (result.error) failure = result.error
          }
          const start = performance.now()
          usedSampler = false
          timestampCount = 0
          captureTimestamps = true
          densityMaps = 0
          await runner.start({
            ...benchmarkSize,
            center: { x: 0, y: 0 },
            zoom: 1,
            escapeRadius: 4,
            gpu: true,
            mode: 'buddha',
            ...options,
          })
          captureTimestamps = false
          const elapsed = performance.now() - start
          assert(!failure && !runner._cpuRunner && usedSampler, failure || 'Sampler did not run on GPU')
          if (name === 'current') assert(densityMaps <= 2, `Unbounded full-image readbacks: ${densityMaps}`)
          let gpuMs = 0
          if (timestamps) {
            const encoder = device.createCommandEncoder()
            encoder.resolveQuerySet(querySet, 0, timestampCount * 2, resolveBuffer, 0)
            encoder.copyBufferToBuffer(resolveBuffer, 0, readBuffer, 0, timestampCount * 16)
            device.queue.submit([encoder.finish()])
            await readBuffer.mapAsync(GPUMapMode.READ)
            const values = new BigUint64Array(readBuffer.getMappedRange(0, timestampCount * 16))
            let total = 0n
            for (let i = 0; i < values.length; i += 2) total += values[i + 1] - values[i]
            gpuMs = Number(total) / 1e6
            readBuffer.unmap()
          }
          if (round >= 2) {
            timings[i].wall.push(elapsed)
            timings[i].gpu.push(gpuMs)
            timings[i].maps.push(densityMaps)
          }
          if (round === 0) {
            const channels = [runner.densityR, runner.densityG, runner.densityB]
            if (!expected) expected = channels.map((channel) => channel.slice())
            else if (compareDensity) {
              let difference = 0
              let total = 0
              let pixels = 0
              for (let ch = 0; ch < channels.length; ch++)
                for (let px = 0; px < channels[ch].length; px++) {
                  const delta = Math.abs(channels[ch][px] - expected[ch][px])
                  difference += delta
                  total += expected[ch][px]
                  if (delta) pixels++
                }
              assert(
                difference === 0,
                `${label}: ${name} changed density: ${pixels} channel/pixels; relative L1=${difference / total}`,
              )
            }
          }
        }
      }
      for (let i = 0; i < runners.length; i++)
        rows.push({
          variant: runners[i].name,
          case: label,
          gpuMs: timestamps ? +median(timings[i].gpu).toFixed(3) : null,
          totalMs: +median(timings[i].wall).toFixed(3),
          densityMaps: median(timings[i].maps),
        })
    }
    // Force periodic readback boundaries without a wall-clock wait. With the
    // same seed each batch, clearing/accumulating must preserve every channel.
    const clockDescriptor = Object.getOwnPropertyDescriptor(performance, 'now')
    const realNow = performance.now.bind(performance)
    let elapsedDisplayTime = 0
    Object.defineProperty(performance, 'now', { configurable: true, value: () => realNow() + elapsedDisplayTime })
    try {
      for (const mode of ['buddha', 'antibuddha']) {
        let expected
        const targetSamples = 4096 * 16
        for (const continuous of [false, true]) {
          const runner = new BuddhabrotWebGPU({ devicePromise: Promise.resolve(probe) })
          let count = 0, frames = 0, failure
          runner.onProgress = data => {
            count += data.delta
            if (continuous) elapsedDisplayTime += 1001
            if (continuous && count >= targetSamples) runner.stop()
          }
          runner.onChunk = () => frames++
          runner.onComplete = result => { if (result.error) failure = result.error }
          await runner.start({ width: 64, height: 64, samples: targetSamples, maxIter: 1000,
            center: { x: 0, y: 0 }, zoom: 1, escapeRadius: 4, gpu: true, mode, continuous })
          assert(!failure && !runner._cpuRunner, failure || 'Continuous test fell back to CPU')
          assert(count === targetSamples, 'Continuous Stop counted the wrong sample total')
          const channels = [runner.densityR, runner.densityG, runner.densityB]
          if (!continuous) expected = channels.map(channel => channel.slice())
          else {
            assert(frames >= 2, 'Periodic GPU readback did not publish updates')
            // Overflow-safe flushing and periodic presentation can group
            // Float32 additions differently while retaining all contributions.
            for (let ch = 0; ch < channels.length; ch++)
              assert(channels[ch].every((value, i) =>
                Math.abs(value - expected[ch][i]) <= Math.max(1 / 1024, Math.abs(expected[ch][i]) * 0.000002)),
                `${mode}: continuous histogram clearing lost or duplicated density`)
          }
        }
      }
    } finally {
      if (clockDescriptor) Object.defineProperty(performance, 'now', clockDescriptor)
      else delete performance.now
    }
    // A locked long orbit can enter a deep viewport after >200 offscreen
    // steps. Its convergence grid must remain at the pre-zoom coordinates.
    for (const zoom of [100, 1000]) {
      let unlockedChannels
      for (const locked of [false, true]) {
        const runner = new BuddhabrotWebGPU({ devicePromise: Promise.resolve(probe) })
        let failure
        runner.onComplete = result => { if (result.error) failure = result.error }
        const params = { gpu: true, width: 160, height: 90, samples: 16,
          maxIter: 600, mode: 'antibuddha', iterationFunction: 'z+0.005', fractalType: 'custom',
          z0Real: 0, escapeRadius: 4, center: { x: 1.5, y: 0 }, zoom,
          sampleSeed: 12345, samplingCenter: { x: 0, y: 0 }, samplingZoom: 1 }
        if (locked) params.convergenceView = { center: params.center, zoom, width: 160, height: 90 }
        await runner.start(params)
        assert(!failure && !runner._cpuRunner, failure || 'Deep zoom test fell back to CPU')
        const channels = [runner.densityR, runner.densityG, runner.densityB]
        assert(channels.some(channel => channel.some(value => value > 0)),
          `Orbit disappeared at zoom ${zoom} before entering the viewport`)
        if (!locked) unlockedChannels = channels.map(channel => channel.slice())
        else assert(channels.every((channel, ch) => channel.every((value, i) => value === unlockedChannels[ch][i])),
          'Enabling Fixed Samples added or removed orbit contributions')
      }
    }
    // Preset-05 must retain late yellow/red bands outside the locked viewport.
    let widePaletteChannels
    for (const locked of [false, true]) {
      const runner = new BuddhabrotWebGPU({ devicePromise: Promise.resolve(probe) })
      let failure
      runner.onComplete = result => { if (result.error) failure = result.error }
      await runner.start({ gpu: true, width: 160, height: 90, samples: 16, maxIter: 600,
        mode: 'antibuddha', iterationFunction: 'z+0.005', fractalType: 'custom',
        z0Real: 0, escapeRadius: 4, center: { x: 1.5, y: 0 }, zoom: 1,
        sampleSeed: 12345, samplingCenter: { x: 0, y: 0 }, samplingZoom: 1,
        palette: getBuddhaPalette('Preset-05'), buddhaBandMode: 'perPoint',
        convergenceView: locked ? { center: { x: 1.5, y: 0 }, zoom: 1000, width: 160, height: 90 } : undefined })
      assert(!failure && !runner._cpuRunner, failure || 'Zoom-out palette test fell back to CPU')
      const channels = [runner.densityR, runner.densityG, runner.densityB]
      if (!locked) widePaletteChannels = channels.map(channel => channel.slice())
      else {
        assert(channels[0].some(value => value > 0), 'Preset-05 lost yellow/red outside the lock')
        assert(channels.every((channel, ch) => channel.every((value, i) => value === widePaletteChannels[ch][i])),
          'Zooming out of the fixed grid changed orbit hits or palette contributions')
      }
    }
    const error = await device.popErrorScope()
    assert(!error, error?.message)
    const validation = runners.length > 1 && compareDensity ? 'fixed-seed density parity' : 'fixed-seed sampling'
    return `PASS: ${validation}; continuous readback/Stop density parity; locked deep-zoom orbit retention; Preset-05 zoom-out palette parity; median of ${measuredRounds} runs after 2 warmups\n${JSON.stringify(rows, null, 2)}`
  } finally {
    for (const { runner } of runners) runner.stop()
    querySet?.destroy()
    resolveBuffer?.destroy()
    readBuffer?.destroy()
    device.destroy()
  }
}

try {
  document.querySelector('#result').textContent = await run()
} catch (error) {
  document.querySelector('#result').textContent = error.stack
  console.error(error)
}

import { BuddhabrotWebGPU } from '../buddhabrotWebGPU.mjs'

const assert = (condition, message) => {
  if (!condition) throw new Error(message)
}
const median = (values) => values.toSorted((a, b) => a - b)[Math.floor(values.length / 2)]

async function run() {
  const adapter = await navigator.gpu?.requestAdapter()
  assert(adapter, 'WebGPU unavailable')
  const timestamps = adapter.features.has('timestamp-query')
  const device = await adapter.requestDevice({ requiredFeatures: timestamps ? ['timestamp-query'] : [] })
  const querySet = timestamps ? device.createQuerySet({ type: 'timestamp', count: 2 }) : null
  const resolveBuffer = timestamps
    ? device.createBuffer({ size: 16, usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC })
    : null
  const readBuffer = timestamps
    ? device.createBuffer({ size: 16, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ })
    : null
  const shaderKinds = new WeakMap()
  let usedSampler = false
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
                return (descriptor = {}) =>
                  object.beginComputePass({
                    ...descriptor,
                    timestampWrites: { querySet, beginningOfPassWriteIndex: 0, endOfPassWriteIndex: 1 },
                  })
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
      const timings = runners.map(() => ({ wall: [], gpu: [] }))
      let expected
      // Alternate versions on every round; exclude compilation/warmup rounds.
      for (let round = 0; round < 9; round++) {
        for (let i = 0; i < runners.length; i++) {
          const { name, runner } = runners[i]
          let failure
          runner.onComplete = (result) => {
            if (result.error) failure = result.error
          }
          const start = performance.now()
          usedSampler = false
          await runner.start({
            width: 640,
            height: 640,
            samples: 250000,
            maxIter: 1000,
            center: { x: 0, y: 0 },
            zoom: 1,
            escapeRadius: 4,
            gpu: true,
            mode: 'buddha',
            ...options,
          })
          const elapsed = performance.now() - start
          assert(!failure && !runner._cpuRunner && usedSampler, failure || 'Sampler did not run on GPU')
          let gpuMs = 0
          if (timestamps) {
            const encoder = device.createCommandEncoder()
            encoder.resolveQuerySet(querySet, 0, 2, resolveBuffer, 0)
            encoder.copyBufferToBuffer(resolveBuffer, 0, readBuffer, 0, 16)
            device.queue.submit([encoder.finish()])
            await readBuffer.mapAsync(GPUMapMode.READ)
            const values = new BigUint64Array(readBuffer.getMappedRange())
            gpuMs = Number(values[1] - values[0]) / 1e6
            readBuffer.unmap()
          }
          if (round >= 2) {
            timings[i].wall.push(elapsed)
            timings[i].gpu.push(gpuMs)
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
        for (const continuous of [false, true]) {
          const runner = new BuddhabrotWebGPU({ devicePromise: Promise.resolve(probe) })
          let count = 0, frames = 0, failure
          runner.onProgress = data => {
            count += data.delta
            if (continuous) elapsedDisplayTime += 1001
            if (continuous && count >= 16384) runner.stop()
          }
          runner.onChunk = () => frames++
          runner.onComplete = result => { if (result.error) failure = result.error }
          await runner.start({ width: 64, height: 64, samples: 16384, maxIter: 1000,
            center: { x: 0, y: 0 }, zoom: 1, escapeRadius: 4, gpu: true, mode, continuous })
          assert(!failure && !runner._cpuRunner, failure || 'Continuous test fell back to CPU')
          assert(count === 16384, 'Continuous Stop counted the wrong sample total')
          const channels = [runner.densityR, runner.densityG, runner.densityB]
          if (!continuous) expected = channels.map(channel => channel.slice())
          else {
            assert(frames >= 2, 'Periodic GPU readback did not publish updates')
            for (let ch = 0; ch < channels.length; ch++)
              assert(channels[ch].every((value, i) => value === expected[ch][i]),
                `${mode}: continuous histogram clearing lost or duplicated density`)
          }
        }
      }
    } finally {
      if (clockDescriptor) Object.defineProperty(performance, 'now', clockDescriptor)
      else delete performance.now
    }
    const error = await device.popErrorScope()
    assert(!error, error?.message)
    const validation = runners.length > 1 && compareDensity ? 'fixed-seed density parity' : 'fixed-seed sampling'
    return `PASS: ${validation}; continuous readback/Stop density parity; median of 7 runs after 2 warmups\n${JSON.stringify(rows, null, 2)}`
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

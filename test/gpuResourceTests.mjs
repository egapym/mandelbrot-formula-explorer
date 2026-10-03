import { MandelbrotCustomWebGPU } from '../mandelbrotCustomWebGPU.mjs'
import { MandelbrotWebGPU } from '../mandelbrotWebGPU.mjs'
import { WorkerContext } from '../workerContext.mjs'

const assert = (condition, message) => { if (!condition) throw new Error(message) }
const equal = (a, b) => {
  for (const name of ['values', 'smooth', 'signs', 'zreal', 'zimag']) {
    if (a[name] === null && b[name] === null) continue
    assert(a[name]?.length === b[name]?.length, `${name}: length`)
    assert(a[name].every((value, i) => Object.is(value, b[name][i])), `${name}: changed output`)
  }
}

async function checkMandelbrotReadback() {
  const renderer = new MandelbrotWebGPU({ onGpuUpdate() {} }, new WorkerContext(), error => { throw new Error(error) })
  const device = await renderer.devicePromise
  const pipeline = renderer.mandelbrotPipeline
  const buffers = []
  const fixtures = [
    ['values', 'Values', new Int32Array([2, 4, 17, 100, 9])],
    ['signs', 'Signs', new Uint32Array([0, 1, 128, 255, 2])],
    ['zreal', 'Zreal', new Float32Array([-1.25, 0, 2.5, 3, -7])],
    ['zimag', 'Zimag', new Float32Array([4, -2, 0, 1.5, 6])],
    ['smooth', 'Smooth', new Int32Array([0, 255, 256, -1, 127])],
  ]
  try {
    for (const [name, suffix, data] of fixtures) {
      const source = device.createBuffer({ size: data.byteLength, usage: GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST })
      const target = device.createBuffer({ size: data.byteLength, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ })
      buffers.push(source, target)
      pipeline[`${name}Buffer`] = source
      pipeline[`result${suffix}Buffer`] = target
      device.queue.writeBuffer(source, 0, data)
    }
    for (const doSmooth of [true, false, true]) {
      const output = await pipeline.readResults({ doSmooth })
      for (const [name, , data] of fixtures) {
        const expected = name === 'signs' ? new Int8Array(data)
          : name === 'smooth' ? (doSmooth ? new Uint8ClampedArray(data) : new Uint8ClampedArray(data.length)) : data
        assert(expected.every((value, i) => Object.is(value, output[name][i])), `Mandelbrot ${name}: readback changed`)
      }
      assert(buffers.every(buffer => buffer.mapState === 'unmapped'), 'Mandelbrot left a mapped buffer')
    }
  } finally {
    for (const buffer of buffers) buffer.destroy()
  }
}

async function checkMandelbrotReuse() {
  const renderer = new MandelbrotWebGPU({ onGpuUpdate() {} }, new WorkerContext(), error => { throw new Error(error) })
  const device = await renderer.devicePromise
  const errors = []
  device.addEventListener('uncapturederror', event => errors.push(event.error.message))
  const pipeline = renderer.mandelbrotPipeline
  const params = { w: 7, h: 5, indices: Uint32Array.from({ length: 35 }, (_, i) => i), max_iter: 32, doSmooth: true, bailout: 16, supersampling: 0 }
  try {
    await pipeline.beforeRun(params)
    const original = new Map(pipeline.buffers)
    for (const name of ['valuesBuffer', 'smoothBuffer', 'signsBuffer', 'zrealBuffer', 'zimagBuffer']) {
      device.queue.writeBuffer(pipeline[name], 0, new Uint32Array(35).fill(123))
    }
    await pipeline.finish()
    await pipeline.beforeRun(params)
    assert([...original].every(([name, buffer]) => pipeline.buffers.get(name) === buffer), 'Mandelbrot recreated buffers')
    const cleared = await pipeline.readResults(params)
    assert(Object.values(cleared).every(channel => channel.every(value => value === 0)), 'Previous render leaked into new output')

    await pipeline.beforeRun({ ...params, max_iter: 64 })
    assert(pipeline.zBuffer !== original.get('zBuffer'), 'Reference storage did not resize')
    assert(pipeline.valuesBuffer === original.get('valuesBuffer'), 'Iteration change reallocated pixel storage')
    await pipeline.beforeRun({ ...params, doSmooth: false, supersampling: 2 })
    assert(pipeline.valuesBuffer === original.get('valuesBuffer'), 'Shader change reallocated pixel storage')
    await pipeline.beforeRun({ ...params, w: 3, h: 3, indices: new Uint32Array(9) })
    assert(pipeline.valuesBuffer.size === 36 && pipeline.indexBuffer.size === 36, 'Resize retained wrong buffer size')
    await device.queue.onSubmittedWorkDone()
    assert(errors.length === 0, errors.join('\n'))
  } finally {
    pipeline.dispose()
    assert(pipeline.buffers.size === 0, 'Disposed pipeline retained resources')
  }
}

async function run() {
  const errors = []
  const renderer = new MandelbrotCustomWebGPU({ onGpuUpdate() {} }, new WorkerContext(), error => errors.push(error))
  const device = await renderer.devicePromise
  assert(device, 'WebGPU is required; this test must not silently pass on CPU')
  device.addEventListener('uncapturederror', event => errors.push(event.error.message))
  const params = {
    w: 63, h: 35, max_iter: 120, refr: -2, refi: -1, ddr0: 0, ddi0: 0,
    ddr: 3 / 63, ddi: 2 / 35, doSmooth: true, bailout: 256,
    supersampling: 0, iterationFunction: 'z*z + c', z0: [0, 0],
  }
  try {
    const reference = await renderer.renderDirect(params)
    assert(reference.values.some(value => value > 4), 'Expected escaping pixels')
    const resources = renderer.pipeline.resources
    equal(reference, await renderer.renderDirect(params))
    assert(renderer.pipeline.resources === resources, 'Repeated render must reuse resources')
    for (const change of [{ doSmooth: false }, { w: 37, h: 19 }, { supersampling: 2 }, { iterationFunction: 'z*z + c + 0.2*zDelay(5)' }]) {
      await renderer.renderDirect({ ...params, ...change })
      equal(reference, await renderer.renderDirect(params))
    }
    const pipeline = renderer.pipeline.pipeline
    await renderer.renderDirect({ ...params, bailout: 64 })
    assert(renderer.pipeline.pipeline === pipeline, 'Uniform-only change recompiled the shader')
    equal(reference, await renderer.renderDirect(params))

    // Exercise device buffer-limit fallback at a small size instead of allocating a huge image.
    renderer.pipeline.disposeResources()
    const limitedDevice = new Proxy(device, {
      get(target, key) {
        if (key === 'limits') return { maxBufferSize: params.w * params.h * 4 + 8 }
        const value = Reflect.get(target, key, target)
        return typeof value === 'function' ? value.bind(target) : value
      },
    })
    renderer.pipeline.devicePromise = Promise.resolve(limitedDevice)
    equal(reference, await renderer.renderDirect(params))
    assert(renderer.pipeline.resources.readBuffers.length === 5, 'Large readback must split at device limit')
    renderer.pipeline.devicePromise = Promise.resolve(device)
    equal(reference, await renderer.renderDirect(params))

    // A mapping rejection must release the cached resources and allow the next render to recover.
    renderer.pipeline.resources.readBuffers[0].mapAsync = async () => { throw new Error('injected mapping failure') }
    let rejected = false
    try { await renderer.renderDirect(params) } catch { rejected = true }
    assert(rejected && renderer.pipeline.resources === null, 'Failed mapping retained cached resources')
    equal(reference, await renderer.renderDirect(params))

    globalThis.fractalGpuPerformance = { enabled: true, samples: [] }
    equal(reference, await renderer.renderDirect(params))
    const sample = fractalGpuPerformance.samples.find(sample => sample.renderer === 'custom.compute-readback')
    assert(sample?.readbackBytes === params.w * params.h * 20, 'Incorrect profile byte count')
    assert(device.features.has('timestamp-query') ? sample.gpuMs >= 0 : sample.gpuMs === null, 'Incorrect timestamp fallback')
    await checkMandelbrotReadback()
    await checkMandelbrotReuse()
    await device.queue.onSubmittedWorkDone()
    assert(errors.length === 0, errors.join('\n'))
    return 'PASS: repeated renders, odd dimensions, resize, smooth, supersampling, history, bailout, device limit, map failure recovery, profiling, Mandelbrot channel readback and buffer reuse'
  } finally {
    renderer.pipeline.dispose()
    delete globalThis.fractalGpuPerformance
  }
}

try {
  document.querySelector('#result').textContent = await run()
  document.documentElement.dataset.result = 'passed'
} catch (error) {
  document.querySelector('#result').textContent = error.stack
  document.documentElement.dataset.result = 'failed'
  console.error(error)
}

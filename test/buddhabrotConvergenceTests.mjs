// Node exercises CPU; the HTML entry point exercises real workers and WebGPU.
import { getBuddhaPalette } from '../buddhaPalettes.mjs'

const assert = (ok, message) => { if (!ok) throw new Error(message) }
const browser = typeof document !== 'undefined'
const palette = getBuddhaPalette('Preset-05')
const cases = [
  { name: 'stationary', expression: '0', next: () => 0, maxIter: 10000 },
  { name: 'two-point cycle', expression: '-z+0.00390625', next: z => -z + 1 / 256 },
  { name: 'subpixel drift', expression: 'z+0.0000152587890625', next: z => z + 1 / 65536 },
  { name: 'motion after 300 stationary steps', expression: 'z+0.00390625*(n/300-fract(n/300))',
    next: (z, n) => z + Math.floor(n / 300) / 256, maxIter: 600 },
  { name: 'escape after subpixel drift', expression: 'z+0.0000152587890625',
    next: z => z + 1 / 65536, escapeRadius: 1 / 256, mode: 'buddha' },
]

async function cpuDensity(params) {
  const channels = Array.from({ length: 3 }, () => new Float64Array(params.width * params.height))
  let worker
  if (browser) {
    worker = new Worker(new URL('../buddhabrotWorker.mjs', import.meta.url), { type: 'module' })
  } else {
    const [{ readFileSync }, vm, parser, projection, config] = await Promise.all([
      import('node:fs'), import('node:vm'), import('../customFunctionParser.mjs'),
      import('../buddhabrotProjection.mjs'), import('../buddhabrotRenderConfig.mjs'),
    ])
    const source = readFileSync(new URL('../buddhabrotWorker.mjs', import.meta.url), 'utf8')
      .replace(/^import .*$/gm, '')
    worker = { terminate() {}, postMessage(data) { context.self.onmessage({ data }) } }
    const context = vm.createContext({ self: {}, console, setTimeout: callback => queueMicrotask(callback),
      ...parser, ...projection, ...config,
      postMessage: data => worker.onmessage({ data }),
    })
    vm.runInContext(source, context)
  }
  try {
    await new Promise((resolve, reject) => {
      worker.onerror = reject
      worker.onmessage = ({ data }) => {
        if (data.type === 'error') reject(new Error(data.message))
        if (data.type === 'done') resolve()
        if (data.type !== 'chunk') return
        const chunk = data.chunk
        for (const [ch, key] of ['r', 'g', 'b'].entries()) {
          for (let i = 0; i < chunk.indices.length; i++) channels[ch][chunk.indices[i]] += chunk[key][i]
        }
      }
      worker.postMessage({ ...params, cmd: 'start', jobId: 1, paletteStops: palette })
    })
    return channels
  } finally { worker.terminate() }
}

// Independent full-orbit oracle, with analytic real-valued recurrence and
// Preset-05's four equal intervals. No renderer convergence logic is reused.
function oracle(test, params) {
  const points = []
  let z = 0, escaped = false
  for (let n = 0; n < params.maxIter; n++) {
    z = test.next(z, n)
    if (Math.abs(z) > params.escapeRadius) { escaped = true; break }
    points.push(z)
  }
  const channels = Array.from({ length: 3 }, () => new Float64Array(params.width * params.height))
  const hits = new Uint32Array(params.width * params.height)
  if (escaped !== (params.mode === 'buddha')) return { channels, hits }
  for (const [k, x] of points.entries()) {
    const px = Math.floor((x - params.center.x) * params.zoom * params.width / 4 + params.width / 2)
    if (px < 0 || px >= params.width) continue
    const idx = params.height / 2 * params.width + px
    hits[idx] += params.samples
    const frac = params.buddhaBandMode === 'perTrajectory' ? points.length / params.maxIter : k / params.maxIter
    const band = Math.min(3, Math.floor(frac * 4))
    const a = palette.bands[band].color, b = palette.bands[Math.min(3, band + 1)].color
    const t = Math.min(1, Math.max(0, frac * 4 - band))
    for (let ch = 0; ch < 3; ch++) channels[ch][idx] += (a[ch] + (b[ch] - a[ch]) * t) / 255 * 1.2 * params.samples
  }
  return { channels, hits }
}

async function run() {
  let device, gpuRunner, checks = 0
  if (browser) {
    const adapter = await navigator.gpu?.requestAdapter()
    assert(adapter, 'WebGPU unavailable')
    device = await adapter.requestDevice()
    const { BuddhabrotWebGPU } = await import('../buddhabrotWebGPU.mjs')
    gpuRunner = new BuddhabrotWebGPU({ devicePromise: Promise.resolve(device) })
  }
  try {
    for (const test of cases) for (const buddhaBandMode of ['perPoint', 'perTrajectory']) {
      // Same seed/range across a wide view and a deep view of the late drift.
      for (const zoom of [1, 1024]) for (const renderDelay of browser ? [0] : [0, 1]) {
        const params = { width: 160, height: 90, samples: 1,
          maxIter: test.maxIter || 1024,
          center: { x: 3 / 256, y: 0 }, zoom, sampleSeed: 12345,
          samplingCenter: { x: 0, y: 0 }, samplingZoom: 1, samplingAspect: 90 / 160,
          escapeRadius: test.escapeRadius || 4, mode: test.mode || 'antibuddha',
          z0Real: 0, z0Imag: 0, fractalType: 'custom', iterationFunction: test.expression,
          buddhaBandMode, renderDelay, palette,
          // Legacy saved grids must have no effect, however coarse or fine.
          convergenceView: { center: { x: 0, y: 0 }, zoom: zoom === 1 ? 1000 : 1, width: 160, height: 90 },
        }
        const outputs = [['CPU', await cpuDensity(params)]]
        if (gpuRunner && renderDelay === 0) {
          // Exceeds u32 histogram capacity without clearing mid-batch. Keep
          // CPU at one orbit to isolate this GPU integer accumulation check.
          const samples = test.name === 'stationary' ? 512 : 1
          let failure
          gpuRunner.onComplete = data => { failure = data.error }
          device.pushErrorScope('validation')
          await gpuRunner.start({ ...params, samples, gpu: true })
          const error = await device.popErrorScope()
          assert(!failure && !error && !gpuRunner._cpuRunner, failure || error?.message || 'GPU fallback')
          outputs.push(['GPU', [gpuRunner.densityR, gpuRunner.densityG, gpuRunner.densityB], samples])
        }
        for (const [backend, actual, samples = 1] of outputs) {
          const expected = oracle(test, { ...params, samples })
          for (let ch = 0; ch < 3; ch++) for (let i = 0; i < expected.hits.length; i++) {
            // GPU quantizes each contribution to 1/1024; CPU buffers Float32.
            const tolerance = 0.001 * expected.hits[i] + 0.0001
            assert(Math.abs(actual[ch][i] - expected.channels[ch][i]) < tolerance,
              `${backend}: ${test.name}, ${buddhaBandMode}, zoom=${zoom}, delay=${renderDelay}, channel=${ch}, pixel=${i}: ${actual[ch][i]} != ${expected.channels[ch][i]}`)
          }
          checks++
        }
      }
    }
    if (gpuRunner) {
      // Carry conversion must round once, including the Float32 midpoints
      // where independently rounding the two words loses a half-unit.
      const { createGpuDensityReadback } = await import('../buddhabrotGpuDensity.mjs')
      const lows = new Uint32Array([0, 1, 0xffffffff, 0x20000101, 0x200000ff, 0x80000000, 0, 0xffffffff])
      const highs = new Uint32Array([0, 0, 0, 1, 1, 3, 1 << 23, 0x80000000])
      const inputs = Array.from({ length: 6 }, (_, i) => {
        const buffer = device.createBuffer({ size: lows.byteLength,
          usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST })
        device.queue.writeBuffer(buffer, 0, i < 3 ? lows : highs)
        return buffer
      })
      const converted = createGpuDensityReadback(device, inputs.slice(0, 3), inputs.slice(3), lows.byteLength, 1024)
      const read = device.createBuffer({ size: lows.byteLength * 3,
        usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST })
      try {
        device.pushErrorScope('validation')
        const encoder = device.createCommandEncoder()
        converted.encode(encoder)
        encoder.copyBufferToBuffer(converted.output, 0, read, 0, lows.byteLength * 3)
        device.queue.submit([encoder.finish()])
        await read.mapAsync(GPUMapMode.READ)
        const values = new Float32Array(read.getMappedRange())
        for (let i = 0; i < values.length; i++) {
          const at = i % lows.length
          const wanted = Math.fround((lows[at] + highs[at] * 4294967296) / 1024)
          assert(values[i] === wanted, `Carry conversion ${at}: ${values[i]} != ${wanted}`)
        }
        read.unmap()
        const error = await device.popErrorScope()
        assert(!error, error?.message)
        checks++
      } finally { read.destroy(); converted.destroy(); for (const buffer of inputs) buffer.destroy() }
      // Force smaller dispatches while keeping the original batch seed. A
      // c-dependent orbit detects missing, duplicated or resampled IDs at the
      // histogram boundary, including a non-workgroup-aligned final chunk.
      const limits = new Proxy(device.limits, { get(target, key) {
        return key === 'maxComputeWorkgroupsPerDimension' ? 1 : Reflect.get(target, key, target)
      } })
      const limited = new Proxy(device, { get(target, key) {
        if (key === 'limits') return limits
        const value = Reflect.get(target, key, target)
        return typeof value === 'function' ? value.bind(target) : value
      } })
      const { BuddhabrotWebGPU } = await import('../buddhabrotWebGPU.mjs')
      const other = new BuddhabrotWebGPU({ devicePromise: Promise.resolve(limited) })
      try {
        for (const size of [
          { width: 160, height: 90, samples: 513, maxIter: 10000 },
          // Exercises compact fast/slow queues across histogram readbacks,
          // compared with the same seeds through the noncompact path.
          { width: 32, height: 32, samples: 65537, maxIter: 64 },
        ]) {
          let expected
          for (const runner of [gpuRunner, other]) {
            let failure
            runner.onComplete = data => { failure = data.error }
            device.pushErrorScope('validation')
            await runner.start({ gpu: true, ...size,
              center: { x: 0, y: 0 }, zoom: 1, sampleSeed: 12345, escapeRadius: 4,
              mode: 'antibuddha', iterationFunction: 'c', fractalType: 'custom',
              palette, buddhaBandMode: 'perPoint' })
            const error = await device.popErrorScope()
            assert(!failure && !error && !runner._cpuRunner, failure || error?.message || 'GPU fallback')
            const channels = [runner.densityR, runner.densityG, runner.densityB]
            if (!expected) expected = channels.map(channel => channel.slice())
            else assert(channels.every((channel, ch) => channel.every((v, i) => v === expected[ch][i])),
              'Dispatch subdivision changed seeded orbit coordinates or palette density')
            checks++
          }
        }
      } finally { other.stop() }
    }
    return `PASS: ${checks} full-orbit density checks; stationary/periodic/subpixel/late-motion/escaping orbits, Preset-05, zoom, both band modes; ${browser ? 'real CPU workers/WebGPU, overflow and seeded subdivision' : 'CPU immediate/delayed replay'}`
  } finally { gpuRunner?.stop(); device?.destroy() }
}

if (browser) {
  try { document.querySelector('#result').textContent = await run() }
  catch (error) { document.querySelector('#result').textContent = `FAIL: ${error.stack || error}` }
} else { console.log(await run()) }

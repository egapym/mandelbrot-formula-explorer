import { BuddhabrotWebGPU } from '../buddhabrotWebGPU.mjs'
import { getBuddhaPalette } from '../buddhaPalettes.mjs'
const result = document.querySelector('#result')
const assert = (ok, message) => { if (!ok) throw new Error(message) }
try {
  const adapter = await navigator.gpu?.requestAdapter()
  assert(adapter, 'WebGPU unavailable')
  const device = await adapter.requestDevice()
  const rows = []
  try {
    for (const [fractalType, maxIter, samples] of [['custom', 1, 20000], ['custom', 64, 40000], ['julia-custom', 64, 40000]]) {
      const runner = new BuddhabrotWebGPU({ devicePromise: Promise.resolve(device) })
      try {
        const params = { width: 512, height: 512, center: { x: 0, y: 0 }, zoom: 0.5,
          samplingCenter: { x: 0, y: 0 }, samplingZoom: 0.5, samplingRadius: 4,
          samples, maxIter, sampleSeed: 12345, gpu: true, reusePipelines: true,
          fractalType, iterationFunction: fractalType === 'custom' ? 'c+0' : 'z+0',
          escapeRadius: 10, mode: 'antibuddha', buddhaBandMode: 'perPoint',
          palette: { bands: [{ color: [255, 255, 255], ratio: 1 }] } }
        let error
        runner.onComplete = data => { error = data.error }
        runner.onProgress = () => {}
        const run = async (options) => {
          device.pushErrorScope('validation')
          await runner.start(options)
          const validation = await device.popErrorScope()
          assert(!error && !validation && !runner._cpuRunner, error || validation?.message || 'CPU fallback')
          return runner.densityR.slice()
        }
        const disk = await run(params)
        let mass = 0, radiusMass = 0, outerMass = 0
        for (let i = 0; i < disk.length; i++) {
          if (!disk[i]) continue
          const x = ((i % 512) + 0.5) / 64 - 4
          const y = (Math.floor(i / 512) + 0.5) / 64 - 4
          const r2 = x * x + y * y
          assert(r2 < 16.1, 'Disk sampled a corner outside radius 4')
          mass += disk[i]; radiusMass += r2 * disk[i]
          if (r2 > 14) outerMass += disk[i]
        }
        assert(mass > 0 && Math.abs(radiusMass / mass - 8) < 0.2, 'Disk must be uniform in area')
        assert(outerMass / mass > 0.09, 'Disk must reach radius 4')
        const rectangle = await run({ ...params, samplingRadius: undefined })
        assert(rectangle.some((value, i) => value > 0 && Math.abs((i % 512) - 256) > 220 && Math.abs(Math.floor(i / 512) - 256) > 220),
          'Current View rectangle must still sample its corners')
        const replay = await run(params)
        assert(replay.every((value, i) => value === disk[i]), 'Switching rectangle/disk reused stale GPU pipelines')
        rows.push({ fractalType, maxIter, samples, meanRadiusSquared: radiusMass / mass })
      } finally { runner.dispose() }
    }
    // User-reported zoom boundary: both sides must retain the left-hand orbits.
    for (const [zoom, x, y] of [
      [0.9566303671497955, -0.9479452767151968, 0.10852589244875467],
      [1.0231724767554273, -0.8962778127181061, 0.10223741821226666],
    ]) {
      const runner = new BuddhabrotWebGPU({ devicePromise: Promise.resolve(device) })
      try {
        let error
        runner.onComplete = data => { error = data.error }
        runner.onProgress = () => {}
        await runner.start({ width: 800, height: 600, samples: 3000000, maxIter: 1000,
          center: { x, y }, zoom, samplingCenter: { x: 0, y: 0 }, samplingZoom: 0.5, samplingRadius: 4,
          sampleSeed: 12345, gpu: true, fractalType: 'custom', iterationFunction: 'conj(z)^z + c',
          z0Real: 0, z0Imag: 0, escapeRadius: 4, mode: 'antibuddha', buddhaBandMode: 'perPoint',
          palette: getBuddhaPalette('Preset-05'), brightness: 6, gamma: 0.1 })
        assert(!error && !runner._cpuRunner, error || 'CPU fallback')
        let leftPixels = 0
        for (let i = 0; i < 800 * 600; i++) {
          if (i % 800 < 700 && runner.densityR[i] + runner.densityG[i] + runner.densityB[i] > 0) leftPixels++
        }
        assert(leftPixels > 10000, `Orbits disappeared at zoom ${zoom}`)
        rows.push({ zoom, leftPixels })
      } finally { runner.dispose() }
    }

  } finally { device.destroy() }
  result.textContent = JSON.stringify({ status: 'PASS', rows }, null, 2)
} catch (error) { result.textContent = `FAIL: ${error.stack}`; console.error(error) }

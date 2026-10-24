import { BuddhabrotWebGPU } from '../buddhabrotWebGPU.mjs'

const assert = (ok, message) => { if (!ok) throw new Error(message) }
const result = document.querySelector('#result')
try {
  const adapter = await navigator.gpu?.requestAdapter()
  assert(adapter, 'WebGPU unavailable')
  const device = await adapter.requestDevice()
  const runner = new BuddhabrotWebGPU({ devicePromise: Promise.resolve(device) })
  try {
    for (const black of [false, true]) {
    const bands = [{ color: black ? [0, 0, 0] : [85, 85, 85], ratio: 1 }, { color: [255, 255, 255], ratio: 1 }]
    for (const [buddhaBandMode, iterationFunction, mode, expected, stepped] of [
      ['perPoint', 'z+0.25', 'antibuddha', 3.6, black ? 2.4 : 3.2],
      ['perTrajectory', 'z+2', 'buddha', 0.8, black ? 0 : 0.4],
    ]) {
      for (const gradient of [true, false, undefined]) for (const paletteBands of [
        bands, bands.map((band, i) => ({ color: band.color, cum: (i + 1) / bands.length })),
      ]) {
        device.pushErrorScope('validation')
        let failure
        runner.onComplete = data => { failure = data.error }
        await runner.start({ width: 16, height: 16, samples: 1, maxIter: 4,
          center: { x: 0, y: 0 }, zoom: 0.5, sampleSeed: 123,
          escapeRadius: 3, z0Real: 0, z0Imag: 0, gpu: true,
          iterationFunction, mode, buddhaBandMode, palette: { bands: paletteBands, gradient } })
        const error = await device.popErrorScope()
        assert(!error && !failure && !runner._cpuRunner, error?.message || failure || 'GPU fallback')
        for (const density of [runner.densityR, runner.densityG, runner.densityB]) {
          const total = density.reduce((a, b) => a + b, 0)
          const wanted = gradient && !black ? expected : stepped
          assert(Math.abs(total - wanted) < 0.004, `${buddhaBandMode}/${gradient}: ${total} != ${wanted}`)
        }
      }
    }
    }
    result.textContent = 'PASS: WebGPU gradient midpoints, endpoints, black-band exclusion, perPoint/perTrajectory and disabled compatibility'
  } finally {
    runner.stop()
    device.destroy()
  }
} catch (error) {
  result.textContent = `FAIL: ${error.stack || error}`
}

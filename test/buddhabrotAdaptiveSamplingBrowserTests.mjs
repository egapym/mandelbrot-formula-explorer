import { BuddhabrotRunner } from '../buddhabrot.mjs'
import { BuddhabrotWebGPU } from '../buddhabrotWebGPU.mjs'
import { getInitialBuddhabrotSamplingRange } from '../buddhabrotSampling.mjs'

const assert = (ok, message) => { if (!ok) throw new Error(message) }
const result = document.querySelector('#result')
try {
  const device = await (await navigator.gpu?.requestAdapter())?.requestDevice()
  assert(device, 'WebGPU unavailable')
  const rows = []
  try {
    // c+5 requires seeds centered at -5. The old radius-4 disk at 0 misses
    // the left half of the projected orbit domain; the derived disk covers all.
    for (const gpu of [false, true]) {
      const runner = gpu
        ? new BuddhabrotWebGPU({ devicePromise: Promise.resolve(device) })
        : new BuddhabrotRunner({ workerCount: 1 })
      try {
        const settings = { iterationFunction: 'c+5', escapeRadius: 4 }
        const params = { ...settings, ...getInitialBuddhabrotSamplingRange(settings),
          width: 128, height: 128, center: { x: -5, y: 0 }, zoom: 0.5,
          samples: 20000, maxIter: 1, sampleSeed: 12345, fractalType: 'custom',
          gpu, mode: 'antibuddha', buddhaBandMode: 'perPoint',
          palette: { bands: [{ color: [255, 255, 255], ratio: 1 }] },
        }
        const render = async (options) => {
          let complete
          const finished = new Promise(resolve => { complete = resolve })
          runner.onComplete = complete
          await runner.start(options)
          const data = await finished
          assert(!data.error && !runner._cpuRunner, data.error || 'Unexpected CPU fallback')
          return runner.densityR.slice()
        }
        const density = await render(params)
        let left = 0, right = 0
        for (let i = 0; i < density.length; i++) {
          if (i % 128 < 64) left += density[i]
          else right += density[i]
        }
        assert(left > 0 && Math.abs(left / (left + right) - 0.5) < 0.03,
          'Derived disk must sample both halves of the orbit domain')
        const legacy = await render({ ...params, samplingCenter: { x: 0, y: 0 }, samplingRadius: 4 })
        assert(!legacy.some((value, i) => value > 0 && i % 128 < 64),
          'Legacy sampling must demonstrate the missing left-hand orbits')
        rows.push({ gpu, leftFraction: left / (left + right), samplingCenter: params.samplingCenter })
      } finally { runner.dispose?.(); runner.terminate?.() }
    }
  } finally { device.destroy() }
  result.textContent = JSON.stringify({ status: 'PASS', rows }, null, 2)
} catch (error) { result.textContent = `FAIL: ${error.stack}`; console.error(error) }

import { MandelbrotCustomWebGPU } from '../mandelbrotCustomWebGPU.mjs'
import { WorkerContext } from '../workerContext.mjs'
import { legacyCustomShader } from './fixtures/legacyCustomShader.mjs'

const assert = (condition, message) => { if (!condition) throw new Error(message) }

export async function checkCustomShaderParity(device) {
  const create = () => new MandelbrotCustomWebGPU({ onGpuUpdate() {} }, new WorkerContext(),
    error => { throw new Error(error) }, { devicePromise: Promise.resolve(device) })
  const baseline = create()
  baseline.pipeline.generateShader = legacyCustomShader
  const optimized = create()
  const expressions = [
    'z*z + c', 'sin(z) + c', 'exp(z) + c', 'conj(z)^2 + c',
    'z*z + c + 0.01*sin(n)', 'z*z + c + 0.2*zDelay(5)',
    'z*z + c - 0.15*zAt(20)', 'z/(c+1) + c',
    'z*z + 0.01*zeta(c)', 'z*1e21 + c',
  ]
  let cases = 0
  device.pushErrorScope('validation')
  try {
    for (const iterationFunction of expressions) {
      // Also cover the shared iterate() helper through the existing SS path.
      for (const supersampling of [0, 2]) for (const doSmooth of [false, true]) {
        for (const [max_iter, z0] of [[1, [0, 0]], [2, [0.25, -0.1]], [1000, [0, 0]], [1000, [32, 0]]]) {
          const params = {
            w: 63, h: 35, max_iter, refr: -2, refi: -1, ddr0: 0, ddi0: 0,
            ddr: 3 / 63, ddi: 3 / 63, doSmooth, bailout: doSmooth ? 256 : 16,
            supersampling, iterationFunction, z0,
          }
          const expected = await baseline.renderDirect(params)
          const actual = await optimized.renderDirect(params)
          assert(baseline.pipeline.pipeline && optimized.pipeline.pipeline, 'Parity test did not render on GPU')
          for (const key of ['values', 'smooth', 'signs', 'zreal', 'zimag']) {
            if (expected[key] === null && actual[key] === null) continue
            assert(expected[key]?.length === actual[key]?.length &&
              expected[key].every((value, i) => Object.is(value, actual[key][i])),
              `Custom shader parity: ${key}, ${iterationFunction}, SS=${supersampling}, smooth=${doSmooth}, maxIter=${max_iter}, z0=${z0}`)
          }
          cases++
        }
      }
    }
    console.info('Custom iteration-limit shader parity', { cases })
  } finally {
    const error = await device.popErrorScope()
    baseline.pipeline.dispose()
    optimized.pipeline.dispose()
    assert(!error, error?.message)
  }
}

import { MandelbrotWebGPU } from '../mandelbrotWebGPU.mjs'
import { WorkerContext } from '../workerContext.mjs'
import * as fxp from '../fxp.mjs'
import { legacyPerturbationShader } from './fixtures/legacyPerturbationShader.mjs'

const assert = (condition, message) => { if (!condition) throw new Error(message) }

export async function checkPerturbationShaderParity(device) {
  const makeRenderer = () => {
    const renderer = new MandelbrotWebGPU({ onGpuUpdate(answer) {
      if (answer.isFinished) renderer.output = answer
    } }, new WorkerContext(), error => { throw new Error(error) }, { devicePromise: Promise.resolve(device) })
    const calculateReference = renderer.calculate_reference.bind(renderer)
    renderer.calculate_reference = async (...args) => {
      const ref = await calculateReference(...args)
      const failure = renderer.failNextReference
      renderer.failNextReference = null
      if (failure === 'length') {
        // Force reference exhaustion in the first pass, then allow normal retry.
        // The backing data stays intact; only the shader-visible length changes.
        return { ...ref, size: 1 }
      }
      if (failure === 'tie') return { ...ref, size: renderer.max_iter }
      if (failure === 'error-bound') {
        return { ...ref, zqErrorBoundBuffer: new Float32Array(ref.size).fill(1e30) }
      }
      return ref
    }
    return renderer
  }
  const baseline = makeRenderer()
  baseline.mandelbrotPipeline.getShadercode = legacyPerturbationShader
  const optimized = makeRenderer()
  const cases = []
  device.pushErrorScope('validation')
  try {
    // Odd sizes exercise the final partial workgroup. Cover every UI sample count,
    // smooth on/off, interior/escaping pixels, and multiple reference passes.
    for (const supersampling of [0, 2, 4, 8, 16, 32]) {
      for (const smooth of [false, true]) {
        for (const [zoom, failure] of [[1, null], [1e4, null], [1e10, null], [1, 'length'], [1, 'error-bound'], [1e10, 'tie']]) {
          const cx = fxp.fromNumber(-0.743643887037151, 80)
          const cy = fxp.fromNumber(0.13182590420533, 80)
          const half = fxp.fromNumber(2 / zoom, 80)
          const task = {
            w: 17, h: 13, frameWidth: 17, frameHeight: 13, xOffset: 0, yOffset: 0,
            frameTopLeft: [cx.subtract(half), cy.subtract(half)],
            frameBottomRight: [cx.add(half), cy.add(half)],
            maxIter: 1000, precision: 80, smooth, supersampling, escapeRadius: 4,
            fractalType: 'mandelbrot', resetCaches: true, paramHash: 'shader-parity',
            skipTopLeft: false, finalOnly: true,
          }
          for (const renderer of [baseline, optimized]) {
            renderer.failNextReference = failure
            await renderer.process({ ...task, jobToken: crypto.randomUUID(), jobId: crypto.randomUUID() })
            assert(renderer.output?.isFinished && !renderer.output.error, 'Shader parity render failed')
          }
          for (const name of ['values', 'smooth', 'signs', 'zreal', 'zimag']) {
            const expected = baseline.output[name]
            const actual = optimized.output[name]
            assert(expected.length === actual.length && expected.every((v, i) => Object.is(v, actual[i])),
              `Shader parity: ${name}, SS=${supersampling}, smooth=${smooth}, zoom=${zoom}, failure=${failure}`)
          }
          assert(baseline.referencePoints.length === optimized.referencePoints.length, 'Reference retry behavior changed')
          if (failure && failure !== 'tie') assert(optimized.referencePoints.length > 1, `Missing ${failure} retry coverage`)
          cases.push({ supersampling, smooth, zoom, failure, references: optimized.referencePoints.length })
        }
      }
    }
    assert(cases.some(test => test.supersampling && test.references > 1), 'Missing failed-sample/retry coverage')
    console.info('Perturbation shader parity', cases)
  } finally {
    const error = await device.popErrorScope()
    baseline.mandelbrotPipeline.dispose()
    optimized.mandelbrotPipeline.dispose()
    assert(!error, error?.message)
  }
}

import * as fxp from '../fxp.mjs'
import { MandelbrotCustom } from '../mandelbrotCustom.mjs'
import { MandelbrotFloat } from '../mandelbrotFloat.mjs'

function assert(condition, message) {
  if (!condition) throw new Error(message)
}

function equal(expected, actual) {
  for (const key of ['values', 'smooth', 'signs', 'zreal', 'zimag', 'otData']) {
    const a = expected[key],
      b = actual[key]
    assert(a?.length === b?.length, `${key}: length`)
    if (a) for (let i = 0; i < a.length; i++) assert(Object.is(a[i], b[i]), `${key}[${i}]`)
  }
}

const worker = new Worker('../worker.js', { type: 'module' })
function send(message) {
  return new Promise((resolve, reject) => {
    worker.onmessage = ({ data }) => (data.type === 'error' ? reject(new Error(data.message)) : resolve(data))
    worker.onerror = reject
    worker.postMessage(message)
  })
}

const jobToken = URL.createObjectURL(new Blob())
const base = {
  type: 'task',
  w: 7,
  h: 5,
  frameWidth: 28,
  frameHeight: 5,
  xOffset: 0,
  yOffset: 0,
  frameTopLeft: [fxp.fromNumber(-1.7), fxp.fromNumber(-0.8)],
  frameBottomRight: [fxp.fromNumber(0.4), fxp.fromNumber(0.7)],
  requiredPrecision: 40,
  maxIter: 40,
  smooth: true,
  supersampling: 0,
  jobToken,
}

try {
  const spec = {
    shape: 'bitmap',
    mode: 'closest',
    angle: 0.2,
    size: 1,
    bitmapWidth: 2,
    bitmapHeight: 1,
    bitmapData: new Uint8ClampedArray([255, 0, 0, 255, 0, 255, 0, 0]),
  }
  for (const [iteration, fractalType] of ['mandelbrot', 'custom', 'julia-custom', 'julia', 'custom'].entries()) {
    if (iteration === 2) Object.assign(spec, { angle: 1.1, size: 0.6, bitmapWidth: 1, bitmapHeight: 2 })
    const useTrap = iteration !== 3
    const tasks = Array.from({ length: 4 }, (_, i) => ({
      ...base,
      xOffset: i * 7,
      fractalType,
      useTrap,
      iterationFunction: 'z*z+c+0.1*zDelay(2)',
      juliaRe: -0.7,
      juliaIm: 0.2,
    }))
    const result = await send({ type: 'batch', tasks, trapSpec: iteration === 0 || iteration === 2 ? spec : undefined })
    assert(result.type === 'batch-answer' && result.answers.length === 4, 'Incorrect batch response')
    for (let i = 0; i < tasks.length; i++) {
      const Cls = fractalType.includes('custom') ? MandelbrotCustom : MandelbrotFloat
      const expected = await new Cls({ shouldStop: () => false }).process({
        ...tasks[i],
        trapSpec: useTrap ? spec : null,
      })
      equal(expected, result.answers[i])
      assert(!('trapSpec' in result.answers[i].task), 'Trap bitmap copied back with a tile')
    }
  }
  // A revoked job must stop even while a long-running task blocks onmessage.
  const stoppedToken = URL.createObjectURL(new Blob())
  URL.revokeObjectURL(stoppedToken)
  const stopped = await send({
    type: 'batch',
    tasks: [
      {
        ...base,
        w: 16,
        h: 16,
        frameWidth: 16,
        frameHeight: 16,
        frameTopLeft: [fxp.fromNumber(0), fxp.fromNumber(0)],
        frameBottomRight: [fxp.fromNumber(0), fxp.fromNumber(0)],
        fractalType: 'mandelbrot',
        maxIter: 10000000,
        jobToken: stoppedToken,
        useTrap: false,
      },
    ],
  })
  assert(stopped.answers[0].values.includes(0), 'Revoked job continued to completion')
  const next = await send({ type: 'batch', tasks: [{ ...base, fractalType: 'mandelbrot', useTrap: false }] })
  equal(await new MandelbrotFloat({ shouldStop: () => false }).process(base), next.answers[0])
  document.querySelector('#result').textContent =
    'PASS: four-tile batches, bitmap reuse/update, history, Julia, trap OFF/ON, transfer channels, cancellation, next job'
  document.documentElement.dataset.result = 'passed'
} catch (error) {
  document.querySelector('#result').textContent = error.stack
  document.documentElement.dataset.result = 'failed'
  console.error(error)
} finally {
  worker.terminate()
  URL.revokeObjectURL(jobToken)
}

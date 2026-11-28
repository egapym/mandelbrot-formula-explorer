import assert from 'node:assert/strict'
import test from 'node:test'
import { compileIterationFunction } from '../customFunctionParser.mjs'
import * as fxp from '../fxp.mjs'
import { MandelbrotCustom } from '../mandelbrotCustom.mjs'
import { calculatePixelOrbitTrap, TRAP_MODE, TRAP_SHAPE } from '../orbitTrap.mjs'
import { OrbitReplay } from '../orbitReplay.mjs'
import {
  COLOR_PATTERN, CustomOrbitTrapPalette, DEFAULT_ORBIT_TRAP_PATTERN_COLORS, getPalette, initPallet, makeOrbitTrapColorFn,
} from '../palette.js'

const context = { shouldStop: () => false }
const baseTask = {
  w: 7,
  h: 5,
  frameWidth: 7,
  frameHeight: 5,
  xOffset: 0,
  yOffset: 0,
  frameTopLeft: [fxp.fromNumber(-1.7), fxp.fromNumber(-0.8)],
  frameBottomRight: [fxp.fromNumber(0.4), fxp.fromNumber(0.7)],
  maxIter: 31,
  smooth: true,
  supersampling: 0,
  z0Real: 0.1,
  z0Imag: -0.05,
  juliaRe: -0.7,
  juliaIm: 0.2,
}

// Always compute the trap in a second pass, independent of the recorded trace.
class SeparateTrapRenderer extends MandelbrotCustom {
  calculateOrbitTraps(...args) {
    this._trapReplay = null
    return super.calculateOrbitTraps(...args)
  }
}

function equalChannels(expected, actual) {
  for (const key of ['values', 'smooth', 'signs', 'zreal', 'zimag', 'otData']) {
    assert.deepEqual(actual[key], expected[key], key)
  }
}

test('custom Cosine RGB interpolates between its selected colors', () => {
  const pattern = COLOR_PATTERN.COSINE_RGB
  const grayscale = makeOrbitTrapColorFn(pattern, [[0, 0, 0], [255, 255, 255]], 1)
  assert.deepEqual(grayscale(0), [0, 0, 0])
  assert.deepEqual(grayscale(0.25), [255, 255, 255])
  assert.deepEqual(grayscale(1), [0, 0, 0])
  const [red, green, blue] = grayscale(0.0625)
  assert.ok(red > 0 && red < 255)
  assert.equal(red, green)
  assert.equal(red, blue)

  const colored = makeOrbitTrapColorFn(pattern, [[200, 0, 0], [0, 0, 240]], 1)
  assert.deepEqual(colored(0.0625), [100, 0, 120])
  assert.deepEqual(makeOrbitTrapColorFn(pattern)(0), [0, 37, 128])
})

test('custom palettes start with the corresponding Mandelbrot and Cross colors', () => {
  assert.deepEqual(initPallet(getPalette('mandelbrot_custom'), 1, 0, 1, 100), initPallet(getPalette('mandelbrot'), 1, 0, 1, 100))

  const customTrap = new CustomOrbitTrapPalette()
  const crossTrap = getPalette('orbit_trap_ring')
  for (const distance of [0, 0.01, 0.0625, 0.25, 1]) {
    assert.deepEqual(customTrap._colorFn(distance), crossTrap._colorFn(distance))
  }
  assert.deepEqual(DEFAULT_ORBIT_TRAP_PATTERN_COLORS[COLOR_PATTERN.COSINE_RGB], [[0, 37, 128], [255, 218, 128]])
  customTrap.setPatternColors(COLOR_PATTERN.COSINE_RGB, [[0, 0, 0], [255, 255, 255]])
  assert.deepEqual(customTrap._colorFn(0), [0, 0, 0])
  customTrap.resetTrapSpec()
  assert.deepEqual(customTrap._colorFn(0), crossTrap._colorFn(0))
})

test('Bitmap Orbit Trap uses Background even with no image selected', () => {
  const palette = new CustomOrbitTrapPalette()
  const rgba = new Uint8ClampedArray(8)
  const values = Int32Array.of(2, 5)
  const trapValues = Float32Array.of(0, 0.25)
  palette.inSetColor = [17, 34, 51]
  palette.updateTrapSpec({ shape: TRAP_SHAPE.BITMAP, bitmapBackgroundColor: '#aabbcc' })
  palette.renderPixels(rgba, null, values, null, null, null, false, null, null, trapValues)
  assert.deepEqual(rgba, Uint8ClampedArray.of(17, 34, 51, 255, 170, 187, 204, 255))
})

test('history-free generic expressions leave supplied history untouched', () => {
  for (const expr of ['cos(z*z)+c', 'z*z+c+0.01*n']) {
    const history = { z: [] }
    compileIterationFunction(expr)(0.1, 0.2, -0.3, 0.4, 3, history)
    assert.deepEqual(history.z, [], expr)
  }
  const history = { z: [] }
  const fn = compileIterationFunction('zAt(3)+zDelay(1)')
  assert.deepEqual(fn(1, 2, 0, 0, 0, history), [0, 0])
  assert.deepEqual(fn(3, 4, 0, 0, 1, history), [1, 2])
  assert.deepEqual(fn(5, 6, 0, 0, 3, history), [5, 6])
})

test('trap reuse preserves every output channel, including history and supersampling', async () => {
  let cases = 0
  for (const fractalType of ['custom', 'julia-custom']) {
    for (const shape of Object.values(TRAP_SHAPE)) {
      for (const mode of Object.values(TRAP_MODE)) {
        for (const supersampling of [0, 2]) {
          const task = {
            ...baseTask,
            fractalType,
            supersampling,
            skipTopLeft: cases % 2 === 0,
            iterationFunction:
              cases % 3 === 0 ? 'z*z+c+0.1*zDelay(2)' : cases % 3 === 1 ? 'cos(z*z)+c' : 'z*z+c+0.01*n',
            trapSpec: {
              shape,
              mode,
              size: 0.7,
              angle: 0.4,
              threshold: 0.8,
              captureStep: 3,
              bitmapWidth: 2,
              bitmapHeight: 1,
              bitmapData: new Uint8ClampedArray([255, 0, 0, 255, 0, 255, 0, 0]),
            },
          }
          equalChannels(
            await new SeparateTrapRenderer(context).process(task),
            await new MandelbrotCustom(context).process(task),
          )
          cases++
        }
      }
    }
  }
  assert.equal(cases, 216)
})

test('long history traces fall back correctly beyond the cached prefix', async () => {
  const task = {
    ...baseTask,
    w: 2,
    h: 2,
    frameWidth: 2,
    frameHeight: 2,
    frameTopLeft: [fxp.fromNumber(-0.02), fxp.fromNumber(-0.02)],
    frameBottomRight: [fxp.fromNumber(0.02), fxp.fromNumber(0.02)],
    maxIter: 5000,
    fractalType: 'custom',
    iterationFunction: '0.7*z+0.01*zDelay(1)+c',
    trapSpec: { shape: 'line', mode: 'average', size: 1, angle: 0.4 },
  }
  equalChannels(
    await new SeparateTrapRenderer(context).process(task),
    await new MandelbrotCustom(context).process(task),
  )
})

test('matching replay avoids evaluation; divergent traces never resume replay', () => {
  let calls = 0
  const replay = new OrbitReplay((zr, zi) => {
    calls++
    return [zr + 1, zi]
  }, 5)
  replay.reset()
  replay.record(0, 0, 0, 0, 1)
  replay.record(1, 0, 0, 0, 2)
  assert.deepEqual([...replay.play(0, 0, 0, 0, 1)], [1, 0])
  assert.equal(calls, 2)
  replay.play(8, 0, 0, 0, 2)
  replay.play(1, 0, 0, 0, 2)
  assert.equal(calls, 4)
  replay.reset()
  replay.record(0, 0, 0, 0, 1)
  replay.play(-0, 0, 0, 0, 1)
  assert.equal(calls, 6, 'signed zero must trigger the original evaluator')
})

test('trap preparation notices mutations of the same spec object', () => {
  const step = (zr, zi, cr, ci) => [zr * zr - zi * zi + cr, 2 * zr * zi + ci]
  for (const shape of ['line', 'square', 'triangle', 'parabola', 'bitmap']) {
    const spec = { shape, mode: 'closest', size: 1, angle: 0.2, bitmapWidth: 2, bitmapHeight: 1 }
    const value = (trap) => calculatePixelOrbitTrap(-0.7, 0.2, 0.1, -0.1, step, 40, trap)
    value(spec)
    Object.assign(spec, { size: 0.8, angle: 1.1, bitmapWidth: 1, bitmapHeight: 3 })
    assert.equal(value(spec), value({ ...spec }))
  }
})

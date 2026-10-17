import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import test from 'node:test'
import vm from 'node:vm'
import { getBuddhabrotProjection } from '../buddhabrotProjection.mjs'
import { DEFAULT_RENDER_POINT_BATCH_SIZE, normalizeRenderPointBatchSize } from '../buddhabrotRenderConfig.mjs'
import { compileIterationFunction } from '../customFunctionParser.mjs'
import * as fxp from '../fxp.mjs'
import { MandelbrotCustom } from '../mandelbrotCustom.mjs'

const workerSource = readFileSync(new URL('../buddhabrotWorker.mjs', import.meta.url), 'utf8')
  .replace(/^import .*$/gm, '')
  .replace('    const keep =', '    observe(escaped, trajLen, cr, ci)\n    const keep =')

async function sample(expr, options = {}, oracle) {
  const orbits = []
  const context = vm.createContext({
    self: {},
    console,
    postMessage() {},
    setTimeout,
    Date,
    Math: Object.assign(Object.create(Math), { random: () => 0.25 }),
    compileIterationFunction,
    getBuddhabrotProjection,
    DEFAULT_RENDER_POINT_BATCH_SIZE,
    normalizeRenderPointBatchSize,
    observe(escaped, length, cr, ci) {
      orbits.push({ escaped, length, c: [cr, ci] })
    },
  })
  vm.runInContext(workerSource, context)
  vm.runInContext('running = true; currentJobId = 1', context)
  const compiled = compileIterationFunction(expr)
  await context.runSampling({
    samples: 128,
    width: 32,
    height: 32,
    maxIter: 64,
    center: { x: 0, y: 0 },
    zoom: 1,
    escapeRadius: 4,
    iterationFunction: expr,
    ...options,
    iterationFunctionCompiled(zr, zi, cr, ci, n, history) {
      const actual = compiled(zr, zi, cr, ci, n, history)
      if (oracle) assert.deepEqual(actual, oracle([zr, zi], [cr, ci]), expr)
      return actual
    },
  })
  return orbits
}

const add = (a, b) => [a[0] + b[0], a[1] + b[1]]
const mul = (a, b) => [a[0] * b[0] - a[1] * b[1], a[0] * b[1] + a[1] * b[0]]
const cases = [
  ['z+i', (z) => add(z, [0, 1])],
  ['c*i', (_z, c) => [-c[1], c[0]]],
  ['c+i', (_z, c) => add(c, [0, 1])],
  ['z*z+c*i', (z, c) => add(mul(z, z), [-c[1], c[0]])],
  ['z*z+c+i', (z, c) => add(add(mul(z, z), c), [0, 1])],
  ['z*z+i+c/i', (z, c) => add(add(mul(z, z), [0, 1]), [c[1], -c[0]])],
]

test('ordinary CPU and Buddhabrot match independent complex arithmetic and escape classification', async () => {
  for (const [expr, oracle] of cases) {
    const renderer = new MandelbrotCustom({})
    renderer.compiledFunction = compileIterationFunction(expr)
    for (const orbit of await sample(expr, {}, oracle)) {
      let z = [0, 0]
      let escaped = false
      let insidePoints = 0
      for (let n = 0; n < 64; n++) {
        z = oracle(z, orbit.c)
        if (z[0] * z[0] + z[1] * z[1] > 16) {
          escaped = true
          break
        }
        insidePoints++
      }
      assert.equal(orbit.escaped, escaped, expr)
      assert.equal(orbit.length, insidePoints, expr)
      assert.equal(renderer.iterate(...orbit.c, 64, 16)[0] !== 2, escaped, expr)
    }
  }
})

test('iteration limits include exactly maxIter evaluations, including Julia', async () => {
  const renderer = new MandelbrotCustom({})
  renderer.compiledFunction = compileIterationFunction('z+i')
  renderer.juliaRe = renderer.juliaIm = 0
  for (const maxIter of [0, 1, 4, 5]) {
    const expected = maxIter >= 5
    assert.equal(renderer.iterate(0, 0, maxIter, 16)[0] !== 2, expected)
    assert.equal(renderer.juliaIterate(0, 0, maxIter, 16)[0] !== 2, expected)
    for (const orbit of await sample('z+i', { maxIter })) assert.equal(orbit.escaped, expected)
  }
})

test('Smooth preserves escape classification for custom/Julia formulas and all sampling modes', async () => {
  for (const fractalType of ['custom', 'julia-custom']) {
    for (const supersampling of [0, 2]) {
      for (const [expr, radius, escaped] of [
        ['c+i', 4, true],
        ['c+i', 8, false],
        ['0.5*i', 0.25, true],
      ]) {
        for (const smooth of [false, true]) {
          const result = await new MandelbrotCustom({ shouldStop: () => false }).process({
            w: 1,
            h: 1,
            frameWidth: 1,
            frameHeight: 1,
            xOffset: 0,
            yOffset: 0,
            frameTopLeft: [fxp.fromNumber(0), fxp.fromNumber(fractalType === 'custom' ? 5 : 0)],
            frameBottomRight: [fxp.fromNumber(0), fxp.fromNumber(fractalType === 'custom' ? 5 : 0)],
            maxIter: 16,
            fractalType,
            iterationFunction: expr,
            escapeRadius: radius,
            juliaRe: 0,
            juliaIm: 5,
            smooth,
            supersampling,
          })
          assert.equal(
            result.values[0] >= 4,
            escaped,
            `${fractalType}, ${expr}, R=${radius}, SS=${supersampling}, smooth=${smooth}`,
          )
        }
      }
    }
  }
})

test('NaN, Infinity and extreme values escape rather than becoming a different formula', async () => {
  for (const expr of ['exp(1000)*i+c', 'exp(1000)', '1e21+i', '1e20+i']) {
    const renderer = new MandelbrotCustom({})
    renderer.compiledFunction = compileIterationFunction(expr)
    assert.ok(renderer.iterate(0, 0, 4, 16)[0] >= 4, expr)
    for (const orbit of await sample(expr, { maxIter: 4 })) {
      assert.equal(orbit.escaped, true, expr)
      assert.equal(orbit.length, 0, expr)
    }
  }
})

test('initial points outside the escape radius produce no Buddhabrot hits', async () => {
  const renderer = new MandelbrotCustom({})
  renderer.z0Imag = 5
  renderer.compiledFunction = compileIterationFunction('0')
  assert.ok(renderer.iterate(0, 0, 4, 16)[0] !== 2)
  for (const orbit of await sample('0', { z0Imag: 5 })) {
    assert.equal(orbit.escaped, true)
    assert.equal(orbit.length, 0)
  }
})

test('fast escape checks retain component limits for very large escape radii', async () => {
  for (const [expr, escapeRadius, escaped] of [
    ['8e19+8e19*i', 2e20, false],
    ['1e20+i', 2e20, true],
    ['exp(1000)', Infinity, true],
    ['exp(1000)*i+c', Infinity, true],
    ['0.5*i', 0, true],
  ]) {
    const orbits = await sample(expr, { escapeRadius, maxIter: 1, samples: 1 })
    assert.equal(orbits.length, 1, expr)
    assert.equal(orbits[0].escaped, escaped, expr)
  }
})

test('Orbit overlay uses the same radius, iteration limit and invalid-value rules', () => {
  const source = readFileSync(new URL('../index.js', import.meta.url), 'utf8')
    .split('function _computeOrbitPoints(')[1]
    .split('function _getOrbitGradientColor(')[0]
  const orbit = vm.runInNewContext(`(function _computeOrbitPoints(${source})`)
  const bailoutSource = readFileSync(new URL('../index.js', import.meta.url), 'utf8')
    .split('function _getOrbitBailout(')[1]
    .split('/**')[0]
  const bailout = vm.runInNewContext(`(function _getOrbitBailout(${bailoutSource})`, { BAILOUT_SMOOTH: 128 })
  assert.equal(bailout({ fractalType: 'custom', smooth: true, escapeRadius: 4 }), 16)
  assert.equal(bailout({ fractalType: 'julia-custom', smooth: true, escapeRadius: 0.25 }), 0.0625)
  for (const [expr, z0, limit, expected] of [
    ['z+i', 0, 4, false],
    ['z+i', 0, 5, true],
    ['0', 5, 4, true],
    ['exp(1000)*i+c', 0, 4, true],
    ['1e21+i', 0, 4, true],
  ]) {
    assert.equal(orbit(0, z0, 0, 0, compileIterationFunction(expr), null, limit, 16).escaped, expected, expr)
  }
})

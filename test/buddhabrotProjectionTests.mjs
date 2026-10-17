import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import test from 'node:test'
import vm from 'node:vm'
import { getBuddhabrotProjection } from '../buddhabrotProjection.mjs'
import { DEFAULT_RENDER_POINT_BATCH_SIZE, normalizeRenderPointBatchSize } from '../buddhabrotRenderConfig.mjs'
import { compileIterationFunction } from '../customFunctionParser.mjs'

const cases = [
  {
    expr: 'z*z+c*i',
    c: [0, -1],
    display: [
      [0, -1],
      [0, -2],
    ],
  },
  {
    expr: 'z*z+c/i',
    c: [0, 1],
    display: [
      [0, 1],
      [0, 2],
    ],
  },
  {
    expr: 'z*z+c+i',
    c: [1, -1],
    display: [
      [1, -1],
      [2, -1],
    ],
  },
  {
    expr: 'z*z+i+c/i',
    c: [1, 1],
    display: [
      [1, 1],
      [1, 2],
    ],
  },
  {
    expr: 'z*z+2*c+1+i',
    c: [0, -0.5],
    display: [
      [0, -0.5],
      [0.5, -0.5],
    ],
  },
  {
    expr: 'z*z+(c+i)*i',
    c: [0, -2],
    display: [
      [0, -2],
      [0, -3],
    ],
  },
]

const normalizeZero = (values) => values.map((v) => (v === 0 ? 0 : v))

test('affine parameter transforms are inverted without altering the iteration function', () => {
  for (const { expr, c, display } of cases) {
    const projection = getBuddhabrotProjection(expr)
    assert.equal(projection.identity, false, expr)
    const fn = compileIterationFunction(expr)
    let z = [0, 0]
    for (const expected of display) {
      z = fn(...z, ...c)
      assert.deepEqual(
        normalizeZero([
          projection.xx * z[0] + projection.xy * z[1] + projection.tx,
          projection.yx * z[0] + projection.yy * z[1] + projection.ty,
        ]),
        expected,
        expr,
      )
    }
  }
})

test('constant rotations, scalar order and equivalent grouping give the same display axes', () => {
  const expected = getBuddhabrotProjection('z*z+c*i')
  for (const expr of ['z^2+i*c', 'i*c+z*z', 'z*z+c*exp(i*pi/2)', 'z*z+c/(0-i)']) {
    const actual = getBuddhabrotProjection(expr)
    for (const key of ['xx', 'xy', 'yx', 'yy', 'tx', 'ty']) {
      assert.ok(Math.abs(actual[key] - expected[key]) < 1e-12, `${expr}: ${key}`)
    }
  }
  const opposite = getBuddhabrotProjection('z*z+c*i^2')
  assert.equal(opposite.xx, -1)
  assert.equal(opposite.yy, -1)
  const fractional = getBuddhabrotProjection('z*z+c*(1+i)^(-1)')
  assert.equal(fractional.xx, 1)
  assert.equal(fractional.yx, 1)
})

test('Julia, nonlinear/mixed c terms and singular transforms are never given a guessed inverse', () => {
  for (const expr of [
    'z*z+c',
    'z+i',
    '(z+i)^2+c',
    'z*c+i',
    'z*z+c*c',
    'z*z+sin(c)',
    'z*z+conj(c)',
    'z*z+0*c+i',
    'z*z+c/0',
    'z*z+c*1e-20',
  ]) {
    assert.equal(getBuddhabrotProjection(expr).identity, true, expr)
  }
  for (const type of ['julia', 'julia-custom']) {
    assert.equal(getBuddhabrotProjection('z*z+c*i', type).identity, true)
  }
})

const workerSource = readFileSync(new URL('../buddhabrotWorker.mjs', import.meta.url), 'utf8')
  .replace(/^import .*$/gm, '')
  .replace(/ {4}const sampleRe = [^\n]+/, '    const sampleRe = fixedC[0]')
  .replace(/ {4}const sampleIm = [^\n]+/, '    const sampleIm = fixedC[1]')
  .replace(
    '      const fx = (displayRe - left)',
    '      observe(pr, pi, displayRe, displayIm)\n      const fx = (displayRe - left)',
  )

test('actual CPU worker projects hits while retaining original orbit and escape decisions', async () => {
  for (const { expr, c, display } of cases) {
    const hits = []
    const raw = []
    const pixels = new Set()
    const context = vm.createContext({
      self: {},
      console,
      setTimeout,
      Date,
      Math,
      fixedC: c,
      postMessage(message) {
        if (message.type === 'chunk') for (const index of message.chunk.indices) pixels.add(index)
      },
      compileIterationFunction,
      getBuddhabrotProjection,
      DEFAULT_RENDER_POINT_BATCH_SIZE,
      normalizeRenderPointBatchSize,
      observe(zr, zi, re, im) {
        raw.push(normalizeZero([zr, zi]))
        hits.push(normalizeZero([re, im]))
      },
    })
    vm.runInContext(workerSource, context)
    vm.runInContext('running = true; currentJobId = 1', context)
    await context.runSampling({
      samples: 1,
      width: 32,
      height: 32,
      maxIter: 16,
      center: { x: 0, y: 0 },
      zoom: 0.25,
      escapeRadius: 4,
      mode: 'buddha',
      iterationFunction: expr,
      iterationFunctionCompiled: compileIterationFunction(expr),
    })
    assert.deepEqual(
      raw,
      [
        [1, 0],
        [2, 0],
      ],
      expr,
    )
    assert.deepEqual(hits, display, expr)
    assert.deepEqual(
      [...pixels].sort((a, b) => a - b),
      display.map(([x, y]) => Math.floor(y * 2 + 16) * 32 + Math.floor(x * 2 + 16)).sort((a, b) => a - b),
      expr,
    )
  }
})

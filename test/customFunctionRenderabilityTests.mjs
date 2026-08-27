/*
 * Custom function renderability tests
 *
 * Based on bertbaron/mandelbrot by Bert Baron
 * This file is part of the Mandelbrot Explorer project.
 * Licensed under GPL-3.0. See LICENSE file for details.
 *
 */

import assert from 'node:assert/strict'
import { compileIterationFunction, getParsedExpression } from '../customFunctionParser.mjs'
import { functionPresets } from '../functionPresets.mjs'
import { CUSTOM_FUNCTION_WGSL_HELPERS, jsExprToWGSL_safe } from '../wgslCompiler.mjs'

function linspace(a, b, n) {
  const out = []
  if (n === 1) return [(a + b) / 2]
  for (let i = 0; i < n; i++) out.push(a + (i * (b - a)) / (n - 1))
  return out
}

function hasBalancedDelimiters(expr) {
  const stack = []
  const pairs = new Map([
    [')', '('],
    [']', '['],
    ['}', '{'],
  ])
  for (const ch of expr) {
    if (ch === '(' || ch === '[' || ch === '{') {
      stack.push(ch)
    } else if (pairs.has(ch)) {
      if (stack.pop() !== pairs.get(ch)) return false
    }
  }
  return stack.length === 0
}

function hasVec2ResultShape(expr) {
  if (!hasBalancedDelimiters(expr)) return false
  const trimmed = expr.trim()
  return /^vec2<f32>\s*\(/.test(trimmed) || /^select\s*\(/.test(trimmed)
}

function testRenderability(expr, opts = {}) {
  const { grid = 21, maxIter = 100, bailout = 4, xmin = -2, xmax = 2, z0Real = 0, z0Imag = 0 } = opts
  // Normalize expression and attempt to compile
  const normalizedExpr = expr.normalize ? expr.normalize('NFKC') : expr
  let compiled
  try {
    compiled = compileIterationFunction(normalizedExpr)
  } catch (e) {
    // console.log(`${expr} : compile FAILED -> ${e.message}`);
    return { expr, compiled: false, renders: false, error: e.message }
  }

  const xs = linspace(xmin, xmax, grid)
  const ys = linspace(xmin, xmax, grid)
  let foundEscape = false
  let foundNonFinite = false
  let runtimeErrors = 0
  outer: for (const cr of xs) {
    for (const ci of ys) {
      // start iteration from preset z0 if provided
      let zR = z0Real,
        zI = z0Imag
      for (let it = 0; it < maxIter; it++) {
        let res
        try {
          res = compiled(zR, zI, cr, ci, it)
        } catch (e) {
          // runtime error for this probe
          runtimeErrors++
          break
        }
        if ((!Array.isArray(res) && !(res instanceof Float64Array)) || res.length !== 2) break
        zR = res[0]
        zI = res[1]
        if (!Number.isFinite(zR) || !Number.isFinite(zI)) {
          foundNonFinite = true
          // treat non-finite as a special case but do not mark as rendering-failure here
          break
        }
        const zq = zR * zR + zI * zI
        if (zq >= bailout) {
          foundEscape = true
          break outer
        }
      }
    }
  }

  // const compiledEmoji = compiled ? "✅" : "❌";
  // const rendersEmoji = foundEscape ? "🟢" : "🔴";
  // const label = expr.length > 40 ? expr : expr.padEnd(40);
  // console.log(`${label} compiled=${compiledEmoji}   renders=${rendersEmoji}   (escape=${foundEscape})`);
  // Per new spec: non-finite occurrences should NOT mark renders as failed.
  const renders = foundEscape || foundNonFinite

  // quick GPU-side WGSL validation for expressions that compiled successfully
  let wgslValid = true
  let wgslError = null
  try {
    const wgsl = jsExprToWGSL_safe(getParsedExpression(normalizedExpr))
    // look for adjacent tokens that would indicate a missing operator
    if (/\)\s+\(/.test(wgsl)) {
      wgslValid = false
      wgslError = 'missing operator between subexpressions'
    } else if (!hasVec2ResultShape(wgsl)) {
      wgslValid = false
      wgslError = 'malformed vec2 expression'
    }
  } catch (e) {
    wgslValid = false
    wgslError = e.message
  }

  return {
    expr,
    compiled: true,
    renders,
    nonFinite: foundNonFinite,
    runtimeErrors,
    wgslValid,
    wgslError,
  }
}

function testRiemannZetaKnownValues() {
  const zeta = compileIterationFunction('zeta(z)')
  const cases = [
    { s: [2, 0], expected: [Math.PI ** 2 / 6, 0], tolerance: 1e-12 },
    { s: [0, 0], expected: [-0.5, 0], tolerance: 1e-12 },
    { s: [-1, 0], expected: [-1 / 12, 0], tolerance: 1e-12 },
    { s: [-2, 0], expected: [0, 0], tolerance: 1e-9 },
    { s: [0.5, 14.1347251417347], expected: [0, 0], tolerance: 2e-9 },
  ]

  for (const { s, expected, tolerance } of cases) {
    const actual = zeta(s[0], s[1], 0, 0, 0)
    assert.ok(Math.abs(actual[0] - expected[0]) <= tolerance, `zeta(${s}) real part: ${actual[0]}`)
    assert.ok(Math.abs(actual[1] - expected[1]) <= tolerance, `zeta(${s}) imaginary part: ${actual[1]}`)
  }

  const pole = zeta(1, 0, 0, 0, 0)
  assert.ok(Math.abs(pole[0]) >= 1e19, 'zeta(1) should be treated as a pole')
  assert.match(CUSTOM_FUNCTION_WGSL_HELPERS, /fn complexZeta\(s: vec2<f32>\)/)
  assert.match(jsExprToWGSL_safe(getParsedExpression('zeta(z)')), /^complexZeta\(z\)$/)
}

function testRemovedFunctionsAreUnavailable() {
  const removedExpressions = ['floor(z) + c', 'min(z, 1) + c', 'max(z, -1) + c', 'clamp(z, -1, 1)^2 + c']
  const originalError = console.error
  console.error = () => {}

  try {
    for (const expr of removedExpressions) {
      assert.throws(() => compileIterationFunction(expr), /Unsupported function/)
      assert.throws(() => jsExprToWGSL_safe(getParsedExpression(expr)), /Unsupported function/)
    }
  } finally {
    console.error = originalError
  }
}

function testIterationIndexVariableAffectsModOrbit() {
  const fn = compileIterationFunction('mod(z*z, n) + c')
  const c = [-0.64, -0.015]
  const withIterationIndex = []
  const withoutIterationIndex = []
  let zrWith = 0
  let ziWith = 0
  let zrWithout = 0
  let ziWithout = 0

  for (let i = 0; i < 4; i++) {
    ;[zrWith, ziWith] = fn(zrWith, ziWith, c[0], c[1], i)
    ;[zrWithout, ziWithout] = fn(zrWithout, ziWithout, c[0], c[1])
    withIterationIndex.push([zrWith, ziWith])
    withoutIterationIndex.push([zrWithout, ziWithout])
  }

  assert.notDeepEqual(
    withIterationIndex,
    withoutIterationIndex,
    'orbit callers must pass the iteration index when an expression uses n',
  )
  assert.deepEqual(
    withoutIterationIndex.slice(1),
    withoutIterationIndex.slice(0, -1),
    'without n, mod(z*z, n) collapses to z=c on every step',
  )
}

function testGpuModuloSelfIdentity() {
  for (const operand of ['z', 'c', 'n']) {
    const wgsl = jsExprToWGSL_safe(getParsedExpression(`z*z + mod(${operand},${operand}) + c`))
    assert.match(wgsl, /^vec2<f32>\(/, `mod(${operand},${operand}) must still produce a complex WGSL expression`)
    assert.doesNotMatch(
      wgsl,
      /floor\(/,
      `mod(${operand},${operand}) must be folded to zero before f32 modulo arithmetic`,
    )
    assert.match(wgsl, /\+ c\.x/)
    assert.match(wgsl, /\+ c\.y/)
  }
}

async function main() {
  testRiemannZetaKnownValues()
  testRemovedFunctionsAreUnavailable()
  testIterationIndexVariableAffectsModOrbit()
  testGpuModuloSelfIdentity()
  const results = []
  // include a couple of expressions that previously triggered GPU bugs
  results.push(testRenderability('sin(Re(z)) + i*cos(Im(z)) + c'))
  results.push(testRenderability('Re(sin(z)) + i*Im(cos(z)) + c'))
  results.push(testRenderability('Re(z+1) + c'))
  results.push(testRenderability('Im(z+1) + c'))
  results.push(testRenderability('Re(z+1) + i*Im(z+c) + c'))
  results.push(testRenderability('|Re(z+1)| + i*|Im(z+c)| + c'))
  results.push(testRenderability('sin(z+1) + c'))
  results.push(testRenderability('sqrt(z+1) + c'))
  results.push(testRenderability('mod(z*z, 1) + c'))
  results.push(testRenderability('mod(z*z, n) + c'))
  results.push(testRenderability('zeta((0.35*z)) + c'))

  for (const preset of functionPresets) {
    const expr = preset.expr || preset
    const opts = {
      z0Real: preset.z0Real || 0,
      z0Imag: preset.z0Imag || 0,
    }
    results.push(testRenderability(expr, opts))
  }

  console.log('\nSummary:\n')
  for (const r of results) {
    const compiledEmoji = r.compiled ? '✅' : '❌'
    const rendersEmoji = r.renders ? '🟢' : '🔴'
    const wgslEmoji = r.wgslValid ? '✅' : '❌'
    const exprStr = String(r.expr)
    const label = exprStr.length > 60 ? exprStr : exprStr.padEnd(60)
    const nf = r.nonFinite ? ' ⚠️ non-finite' : ''
    const re = r.runtimeErrors ? ` (runtimeErr=${r.runtimeErrors})` : ''
    const wgslMsg = r.wgslValid ? '' : ` WGSL_FAIL:${r.wgslError}`
    console.log(`${label} compiled=${compiledEmoji}   renders=${rendersEmoji}${nf}${re}   wgsl=${wgslEmoji}${wgslMsg}`)
  }

  const compiledCount = results.filter((r) => r.compiled).length
  const rendersCount = results.filter((r) => r.renders).length
  const total = results.length
  console.log(`\ncompiled:${compiledCount}/${total}, renders:${rendersCount}/${total}`)

  const failures = results.filter((r) => !r.compiled || !r.renders || !r.wgslValid)
  if (failures.length > 0) {
    console.error(`\nFAILED:${failures.length}/${total}`)
    process.exitCode = 1
  }
}

if (import.meta.url === `file://${process.cwd()}/test/customFunctionRenderabilityTests.mjs`) {
  main()
} else {
  main()
}

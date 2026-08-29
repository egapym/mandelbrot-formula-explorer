/*
 * Fold pattern equivalence tests
 *
 * Based on bertbaron/mandelbrot by Bert Baron
 * This file is part of the Mandelbrot Formula Explorer project.
 * Licensed under GPL-3.0. See LICENSE file for details.
 *
 */

import assert from 'node:assert/strict'

import { compileIterationFunction, functionCache, getParsedExpression } from '../customFunctionParser.mjs'
import { jsExprToWGSL_safe } from '../wgslCompiler.mjs'

const burningShipExpr = '(|Re(z)| + i*|Im(z)|)^2 + c'
const foldExprs = ['fold(z)^2 + c', 'fold(z) * fold(z) + c']

const expectedParsed = getParsedExpression(burningShipExpr)
const expectedWgsl = jsExprToWGSL_safe(expectedParsed)

for (const expr of foldExprs) {
  functionCache.clear()
  const foldedParsed = getParsedExpression(expr)
  assert.equal(foldedParsed, expectedParsed, `${expr} should use the burning ship parsed expression`)
  assert.equal(jsExprToWGSL_safe(foldedParsed), expectedWgsl, `${expr} should use the burning ship WGSL expression`)

  const foldedFn = compileIterationFunction(expr)
  const burningShipFn = compileIterationFunction(burningShipExpr)
  for (const [zr, zi, cr, ci] of [
    [-1.2, -0.7, 0.1, 0.2],
    [0, 0, -0.4, 0.6],
    [1.5, -2.25, -0.3, -0.8],
  ]) {
    assert.deepEqual(
      Array.from(foldedFn(zr, zi, cr, ci)),
      Array.from(burningShipFn(zr, zi, cr, ci)),
      `${expr} should match burning ship at z=${zr}+${zi}i, c=${cr}+${ci}i`,
    )
  }
}

console.log('fold pattern equivalence ok')

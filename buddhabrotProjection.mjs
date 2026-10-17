import { getParsedExpression } from './customFunctionParser.mjs'

const IDENTITY = Object.freeze({ xx: 1, xy: 0, yx: 0, yy: 1, tx: 0, ty: 0, identity: true })
const projectionCache = new Map()
const add = (a, b) => [a[0] + b[0], a[1] + b[1]]
const multiply = (a, b) => [a[0] * b[0] - a[1] * b[1], a[0] * b[1] + a[1] * b[0]]
const hasC = (value) => value.a[0] !== 0 || value.a[1] !== 0
const constant = (b) => ({ a: [0, 0], b, residual: false })

function splitArgs(text) {
  const parts = []
  let depth = 0
  let start = 0
  for (let i = 0; i < text.length; i++) {
    if (text[i] === '(' || text[i] === '[') depth++
    else if (text[i] === ')' || text[i] === ']') depth--
    else if (text[i] === ',' && depth === 0) {
      parts.push(text.slice(start, i).trim())
      start = i + 1
    }
  }
  parts.push(text.slice(start).trim())
  return parts
}

function scalar(text) {
  if (text === 'Math.PI') return Math.PI
  if (text === 'Math.E') return Math.E
  return /^-?(?:\d+\.?\d*|\.\d+)(?:e[+-]?\d+)?$/i.test(text) ? Number(text) : NaN
}

// Inspect the parser's structured complex operations, never estimate a
// derivative from sample values: nonlinear/mixed c expressions must stay raw.
// A residual is a z-dependent term that is independent of c.
function affine(expression) {
  const number = scalar(expression)
  if (Number.isFinite(number)) return constant([number, 0])
  if (expression.replace(/\s/g, '') === '[cReal,cImag]') {
    return { a: [1, 0], b: [0, 0], residual: false }
  }
  if (expression.startsWith('[') && expression.endsWith(']')) {
    const pair = splitArgs(expression.slice(1, -1)).map(scalar)
    if (pair.length === 2 && pair.every(Number.isFinite)) return constant(pair)
  }
  const match = expression.match(/^(complex\w+)\(([\s\S]*)\)$/)
  if (match) {
    const args = splitArgs(match[2])
    const left = affine(args[0])
    const right = args.length === 2 ? affine(args[1]) : null
    if (left && right) {
      if (match[1] === 'complexAdd' || match[1] === 'complexSubtract') {
        const sign = match[1] === 'complexAdd' ? 1 : -1
        return {
          a: add(
            left.a,
            right.a.map((v) => sign * v),
          ),
          b: add(
            left.b,
            right.b.map((v) => sign * v),
          ),
          residual: left.residual || right.residual,
        }
      }
      if (match[1] === 'complexMultiply') {
        const scale = !hasC(left) && !left.residual ? left : !hasC(right) && !right.residual ? right : null
        if (scale) {
          const value = scale === left ? right : left
          return { a: multiply(value.a, scale.b), b: multiply(value.b, scale.b), residual: value.residual }
        }
      }
      if (match[1] === 'complexDivide' && !hasC(right) && !right.residual) {
        const norm = right.b[0] ** 2 + right.b[1] ** 2
        if (norm >= 1e-12) {
          const inverse = [right.b[0] / norm, -right.b[1] / norm]
          return { a: multiply(left.a, inverse), b: multiply(left.b, inverse), residual: left.residual }
        }
      }
      if (match[1] === 'complexPower' && !hasC(right) && !right.residual && right.b[1] === 0) {
        const exponent = right.b[0]
        if (exponent === 1) return left
        if (!hasC(left) && !left.residual && Number.isInteger(exponent) && Math.abs(exponent) <= 64) {
          let result = [1, 0]
          for (let n = 0; n < Math.abs(exponent); n++) result = multiply(result, left.b)
          if (exponent >= 0) return constant(result)
          const norm = result[0] ** 2 + result[1] ** 2
          if (norm >= 1e-12) return constant([result[0] / norm, -result[1] / norm])
        }
      }
    }
    if (left && !hasC(left) && !left.residual && args.length === 1) {
      const [x, y] = left.b
      if (match[1] === 'complexExp') return constant([Math.exp(x) * Math.cos(y), Math.exp(x) * Math.sin(y)])
      if (match[1] === 'complexSin') return constant([Math.sin(x) * Math.cosh(y), Math.cos(x) * Math.sinh(y)])
      if (match[1] === 'complexCos') return constant([Math.cos(x) * Math.cosh(y), -Math.sin(x) * Math.sinh(y)])
      if (match[1] === 'complexConj') return constant([x, -y])
    }
  }
  if (/\b(?:cReal|cImag)\b/.test(expression)) return null
  // Keep complete nonlinear z subexpressions intact; constants inside them
  // belong to f(z), not to the separately added parameter transformation.
  if (/\b(?:z|zReal|zImag|iterIndex|history|complexZAt|complexZDelay)\b/.test(expression)) {
    return { a: [0, 0], b: [0, 0], residual: true }
  }
  return null
}

/**
 * For f(z) + a*c + b, display orbit hits at (z-b)/a so their axes match
 * ordinary parameter-plane rendering. Iteration, sampling and escape tests
 * still use the original formula/coordinates. Julia's axes already represent
 * z0, so Julia and expressions without a proven invertible affine c term keep
 * their original projection.
 */
function calculateProjection(expression) {
  try {
    const value = affine(getParsedExpression(expression))
    if (!value || !hasC(value)) return IDENTITY
    const [ar, ai] = value.a
    const norm = ar * ar + ai * ai
    if (!(norm >= 1e-12) || !Number.isFinite(norm)) return IDENTITY
    const xx = ar / norm
    const yx = -ai / norm
    const tx = -(xx * value.b[0] - yx * value.b[1])
    const ty = -(yx * value.b[0] + xx * value.b[1])
    if (![xx, yx, tx, ty].every((v) => Number.isFinite(v) && Math.abs(v) < 1e20)) return IDENTITY
    if (xx === 1 && yx === 0 && tx === 0 && ty === 0) return IDENTITY
    return { xx, xy: -yx, yx, yy: xx, tx, ty, identity: false }
  } catch {
    return IDENTITY
  }
}

export function getBuddhabrotProjection(expression, fractalType = 'custom') {
  if (!expression || fractalType === 'julia' || fractalType === 'julia-custom') return IDENTITY
  if (projectionCache.has(expression)) return projectionCache.get(expression)
  const projection = Object.freeze(calculateProjection(expression))
  if (projectionCache.size >= 100) projectionCache.delete(projectionCache.keys().next().value)
  projectionCache.set(expression, projection)
  return projection
}

// Emit only necessary operations. In particular, a quarter-turn becomes y/-x,
// without zero multiplies, unit multiplies or a general matrix in the GPU loop.
export function getBuddhabrotProjectionWGSL(projection) {
  const coordinate = (real, imag, offset) => {
    const terms = []
    for (const [coefficient, name] of [
      [real, 'x'],
      [imag, 'y'],
    ]) {
      if (coefficient === 1) terms.push(name)
      else if (coefficient === -1) terms.push(`(-${name})`)
      else if (coefficient !== 0) terms.push(`(${coefficient.toExponential(16)} * ${name})`)
    }
    if (offset !== 0) terms.push(offset.toExponential(16))
    return terms.length ? terms.join(' + ') : '0.0'
  }
  return {
    real: coordinate(projection.xx, projection.xy, projection.tx),
    imag: coordinate(projection.yx, projection.yy, projection.ty),
  }
}

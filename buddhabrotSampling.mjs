/** This file is part of Mandelbrot Formula Explorer. Licensed under GPL-3.0. */
import { compileIterationFunction } from './customFunctionParser.mjs'
import { getBuddhabrotParameterCoefficient } from './buddhabrotProjection.mjs'

// Only first-step survivors can contribute orbit points: the escaping point
// itself is excluded by both samplers. For z1=a*c+b their domain is an exact disk.
export function getInitialBuddhabrotSamplingRange({
  iterationFunction = 'z*z + c', fractalType = 'custom',
  z0Real = 0, z0Imag = 0, escapeRadius = 4,
} = {}) {
  const disk = (x, y, radius) => ({
    samplingCenter: { x: x || 0, y: y || 0 }, samplingZoom: 2 / radius, samplingRadius: radius,
  })
  const fallback = () => disk(0, 0, 4)
  if (!(escapeRadius > 0) || !Number.isFinite(escapeRadius)) return fallback()
  // Julia samples z0 rather than c; initial points outside R never contribute.
  if (fractalType === 'julia' || fractalType === 'julia-custom') return disk(0, 0, escapeRadius)
  try {
    const a = getBuddhabrotParameterCoefficient(iterationFunction)
    if (!a || ![z0Real, z0Imag].every(Number.isFinite)) return fallback()
    const b = compileIterationFunction(iterationFunction)(z0Real, z0Imag, 0, 0, 0, { z: [] })
    if (!b.every((value) => Number.isFinite(value) && Math.abs(value) < 1e20)) return fallback()
    const magnitude = Math.hypot(...a)
    const radius = escapeRadius / magnitude
    const x = -(b[0] * (a[0] / magnitude) + b[1] * (a[1] / magnitude)) / magnitude
    const y = -(b[1] * (a[0] / magnitude) - b[0] * (a[1] / magnitude)) / magnitude
    // The same domain is sent to float32 GPU uniforms/shader constants.
    if (![x, y, radius, 2 / radius].every((value) => Number.isFinite(Math.fround(value))) ||
        Math.fround(radius) <= 0 || Math.fround(2 / radius) <= 0) return fallback()
    return disk(x, y, radius)
  } catch {
    return fallback()
  }
}

// Replaying seeds avoids storing millions of coordinate pairs. Distinct worker
// and GPU batch seeds preserve the full finite sampling sequence.
export function getBuddhabrotSampleSeed(seed, stream) {
  let value = ((seed >>> 0) + Math.imul(stream + 1, 0x9e3779b9)) >>> 0
  value = Math.imul(value ^ (value >>> 16), 0x21f0aaad)
  value = Math.imul(value ^ (value >>> 15), 0x735a2d97)
  return ((value ^ (value >>> 15)) >>> 0) || 1
}

export class FixedBuddhabrotSamples {
  constructor(createSeed = () => crypto.getRandomValues(new Uint32Array(1))[0]) {
    this.createSeed = createSeed
    this.views = new Map()
  }
  clear() { this.views.clear() }
  reseed(kind) {
    const view = this.views.get(kind)
    if (view) view.sampleSeed = this.createSeed()
  }
  apply(params, kind, baseline = params) {
    if (!this.views.has(kind)) {
      this.views.set(kind, {
        sampleSeed: baseline.sampleSeed ?? this.createSeed(),
        samplingCenter: { ...(baseline.samplingCenter || baseline.center) },
        samplingZoom: baseline.samplingZoom || baseline.zoom,
        samplingRadius: baseline.samplingRadius,
        samplingAspect: baseline.samplingAspect ?? baseline.height / baseline.width,
      })
    }
    return { ...params, ...this.views.get(kind) }
  }
}

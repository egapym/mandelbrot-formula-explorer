/** This file is part of Mandelbrot Formula Explorer. Licensed under GPL-3.0. */
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
        samplingAspect: baseline.samplingAspect ?? baseline.height / baseline.width,
        convergenceView: baseline.convergenceView || {
          center: { ...baseline.center }, zoom: baseline.zoom,
          width: baseline.width, height: baseline.height,
        },
      })
    }
    return { ...params, ...this.views.get(kind) }
  }
}

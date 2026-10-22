export const BUDDHA_DEFAULT_BRIGHTNESS = 1.0
export const BUDDHA_DEFAULT_GAMMA = 1.5
export const BUDDHA_TARGET_BRIGHTNESS = 0.1

// Keep metering fixed so changing the display Gamma is not canceled by exposure.
const METERING_GAMMA = 4.8
export const BUDDHA_REFERENCE_BRIGHTNESS = 1.2

// Meter occupied pixels only: empty background must stay black and must not
// increase exposure when an orbit covers only a small part of the canvas.
// Log-density bins make metering robust across the full Float32 density range.
export class DensityExposure {
  constructor() {
    this.histogram = new Map()
    this.count = 0
    this.logSum = 0
  }

  add(value, count = 1) {
    if (!(value > 0) || !Number.isFinite(value)) return
    const bin = Math.round(Math.log2(value) * 128)
    const next = (this.histogram.get(bin) || 0) + count
    if (next) this.histogram.set(bin, next)
    else this.histogram.delete(bin)
    this.count += count
    this.logSum += bin * count
  }

  scale() {
    if (!this.count) return 0
    const center = this.logSum / this.count / 128
    const bins = Array.from(this.histogram, ([bin, count]) => [2 ** (bin / 128 - center), count])
    // Meter using a fixed reference curve. User Gamma must never feed back
    // into exposure, otherwise automatic metering cancels manual adjustments.
    // A low midpoint keeps background noise dark while retaining bright orbits.
    let low = -80,
      high = 80
    for (let step = 0; step < 32; step++) {
      const mid = (low + high) / 2
      const exposure = 2 ** mid
      let sum = 0
      for (const [density, count] of bins) {
        const x = density * exposure
        sum += count * (-Math.expm1(-x)) ** METERING_GAMMA
      }
      if (sum / this.count < BUDDHA_TARGET_BRIGHTNESS) low = mid
      else high = mid
    }
    return 2 ** ((low + high) / 2 - center)
  }
}

export function getDensityExposure(r, g, b) {
  const meter = new DensityExposure()
  for (let i = 0; i < r.length; i++) meter.add(Math.max(r[i] || 0, g[i] || 0, b[i] || 0))
  return meter.scale()
}

// A smooth highlight shoulder avoids abrupt white clipping as exposure rises.
export function mapDensityChannel(value, scale, brightness, gamma) {
  const x = Math.max(0, value) * scale * (brightness / BUDDHA_REFERENCE_BRIGHTNESS)
  return (-Math.expm1(-x)) ** gamma
}

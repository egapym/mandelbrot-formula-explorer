import { DensityExposure, mapDensityChannel } from './buddhabrotExposure.mjs'

// Incremental CPU exposure mapping. Density only grows between sparse updates;
// a changed exposure or display setting requires recoloring occupied pixels.
export class CpuDensityColorMap {
  constructor(r, g, b, width, height, pixels) {
    this.r = r
    this.g = g
    this.b = b
    this.width = width
    this.height = height
    this.pixels = pixels
    this.max = 0
    this.meter = new DensityExposure()
    this.peaks = new Float32Array(width * height)
    this.initialized = false
    this.active = new Set()
  }

  update(brightness, gamma, chunks) {
    const { r, g, b, width, height, pixels } = this
    const length = width * height
    const sparse = this.initialized && chunks?.length > 0 &&
      chunks.every((chunk) => chunk.indices?.length > 0)
    const dirty = sparse ? new Set() : null
    let max = sparse ? this.max : 0
    if (sparse) {
      for (const chunk of chunks) {
        for (const index of chunk.indices) {
          const i = chunk.y * width + chunk.x + index
          dirty.add(i)
          const value = (r[i] || 0) + (g[i] || 0) + (b[i] || 0)
          if (value > 0) this.active.add(i)
          if (value > max) max = value
        }
      }
    } else {
      this.active.clear()
      pixels.fill(0)
      for (let i = 0; i < length; i++) {
        pixels[i * 4 + 3] = 255
        const value = (r[i] || 0) + (g[i] || 0) + (b[i] || 0)
        if (value > 0) this.active.add(i)
        if (value > max) max = value
      }
    }
    if (!sparse) {
      this.meter = new DensityExposure()
      this.peaks.fill(0)
    }
    for (const i of sparse ? dirty : this.active) {
      if (sparse) this.meter.add(this.peaks[i], -1)
      this.peaks[i] = Math.max(r[i] || 0, g[i] || 0, b[i] || 0)
      this.meter.add(this.peaks[i])
    }
    const scale = this.meter.scale()
    const full = !sparse || scale !== this.scale || brightness !== this.brightness || gamma !== this.gamma
    const color = (i) => {
      const di = i * 4
      pixels[di] = (255 * mapDensityChannel(r[i] || 0, scale, brightness, gamma)) | 0
      pixels[di + 1] = (255 * mapDensityChannel(g[i] || 0, scale, brightness, gamma)) | 0
      pixels[di + 2] = (255 * mapDensityChannel(b[i] || 0, scale, brightness, gamma)) | 0
      pixels[di + 3] = 255
    }
    let x = width, y = height, right = 0, bottom = 0
    if (full) {
      // Zero-density pixels stay black for the supported positive gamma range.
      // Even a normalization change only needs to revisit occupied pixels.
      if (gamma > 0) {
        for (const i of this.active) color(i)
      } else {
        for (let i = 0; i < length; i++) color(i)
      }
      x = y = 0
      right = width
      bottom = height
    } else {
      for (const i of dirty) {
        color(i)
        const px = i % width, py = Math.floor(i / width)
        x = Math.min(x, px)
        y = Math.min(y, py)
        right = Math.max(right, px + 1)
        bottom = Math.max(bottom, py + 1)
      }
    }
    this.max = max
    this.scale = scale
    this.brightness = brightness
    this.gamma = gamma
    this.initialized = true
    return { x, y, width: right - x, height: bottom - y, max, full }
  }
}

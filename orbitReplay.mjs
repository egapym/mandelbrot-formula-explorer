/**
 * This file is part of the Mandelbrot Formula Explorer project.
 * Licensed under GPL-3.0.
 */

// Reuse matching iteration steps between normal coloring and Orbit Trap.
// Each pixel starts a new trace. Only a bounded prefix is retained, and a
// different trajectory (including sentinel substitution) falls back to fn.
export class OrbitReplay {
  constructor(fn, maxIter, usesHistory = false) {
    this.fn = fn
    this.usesHistory = usesHistory
    this.capacity = Math.min(4096, maxIter + 1)
    this.steps = new Float64Array(this.capacity * 4)
    this.pair = new Float64Array(2)
    this.length = 0
    this.record = (zr, zi, cr, ci, n, history) => {
      const result = this.fn(zr, zi, cr, ci, n, history)
      this.recordStep(zr, zi, n, result[0], result[1])
      return result
    }
    this.play = (zr, zi, cr, ci, n, history) => {
      const offset = n * 4
      if (
        this.replayValid &&
        n < this.length &&
        Object.is(zr, this.steps[offset]) &&
        Object.is(zi, this.steps[offset + 1])
      ) {
        if (this.usesHistory) history.z[n] = [zr, zi]
        this.pair[0] = this.steps[offset + 2]
        this.pair[1] = this.steps[offset + 3]
        return this.pair
      }
      this.replayValid = false
      return this.fn(zr, zi, cr, ci, n, history)
    }
  }

  reset() {
    this.length = 0
    this.replayValid = true
  }

  recordStep(zr, zi, n, nextReal, nextImag) {
    if (n >= this.capacity) return
    const offset = n * 4
    this.steps[offset] = zr
    this.steps[offset + 1] = zi
    this.steps[offset + 2] = nextReal
    this.steps[offset + 3] = nextImag
    this.length = n + 1
  }
}

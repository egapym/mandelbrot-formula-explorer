/**
 * This file is part of the Mandelbrot Formula Explorer project.
 * Licensed under GPL-3.0.
 */

// Keep only the latest requested view and run one configured-quality render at a time.
// CPU start() resolves after dispatch; GPU start() resolves after completion.
// In both cases onComplete, rather than the start promise, advances the queue.
export class BuddhabrotRealtimeRenderer {
  constructor({ createRunner, draw, onError }) {
    this.createRunner = createRunner
    this.draw = draw
    this.onError = onError
    this.closed = false
    this.busy = false
    this.pending = null
    this.generation = 0
  }

  request(params) {
    if (this.closed) return
    this.pending = params
    if (this.runner?.continuous && this.busy) {
      // A continuous view never completes. Replace it when navigation requests
      // new coordinates; use a fresh runner so old GPU readbacks cannot race it.
      this.generation++
      this.runner.terminate()
      this.runner = null
      this.busy = false
    }
    if (!this.busy) void this._pump()
  }

  async _pump() {
    if (this.closed || this.busy || !this.pending) return
    this.busy = true
    const generation = ++this.generation
    try {
      if (!this.runner) {
        const runner = await this.createRunner()
        if (this.closed || generation !== this.generation) {
          runner.stop?.()
          runner.terminate?.()
          return
        }
        this.runner = runner
      }
      const params = this.pending
      this.pending = null
      const runner = this.runner
      runner.onChunk = () => {
        if (!this.closed && generation === this.generation) this.draw(runner, params)
      }
      runner.onComplete = (result) => {
        if (this.closed || generation !== this.generation) return
        if (result?.error) {
          this.cancel()
          this.onError(result.error)
          return
        }
        this.draw(runner, params)
        this.busy = false
        // Avoid reentering start() from inside a runner's completion handler.
        queueMicrotask(() => void this._pump())
      }
      await runner.start(params)
    } catch (error) {
      if (this.closed || generation !== this.generation) return
      this.cancel()
      this.onError(error)
    }
  }

  cancel() {
    this.closed = true
    this.generation++
    this.pending = null
    this.runner?.stop?.()
    this.runner?.terminate?.()
    this.runner = null
  }
}

// Map a completed density frame into the current pan/zoom without changing axes.
export function getBuddhabrotFrameTransform(width, height, from, to) {
  const scale = to.zoom / from.zoom
  const pixelsPerUnit = to.zoom * width / 4
  return {
    scale,
    x: width / 2 * (1 - scale) + (from.center.x - to.center.x) * pixelsPerUnit,
    y: height / 2 * (1 - scale) + (from.center.y - to.center.y) * pixelsPerUnit,
  }
}

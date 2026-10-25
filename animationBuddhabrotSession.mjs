import { checkSignal, IMAGE_BUDGET } from './animation.mjs'
import { createBuddhaRunner } from './buddhabrot.mjs'
import { mapBuddhabrotFrame } from './buddhabrotCpuColorMap.mjs'

// A dedicated runner replaces its density for each frame. No callback can repaint
// the visible canvas, and Stop cancels both worker startup and active sampling.
export class AnimationBuddhabrotSession {
  constructor({ view, params, samplingRange, workerCount, createRunner = createBuddhaRunner }) {
    const bytes = params.width * params.height * 4
    if (!Number.isSafeInteger(bytes) || bytes <= 0 || bytes * 2 > IMAGE_BUDGET)
      throw new Error('Animation resolution exceeds the 128 MiB image budget')
    Object.assign(this, { view, params, samplingRange, workerCount, createRunner })
  }

  async render(frame, signal) {
    return (await this.prepareFrame(frame, signal)).blob
  }

  async prepareFrame(frame, signal) {
    checkSignal(signal)
    const view = this.view
    view.setZoom(frame.zoom)
    view.setCenter([...frame.center])
    const params = {
      ...this.params,
      reusePipelines: true,
      center: { x: view.center[0].toNumber(), y: view.center[1].toNumber() },
      zoom: view.zoom.toNumber(),
      ...this.samplingRange(view),
    }
    // Fixed Samples retains the original sampling rectangle as well as seed.
    if (this.params.sampleSeed !== undefined && this.params.samplingAspect !== undefined) {
      for (const key of ['samplingCenter', 'samplingZoom', 'samplingAspect']) params[key] = this.params[key]
    }
    params.orbitCacheBytes = params.samplingCenter && params.sampleSeed !== undefined
      ? Math.floor(32 * 1024 ** 2 / Math.max(1, this.workerCount || 1)) : 0
    let complete, fail
    const finished = new Promise((resolve, reject) => { complete = resolve; fail = reject })
    const runner = this.runner || await this.createRunner({
      ...params, useGpu: params.gpu, workerCount: this.workerCount,
    })
    this.runner = runner
    runner.onComplete = (result) => result.error ? fail(new Error(result.error)) : complete(result)
    const cancel = () => {
      runner.stop()
      fail(new DOMException('Animation stopped', 'AbortError'))
    }
    signal.addEventListener('abort', cancel, { once: true })
    let start
    try {
      checkSignal(signal)
      start = Promise.resolve(runner.start(params))
      const [result] = await Promise.all([finished, start])
      checkSignal(signal)
      if (result.aborted) throw new DOMException('Animation stopped', 'AbortError')
      if (!this.canvas) {
        this.canvas = document.createElement('canvas')
        this.canvas.width = params.width
        this.canvas.height = params.height
        this.context = this.canvas.getContext('2d')
        this.image = this.context.createImageData(params.width, params.height)
      }
      const { canvas, context, image } = this
      mapBuddhabrotFrame(runner.densityR, runner.densityG, runner.densityB,
        image.data, params.brightness, params.gamma)
      context.putImageData(image, 0, 0)
      return { blob: new Promise((resolve, reject) => canvas.toBlob(
        (blob) => blob ? resolve(blob) : reject(new Error('Could not encode animation frame')), 'image/png',
      )) }
    } finally {
      signal.removeEventListener('abort', cancel)
      await start?.catch(() => {})
    }
  }

  dispose() {
    this.runner?.stop()
    this.runner?.dispose?.()
    this.runner?.terminate?.()
    this.runner?._cpuRunner?.terminate?.()
    this.runner?.devicePromise?.then((device) => device?.destroy()).catch(() => {})
    this.runner = null
    if (this.canvas) this.canvas.width = this.canvas.height = 1
    this.canvas = this.context = this.image = null
  }
}

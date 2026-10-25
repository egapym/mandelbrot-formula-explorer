import { checkSignal, IMAGE_BUDGET } from './animation.mjs'
import { createBuddhaRunner } from './buddhabrot.mjs'
import { CpuDensityColorMap } from './buddhabrotCpuColorMap.mjs'

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
    checkSignal(signal)
    const view = this.view
    view.setZoom(frame.zoom)
    view.setCenter([...frame.center])
    const params = {
      ...this.params,
      center: { x: view.center[0].toNumber(), y: view.center[1].toNumber() },
      zoom: view.zoom.toNumber(),
      ...this.samplingRange(view),
    }
    // Fixed Samples retains the original sampling rectangle as well as seed.
    if (this.params.sampleSeed !== undefined && this.params.samplingAspect !== undefined) {
      for (const key of ['samplingCenter', 'samplingZoom', 'samplingAspect']) params[key] = this.params[key]
    }
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
      const canvas = document.createElement('canvas')
      canvas.width = params.width
      canvas.height = params.height
      const context = canvas.getContext('2d')
      const image = context.createImageData(canvas.width, canvas.height)
      new CpuDensityColorMap(runner.densityR, runner.densityG, runner.densityB,
        canvas.width, canvas.height, image.data).update(params.brightness, params.gamma)
      context.putImageData(image, 0, 0)
      return await new Promise((resolve, reject) => canvas.toBlob(
        (blob) => blob ? resolve(blob) : reject(new Error('Could not encode animation frame')), 'image/png',
      ))
    } finally {
      signal.removeEventListener('abort', cancel)
      await start?.catch(() => {})
    }
  }

  dispose() {
    this.runner?.stop()
    this.runner?.terminate?.()
    this.runner?._cpuRunner?.terminate?.()
    this.runner?.devicePromise?.then((device) => device?.destroy()).catch(() => {})
    this.runner = null
  }
}

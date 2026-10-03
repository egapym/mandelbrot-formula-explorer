import { checkSignal, IMAGE_BUDGET, validateAnimationSize } from './animation.mjs'
import { MandelbrotCustomWebGPU } from './mandelbrotCustomWebGPU.mjs'
import { MandelbrotWebGPU } from './mandelbrotWebGPU.mjs'
import { OrbitTrapWebGPU } from './orbitTrapBitmapWebGPU.mjs'
import { WorkerContext } from './workerContext.mjs'

// Dedicated buffers and callbacks, shared devices. The main renderer cannot
// publish partial preparation frames or fall back to CPU on our behalf.
export class AnimationGpuSession {
  constructor({ view, sources, select, createScreen, z0, colorPatternId }) {
    Object.assign(this, { view, sources, select, createScreen, z0, colorPatternId })
    this.renderers = new Map()
    this.listeners = []
    this.closed = false
  }

  async rendererFor(kind) {
    if (this.renderers.has(kind)) return this.renderers.get(kind)
    const device = await this.sources[kind].devicePromise
    if (!device || !this.sources[kind].available) throw new Error('WebGPU is unavailable for this animation')
    const onError = (error) => {
      this.error = new Error(String(error))
    }
    const onGpuUpdate = (answer) => {
      if (answer.jobToken !== this.token) return
      if (answer.error) this.error = new Error(answer.error)
      if (answer.isFinished) this.result = answer
    }
    const options = { devicePromise: Promise.resolve(device) }
    const renderer =
      kind === 'orbit'
        ? new OrbitTrapWebGPU(onError, options)
        : new (kind === 'direct' ? MandelbrotCustomWebGPU : MandelbrotWebGPU)(
            { onGpuUpdate },
            new WorkerContext(),
            onError,
            options,
          )
    const errorListener = (event) => {
      onError(event.error.message)
      renderer.newTask = null
    }
    device.addEventListener('uncapturederror', errorListener)
    this.listeners.push(() => device.removeEventListener('uncapturederror', errorListener))
    device.lost.then(() => {
      if (!this.closed) {
        onError('WebGPU device lost')
        renderer.newTask = null
      }
    })
    // Bound the perturbation reference cache, including JS reference sequence overhead.
    if (kind === 'perturbation') {
      const calculateReference = renderer.calculate_reference.bind(renderer)
      renderer.calculate_reference = async (...args) => {
        const bytes = renderer.referencePoints.reduce(
          (sum, ref) => sum + ref.zBuffer.byteLength + ref.zqErrorBoundBuffer.byteLength,
          0,
        )
        if (bytes + (renderer.max_iter + 2) * 64 > IMAGE_BUDGET)
          throw new Error('Animation reference memory limit reached')
        return calculateReference(...args)
      }
    }
    const entry = { renderer, device, onGpuUpdate }
    this.renderers.set(kind, entry)
    return entry
  }

  async render(frame, signal) {
    checkSignal(signal)
    const view = this.view
    view.setZoom(frame.zoom)
    view.setCenter([...frame.center])
    const kind = this.select(view)
    if (!kind) throw new Error('This frame cannot be rendered with WebGPU')
    const { renderer, device, onGpuUpdate } = await this.rendererFor(kind)
    checkSignal(signal)
    if (this.error) throw this.error
    const trapSpec = view.paletteComponent.palette.trapSpec ?? null
    validateAnimationSize(
      {
        width: view.width,
        height: view.height,
        maxIter: view.max_iter,
        bitmapBytes: trapSpec?.bitmapData?.byteLength || 4,
        perturbation: kind === 'perturbation',
      },
      device.limits,
    )
    if (!this.screen) this.screen = this.createScreen()
    this.result = null
    this.token = crypto.randomUUID()
    const task = {
      jobToken: this.token,
      jobId: this.token,
      viewRevision: 0,
      w: view.width,
      h: view.height,
      frameWidth: view.width,
      frameHeight: view.height,
      xOffset: 0,
      yOffset: 0,
      pixelSize: 1,
      skipTopLeft: false,
      frameTopLeft: view.canvas2complex(0, 0),
      frameBottomRight: view.canvas2complex(view.width, view.height),
      maxIter: view.max_iter,
      smooth: view.smooth,
      supersampling: view.supersampling,
      precision: view.precision,
      requiredPrecision: view.requiredPrecision,
      fractalType: view.fractalType,
      iterationFunction: view.iterationFunction,
      escapeRadius: view.escapeRadius,
      z0: this.z0,
      z0Real: this.z0[0],
      z0Imag: this.z0[1],
      trapSpec,
      colorPatternId: this.colorPatternId,
      animationQuick: false,
      finalOnly: true,
      // Never retain references from earlier frames.
      paramHash: this.token,
      resetCaches: true,
      onUpdate: onGpuUpdate,
    }
    const cancel = () => {
      renderer.newTask = null
    }
    signal.addEventListener('abort', cancel, { once: true })
    device.pushErrorScope('out-of-memory')
    device.pushErrorScope('validation')
    let failure
    try {
      // process resolves only after readback/unmap, even if Stop was requested.
      await renderer.process(task)
    } catch (error) {
      failure = error
    } finally {
      signal.removeEventListener('abort', cancel)
      const validation = await device.popErrorScope()
      const allocation = await device.popErrorScope()
      failure ||= validation || allocation
    }
    checkSignal(signal)
    if (failure || this.error) throw failure || this.error
    const result = this.result
    if (!result?.isFinished || result.error) throw new Error(result?.error || 'GPU did not finish the animation frame')
    const screen = this.screen
    if (result.rgba) screen.renderRgba(result.rgba)
    else {
      for (const key of ['values', 'smooth', 'signs', 'zreal', 'zimag', 'otData']) {
        if (result[key]) screen[key].set(result[key])
      }
      screen.render(view.palette, view.max_iter, view.smooth, view.paletteComponent.palette)
    }
    this.result = null
    return new Promise((resolve, reject) =>
      screen.canvas.toBlob((blob) => {
        if (blob) resolve(blob)
        else reject(new Error('Could not encode animation frame'))
      }, 'image/png'),
    )
  }

  dispose() {
    // Called only after prepare has settled, never while buffers are in flight.
    this.closed = true
    for (const remove of this.listeners) remove()
    for (const { renderer } of this.renderers.values()) {
      renderer.pipeline?.dispose?.()
      renderer.mandelbrotPipeline?.dispose()
      renderer._destroyOutputResources?.()
      renderer._destroyBitmapBuffer?.()
      renderer.referencePoints = []
    }
    this.renderers.clear()
    if (this.screen) {
      for (const canvas of [this.screen.canvas, this.screen.offscreen, this.screen.smoothscreen])
        canvas.width = canvas.height = 1
    }
    this.screen = null
  }
}

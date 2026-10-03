/**
 * @author Bert Baron
 * @modified
 */

import * as fxp from './fxp.mjs'
import { beginGpuProfile, endGpuProfile } from './gpuPerformance.mjs'
import { BAILOUT_SMOOTH } from './sharedCalculations.mjs'
import { smoothen, WorkerContext } from './workerContext.mjs'

const USE_GPU = true

export class MandelbrotWebGPU {
  /**
   * @param {WorkerContext} ctx
   */
  constructor(p, ctx, errorCallback) {
    this.p = p
    this.ctx = ctx
    this.errorCallback = errorCallback
    this.paramHash = null
    this.jobId = null
    this.referencePoints = []
    this.fractalType = 'mandelbrot' // 既定のフラクタル種別
    // 利用可能な GPU / device があるかどうかを表す
    this.available = true
    this.devicePromise = this.initGpu()
    this.mandelbrotPipeline = this.createPipeline()
    this.running = Promise.resolve()
    this.currentTask = null
    this.newTask = null
    // 前回描画時のフレームサイズを保持し、大きなズームアウトを検出する
    this.lastFrameWidth = null
    this.lastFrameHeight = null
  }

  async initGpu() {
    const adapter = await navigator.gpu?.requestAdapter({
      powerPreference: 'high-performance',
      // powerPreference: 'low-power'
    })
    const device = await adapter?.requestDevice()
    if (!device) {
      // CPU へ安全にフォールバックできるよう利用不可にする
      this.available = false
      this.errorCallback('need a browser that supports WebGPU')
      return null
    }

    // GPU が利用可能
    this.available = true

    try {
      const info = await device?.adapterInfo
      console.log(`GPU Adapter: ${info.vendor}:${info.architecture}:${info.device} ${info.description}`)
    } catch (error) {
      console.log(`Failed to get adapter info: ${error}`)
    }

    device.lost.then(() => {
      console.log('GPU lost, reloading')
      // TODO: 実際にこの再初期化で十分か確認する
      this.devicePromise = this.initGpu()
      // 再初期化完了までは一時的に利用不可とみなす
      this.available = false
      this.mandelbrotPipeline.dispose()
      this.mandelbrotPipeline = this.createPipeline()
    })
    return device
  }

  createPipeline() {
    // 常に GPU パイプラインを使う。現在はスーパーサンプリングにも対応している
    return USE_GPU ? new MandelbrotPipeline(this, this.devicePromise) : new MandelbrotReference()
  }

  shouldStop() {
    return this.currentTask !== this.newTask
  }

  async process(task) {
    this.newTask = task.jobToken
    await this.running
    if (task.jobToken !== this.newTask) {
      return
    }
    this.currentTask = task.jobToken

    // タスクからフラクタル種別を受け取る
    this.fractalType = task.fractalType || 'mandelbrot'

    this.max_iter = task.maxIter
    const w = task.w
    const h = task.h

    this.running = this.calculate(w, h, task.skipTopLeft, task)
    await this.running
  }

  async calculate(w, h, skipTopLeft, task) {
    const scale = task.precision
    const bigScale = BigInt(scale)
    const rmin = task.frameTopLeft[0]
    const rmax = task.frameBottomRight[0]
    const imin = task.frameTopLeft[1]
    const imax = task.frameBottomRight[1]

    // 暗黙の指数 2^-scale を持つ複素平面上の描画範囲サイズ
    const cWidth = Number(rmax.subtract(rmin).bigInt)
    const cHeight = Number(imax.subtract(imin).bigInt)
    const refr = rmin.bigInt
    const refi = imin.bigInt

    const bailout = task.smooth ? BAILOUT_SMOOTH : task.escapeRadius !== undefined ? task.escapeRadius ** 2 : 16

    this.updateCache(task)
    // this.referencePoints = []  // デバッグ用

    if (this.referencePoints.length === 0) {
      const x = Math.trunc(w / 2)
      const y = Math.trunc(h / 2)
      const rr = refr + BigInt(Math.trunc((x / w) * cWidth))
      const ri = refi + BigInt(Math.trunc((y / h) * cHeight))
      this.referencePoints.push(await this.calculate_reference(rr, ri, bigScale, scale, bailout))
      if (this.shouldStop()) {
        // 空の結果を返しつつ完了扱いにして、進捗表示だけは確実に閉じる
        const values = new Int32Array(w * h)
        const smooth = task.smooth ? new Uint8ClampedArray(w * h) : null
        const signs = new Int8Array(w * h)
        const zreal = new Float32Array(w * h)
        const zimag = new Float32Array(w * h)
        this.p.onGpuUpdate({
          jobToken: task.jobToken,
          values,
          smooth,
          signs,
          zreal,
          zimag,
          renderedPixels: values.length,
          isFinished: true,
        })
        return {
          values,
          smooth,
          signs,
          zreal,
          zimag,
          error: 'Stopped',
        }
      }
    }
    const ddr = cWidth / task.frameWidth
    const ddi = cHeight / task.frameHeight
    const ddr0 = task.xOffset * ddr
    const ddi0 = task.yOffset * ddi

    let solved = false
    let refIdx = 0
    let passnr = 0
    let indices = this.getInitialIndices(w, h)
    const refValues = []
    let values = []
    let smooth = []
    let signs = []
    let zreal = null
    let zimag = null
    // アニメーション中は更新間隔を短くして、見た目の追従性を上げる
    const animationQuick = !!task.animationQuick
    const updateIntervalMs = animationQuick ? 16 : 100
    let lastUpdate = performance.now()
    while (!solved) {
      const ref = this.referencePoints[refIdx]
      const result = await this.perturbationPass({
        passnr,
        w,
        h,
        indices,
        zBuffer: ref.zBuffer,
        zqErrorBoundBuffer: ref.zqErrorBoundBuffer,
        max_iter: task.maxIter,
        dExp: -task.precision,
        refr: Number(ref.rr - refr),
        refi: Number(ref.ri - refi),
        refsize: ref.size,
        ddr0,
        ddi0,
        ddr,
        ddi,
        doSmooth: task.smooth,
        bailout,
        skipTopLeft,
        supersampling: task.supersampling,
      })
      const remainingIndices = result.indices
      values = result.values
      smooth = result.smooth
      signs = result.signs
      zreal = result.zreal
      zimag = result.zimag
      if (this.shouldStop()) {
        this.p.onGpuUpdate({
          jobToken: task.jobToken,
          values,
          smooth,
          signs,
          zreal,
          zimag,
          renderedPixels: values.length,
          isFinished: true,
        })
        return {
          values,
          smooth,
          signs,
          zreal,
          zimag,
          error: 'Stopped',
        }
      }
      // スーパーサンプリングが高いほど必要な pass 数も増える
      // const maxPasses = task.supersampling > 0 ? 200 : 100;
      // if (passnr > maxPasses) {
      //   console.log(`Too many passes (${passnr}), giving up`);
      //   this.p.onGpuUpdate({
      //     jobToken: task.jobToken,
      //     values,
      //     smooth,
      //     renderedPixels: values.length,
      //     isFinished: true,
      //   });
      //   return {
      //     values,
      //     smooth,
      //     // Don't set error field - rendering is successful even if some pixels remain
      //   };
      // }
      indices = remainingIndices

      if (indices.length > 0) {
        const newRefPoint = indices[Math.trunc(indices.length / 2)]
        refIdx++
        if (refIdx >= this.referencePoints.length) {
          const x = newRefPoint % w
          const y = Math.floor(newRefPoint / w)
          const rr = refr + BigInt(Math.trunc((x / w) * cWidth))
          const ri = refi + BigInt(Math.trunc((y / h) * cHeight))
          const ref = await this.calculate_reference(rr, ri, bigScale, scale, bailout)
          if (this.shouldStop()) {
            this.p.onGpuUpdate({
              jobToken: task.jobToken,
              values,
              smooth,
              signs,
              zreal,
              zimag,
              renderedPixels: values.length,
              isFinished: true,
            })
            return {
              values,
              smooth,
              signs,
              zreal,
              zimag,
              error: 'Stopped',
            }
          }
          this.referencePoints.push(ref)

          // 摂動計算が失敗しても前進できるよう、参照点候補は一度 indices から外す
          // 実際の値は refValues に保持して後で戻す
          indices = indices.filter((idx) => idx !== newRefPoint)
          refValues.push([newRefPoint, ref.iter, ref.zq, ref.escZr || 0, ref.escZi || 0])
        }
        // this.ctx.stats.numberOfLowPrecisionMisses += indices.length
      }
      solved = indices.length === 0
      const now = performance.now()
      if (!solved && now - lastUpdate > updateIntervalMs) {
        this.intermediateUpdate(values, smooth, signs, zreal, zimag)
        lastUpdate = now
      }
      passnr++
    }
    await this.mandelbrotPipeline.finish()

    for (const [offset, iter, zq, escZr, escZi] of refValues) {
      values[offset] = smoothen(smooth, offset, iter, zq)
      if (zreal) zreal[offset] = escZr
      if (zimag) zimag[offset] = escZi
    }
    this.p.onGpuUpdate({
      jobToken: task.jobToken,
      values,
      smooth,
      signs,
      zreal,
      zimag,
      renderedPixels: values.length,
      isFinished: true,
    })
    return { values, smooth, signs, zreal, zimag }
  }

  getInitialIndices(w, h) {
    const key = `${w}:${h}`
    if (this.initialIndicesKey === key) {
      return this.initialIndices
    }
    const indices = this.createZOrderCurve(w, h)
    this.initialIndicesKey = key
    this.initialIndices = indices
    return indices
  }

  createZOrderCurve(w, h) {
    const withZValue = []
    for (let y = 0; y < h; y++) {
      for (let x = 0; x < w; x++) {
        const index = y * w + x
        const z = this.interleaveBits(x, y)
        withZValue.push([index, z])
      }
    }
    withZValue.sort((a, b) => a[1] - b[1])
    return new Uint32Array(withZValue.map(([index, _]) => index))
  }

  interleaveBits(x, y) {
    let z = 0
    for (let i = 0; i < 32; i++) {
      z |= ((x & (1 << i)) << i) | ((y & (1 << i)) << (i + 1))
    }
    return z
  }

  intermediateUpdate(values, smooth, signs, zreal, zimag) {
    this.p.onGpuUpdate({
      jobToken: this.currentTask,
      values,
      smooth,
      signs,
      zreal,
      zimag,
      isFinished: false,
    })
  }

  updateCache(task) {
    if (task.jobId !== this.jobId) {
      this.jobId = task.jobId
      if (this.paramHash !== task.paramHash || this.referencePoints.length === 0 || task.resetCaches) {
        this.paramHash = task.paramHash
        this.referencePoints = []
      } else {
        // ジョブ条件が変わらない場合は、表示範囲内の参照点だけを再利用する
        const oldReferencePoints = this.referencePoints
        this.referencePoints = []
        const oldPrecision = this.precision
        const newPrecision = task.precision
        if (newPrecision === oldPrecision) {
          // 大きなズームアウトを検出する。
          // 前回よりフレーム幅が 2 倍以上に広がると、深いズーム用の参照点は使えない。
          const newFrameWidth = task.frameBottomRight[0].bigInt - task.frameTopLeft[0].bigInt
          const zoomedOutSignificantly = this.lastFrameWidth !== null && newFrameWidth > this.lastFrameWidth * 2n

          if (!zoomedOutSignificantly) {
            // 暗黙スケールの補正は未対応なので、範囲内の参照点だけを残す
            for (const ref of oldReferencePoints) {
              if (
                ref.rr >= task.frameTopLeft[0].bigInt &&
                ref.rr <= task.frameBottomRight[0].bigInt &&
                ref.ri >= task.frameTopLeft[1].bigInt &&
                ref.ri <= task.frameBottomRight[1].bigInt
              ) {
                this.referencePoints.push(ref)
              }
            }
          }
          // } else {
          //     console.log(`Clearing caches because precision changed ${oldPrecision} -> ${newPrecision}`)
        }
      }
      this.precision = task.precision
      // 次回のズームアウト判定に使うため、今回のフレーム幅を記録する
      this.lastFrameWidth = task.frameBottomRight[0].bigInt - task.frameTopLeft[0].bigInt
      this.lastFrameHeight = task.frameBottomRight[1].bigInt - task.frameTopLeft[1].bigInt
    }
  }

  async perturbationPass(data) {
    if (data.passnr === 0) {
      await this.mandelbrotPipeline.beforeRun(data)
    }
    return await this.mandelbrotPipeline.run(data)
  }

  /**
   * @param {BigInt} rr 参照点の実部
   * @param {BigInt} ri 参照点の虚部
   * @param {BigInt} bigScale
   * @param {number} scale
   * @param {number} bailout
   */
  async calculate_reference(rr, ri, bigScale, scale, bailout) {
    const [iter, zq, seq] = this.mandelbrot_high_precision(rr, ri, this.max_iter, bailout, bigScale, scale)

    const iterations = seq.length
    const zBuffer = new Float32Array(iterations * 2)
    const zqErrorBoundBuffer = new Float32Array(iterations)

    seq.forEach(([zr, zi, zq], idx) => {
      zBuffer[idx * 2] = zr
      zBuffer[idx * 2 + 1] = zi
      zqErrorBoundBuffer[idx] = zq * 0.000001
    })
    const lastPair = seq.length > 0 ? seq[seq.length - 1] : [0, 0, 0]
    const escZr = iter !== 2 ? lastPair[0] : 0
    const escZi = iter !== 2 ? lastPair[1] : 0
    // console.log(`Calculated reference point in ${(end - start).toFixed(1)}ms`)
    // this.ctx.stats.timeSpendInHighPrecision += end - start
    // this.ctx.stats.numberOfHighPrecisionPoints++
    return {
      rr,
      ri,
      iter,
      zq,
      size: zqErrorBoundBuffer.length,
      zBuffer,
      zqErrorBoundBuffer,
      signBuffer: null, // No sign buffer needed for Mandelbrot
      escZr,
      escZi,
    }
  }

  /**
   * @param {BigInt} re
   * @param {BigInt} im
   * @param {number} max_iter
   * @param {number} bailout
   * @param {BigInt} bigScale
   * @param {number} scale
   * @returns {[number, BigInt, [number, number, zq][]]} [iterations, zq, sequence] where sequence is a list of [zr, zi, zq] tuples
   */
  mandelbrot_high_precision(re, im, max_iter, bailout, bigScale, scale) {
    const scale_1 = bigScale - 1n
    let zr = 0n
    let zi = 0n
    let iter = -1
    let zrq = 0n
    let ziq = 0n
    let zq = 0
    const seq = []
    while (zq <= bailout) {
      if (iter++ === max_iter) {
        return [2, 0, seq]
      }
      zi = ((zr * zi) >> scale_1) + im
      zr = zrq - ziq + re
      zrq = (zr * zr) >> bigScale
      ziq = (zi * zi) >> bigScale
      const z_real = fxp.toNumber(zr, scale)
      const z_imag = fxp.toNumber(zi, scale)
      zq = z_real * z_real + z_imag * z_imag
      seq.push([z_real, z_imag, zq])
    }
    zi = ((zr * zi) >> scale_1) + im
    zr = zrq - ziq + re
    const z_real = fxp.toNumber(zr, scale)
    const z_imag = fxp.toNumber(zi, scale)
    seq.push([z_real, z_imag, z_real * z_real + z_imag * z_imag])
    return [iter + 4, zq, seq]
  }
}

const SPEC_SIZE = 14 * 4
class MandelbrotPipeline {
  constructor(ctx, devicePromise) {
    this.ctx = ctx
    this.devicePromise = devicePromise
    this.pipeline = null
    this.pipelineKey = null

    this.workgroupSize = 64 // recommended default
    this.testsem = 0
    this.buffers = new Map()
    this.bufferDevice = null
  }

  /**
   * Prepares a new rendering run. Creates a bindgroup and fill all the buffers that will not change during the different
   * passes.
   * @param data
   * @returns {Promise<void>}
   */
  async beforeRun(data) {
    // Safety net: max_iter must be a finite positive integer.
    // If the user cleared the input field, NaN can reach here; clamp it to avoid
    // createBuffer receiving NaN as size (which throws a TypeError).
    if (!Number.isFinite(data.max_iter) || data.max_iter < 1) {
      data.max_iter = 1000
    }
    const device = await this.devicePromise
    if (this.bufferDevice !== device) {
      this.dispose()
      this.bufferDevice = device
    }
    const pipeline = await this.getPipeline(
      device,
      this.workgroupSize,
      data.doSmooth,
      data.bailout,
      data.supersampling,
      this.ctx.fractalType,
    )
    this.doSmooth = data.doSmooth

    this.specBuffer = this.reuseBuffer(device, 'specBuffer', {
      size: SPEC_SIZE,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    })

    this.indexBuffer = this.reuseBuffer(device, 'indexBuffer', {
      label: 'index buffer',
      size: data.indices.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
    })
    this.valuesBuffer = this.reuseBuffer(device, 'valuesBuffer', {
      label: 'values buffer',
      size: data.w * data.h * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
    })

    this.zBuffer = this.reuseBuffer(device, 'zBuffer', {
      label: 'zr buffer',
      size: 4 * (data.max_iter + 1) * 2,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    })
    this.zqErrorBoundBuffer = this.reuseBuffer(device, 'zqErrorBoundBuffer', {
      label: 'zq error bound buffer',
      size: 4 * (data.max_iter + 1),
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    })
    this.smoothBuffer = this.reuseBuffer(device, 'smoothBuffer', {
      label: 'smooth buffer',
      size: data.w * data.h * 4, // u32, WebGPU does not support u8 or similar
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
    })
    this.signsBuffer = this.reuseBuffer(device, 'signsBuffer', {
      label: 'signs buffer',
      size: data.w * data.h * 4, // u32 per pixel (0=in-set, 1=same-sign, 2=diff-sign)
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
    })
    this.zrealBuffer = this.reuseBuffer(device, 'zrealBuffer', {
      label: 'zreal buffer',
      size: data.w * data.h * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
    })
    this.zimagBuffer = this.reuseBuffer(device, 'zimagBuffer', {
      label: 'zimag buffer',
      size: data.w * data.h * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
    })

    // Pre-create the buffers that will be used later to copy the results into
    this.resultIndexBuffer = this.reuseBuffer(device, 'resultIndexBuffer', {
      label: 'result index buffer',
      size: data.indices.byteLength,
      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    })
    this.resultValuesBuffer = this.reuseBuffer(device, 'resultValuesBuffer', {
      label: 'result buffer',
      size: data.w * data.h * 4,
      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    })
    this.resultSmoothBuffer = this.reuseBuffer(device, 'resultSmoothBuffer', {
      label: 'smooth result buffer',
      size: data.w * data.h * 4,
      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    })
    this.resultSignsBuffer = this.reuseBuffer(device, 'resultSignsBuffer', {
      label: 'signs result buffer',
      size: data.w * data.h * 4,
      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    })
    this.resultZrealBuffer = this.reuseBuffer(device, 'resultZrealBuffer', {
      label: 'zreal result buffer',
      size: data.w * data.h * 4,
      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    })
    this.resultZimagBuffer = this.reuseBuffer(device, 'resultZimagBuffer', {
      label: 'zimag result buffer',
      size: data.w * data.h * 4,
      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    })

    // Fresh allocations are zeroed by WebGPU. Reused output buffers must have
    // the same initial contents, including pixels not yet solved by perturbation.
    const clear = device.createCommandEncoder()
    for (const buffer of [this.valuesBuffer, this.smoothBuffer, this.signsBuffer, this.zrealBuffer, this.zimagBuffer]) {
      clear.clearBuffer(buffer)
    }
    device.queue.submit([clear.finish()])

    const bindGroupEntries = [
      { binding: 0, resource: { buffer: this.specBuffer } },
      { binding: 1, resource: { buffer: this.indexBuffer } },
      { binding: 2, resource: { buffer: this.valuesBuffer } },
      { binding: 3, resource: { buffer: this.zBuffer } },
      { binding: 4, resource: { buffer: this.zqErrorBoundBuffer } },
      { binding: 5, resource: { buffer: this.smoothBuffer } },
      { binding: 6, resource: { buffer: this.signsBuffer } },
      { binding: 7, resource: { buffer: this.zrealBuffer } },
      { binding: 8, resource: { buffer: this.zimagBuffer } },
    ]

    this.bindGroup = device.createBindGroup({
      label: 'bindGroup for work buffer',
      layout: pipeline.getBindGroupLayout(0),
      entries: bindGroupEntries,
    })
  }

  /**
   * @param data
   * @returns {Promise<{indices: Uint32Array, values: Int32Array, smooth: Uint8ClampedArray}>}
   */
  async run(data) {
    const device = await this.devicePromise
    device.queue.writeBuffer(this.zBuffer, 0, data.zBuffer)
    device.queue.writeBuffer(this.zqErrorBoundBuffer, 0, data.zqErrorBoundBuffer)

    // split data.indices into chunks so that we are more responsive
    // Reduce chunk size during interactive animation to improve responsiveness
    const maxWorkersCount = data.animationQuick ? 1 << 16 : 2 ** 18
    const indices = data.indices
    const remainingIndices = []
    for (let i = 0; i < indices.length; i += maxWorkersCount) {
      const chunk = indices.slice(i, i + maxWorkersCount)
      const remainingChunk = await this.doRun(data, chunk)
      // here we can already start calculating a new reference point in another thread if needed
      for (const element of remainingChunk) {
        remainingIndices.push(element)
      }
      if (this.ctx.shouldStop()) {
        const result = await this.readResults(data)
        await this.finish()
        return {
          indices: [],
          ...result,
        }
      }
    }
    const result = await this.readResults(data)
    return {
      indices: new Uint32Array(remainingIndices),
      ...result,
    }
  }

  async doRun(data, indices) {
    const device = await this.devicePromise

    const workgroupCount = Math.ceil(indices.length / this.workgroupSize)
    const specSize = SPEC_SIZE // max_iter, size, refSize, w, h, refr, refi, dExp
    const specArray = new Int32Array(specSize / 4)
    const specFloatView = new Float32Array(specArray.buffer)
    specArray.set([data.max_iter, indices.length, data.refsize, data.w, data.h, 0], 0)
    specFloatView.set(
      [
        data.refr,
        data.refi,
        data.ddr0,
        data.ddi0,
        data.ddr,
        data.ddi,
        data.dExp,
        2 ** data.dExp, // Can be pre-calculated of course
      ],
      6,
    )
    device.queue.writeBuffer(this.specBuffer, 0, specArray)
    device.queue.writeBuffer(this.indexBuffer, 0, indices)
    // device.queue.writeBuffer(this.debugBuffer, 0, new Float32Array(data.w * data.h))

    const encoder = device.createCommandEncoder({
      label: 'mandelbrot encoder',
    })
    const pass = encoder.beginComputePass({
      label: 'mandelbrot compute pass',
    })
    pass.setPipeline(this.pipeline)
    pass.setBindGroup(0, this.bindGroup)
    // Use safe dispatch to avoid exceeding device limits for workgroups per dimension
    const maxPerDim = (device.limits?.maxComputeWorkgroupsPerDimension) || 65535
    let remainingWg = workgroupCount
    while (remainingWg > 0) {
      const chunk = Math.min(remainingWg, maxPerDim)
      pass.dispatchWorkgroups(chunk, 1, 1)
      remainingWg -= chunk
      // Note: we keep the same pass/pipeline; some drivers may require separate passes
      // but most accept multiple dispatches within the same pass.
    }
    pass.end()

    encoder.copyBufferToBuffer(this.indexBuffer, 0, this.resultIndexBuffer, 0, indices.length * 4)
    const commandBuffer = encoder.finish()
    device.queue.submit([commandBuffer])

    await this.resultIndexBuffer.mapAsync(GPUMapMode.READ)
    const remainingIndices = []
    const resultIndex = new Int32Array(this.resultIndexBuffer.getMappedRange())
    for (let i = 0; i < indices.length; i++) {
      if (resultIndex[i] !== -1) {
        remainingIndices.push(resultIndex[i])
      }
    }
    this.resultIndexBuffer.unmap()

    return remainingIndices
  }

  async readResults(data) {
    const profile = beginGpuProfile('mandelbrot.readback')
    const device = await this.devicePromise
    const encoder = device.createCommandEncoder({ label: 'mandelbrot result encoder' })
    const pairs = [
      [this.valuesBuffer, this.resultValuesBuffer],
      [this.signsBuffer, this.resultSignsBuffer],
      [this.zrealBuffer, this.resultZrealBuffer],
      [this.zimagBuffer, this.resultZimagBuffer],
    ]
    if (data.doSmooth) pairs.push([this.smoothBuffer, this.resultSmoothBuffer])
    for (const [source, target] of pairs) encoder.copyBufferToBuffer(source, 0, target, 0, target.size)
    device.queue.submit([encoder.finish()])

    // Start all mappings together; wait for every settlement before cleanup on failure.
    try {
      const results = await Promise.allSettled(pairs.map(([, target]) => target.mapAsync(GPUMapMode.READ)))
      const failure = results.find((result) => result.status === 'rejected')
      if (failure) throw failure.reason
      const mappedAt = performance.now()
      const values = new Int32Array(new Int32Array(this.resultValuesBuffer.getMappedRange()))
      const signs = new Int8Array(new Uint32Array(this.resultSignsBuffer.getMappedRange()))
      const smooth = new Uint8ClampedArray(this.resultSmoothBuffer.size / 4)
      if (data.doSmooth) smooth.set(new Int32Array(this.resultSmoothBuffer.getMappedRange()))
      const zreal = new Float32Array(new Float32Array(this.resultZrealBuffer.getMappedRange()))
      const zimag = new Float32Array(new Float32Array(this.resultZimagBuffer.getMappedRange()))
      endGpuProfile(profile, {
        waitAndCopyMs: profile ? mappedAt - profile.started : 0,
        decodeMs: performance.now() - mappedAt,
        readbackBytes: pairs.reduce((sum, [, target]) => sum + target.size, 0),
      })
      return { values, smooth, signs, zreal, zimag }
    } finally {
      for (const [, target] of pairs) if (target.mapState === 'mapped') target.unmap()
    }
  }

  reuseBuffer(device, name, descriptor) {
    const cached = this.buffers.get(name)
    if (cached?.size === descriptor.size && cached.usage === descriptor.usage) return cached
    cached?.destroy()
    this.buffers.delete(name)
    const buffer = device.createBuffer(descriptor)
    this.buffers.set(name, buffer)
    return buffer
  }

  async finish() {
    // Keep allocations for the next serialized render. beforeRun clears output
    // channels once per job; subsequent perturbation passes retain their results.
  }

  dispose() {
    for (const buffer of this.buffers.values()) buffer.destroy()
    this.buffers.clear()
    this.bufferDevice = null
    this.bindGroup = null
    this.pipeline = null
    this.pipelineKey = null
  }

  async getPipeline(device, workgroupSize, smooth, bailout, supersampling, fractalType = 'mandelbrot') {
    const key = `${workgroupSize}:${smooth}:${bailout}:${supersampling}:${fractalType}`
    if (this.pipelineKey === key) {
      return this.pipeline
    }

    const module = device.createShaderModule({
      code: this.getShadercode(workgroupSize, smooth, bailout, supersampling, fractalType),
    })
    // const module = device.createShaderModule({code: this.originalGetShadercode(workgroupSize, smooth, bailout)})
    const pipeline = device.createComputePipeline({
      label: 'mandelbrot',
      layout: 'auto',
      compute: {
        module,
      },
    })
    this.pipelineKey = key
    this.pipeline = pipeline
    return pipeline
  }

  getShadercode(workgroupSize, smooth, bailout, supersampling) {
    let smoothCode = ''
    if (smooth)
      smoothCode = `
            var nu = log2(log2(zzq)) - 1;
            var modf = modf(nu);
            iter = iter - i32(modf.whole);
            smoothBuffer[i] = u32(255.0 * (1.0 - modf.fract));
        `

    // Perturbation formula for Mandelbrot — also tracks lastZz for sign computation
    const perturbationCode = `
                    // Mandelbrot perturbation
                    let zz = z + ez * eExpFactor;
                    lastZz = zz;
                    zzq = dot(zz, zz);
                    if (zzq < zqErrorBound) {
                        ${supersampling > 0 ? 'allConverged = false; break;' : 'return;'}
                    }

                    let ez_2z = z + zz;
                    ez = vec2f(dot(ez_2z, vec2f(ez.x, -ez.y)), dot(ez_2z, vec2f(ez.y, ez.x))) + dc;
                `

    // Generate compute function based on supersampling mode
    let computeFunction = ''
    if (supersampling > 0) {
      const samples = supersampling
      computeFunction = `
            @compute @workgroup_size(${workgroupSize}) fn computeSomething(
              @builtin(global_invocation_id) id: vec3u
            ) {
                let iid = id.x;
                if (iid >= spec.size) {
                    return;
                }
                let i = u32(indexBuffer[iid]);  // input will always be >=0
                let xy = vec2f(f32(i % spec.w), f32(i / spec.w));

                // Supersampling: ${samples}x${samples} samples per pixel
                var totalIter = 0.0;
                var totalSmooth = 0.0;
                var sampleCount = 0;
                var allConverged = true;
                var capturedSign = 0u;
                var capturedZr = 0.0;
                var capturedZi = 0.0;

                for (var sy = 0; sy < ${samples}; sy++) {
                    for (var sx = 0; sx < ${samples}; sx++) {
                        let offset = vec2f((f32(sx) + 0.5) / ${samples}.0, (f32(sy) + 0.5) / ${samples}.0);
                        var dc = fma(xy + offset, spec.dd, spec.dd0) - spec.reff;

                        var eExp = spec.dExp;
                        var eExpFactor = spec.dExpFactor;
                        var ez = dc;
                        var lastZz = vec2f(0.0, 0.0);

                        var iter = -1;
                        var zzq = 0.0;
                        while (zzq <= ${bailout}) {
                            iter = iter + 1;
                            if (iter == spec.max_iter) {
                                totalIter += 2.0;
                                sampleCount++;
                                break;
                            }
                            if (iter >= spec.refSize) {
                                allConverged = false;
                                break;
                            }

                            while (max(abs(ez.x), abs(ez.y)) > 2) {
                                eExp = eExp + 1.0;
                                ez = ez * 0.5;
                                dc = dc * 0.5;
                                eExpFactor = eExpFactor * 2.0;
                                if (eExp == -126.0) {
                                    eExpFactor = 0x1.0p-126;
                                }
                            }

                            let z = zBuffer[iter];
                            let zqErrorBound = zqErrorBoundBuffer[iter];

                            ${perturbationCode}
                        }

                        if (iter >= 0 && iter < spec.max_iter && zzq > ${bailout}) {
                            // Capture sign from first sample (sy=0, sx=0)
                            if (sy == 0 && sx == 0) {
                                let sameSign = (lastZz.x >= 0.0) == (lastZz.y >= 0.0);
                                capturedSign = select(2u, 1u, sameSign);
                                capturedZr = lastZz.x;
                                capturedZi = lastZz.y;
                            }
                            ${
                              smooth
                                ? `
                            var nu = log2(log2(zzq)) - 1;
                            var modf_result = modf(nu);
                            let adjustedIter = f32(iter) - modf_result.whole;
                            let smoothValue = 255.0 * (1.0 - modf_result.fract);
                            totalIter += adjustedIter + 4.0;
                            totalSmooth += smoothValue;
                            `
                                : `
                            totalIter += f32(iter) + 4.0;
                            `
                            }
                            sampleCount++;
                        }
                    }
                }

                if (!allConverged || sampleCount == 0) {
                    return;
                }

                values[i] = i32(round(totalIter / f32(sampleCount)));
                ${smooth ? 'smoothBuffer[i] = u32(round(totalSmooth / f32(sampleCount)));' : 'smoothBuffer[i] = 0;'}
                signsBuffer[i] = capturedSign;
                zrealBuffer[i] = capturedZr;
                zimagBuffer[i] = capturedZi;
                indexBuffer[iid] = -1;
            }
            `
    } else {
      computeFunction = `
            @compute @workgroup_size(${workgroupSize}) fn computeSomething(
              @builtin(global_invocation_id) id: vec3u
            ) {
                let iid = id.x;
                if (iid >= spec.size) {
                    return;
                }
                let i = u32(indexBuffer[iid]);  // input will always be >=0
                let xy = vec2f(f32(i % spec.w), f32(i / spec.w));
                var dc = fma(xy, spec.dd, spec.dd0) - spec.reff;

                var eExp = spec.dExp;
                var eExpFactor = spec.dExpFactor;

                var ez = dc;
                var lastZz = vec2f(0.0, 0.0);

                var iter = -1;
                var zzq = 0.0;
                while (zzq <= ${bailout}) {
                    iter = iter + 1;
                    if (iter == spec.max_iter) {
                        values[i] = 2;
                        smoothBuffer[i] = 0;
                        indexBuffer[iid] = -1;
                        return;
                    }
                    if (iter >= spec.refSize) {
                        return;
                    }

                    while (max(abs(ez.x), abs(ez.y)) > 2) {
                        eExp = eExp + 1.0;
                        ez = ez * 0.5;
                        dc = dc * 0.5;
                        eExpFactor = eExpFactor * 2.0;
                        if (eExp == -126.0) {
                            eExpFactor = 0x1.0p-126;
                        }
                    }

                    let z = zBuffer[iter];
                    let zqErrorBound = zqErrorBoundBuffer[iter];

                    ${perturbationCode}
                }

                ${smoothCode}

                let sameSign = (lastZz.x >= 0.0) == (lastZz.y >= 0.0);
                signsBuffer[i] = select(2u, 1u, sameSign);
                zrealBuffer[i] = lastZz.x;
                zimagBuffer[i] = lastZz.y;
                values[i] = iter + 4;
                indexBuffer[iid] = -1;
            }
            `
    }

    //language=WGSL
    return `
            struct Spec {
                max_iter: i32,
                size: u32,
                refSize: i32,
                w: u32,
                h: u32,
                padd0: u32,
                reff: vec2f,
                dd0: vec2f,
                dd: vec2f,
                dExp: f32,
                dExpFactor: f32,
            };
            @group(0) @binding(0) var<uniform> spec: Spec;
            @group(0) @binding(1) var<storage, read_write> indexBuffer: array<i32>;
            @group(0) @binding(2) var<storage, read_write> values: array<i32>;
            @group(0) @binding(3) var<storage, read> zBuffer: array<vec2f>;
            @group(0) @binding(4) var<storage, read> zqErrorBoundBuffer: array<f32>;
            @group(0) @binding(5) var<storage, read_write> smoothBuffer: array<u32>;
            @group(0) @binding(6) var<storage, read_write> signsBuffer: array<u32>;
            @group(0) @binding(7) var<storage, read_write> zrealBuffer: array<f32>;
            @group(0) @binding(8) var<storage, read_write> zimagBuffer: array<f32>;

            /**
             * This is the authors own code. In particular, the idea to use an implicit
             * extended exponent to overcome the limit of float32 is the authors own.
             * Feel free to use or adapt this code in your own projects.
             * If you do, I would greatly appreciate it if you could reference the original source.
             * Thank you!
             */

            ${computeFunction}
        `
  }
}

/**
 * Reference implementation in javascript
 */
class MandelbrotReference {
  constructor() {}

  async beforeRun(data) {
    this.values = new Int32Array(data.w * data.h)

    // Split interleaved zBuffer into separate real and imaginary buffers
    const size = data.zBuffer.length / 2
    this.zrBuffer = new Float32Array(size)
    this.ziBuffer = new Float32Array(size)
    for (let i = 0; i < size; i++) {
      this.zrBuffer[i] = data.zBuffer[i * 2]
      this.ziBuffer[i] = data.zBuffer[i * 2 + 1]
    }
  }

  /**
   * @param data
   * @returns {Promise<{indices: Uint32Array, values: Int32Array, smooth: Uint8ClampedArray}>}
   */
  async run(data) {
    const indices = data.indices
    const smooth = data.doSmooth ? new Uint8ClampedArray(data.w * data.h) : null
    const remainingIndices = []

    if (data.supersampling > 0) {
      // Supersampling: data.supersampling x data.supersampling samples per pixel
      for (const offset of indices) {
        const x = offset % data.w
        const y = Math.floor(offset / data.w)

        const samples = data.supersampling
        let totalIter = 0
        let totalZq = 0
        let sampleCount = 0
        let allPositive = true

        for (let sy = 0; sy < samples; sy++) {
          for (let sx = 0; sx < samples; sx++) {
            const offsetX = (sx + 0.5) / samples
            const offsetY = (sy + 0.5) / samples
            const dcr = data.ddr0 + (x + offsetX) * data.ddr - data.refr
            const dci = data.ddi0 + (y + offsetY) * data.ddi - data.refi
            const [iter, zq] = this.mandlebrot_perturbation(
              offset,
              data.dExp,
              dcr,
              dci,
              data.max_iter,
              data.bailout,
              data.refsize,
              this.zrBuffer,
              this.ziBuffer,
              data.zqErrorBoundBuffer,
            )

            if (iter >= 0) {
              totalIter += iter
              totalZq += zq
              sampleCount++
            } else {
              allPositive = false
            }
          }
        }

        if (allPositive && sampleCount > 0) {
          const avgIter = Math.floor(totalIter / sampleCount)
          const avgZq = totalZq / sampleCount
          this.values[offset] = smoothen(smooth, offset, avgIter, avgZq)
        } else {
          remainingIndices.push(offset)
        }
      }
    } else {
      // Normal rendering without supersampling
      for (const offset of indices) {
        const x = offset % data.w
        const y = Math.floor(offset / data.w)
        const dcr = data.ddr0 + x * data.ddr - data.refr
        const dci = data.ddi0 + y * data.ddi - data.refi
        const [iter, zq] = this.mandlebrot_perturbation(
          offset,
          data.dExp,
          dcr,
          dci,
          data.max_iter,
          data.bailout,
          data.refsize,
          this.zrBuffer,
          this.ziBuffer,
          data.zqErrorBoundBuffer,
        )
        if (iter >= 0) {
          this.values[offset] = smoothen(smooth, offset, iter, zq)
        } else {
          remainingIndices.push(offset)
        }
      }
    }

    return {
      indices: new Uint32Array(remainingIndices),
      values: this.values,
      smooth,
    }
  }

  /**
   * @param {number} idx the pixel index
   * @param {number} dExp
   * @param {number} dcr
   * @param {number} dci
   * @param {number} max_iter
   * @param {number} bailout
   * @param {number} refsize
   * @param {Float32Array} zrBuffer
   * @param {Float32Array} ziBuffer
   * @param {Float32Array} zqErrorBoundBuffer
   * @param {Float32Array} eExpFactorBuffer
   * @param {Float32Array} eEzpDeltaFactorBuffer
   * @returns {[number, number]} [iter, zq]
   */
  mandlebrot_perturbation(idx, dExp, dcr, dci, max_iter, bailout, refsize, zrBuffer, ziBuffer, zqErrorBoundBuffer) {
    dcr = f32(dcr)
    dci = f32(dci)

    let eExp = dExp
    let eExpFactor = f32(2 ** eExp)

    // ε₀ = δ
    let ezr = dcr
    let ezi = dci

    let iter = -1
    let zzq = 0
    const debug = []
    while (zzq <= bailout) {
      if (iter++ === max_iter) {
        return [2, 0]
      }

      if (iter >= refsize) {
        return [-1, zzq]
      }

      while (Math.max(ezr, ezi) > 2) {
        eExp += 1
        if (eExp === -126) {
          eExpFactor = 2 ** -126
        } else {
          eExpFactor *= 2
        }
        ezr = f32(ezr * 0.5)
        ezi = f32(ezi * 0.5)
        dcr = f32(dcr * 0.5)
        dci = f32(dci * 0.5)
      }

      // Zₙ
      const zr = f32(zrBuffer[iter])
      const zi = f32(ziBuffer[iter])
      const zqErrorBound = f32(zqErrorBoundBuffer[iter])

      // Z'ₙ = Zₙ + εₙ
      const zzr = f32(zr + f32(ezr * eExpFactor))
      const zzi = f32(zi + f32(ezi * eExpFactor))
      zzq = f32(f32(zzr * zzr) + f32(zzi * zzi))
      if (zzq < zqErrorBound) {
        return [-1, 0]
      }

      // εₙ₊₁ = 2·zₙ·εₙ + εₙ² + δ = (2·zₙ + εₙ)·εₙ + δ
      const zr_ezr_2 = f32(zr + zzr)
      const zi_ezi_2 = f32(zi + zzi)
      const _ezr = f32(f32(zr_ezr_2 * ezr) - f32(zi_ezi_2 * ezi))
      const _ezi = f32(f32(zr_ezr_2 * ezi) + f32(zi_ezi_2 * ezr))
      ezr = f32(_ezr + dcr)
      ezi = f32(_ezi + dci)
      if (idx === 0) {
        debug.push(eExp)
      }
    }
    if (idx === 0) {
      console.log(`debug: ${debug}`)
    }
    return [iter + 4, zzq]
  }

  async finish() {
    // nothing to do
  }
}

const _f32buf = new Float32Array(1)
function f32(f) {
  _f32buf[0] = f
  const result = _f32buf[0]
  // validate if f is a valid float32 (not inf, -inf, nan)
  // if (!Number.isFinite(result)) {
  //     console.error(`Invalid float32: ${result}`)
  // }
  return result
}

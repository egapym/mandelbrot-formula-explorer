/**
 * @author Bert Baron
 * @modified
 */

import { BAILOUT_SMOOTH } from './sharedCalculations.mjs'
import { smoothen, WorkerContext } from './workerContext.mjs'

export class MandelbrotPerturbation {
  /**
   * @param {WorkerContext} ctx
   */
  constructor(ctx) {
    this.ctx = ctx
    this.paramHash = null
    this.jobId = null
    this.referencePoints = []
    this.fractalType = 'mandelbrot' // 既定のフラクタル種別
  }

  async process(task) {
    // タスクからフラクタル種別を受け取る
    this.fractalType = task.fractalType || 'mandelbrot'

    this.max_iter = task.maxIter
    const w = task.w
    const h = task.h

    const values = new Int32Array(w * h)
    const smooth = task.smooth ? new Uint8ClampedArray(w * h) : null
    const signs = new Int8Array(w * h)
    const zreal = new Float32Array(w * h)
    const zimag = new Float32Array(w * h)
    this.calculate(values, smooth, signs, zreal, zimag, w, h, task.skipTopLeft, task.supersampling, task)

    return {
      type: 'answer',
      task: task,
      values: values,
      smooth: smooth,
      signs: signs,
      zreal: zreal,
      zimag: zimag,
      stats: {
        highPrecisionCalculations: this.ctx.stats.numberOfHighPrecisionPoints,
        lowPrecisionMisses: this.ctx.stats.numberOfLowPrecisionMisses,
      },
    }
  }

  calculate(values, smooth, signs, zreal, zimag, w, h, skipTopLeft, supersampling, task) {
    const stats = this.ctx.stats
    const scale = task.precision
    const scaleFactor = 2 ** Number(scale)
    const bigScale = BigInt(scale)
    const rmin = task.frameTopLeft[0]
    const rmax = task.frameBottomRight[0]
    const imin = task.frameTopLeft[1]
    const imax = task.frameBottomRight[1]

    // 複素平面上での描画範囲サイズ
    const cWidth = Number(rmax.subtract(rmin).bigInt) / scaleFactor
    const cHeight = Number(imax.subtract(imin).bigInt) / scaleFactor
    const refr = rmin.bigInt
    const refi = imin.bigInt

    const bailout = smooth ? BAILOUT_SMOOTH : task.escapeRadius !== undefined ? task.escapeRadius ** 2 : 16
    const bigBailout = BigInt(Math.trunc(bailout)) << bigScale

    this.updateCache(task, cWidth, cHeight, scaleFactor)

    if (this.referencePoints.length === 0) {
      const x = Math.trunc(w / 2)
      const y = Math.trunc(h / 2)
      const dr = ((task.xOffset + x) / task.frameWidth) * cWidth
      const di = ((task.yOffset + y) / task.frameHeight) * cHeight
      this.referencePoints.push(this.calculate_reference(refr, refi, dr, di, bigScale, scaleFactor, bigBailout))
      if (this.ctx.shouldStop()) return
    }

    const pixelWidth = cWidth / task.frameWidth
    const pixelHeight = cHeight / task.frameHeight

    // 参照点は LRU 順で扱い、head は最後に使われた時刻が最も古い点を指す
    let head = this.referencePoints.length - 1
    for (let y = 0; y < h; y++) {
      const di = ((task.yOffset + y) / task.frameHeight) * cHeight
      const skipLeft = skipTopLeft && y % 2 === 0

      for (let x = 0; x < w; x++) {
        if (skipLeft && x % 2 === 0) {
          // スキップする
        } else {
          const offset = y * w + x

          if (supersampling > 0) {
            this.calculatePixelSupersampled(
              x,
              y,
              w,
              task,
              cWidth,
              cHeight,
              pixelWidth,
              pixelHeight,
              values,
              smooth,
              bailout,
              head,
              supersampling,
              zreal,
              zimag,
            )
            signs[y * w + x] = 0 // スーパーサンプリング時は符号情報を持たない
          } else {
            const dr = ((task.xOffset + x) / task.frameWidth) * cWidth

            let found = false

            let refIndex = head
            for (const _ignored of this.referencePoints) {
              const referencePoint = this.referencePoints[refIndex]
              const refDr = referencePoint[0][0]
              const refDi = referencePoint[0][1]
              const zs = referencePoint[3]

              const [iter, zq, zzr, zzi] = this.mandlebrot_perturbation(
                dr - refDr,
                di - refDi,
                this.max_iter,
                bailout,
                zs,
              )
              if (iter >= 0) {
                values[offset] = smoothen(smooth, offset, iter, zq)
                signs[offset] = iter >= 4 ? (zzr >= 0 === zzi >= 0 ? 1 : 2) : 0
                zreal[offset] = iter > 3 ? zzr : 0
                zimag[offset] = iter > 3 ? zzi : 0
                found = true
                stats.numberOfLowPrecisionPoints++
                if (refIndex < head) {
                  head--
                  this.referencePoints[refIndex] = this.referencePoints[head]
                  this.referencePoints[head] = referencePoint
                } else if (refIndex > head) {
                  for (let i = refIndex; i > head; i--) {
                    this.referencePoints[i] = this.referencePoints[i - 1]
                  }
                  this.referencePoints[head] = referencePoint
                }
                break
              }
              stats.numberOfLowPrecisionMisses++
              refIndex = (refIndex + 1) % this.referencePoints.length
            }
            if (!found) {
              const newRef = this.calculate_reference(refr, refi, dr, di, bigScale, scaleFactor, bigBailout)
              values[offset] = smoothen(smooth, offset, newRef[1], Number(newRef[2]) / scaleFactor)
              signs[offset] = newRef[4]
              zreal[offset] = newRef[5]
              zimag[offset] = newRef[6]
              this.referencePoints.unshift(newRef)
              this.referencePoints[0] = this.referencePoints[head]
              this.referencePoints[head] = newRef
              if (this.ctx.shouldStop()) return
            }
          }
        }
      }
      if (this.ctx.shouldStop()) return
    }
  }

  calculatePixelSupersampled(
    x,
    y,
    w,
    task,
    cWidth,
    cHeight,
    _pixelWidth,
    _pixelHeight,
    values,
    smooth,
    bailout,
    head,
    supersampling,
    _zreal,
    _zimag,
  ) {
    const stats = this.ctx.stats
    const offset = y * w + x
    const samples = supersampling
    let totalIter = 0
    let totalSmoothValue = 0
    let sampleCount = 0

    for (let sy = 0; sy < samples; sy++) {
      for (let sx = 0; sx < samples; sx++) {
        // サブピクセル中心を使ってサンプリングする
        const offsetX = (sx + 0.5) / samples
        const offsetY = (sy + 0.5) / samples
        const sampleX = x + offsetX
        const sampleY = y + offsetY
        const dr = ((task.xOffset + sampleX) / task.frameWidth) * cWidth
        const di = ((task.yOffset + sampleY) / task.frameHeight) * cHeight

        let found = false

        let refIndex = head
        for (const _ignored of this.referencePoints) {
          const referencePoint = this.referencePoints[refIndex]
          const refDr = referencePoint[0][0]
          const refDi = referencePoint[0][1]
          const zs = referencePoint[3]

          const [iter, zq] = this.mandlebrot_perturbation(dr - refDr, di - refDi, this.max_iter, bailout, zs)
          if (iter >= 0) {
            if (smooth && iter > 3) {
              const log_zn = Math.log(zq) / 2
              let nu = Math.log(log_zn / Math.log(2)) / Math.log(2)
              const smoothIter = Math.floor(iter + 1 - nu)
              nu = nu - Math.floor(nu)
              totalIter += smoothIter
              totalSmoothValue += 1 - nu
            } else {
              totalIter += iter
            }
            found = true
            stats.numberOfLowPrecisionPoints++
            break
          }
          stats.numberOfLowPrecisionMisses++
          refIndex = (refIndex + 1) % this.referencePoints.length
        }
        if (!found) {
          const scale = task.precision
          const scaleFactor = 2 ** Number(scale)
          const bigScale = BigInt(scale)
          const refr = task.frameTopLeft[0].bigInt
          const refi = task.frameTopLeft[1].bigInt
          const bigBailout = BigInt(bailout) << bigScale
          const newRef = this.calculate_reference(refr, refi, dr, di, bigScale, scaleFactor, bigBailout)
          const rawIter = newRef[1]
          if (smooth && rawIter > 3) {
            const zq = Number(newRef[2]) / scaleFactor
            const log_zn = Math.log(zq) / 2
            let nu = Math.log(log_zn / Math.log(2)) / Math.log(2)
            const smoothIter = Math.floor(rawIter + 1 - nu)
            nu = nu - Math.floor(nu)
            totalIter += smoothIter
            totalSmoothValue += 1 - nu
          } else {
            totalIter += rawIter
          }
        }
        sampleCount++
      }
    }

    values[offset] = Math.round(totalIter / sampleCount)
    if (smooth) {
      smooth[offset] = Math.round(255 * (totalSmoothValue / sampleCount))
    }
  }

  updateCache(task, cWidth, cHeight, scaleFactor) {
    if (task.jobId !== this.jobId) {
      this.jobId = task.jobId
      if (this.paramHash !== task.paramHash || this.referencePoints.length === 0 || task.resetCaches) {
        this.paramHash = task.paramHash
        this.referencePoints = []
      } else {
        // Keep reference points that are within the total frame when job parameters did not change
        const oldReferencePoints = this.referencePoints
        this.referencePoints = []
        const oldPrecision = this.precision
        const newPrecision = task.precision
        if (newPrecision === oldPrecision) {
          const deltar = Number(task.frameTopLeft[0].subtract(this.topLeft[0]).bigInt) / scaleFactor
          const deltai = Number(task.frameTopLeft[1].subtract(this.topLeft[1]).bigInt) / scaleFactor
          for (const referencePoint of oldReferencePoints) {
            const dr = referencePoint[0][0] - deltar
            const di = referencePoint[0][1] - deltai
            if (dr < cWidth && di < cHeight) {
              referencePoint[0] = [dr, di]
              this.referencePoints.push(referencePoint)
            }
          }
        }
      }
      this.precision = task.precision
      this.topLeft = task.frameTopLeft
    }
  }

  /**
   * @param {number} dcr
   * @param {number} dci
   * @param {number} max_iter
   * @param {number} bailout
   * @param {[number, number, number][]} zs
   * @returns {(number|number)[]|number[]}
   */
  mandlebrot_perturbation(dcr, dci, max_iter, bailout, zs) {
    // ε₀ = δ
    let ezr = dcr
    let ezi = dci

    let iter = -1
    let zzq = 0
    let lastZzr = dcr
    let lastZzi = dci
    while (zzq <= bailout) {
      if (iter++ === max_iter) {
        return [2, 0, 0, 0]
      }
      if (iter >= zs.length) {
        return [-1, zzq, 0, 0]
      }

      // Zₙ
      const _zsvalues = zs[iter]
      const zr = _zsvalues[0]
      const zi = _zsvalues[1]
      const zqErrorBound = _zsvalues[2]

      // Z'ₙ = Zₙ + εₙ
      const zzr = zr + ezr
      const zzi = zi + ezi
      zzq = zzr * zzr + zzi * zzi
      if (zzq < zqErrorBound) {
        return [-1, 0, 0, 0]
      }
      lastZzr = zzr
      lastZzi = zzi

      // εₙ₊₁ = 2·zₙ·εₙ + εₙ² + δ = (2·zₙ + εₙ)·εₙ + δ
      const zr_ezr_2 = zr + zzr
      const zi_ezi_2 = zi + zzi
      const _ezr = zr_ezr_2 * ezr - zi_ezi_2 * ezi
      const _ezi = zr_ezr_2 * ezi + zi_ezi_2 * ezr
      ezr = _ezr + dcr
      ezi = _ezi + dci
    }
    return [iter + 4, zzq, lastZzr, lastZzi]
  }

  /**
   * @param {BigInt} refr
   * @param {BigInt} refi
   * @param {number} dr
   * @param {number} di
   * @param {BigInt} bigScale
   * @param {number} scaleFactor
   * @param {BigInt} bailout
   * @returns {[[number, number], number, BigInt, [number, number, number, number, number][]]} [rr, ri], iter, zq, sequence where sequence is a list of [zr, zi, errorbound, zr_sign, zi_sign] tuples
   */
  calculate_reference(refr, refi, dr, di, bigScale, scaleFactor, bailout) {
    const rr = refr + BigInt(Math.round(dr * scaleFactor))
    const ri = refi + BigInt(Math.round(di * scaleFactor))

    const [iter, zq, seq] = this.mandelbrot_high_precision(rr, ri, this.max_iter, bailout, bigScale)
    const zs = seq.map(([zr, zi]) => {
      const z_real = Number(zr) / scaleFactor
      const z_imag = Number(zi) / scaleFactor
      return [z_real, z_imag, (z_real * z_real + z_imag * z_imag) * 0.000001, 1, 1]
    })
    const lastPair = seq.length > 0 ? seq[seq.length - 1] : [0n, 0n]
    const refSign = iter === 2 ? 0 : lastPair[0] >= 0n === lastPair[1] >= 0n ? 1 : 2
    const escZr = iter !== 2 ? Number(lastPair[0]) / scaleFactor : 0
    const escZi = iter !== 2 ? Number(lastPair[1]) / scaleFactor : 0
    this.ctx.stats.numberOfHighPrecisionPoints++
    return [[dr, di], iter, zq, zs, refSign, escZr, escZi]
  }

  /**
   * @param {BigInt} re
   * @param {BigInt} im
   * @param {number} max_iter
   * @param {BigInt} bailout
   * @param {BigInt} scale
   * @returns {[number, BigInt, [BigInt, BigInt][]]} [iterations, zq, sequence] where sequence is a list of [zr, zi] points
   */
  mandelbrot_high_precision(re, im, max_iter, bailout, scale) {
    const scale_1 = scale - 1n
    let zr = 0n
    let zi = 0n
    let iter = -1
    let zrq = 0n
    let ziq = 0n
    let zq = 0n
    const seq = []
    while (zq <= bailout) {
      if (iter++ === max_iter) {
        return [2, 0n, seq]
      }
      zi = ((zr * zi) >> scale_1) + im
      zr = zrq - ziq + re
      seq.push([zr, zi])
      zrq = (zr * zr) >> scale
      ziq = (zi * zi) >> scale
      zq = zrq + ziq
    }
    zi = ((zr * zi) >> scale_1) + im
    zr = zrq - ziq + re
    seq.push([zr, zi])
    return [iter + 4, zq, seq]
  }
}

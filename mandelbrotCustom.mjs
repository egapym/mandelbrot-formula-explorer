/**
 * Based on bertbaron/mandelbrot by Bert Baron
 * This file is part of the Mandelbrot Formula Explorer project.
 * Licensed under GPL-3.0.
 */

import { compileIterationFunction } from './customFunctionParser.mjs'
import { calculatePixelOrbitTrap } from './orbitTrap.mjs'
import { BAILOUT_MIN, BAILOUT_SMOOTH } from './sharedCalculations.mjs'
import { WorkerContext } from './workerContext.mjs'

// ============================================================================
// 定数
// ============================================================================

// CPU 側のセンチネル値を WGSL 側とそろえる
const SAFE_SENTINEL = 1e20

const ITERATION_CONFIG = {
  DEFAULT_BAILOUT_SMOOTH: BAILOUT_SMOOTH,
  DEFAULT_BAILOUT: BAILOUT_MIN,
  IN_SET_INDEX: 2,
  ESCAPE_OFFSET: 4,
  MIN_ITER_FOR_SMOOTH: 3,
  SMOOTH_SCALE: 255,
  // signs[] 値: 0 = 未脱出, 1 = 同一象限で脱出, 2 = 異なる象限で脱出
  SIGN_NOT_ESCAPED: 0,
  SIGN_SAME_QUADRANT: 1,
  SIGN_DIFF_QUADRANT: 2,
}

const DEFAULT_FUNCTION = 'z*z + c'
const DEFAULT_ESCAPE_RADIUS = 4.0
const DEFAULT_JULIA_PARAM = 0.0
const DEFAULT_Z0_COMPONENT = 0.0

// ============================================================================
// エラー処理ユーティリティ
// ============================================================================

const ErrorHelpers = {
  /**
   * エラーメッセージを文字列に整える
   */
  format(error) {
    return error?.message ? error.message : String(error)
  },

  /**
   * コンパイルエラーを記録する
   */
  logCompileError(error) {
    console.error('Failed to compile custom function:', this.format(error))
  },

  /**
   * エラーをメインスレッドへ通知する
   */
  sendError(message) {
    self.postMessage({ type: 'error', message })
  },
}

/**
 * 既定の Mandelbrot 反復関数を作る
 */
function createDefaultIterationFunction() {
  return (zReal, zImag, cReal, cImag) => {
    const zReal2 = zReal * zReal - zImag * zImag + cReal
    const zImag2 = 2 * zReal * zImag + cImag
    return [zReal2, zImag2]
  }
}

/**
 * ユーザー定義の反復関数を使うカスタムフラクタルレンダラー
 *
 * 任意の数式をコンパイルし、フラクタル描画に使います。
 */
export class MandelbrotCustom {
  /**
   * @param {WorkerContext} ctx - worker の実行コンテキスト
   */
  constructor(ctx) {
    this.ctx = ctx
    this.compiledFunction = null
    this.currentFunctionStr = null
  }

  /**
   * 描画タスクを処理する
   * @param {Object} task - 描画タスク
   * @returns {Promise<Object>} 描画結果
   */

  async process(task) {
    this.max_iter = task.maxIter
    this.fractalType = task.fractalType || 'custom'
    // タスクで z0 が指定されていれば使う
    if (task.z0Real !== undefined && task.z0Imag !== undefined) {
      this.z0Real = task.z0Real
      this.z0Imag = task.z0Imag
    } else {
      // 前回の上書き値を消して既定動作へ戻す
      this.z0Real = undefined
      this.z0Imag = undefined
    }
    // 脱出半径を受け取る
    this.escapeRadius = task.escapeRadius !== undefined ? task.escapeRadius : DEFAULT_ESCAPE_RADIUS
    // 'julia-custom' 用の固定 c を受け取る
    if (task.fractalType === 'julia-custom') {
      this.juliaRe = task.juliaRe !== undefined ? task.juliaRe : DEFAULT_JULIA_PARAM
      this.juliaIm = task.juliaIm !== undefined ? task.juliaIm : DEFAULT_JULIA_PARAM
    }
    const w = task.w
    const h = task.h

    // 必要なら反復関数をコンパイルする
    const functionStr = task.iterationFunction || DEFAULT_FUNCTION
    if (functionStr !== this.currentFunctionStr) {
      this.currentFunctionStr = functionStr
      try {
        this.compiledFunction = compileIterationFunction(functionStr)
      } catch (e) {
        ErrorHelpers.logCompileError(e)
        ErrorHelpers.sendError(ErrorHelpers.format(e))
        // 失敗時は既定の Mandelbrot 関数へ戻す
        this.compiledFunction = createDefaultIterationFunction()
      }
    }

    const frameTopLeftFloat = task.frameTopLeft.map((fixed) => fixed.toNumber())
    const frameBottomRightFloat = task.frameBottomRight.map((fixed) => fixed.toNumber())
    const topLeftFloat = [
      frameTopLeftFloat[0] + (task.xOffset * (frameBottomRightFloat[0] - frameTopLeftFloat[0])) / task.frameWidth,
      frameTopLeftFloat[1] + (task.yOffset * (frameBottomRightFloat[1] - frameTopLeftFloat[1])) / task.frameHeight,
    ]
    const bottomRightFloat = [
      frameTopLeftFloat[0] + ((task.xOffset + w) * (frameBottomRightFloat[0] - frameTopLeftFloat[0])) / task.frameWidth,
      frameTopLeftFloat[1] +
        ((task.yOffset + h) * (frameBottomRightFloat[1] - frameTopLeftFloat[1])) / task.frameHeight,
    ]

    const values = new Int32Array(w * h)
    const smooth = task.smooth ? new Uint8ClampedArray(w * h) : null
    const signs = new Int8Array(w * h)
    const zreal = new Float32Array(w * h)
    const zimag = new Float32Array(w * h)
    this.calculate(
      values,
      smooth,
      signs,
      zreal,
      zimag,
      w,
      h,
      topLeftFloat,
      bottomRightFloat,
      task.skipTopLeft,
      task.supersampling,
      task.jobToken,
    )

    // trapSpec を持つパレットのときだけ Orbit trap を計算する
    let otData = null
    if (task.trapSpec) {
      otData = new Float32Array(w * h)
      this.calculateOrbitTraps(
        otData,
        w,
        h,
        topLeftFloat,
        bottomRightFloat,
        task.trapSpec,
        task.supersampling,
        task.jobToken,
      )
    }

    return {
      type: 'answer',
      task: task,
      values: values,
      smooth: smooth,
      signs: signs,
      zreal: zreal,
      zimag: zimag,
      otData: otData,
    }
  }

  /**
   * カスタム反復関数を使ってOrbit trapデータを計算する。
   * trapSpec によって形状・モードが決まるため、パレット設定に応じた計算を行う。
   * 'julia-custom': z0=ピクセル座標, c=固定値 / 'custom': z0=(z0Real, z0Imag), c=ピクセル座標
   *
   * @param {Float32Array} otData
   * @param {number} w
   * @param {number} h
   * @param {[number,number]} topleft
   * @param {[number,number]} bottomright
   * @param {import('./orbitTrap.mjs').TrapSpec} trapSpec
   * @param {string} jobToken
   */
  calculateOrbitTraps(otData, w, h, topleft, bottomright, trapSpec, supersampling, jobToken) {
    const rmin = topleft[0]
    const rmax = bottomright[0]
    const imin = topleft[1]
    const imax = bottomright[1]
    const dr = (rmax - rmin) / w
    const di = (imax - imin) / h
    const isJulia = this.fractalType === 'julia-custom'
    const jRe = isJulia ? (this.juliaRe ?? 0) : 0
    const jIm = isJulia ? (this.juliaIm ?? 0) : 0
    const escapeRadius = this.escapeRadius ?? DEFAULT_ESCAPE_RADIUS
    // コンパイル済み関数を反復ステップとして使う
    const fn = this.compiledFunction
    // supersampling=0 は OFF、1 以上なら samples×samples 個のサブサンプルを使う
    const samples = supersampling > 0 ? supersampling : 1
    const sampleStep = 1 / samples
    for (let y = 0; y < h; y++) {
      if (this.ctx.shouldStop(jobToken)) return
      const im = imin + di * y
      for (let x = 0; x < w; x++) {
        const re = rmin + dr * x
        if (samples > 1) {
          // サブピクセルごとの値を平均する
          let sum = 0
          for (let sy = 0; sy < samples; sy++) {
            for (let sx = 0; sx < samples; sx++) {
              const sRe = re + dr * (sx + 0.5) * sampleStep
              const sIm = im + di * (sy + 0.5) * sampleStep
              const cr = isJulia ? jRe : sRe
              const ci = isJulia ? jIm : sIm
              const z0r = isJulia ? sRe : (this.z0Real ?? 0.0)
              const z0i = isJulia ? sIm : (this.z0Imag ?? 0.0)
              sum += calculatePixelOrbitTrap(cr, ci, z0r, z0i, fn, this.max_iter, trapSpec, escapeRadius)
            }
          }
          otData[y * w + x] = sum / (samples * samples)
        } else {
          const cr = isJulia ? jRe : re
          const ci = isJulia ? jIm : im
          const z0r = isJulia ? re : (this.z0Real ?? 0.0)
          const z0i = isJulia ? im : (this.z0Imag ?? 0.0)
          otData[y * w + x] = calculatePixelOrbitTrap(cr, ci, z0r, z0i, fn, this.max_iter, trapSpec, escapeRadius)
        }
      }
    }
  }

  /**
   * @param {Int32Array} values
   * @param {number} h
   * @param {[number, number]} topleft
   * @param {[number, number]} bottomright
   * @param {boolean} skipTopLeft
   * @param {number} supersampling
   * @param {string} jobToken
   */
  calculate(values, smooth, signs, zreal, zimag, w, h, topleft, bottomright, skipTopLeft, supersampling, jobToken) {
    const rmin = topleft[0]
    const rmax = bottomright[0]
    const imin = topleft[1]
    const imax = bottomright[1]
    const dr = (rmax - rmin) / w
    const di = (imax - imin) / h
    for (let y = 0; y < h; y++) {
      if (this.ctx.shouldStop(jobToken)) {
        return
      }
      const im = imin + di * y
      if (skipTopLeft && y % 2 === 0) {
        for (let x = 1; x < w; x += 2) {
          this.calculatePixel(y, w, x, rmin, dr, di, im, values, smooth, signs, zreal, zimag, supersampling)
        }
      } else {
        for (let x = 0; x < w; x++) {
          this.calculatePixel(y, w, x, rmin, dr, di, im, values, smooth, signs, zreal, zimag, supersampling)
        }
      }
    }
  }

  /**
   * @param {number} y
   * @param {number} w
   * @param {number} x
   * @param {number} rmin
   * @param {number} dr
   * @param {number} di
   * @param {number} im
   * @param {Int32Array} values
   * @param {Uint8ClampedArray|null} smooth
   * @param {number} supersampling - 0=OFF, 2=2x2, 3=3x3, 4=4x4
   */
  calculatePixel(y, w, x, rmin, dr, di, im, values, smooth, signs, zreal, zimag, supersampling) {
    const offset = y * w + x
    const re = rmin + dr * x

    // Julia 集合ではピクセル座標を z0 とし、c は固定値を使う
    if (this.fractalType === 'julia-custom') {
      if (supersampling > 0) {
        const samples = supersampling
        let totalIter = 0
        let totalNu = 0
        let sampleCount = 0
        let capturedSign = 0
        for (let sy = 0; sy < samples; sy++) {
          for (let sx = 0; sx < samples; sx++) {
            const sampleRe = re + (dr * (sx + 0.5)) / samples
            const sampleIm = im + (di * (sy + 0.5)) / samples
            if (smooth) {
              let [iter, zq, zrE, ziE] = this.juliaIterate(
                sampleRe,
                sampleIm,
                this.max_iter,
                ITERATION_CONFIG.DEFAULT_BAILOUT_SMOOTH,
              )
              let nu = 1
              if (iter > ITERATION_CONFIG.MIN_ITER_FOR_SMOOTH) {
                const log_zn = Math.log(zq) / 2
                nu = Math.log(log_zn / Math.log(2)) / Math.log(2)
                iter = Math.floor(iter + 1 - nu)
                nu = nu - Math.floor(nu)
              }
              totalIter += iter
              totalNu += nu
              if (sy === 0 && sx === 0)
                capturedSign =
                  iter >= ITERATION_CONFIG.ESCAPE_OFFSET
                    ? zrE >= 0 === ziE >= 0
                      ? ITERATION_CONFIG.SIGN_SAME_QUADRANT
                      : ITERATION_CONFIG.SIGN_DIFF_QUADRANT
                    : ITERATION_CONFIG.SIGN_NOT_ESCAPED
            } else {
              const [si, , sr, si2] = this.juliaIterate(
                sampleRe,
                sampleIm,
                this.max_iter,
                this.escapeRadius * this.escapeRadius,
              )
              totalIter += si
              if (sy === 0 && sx === 0)
                capturedSign =
                  si >= ITERATION_CONFIG.ESCAPE_OFFSET
                    ? sr >= 0 === si2 >= 0
                      ? ITERATION_CONFIG.SIGN_SAME_QUADRANT
                      : ITERATION_CONFIG.SIGN_DIFF_QUADRANT
                    : ITERATION_CONFIG.SIGN_NOT_ESCAPED
            }
            sampleCount++
          }
        }
        values[offset] = Math.floor(totalIter / sampleCount)
        signs[offset] = capturedSign
        if (smooth)
          smooth[offset] = Math.floor(
            ITERATION_CONFIG.SMOOTH_SCALE - ITERATION_CONFIG.SMOOTH_SCALE * (totalNu / sampleCount),
          )
      } else {
        if (smooth) {
          let [iter, zq, zrE, ziE] = this.juliaIterate(re, im, this.max_iter, ITERATION_CONFIG.DEFAULT_BAILOUT_SMOOTH)
          let nu = 1
          if (iter > ITERATION_CONFIG.MIN_ITER_FOR_SMOOTH) {
            const log_zn = Math.log(zq) / 2
            nu = Math.log(log_zn / Math.log(2)) / Math.log(2)
            iter = Math.floor(iter + 1 - nu)
            nu = nu - Math.floor(nu)
          }
          smooth[offset] = Math.floor(ITERATION_CONFIG.SMOOTH_SCALE - ITERATION_CONFIG.SMOOTH_SCALE * nu)
          values[offset] = iter
          signs[offset] =
            iter >= ITERATION_CONFIG.ESCAPE_OFFSET
              ? zrE >= 0 === ziE >= 0
                ? ITERATION_CONFIG.SIGN_SAME_QUADRANT
                : ITERATION_CONFIG.SIGN_DIFF_QUADRANT
              : ITERATION_CONFIG.SIGN_NOT_ESCAPED
          zreal[offset] = zrE
          zimag[offset] = ziE
        } else {
          const [rawIter, _zqE, zrE, ziE] = this.juliaIterate(
            re,
            im,
            this.max_iter,
            this.escapeRadius * this.escapeRadius,
          )
          values[offset] = rawIter
          signs[offset] =
            rawIter >= ITERATION_CONFIG.ESCAPE_OFFSET
              ? zrE >= 0 === ziE >= 0
                ? ITERATION_CONFIG.SIGN_SAME_QUADRANT
                : ITERATION_CONFIG.SIGN_DIFF_QUADRANT
              : ITERATION_CONFIG.SIGN_NOT_ESCAPED
          zreal[offset] = zrE
          zimag[offset] = ziE
        }
      }
      return
    }

    if (supersampling > 0) {
      const samples = supersampling
      let totalIter = 0
      let totalNu = 0
      let sampleCount = 0
      let capturedSign = 0

      for (let sy = 0; sy < samples; sy++) {
        for (let sx = 0; sx < samples; sx++) {
          const offsetX = (sx + 0.5) / samples
          const offsetY = (sy + 0.5) / samples
          const sampleRe = re + dr * offsetX
          const sampleIm = im + di * offsetY

          if (smooth) {
            let [iter, zq, zrE, ziE] = this.iterate(
              sampleRe,
              sampleIm,
              this.max_iter,
              ITERATION_CONFIG.DEFAULT_BAILOUT_SMOOTH,
            )
            let nu = 1
            if (iter > ITERATION_CONFIG.MIN_ITER_FOR_SMOOTH) {
              const log_zn = Math.log(zq) / 2
              nu = Math.log(log_zn / Math.log(2)) / Math.log(2)
              iter = Math.floor(iter + 1 - nu)
              nu = nu - Math.floor(nu)
            }
            totalIter += iter
            totalNu += nu
            if (sy === 0 && sx === 0)
              capturedSign =
                iter >= ITERATION_CONFIG.ESCAPE_OFFSET
                  ? zrE >= 0 === ziE >= 0
                    ? ITERATION_CONFIG.SIGN_SAME_QUADRANT
                    : ITERATION_CONFIG.SIGN_DIFF_QUADRANT
                  : ITERATION_CONFIG.SIGN_NOT_ESCAPED
          } else {
            const [si, , sr, si2] = this.iterate(
              sampleRe,
              sampleIm,
              this.max_iter,
              this.escapeRadius * this.escapeRadius,
            )
            totalIter += si
            if (sy === 0 && sx === 0)
              capturedSign =
                si >= ITERATION_CONFIG.ESCAPE_OFFSET
                  ? sr >= 0 === si2 >= 0
                    ? ITERATION_CONFIG.SIGN_SAME_QUADRANT
                    : ITERATION_CONFIG.SIGN_DIFF_QUADRANT
                  : ITERATION_CONFIG.SIGN_NOT_ESCAPED
          }
          sampleCount++
        }
      }

      values[offset] = Math.floor(totalIter / sampleCount)
      signs[offset] = capturedSign
      if (smooth) {
        smooth[offset] = Math.floor(
          ITERATION_CONFIG.SMOOTH_SCALE - ITERATION_CONFIG.SMOOTH_SCALE * (totalNu / sampleCount),
        )
      }
    } else {
      // Normal single sample
      if (smooth) {
        let [iter, zq, zrE, ziE] = this.iterate(re, im, this.max_iter, ITERATION_CONFIG.DEFAULT_BAILOUT_SMOOTH)
        let nu = 1
        if (iter > ITERATION_CONFIG.MIN_ITER_FOR_SMOOTH) {
          const log_zn = Math.log(zq) / 2
          nu = Math.log(log_zn / Math.log(2)) / Math.log(2)
          iter = Math.floor(iter + 1 - nu)
          nu = nu - Math.floor(nu)
        }
        smooth[offset] = Math.floor(ITERATION_CONFIG.SMOOTH_SCALE - ITERATION_CONFIG.SMOOTH_SCALE * nu)
        values[offset] = iter
        signs[offset] =
          iter >= ITERATION_CONFIG.ESCAPE_OFFSET
            ? zrE >= 0 === ziE >= 0
              ? ITERATION_CONFIG.SIGN_SAME_QUADRANT
              : ITERATION_CONFIG.SIGN_DIFF_QUADRANT
            : ITERATION_CONFIG.SIGN_NOT_ESCAPED
        zreal[offset] = zrE
        zimag[offset] = ziE
      } else {
        const [rawIter, _zqE, zrE, ziE] = this.iterate(re, im, this.max_iter, this.escapeRadius * this.escapeRadius)
        values[offset] = rawIter
        signs[offset] =
          rawIter >= ITERATION_CONFIG.ESCAPE_OFFSET
            ? zrE >= 0 === ziE >= 0
              ? ITERATION_CONFIG.SIGN_SAME_QUADRANT
              : ITERATION_CONFIG.SIGN_DIFF_QUADRANT
            : ITERATION_CONFIG.SIGN_NOT_ESCAPED
        zreal[offset] = zrE
        zimag[offset] = ziE
      }
    }
  }

  /**
   * Julia iteration using the compiled custom function.
   * z0 = (z0re, z0im) is the pixel coordinate; c = (juliaRe, juliaIm) is fixed.
   * @param {number} z0re - starting real part
   * @param {number} z0im - starting imaginary part
   * @param {number} max_iter
   * @param {number} bailout - bailout radius squared
   * @returns {[number, number]} - [iterations, |z|^2]
   */
  juliaIterate(z0re, z0im, max_iter, bailout) {
    const cReal = this.juliaRe
    const cImag = this.juliaIm
    let zReal = z0re
    let zImag = z0im
    let iter = -1
    let zq = zReal * zReal + zImag * zImag
    while (zq <= bailout) {
      if (iter++ === max_iter) {
        return [ITERATION_CONFIG.IN_SET_INDEX, 0, 0, 0]
      }
      const result = this.compiledFunction(zReal, zImag, cReal, cImag, iter)
      let r0 = result[0]
      let r1 = result[1]
      if (!Number.isFinite(r0) || !Number.isFinite(r1) || Number.isNaN(r0) || Number.isNaN(r1)) {
        r0 = SAFE_SENTINEL
        r1 = 0.0
      }
      zReal = r0
      zImag = r1
      zq = zReal * zReal + zImag * zImag
      if (!Number.isFinite(zq) || Number.isNaN(zq)) {
        zq = SAFE_SENTINEL * SAFE_SENTINEL
      }
    }
    return [iter + ITERATION_CONFIG.ESCAPE_OFFSET, zq, zReal, zImag]
  }

  /**
   * Iterate the custom function
   * @param {number} cReal - Real part of c
   * @param {number} cImag - Imaginary part of c
   * @param {number} max_iter - Maximum iterations
   * @param {number} bailout - Bailout radius squared
   * @returns {[number, number]} - [iterations, |z|^2]
   */
  iterate(cReal, cImag, max_iter, bailout) {
    // Allow caller to provide initial z0 via task (z0Real, z0Imag) when creating
    // the task. If not provided, default to 0.0 to preserve existing behavior.
    // Note: the public API (worker task) should pass z0Real/z0Imag fields.
    let zReal = this.z0Real !== undefined ? this.z0Real : DEFAULT_Z0_COMPONENT
    let zImag = this.z0Imag !== undefined ? this.z0Imag : DEFAULT_Z0_COMPONENT
    let iter = -1
    let zq = 0.0

    while (zq <= bailout) {
      if (iter++ === max_iter) {
        return [ITERATION_CONFIG.IN_SET_INDEX, 0, 0, 0]
      }

      // Apply the custom iteration function
      const result = this.compiledFunction(zReal, zImag, cReal, cImag, iter)

      // Normalize result to finite sentinel when GPU uses large finite sentinel
      let r0 = result[0]
      let r1 = result[1]
      if (!Number.isFinite(r0) || !Number.isFinite(r1) || Number.isNaN(r0) || Number.isNaN(r1)) {
        // Replace Infinity/NaN with large finite sentinel to match WGSL behavior
        r0 = SAFE_SENTINEL
        r1 = 0.0
      }

      zReal = r0
      zImag = r1

      // Ensure we compute a finite magnitude for bailout checks
      zq = zReal * zReal + zImag * zImag
      if (!Number.isFinite(zq) || Number.isNaN(zq)) {
        // coerce to large finite number
        zq = SAFE_SENTINEL * SAFE_SENTINEL
      }
    }
    return [iter + ITERATION_CONFIG.ESCAPE_OFFSET, zq, zReal, zImag]
  }
}

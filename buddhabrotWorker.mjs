/**
 * Based on bertbaron/mandelbrot by Bert Baron
 * This file is part of the Mandelbrot Formula Explorer project.
 * Licensed under GPL-3.0.
 */

import { DEFAULT_RENDER_POINT_BATCH_SIZE, normalizeRenderPointBatchSize } from './buddhabrotRenderConfig.mjs'
import { compileIterationFunction } from './customFunctionParser.mjs'

// ============================================================================
// 定数
// ============================================================================

const SAMPLING_CONFIG = {
  DEFAULT_SAMPLES: 10000,
  DEFAULT_MAX_ITER: 1000,
  DEFAULT_CENTER_X: -0.5,
  DEFAULT_CENTER_Y: 0,
  DEFAULT_ZOOM: 1,
  DEFAULT_MODE: 'buddha',
  DEFAULT_BAND_MODE: 'perPoint',
  FLUSH_INTERVAL_DIVISOR: 2000,
  MIN_FLUSH_INTERVAL: 1000,
  FLUSH_DIRTY_THRESHOLD: 2000,
  COOPERATIVE_SAMPLE_INTERVAL: 100,
  COOPERATIVE_TIME_SLICE_MS: 8,
  VIEW_SPAN: 2.0,
  MAX_ITER_FLOAT32_THRESHOLD: 5000,
}

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
   * 文脈付きの警告を出力する
   */
  warn(context, error) {
    console.warn(`[${context}]`, this.format(error))
  },

  /**
   * エラーをメインスレッドへ送る
   */
  sendError(message) {
    try {
      postMessage({ type: 'error', message })
    } catch (e) {
      this.warn('Send Error Message', e)
    }
  },

  /**
   * コンパイル結果を送る
   */
  sendCompileStatus(ok, message = null) {
    try {
      postMessage({ type: 'compile', ok, message })
    } catch (e) {
      this.warn('Send Compile Status', e)
    }
  },
}

// ============================================================================
// ジョブコンテキスト
// ============================================================================

/**
 * 1 件の描画ジョブの状態を管理する
 * job ID の検証とメッセージ送信をまとめて扱う
 */
class JobContext {
  constructor(jobId) {
    this.jobId = jobId
    this.globalRunning = () => running
    this.globalCurrentJobId = () => currentJobId
  }

  /**
   * このジョブがまだ有効か確認する
   * @returns {boolean} 続行できるなら true
   */
  isActive() {
    return this.globalRunning() && this.globalCurrentJobId() === this.jobId
  }

  /**
   * ジョブが有効なときだけ進捗を送る
   * @param {number} done - 完了サンプル数
   * @param {number} total - 総サンプル数
   */
  sendProgress(done, total) {
    if (this.isActive()) {
      postMessage({ type: 'progress', jobId: this.jobId, done, total })
    }
  }

  /**
   * ジョブが有効なときだけチャンクを送る
   * @param {Object} chunk - チャンクデータ
   * @param {Array<ArrayBuffer>} transfers - 必要なら transferable を渡す
   */
  sendChunk(chunk, transfers = null) {
    if (!this.isActive()) return

    const message = {
      type: 'chunk',
      jobId: this.jobId,
      chunk,
    }

    try {
      if (transfers) {
        postMessage(message, transfers)
      } else {
        postMessage(message)
      }
    } catch (_e) {
      // transfer に失敗した場合は通常送信に戻す
      postMessage(message)
    }
  }

  /**
   * ジョブを止めるべきか確認する
   * @returns {boolean} 停止すべきなら true
   */
  shouldStop() {
    return !this.isActive()
  }
}

let running = false
let renderDelay = 0 // 描画点のバッチごとの待ち時間（ミリ秒）
let renderPointBatchSize = DEFAULT_RENDER_POINT_BATCH_SIZE
let trajectorySlots = null
let wakeSlots = null
function releaseSlotWait() {
  wakeSlots?.()
  wakeSlots = null
}
const activeSlotCount = () => trajectorySlots ?? renderPointBatchSize
let currentJobId = 0 // メッセージ検証用の現在の job ID
let nextPresentationId = 0
const pendingPresentations = new Map()

function releasePresentations() {
  for (const resolve of pendingPresentations.values()) resolve()
  pendingPresentations.clear()
}

self.onmessage = (e) => {
  const data = e.data
  if (data.cmd === 'start') {
    releasePresentations()
    releaseSlotWait()
    running = true
    renderDelay = data.renderDelay ?? 0 // start 時に待ち時間を初期化する
    renderPointBatchSize = normalizeRenderPointBatchSize(data.renderPointBatchSize)
    trajectorySlots = data.trajectorySlots ?? null
    currentJobId = data.jobId ?? 0 // この描画セッションの job ID を保持する
    const jobId = currentJobId
    const iterFnStr = data.iterationFunction || null
    let compiledIter = null
    if (iterFnStr) {
      try {
        compiledIter = compileIterationFunction(iterFnStr)
        ErrorHelpers.sendCompileStatus(true)
      } catch (err) {
        compiledIter = null
        ErrorHelpers.sendError('Failed to compile iteration function in worker: ' + ErrorHelpers.format(err))
        ErrorHelpers.sendCompileStatus(false, ErrorHelpers.format(err))
      }
    }
    // 本番ではデバッグメッセージは出さない
    runSampling({ ...data, iterationFunctionCompiled: compiledIter }).then(() => {
      // An old async sampling loop can resume after a newer start message.
      // It must not emit completion for the newer job.
      if (running && currentJobId === jobId) {
        postMessage({ type: 'done', jobId })
      }
    })
  } else if (data.cmd === 'stop') {
    releasePresentations()
    releaseSlotWait()
    running = false
    // 古いメッセージを無視できるよう job ID を無効化する
    currentJobId = -1
    if (data.releaseToPool === true) postMessage({ type: 'released' })
  } else if (data.cmd === 'setSpeed') {
    // 実行中に待ち時間を更新する
    renderDelay = data.renderDelay ?? 0
    releaseSlotWait()
  } else if (data.cmd === 'setPointBatchSize') {
    renderPointBatchSize = normalizeRenderPointBatchSize(data.renderPointBatchSize)
    trajectorySlots = data.trajectorySlots ?? null
    releaseSlotWait()
  } else if (data.cmd === 'presented' && data.jobId === currentJobId) {
    pendingPresentations.get(data.presentationId)?.()
    pendingPresentations.delete(data.presentationId)
  }
}

/**
 * Buddhabrot 軌道生成のメインサンプリング処理
 * @param {Object} opts - サンプリング設定
 */
async function runSampling(opts) {
  // 今回の実行用ジョブコンテキストを作る
  const jobCtx = new JobContext(currentJobId)

  const samples = opts.samples ?? SAMPLING_CONFIG.DEFAULT_SAMPLES
  const maxIter = opts.maxIter ?? SAMPLING_CONFIG.DEFAULT_MAX_ITER
  const width = opts.width
  const height = opts.height
  const center = opts.center || {
    x: SAMPLING_CONFIG.DEFAULT_CENTER_X,
    y: SAMPLING_CONFIG.DEFAULT_CENTER_Y,
  }
  const zoom = opts.zoom || SAMPLING_CONFIG.DEFAULT_ZOOM
  const mode = opts.mode || SAMPLING_CONFIG.DEFAULT_MODE
  const paletteStops = opts.paletteStops || null
  // band の色分け方法。
  // 'perPoint' は各点ごと、'perTrajectory' は軌道全体に 1 つの band を使う。
  const buddhaBandMode = opts.buddhaBandMode || SAMPLING_CONFIG.DEFAULT_BAND_MODE
  const compiledIter = opts.iterationFunctionCompiled || null
  const escapeRadiusSq = opts.escapeRadius !== undefined ? opts.escapeRadius * opts.escapeRadius : 4
  const fractalType = opts.fractalType || 'mandelbrot'
  const isJulia = fractalType === 'julia' || fractalType === 'julia-custom'
  const fixedJuliaRe = typeof opts.juliaRe === 'number' ? opts.juliaRe : 0
  const fixedJuliaIm = typeof opts.juliaIm === 'number' ? opts.juliaIm : 0

  // パレット入力を正規化する。対応形式は { bands: [{ color, ratio }, ...] } のみ
  let bands = null
  if (paletteStops?.bands && Array.isArray(paletteStops.bands) && paletteStops.bands.length > 0) {
    bands = paletteStops.bands.map((b) => ({
      color: b.color.slice(),
      ratio: Number(b.ratio) || 0,
    }))
    const bsum = bands.reduce((a, b) => a + b.ratio, 0)
    if (bsum <= 0) {
      bands = null
    } else {
      // 正規化して累積値を作る
      let acc = 0
      for (let i = 0; i < bands.length; i++) {
        bands[i].ratio = bands[i].ratio / bsum
        acc += bands[i].ratio
        bands[i].cum = acc
      }
      // 正規化完了
    }
  }
  // メイン描画と同じ表示範囲を使う。半幅は 2 / zoom
  const left = center.x - (1 / zoom) * SAMPLING_CONFIG.VIEW_SPAN
  const right = center.x + (1 / zoom) * SAMPLING_CONFIG.VIEW_SPAN
  const aspect = height / width
  const top = center.y - (1 / zoom) * SAMPLING_CONFIG.VIEW_SPAN * aspect
  const bottom = center.y + (1 / zoom) * SAMPLING_CONFIG.VIEW_SPAN * aspect

  const chunkW = width
  const chunkH = height
  let localR = new Float32Array(chunkW * chunkH)
  let localG = new Float32Array(chunkW * chunkH)
  let localB = new Float32Array(chunkW * chunkH)
  const isDirty = new Uint8Array(chunkW * chunkH)
  let dirtyList = []
  // 非ゼロセル数を追跡し、毎回バッファ全体を走査しないようにする
  let pendingNonzeroCount = 0
  let presentationWait = null
  // 再利用可能な軌道バッファ。通常は Float32 を使い、
  // 反復回数が非常に大きいときだけ Float64 に切り替える。
  const trajBuf =
    maxIter <= SAMPLING_CONFIG.MAX_ITER_FLOAT32_THRESHOLD
      ? new Float32Array((maxIter + (isJulia ? 1 : 0)) * 2)
      : new Float64Array((maxIter + (isJulia ? 1 : 0)) * 2)

  // 軽量な xorshift32 乱数生成器を worker 実行ごとに 1 回だけ初期化する。
  // ホットループ内で Math.random() を何度も呼ばないための工夫。
  let _prng_state = ((Math.random() * 0xffffffff) | 0) >>> 0
  function rand32() {
    _prng_state ^= _prng_state << 13
    _prng_state = _prng_state >>> 0
    _prng_state ^= _prng_state >>> 17
    _prng_state ^= _prng_state << 5
    _prng_state = _prng_state >>> 0
    return _prng_state >>> 0
  }
  function randf() {
    return (rand32() >>> 0) / 4294967295
  }

  /**
   * 疎な更新ピクセルだけを送る補助関数
   */
  const flushSparse = (jobCtx) => {
    if (dirtyList.length === 0 || !jobCtx.isActive()) return false

    const n = dirtyList.length
    const indices = new Uint32Array(n)
    const rvals = new Float32Array(n)
    const gvals = new Float32Array(n)
    const bvals = new Float32Array(n)
    // 黒の band は密度に値を加えない。初回表示を待たせないため、
    // 実際に可視の寄与を含むチャンクだけを描画側へ渡す。
    let hasVisibleDensity = false

    for (let i = 0; i < n; i++) {
      const idx = dirtyList[i]
      indices[i] = idx
      rvals[i] = localR[idx]
      gvals[i] = localG[idx]
      bvals[i] = localB[idx]
      if (rvals[i] !== 0 || gvals[i] !== 0 || bvals[i] !== 0) hasVisibleDensity = true
      // 転送後はローカル値を 0 に戻す
      localR[idx] = 0
      localG[idx] = 0
      localB[idx] = 0
      if (isDirty[idx]) {
        isDirty[idx] = 0
        pendingNonzeroCount--
      }
    }
    dirtyList = []

    if (!hasVisibleDensity) return false

    const presentationId = opts.waitForPresentation && renderDelay > 0 ? ++nextPresentationId : undefined
    presentationWait = presentationId === undefined ? null : new Promise((resolve) => {
      pendingPresentations.set(presentationId, resolve)
    })

    jobCtx.sendChunk(
      {
        x: 0,
        y: 0,
        w: chunkW,
        h: chunkH,
        indices: indices,
        r: rvals,
        g: gvals,
        b: bvals,
        presentationId,
      },
      [indices.buffer, rvals.buffer, gvals.buffer, bvals.buffer],
    )
    return true
  }

  /**
   * 密度バッファ全体を送る補助関数
   */
  const flushFull = (jobCtx) => {
    if (!jobCtx.isActive()) return

    jobCtx.sendChunk(
      {
        x: 0,
        y: 0,
        w: chunkW,
        h: chunkH,
        r: localR,
        g: localG,
        b: localB,
      },
      [localR.buffer, localG.buffer, localB.buffer],
    )

    // 転送後に再確保する
    localR = new Float32Array(chunkW * chunkH)
    localG = new Float32Array(chunkW * chunkH)
    localB = new Float32Array(chunkW * chunkH)
    // バッファを空にしたので dirty 状態もリセットする
    isDirty.fill(0)
    pendingNonzeroCount = 0
  }

  const flushIntervalSamples = Math.max(
    SAMPLING_CONFIG.MIN_FLUSH_INTERVAL,
    Math.floor(samples / SAMPLING_CONFIG.FLUSH_INTERVAL_DIVISOR),
  )
  const flushWhenDirtyCount = SAMPLING_CONFIG.FLUSH_DIRTY_THRESHOLD

  // 内側ループで頻繁に使う値をローカルへ退避する
  const invWSpan = 1 / (right - left)
  const invHSpan = 1 / (bottom - top)
  const chunkW_local = chunkW
  const chunkH_local = chunkH
  const width_local = width
  const height_local = height

  // band 情報を typed array 化し、内側ループの参照を軽くする
  let bandsLen = 0
  let bandColors = null
  let bandCums = null
  if (Array.isArray(bands) && bands.length > 0) {
    bandsLen = bands.length
    bandColors = new Float32Array(bandsLen * 3)
    bandCums = new Float32Array(bandsLen)
    for (let bi = 0; bi < bandsLen; bi++) {
      const b = bands[bi]
      const col = b.color || [255, 255, 255]
      bandColors[bi * 3 + 0] = (col[0] || 0) / 255
      bandColors[bi * 3 + 1] = (col[1] || 0) / 255
      bandColors[bi * 3 + 2] = (col[2] || 0) / 255
      bandCums[bi] = Number(b.cum) || 0
    }
  }

  const useBands = bandsLen > 0
  const perTrajectory = buddhaBandMode === 'perTrajectory'
  const iterationDenominator = Math.max(1, maxIter)
  const band1 = Math.max(1, Math.floor(maxIter * 0.01))
  const band2 = Math.max(band1 + 1, Math.floor(maxIter * 0.1))

  function* drawTrajectory(stepped = false) {
    // 表示領域内のランダム点を選ぶ（PRNG を使用）
    const sampleRe = left + randf() * (right - left)
    const sampleIm = top + randf() * (bottom - top)
    const cr = isJulia ? fixedJuliaRe : sampleRe
    const ci = isJulia ? fixedJuliaIm : sampleIm

    // 反復を開始
    // Julia は画素側を z0 とし、Mandelbrot / Custom は従来どおり固定 z0 から始める
    let zx = isJulia ? sampleRe : typeof opts.z0Real === 'number' ? opts.z0Real : 0
    let zy = isJulia ? sampleIm : typeof opts.z0Imag === 'number' ? opts.z0Imag : 0
    let escaped = false
    let trajLen = 0
    const history = { z: [] }
    if (isJulia) {
      trajBuf[0] = sampleRe
      trajBuf[1] = sampleIm
      trajLen = 1
    }
    // trajBuf を再利用し、長さを trajLen で管理する
    for (let iter = 0; iter < maxIter; iter++) {
      // コンパイル済み反復関数があればそれを使い、なければ z^2 + c を使う
      let x2, y2
      if (compiledIter) {
        try {
          const res = compiledIter(zx, zy, cr, ci, iter, history)
          x2 = Number(res[0])
          y2 = Number(res[1])
        } catch (_e) {
          // コンパイル済み関数の実行に失敗した場合は通常の z^2 + c に戻す
          x2 = zx * zx - zy * zy + cr
          y2 = 2 * zx * zy + ci
        }
      } else {
        // z = z^2 + c  (実部/虚部)
        x2 = zx * zx - zy * zy + cr
        y2 = 2 * zx * zy + ci
      }
      if (x2 * x2 + y2 * y2 > escapeRadiusSq) {
        // 次の z が発散する場合は脱出とみなし、脱出点自体は含めない
        escaped = true
        break
      }
      zx = x2
      zy = y2
      // 再利用可能なバッファに保存する
      const ti = trajLen * 2
      trajBuf[ti] = zx
      trajBuf[ti + 1] = zy
      trajLen++
    }

    const keep = (mode === 'buddha' && escaped) || (mode === 'antibuddha' && !escaped)
    if (!keep || trajLen === 0) return

    // Color components retain the original Float32 palette rounding, followed
    // by double-precision multiplication and Float32 density accumulation.
    let colorR = 0
    let colorG = 0
    let colorB = 0
    let pointBand = 0
    if (useBands && perTrajectory) {
      const fracTraj = trajLen / iterationDenominator
      let bi = 0
      while (bi < bandsLen && fracTraj >= bandCums[bi]) bi++
      const offset = Math.min(bi, bandsLen - 1) * 3
      colorR = (bandColors[offset] || 0) * 1.2
      colorG = (bandColors[offset + 1] || 0) * 1.2
      colorB = (bandColors[offset + 2] || 0) * 1.2
    }

    // Track the last occurrences of the three most recent distinct pixels.
    // The third timestamp determines whether the 50-point window has >2 IDs,
    // without an orbit buffer or a Set scan at every convergence check.
    // convergence state begin
    let newestPixel = -1
    let secondPixel = -1
    let newestAt = -1
    let secondAt = -1
    let thirdAt = -1
    // convergence state end

    const points = stepped ? trajBuf.slice(0, trajLen * 2) : trajBuf
    // Prime the trajectory without drawing: every active orbit starts at step zero.
    if (stepped) yield true

    // Buddhabrot worker ではスーパーサンプリングを省略し、
    // 1 サンプルマッピングのみを使う
    for (let k = 0; k < trajLen; k++) {
      // 軌道点の描画ごとにジョブ停止を確認する
      if (jobCtx.shouldStop()) return

      const pr = points[k * 2]
      const pi = points[k * 2 + 1]
      const fx = (pr - left) * invWSpan
      const fy = (pi - top) * invHSpan
      const px = Math.floor(fx * width_local)
      const py = Math.floor(fy * height_local)

      let currentPixelIdx = -1 // 収束判定用に現在のピクセルを追跡

      if (px >= 0 && px < chunkW_local && py >= 0 && py < chunkH_local) {
        const idx = py * chunkW_local + px
        currentPixelIdx = idx // 収束追跡のために保持
        if (useBands) {
          if (!perTrajectory) {
            // k increases monotonically: an earlier band cannot match again.
            const frac = k / iterationDenominator
            while (pointBand < bandsLen && frac >= bandCums[pointBand]) pointBand++
            const offset = Math.min(pointBand, bandsLen - 1) * 3
            colorR = bandColors[offset] * 1.2
            colorG = bandColors[offset + 1] * 1.2
            colorB = bandColors[offset + 2] * 1.2
          }
        } else {
          colorR = k < band1 ? 1.2 : 0
          colorG = k >= band1 && k < band2 ? 1.2 : 0
          colorB = k >= band2 ? 1.2 : 0
        }
        localR[idx] += colorR
        localG[idx] += colorG
        localB[idx] += colorB
        if (!isDirty[idx]) {
          isDirty[idx] = 1
          dirtyList.push(idx)
          pendingNonzeroCount++
        }
      }

      // Offscreen points advance the window without counting as a pixel ID.
      // convergence update begin
      if (trajLen > 200) {
        if (currentPixelIdx >= 0) {
          if (currentPixelIdx === newestPixel) {
            newestAt = k
          } else {
            if (currentPixelIdx !== secondPixel) thirdAt = secondAt
            secondPixel = newestPixel
            secondAt = newestAt
            newestPixel = currentPixelIdx
            newestAt = k
          }
        }
        if (k >= 200 && k % 10 === 0 && thirdAt <= k - 50) {
          if (stepped) yield false
          return
        }
      }
      // convergence update end

      if (stepped) yield k + 1 < trajLen
    }
  }

  const trajectories = []
  let nextSample = 0
  let completedSamples = 0
  let invisibleSteps = 0
  // Yield often enough for Stop/live controls, without paying a timer roundtrip
  // for every 100 cheap samples. This clock controls scheduling only.
  let sliceStarted = Date.now()

  while (jobCtx.isActive() && (nextSample < samples || trajectories.length > 0)) {
    if (renderDelay === 0) {
      // Keep immediate rendering on the original full-trajectory mapping path.
      if (trajectories.length > 0) {
        const trajectory = trajectories.shift()
        while (!trajectory.next().done && jobCtx.isActive()) {}
      } else {
        nextSample++
        drawTrajectory().next()
      }
      completedSamples++
      const shouldFlush = dirtyList.length >= flushWhenDirtyCount ||
        (completedSamples > 0 && completedSamples % flushIntervalSamples === 0)
      if (shouldFlush) flushSparse(jobCtx)
      if (
        (shouldFlush || completedSamples % SAMPLING_CONFIG.COOPERATIVE_SAMPLE_INTERVAL === 0) &&
        Date.now() - sliceStarted >= SAMPLING_CONFIG.COOPERATIVE_TIME_SLICE_MS
      ) {
        await new Promise((resolve) => setTimeout(resolve, 0))
        if (jobCtx.shouldStop()) break
        sliceStarted = Date.now()
        jobCtx.sendProgress(completedSamples, samples)
      }
      continue
    }

    if (activeSlotCount() === 0) {
      await new Promise((resolve) => { wakeSlots = resolve })
      continue
    }

    // Batch size controls concurrent sampled coordinates, never steps along one orbit.
    // On a decrease, extra trajectories wait; on an increase, new slots start at step zero.
    while (trajectories.length < activeSlotCount() && nextSample < samples && jobCtx.isActive()) {
      nextSample++
      const trajectory = drawTrajectory(true)
      if (trajectory.next().done) completedSamples++
      else trajectories.push(trajectory)
      if (nextSample % SAMPLING_CONFIG.COOPERATIVE_SAMPLE_INTERVAL === 0) {
        await new Promise((resolve) => setTimeout(resolve, 0))
        if (jobCtx.shouldStop()) break
        jobCtx.sendProgress(completedSamples, samples)
      }
    }
    if (jobCtx.shouldStop()) break
    if (renderDelay === 0) continue

    const activeCount = Math.min(trajectories.length, activeSlotCount())
    const finished = []
    for (let i = 0; i < activeCount; i++) {
      const step = trajectories[i].next()
      if (step.done || step.value === false) finished.push(i)
    }
    for (let i = finished.length - 1; i >= 0; i--) trajectories.splice(finished[i], 1)
    completedSamples += finished.length

    if (flushSparse(jobCtx)) {
      if (presentationWait) await presentationWait
      if (jobCtx.shouldStop()) break
      if (!opts.waitForPresentation) await new Promise((resolve) => setTimeout(resolve, renderDelay))
      if (jobCtx.shouldStop()) break
      jobCtx.sendProgress(completedSamples, samples)
    } else if (++invisibleSteps % SAMPLING_CONFIG.COOPERATIVE_SAMPLE_INTERVAL === 0) {
      // Invisible or black-band trajectories still accept Stop and live settings.
      await new Promise((resolve) => setTimeout(resolve, 0))
      if (jobCtx.shouldStop()) break
      jobCtx.sendProgress(completedSamples, samples)
    }
  }

  // 最終フラッシュ
  if (flushSparse(jobCtx) && presentationWait) await presentationWait
  if (pendingNonzeroCount > 0)
    // 残っているフルバッファも送る（疎なフラッシュで空になっているはず）
    flushFull(jobCtx)
  jobCtx.sendProgress(samples, samples)
}

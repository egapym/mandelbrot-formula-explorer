/**
 * Based on bertbaron/mandelbrot by Bert Baron
 * This file is part of the Mandelbrot Formula Explorer project.
 * Licensed under GPL-3.0.
 */

import { getRenderPointBatchLimit, normalizeRenderPointBatchSize } from './buddhabrotRenderConfig.mjs'
import { createWorkerFrom } from './workerLoader.mjs'

// CPU Buddhabrot is recreated when its view is rerendered, but a Worker does
// not need to be recreated with it.  Keeping idle workers avoids refetching
// the module (and its imports) for every Render click.  Workers are released
// only after stop/terminate and remain owned by this page until it unloads.
const idleWorkers = []
let nextRunnerId = 0

async function acquireWorker() {
  return idleWorkers.pop() || createWorkerFrom('buddhabrotWorker.mjs', { type: 'module' })
}

function releaseWorker(worker) {
  try {
    // Wait for the worker's message handler to stop its active job before
    // allowing another runner to acquire it from the pool.
    worker.postMessage({ cmd: 'stop', releaseToPool: true })
  } catch (_e) {
    // A worker that is no longer usable must not be put back in the pool.
    try {
      worker.terminate()
    } catch (_ignored) {}
  }
}

// ============================================================================
// 定数
// ============================================================================

const DEFAULT_CONFIG = {
  WORKER_COUNT: navigator.hardwareConcurrency || 4,
  WIDTH: 800,
  HEIGHT: 600,
  MAX_ITER: 1000,
  SAMPLES: 100000,
  CENTER_X: -0.5,
  CENTER_Y: 0,
  ZOOM: 1,
  BRIGHTNESS: 1.8,
  GAMMA: 0.8,
  MODE: 'buddha',
  BAND_MODE: 'perPoint',
}

const DEFAULT_PALETTE = [
  { color: [255, 255, 255], weight: 0.9 },
  { color: [0, 0, 255], weight: 0.085 },
  { color: [128, 0, 128], weight: 0.015 },
]

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
   * 詳細付きの警告を出力する
   */
  warn(context, error) {
    console.warn(`[${context}]`, this.format(error))
  },
}

// ============================================================================
// 密度バッファ用ユーティリティ
// ============================================================================

const DensityHelpers = {
  /**
   * 指定サイズの密度バッファを作成する
   */
  createBuffers(width, height) {
    const total = Math.max(0, width * height)
    return {
      densityMap: new Float32Array(total),
      densityR: new Float32Array(total),
      densityG: new Float32Array(total),
      densityB: new Float32Array(total),
    }
  },

  /**
   * すべての密度バッファを 0 に戻す
   */
  resetBuffers(buffers) {
    if (buffers.densityMap) buffers.densityMap.fill(0)
    if (buffers.densityR) buffers.densityR.fill(0)
    if (buffers.densityG) buffers.densityG.fill(0)
    if (buffers.densityB) buffers.densityB.fill(0)
  },
}

/**
 * Web Worker でサンプル生成を並列化する Buddhabrot レンダラー
 *
 * Mandelbrot 集合から脱出した点の軌道をたどり、
 * 密度マップを生成するワーカープールを管理します。
 */
export class BuddhabrotRunner {
  /**
   * @param {Object} options - 設定
   * @param {number} [options.workerCount] - 使用するワーカー数
   * @param {number} [options.width] - キャンバスの幅
   * @param {number} [options.height] - キャンバスの高さ
   * @param {number} [options.maxIter] - 最大反復回数
   * @param {number} [options.samples] - 生成するサンプル数
   * @param {Function} [options.onProgress] - 進捗通知コールバック
   * @param {Function} [options.onChunk] - チャンク完了時のコールバック
   * @param {Function} [options.onComplete] - 完了時のコールバック
   */
  constructor(options = {}) {
    this.workers = []
    this.workerCount = options.workerCount || DEFAULT_CONFIG.WORKER_COUNT
    this.width = options.width || DEFAULT_CONFIG.WIDTH
    this.height = options.height || DEFAULT_CONFIG.HEIGHT
    this.maxIter = options.maxIter ?? DEFAULT_CONFIG.MAX_ITER
    this.samples = options.samples ?? DEFAULT_CONFIG.SAMPLES
    this.onProgress = options.onProgress || (() => {})
    this.onChunk = options.onChunk || (() => {})
    this.onComplete = options.onComplete || (() => {})

    // 密度バッファを初期化
    const buffers = DensityHelpers.createBuffers(this.width, this.height)
    this.densityMap = buffers.densityMap
    this.densityR = buffers.densityR
    this.densityG = buffers.densityG
    this.densityB = buffers.densityB

    this.running = false
    // 各軌道を 1 ステップずつ進める間隔（ミリ秒）。
    this.renderDelay = options.renderDelay ?? 0
    this.renderPointBatchSize = normalizeRenderPointBatchSize(options.renderPointBatchSize, this.maxRenderPointBatchSize)
    // 描画セッション管理用の ID。古い worker メッセージを除外する
    this._runnerId = ++nextRunnerId
    this._currentJobId = null
    this._startToken = 0
    this._initWorkers()
  }

  /**
   * ワーカープールを初期化する
   * @private
   */
  _initWorkers() {
    this.terminate()
    this._terminated = false
    // プール済みの Worker を再利用し、worker スクリプトの取得回数を抑える
    const creates = []
    for (let i = 0; i < this.workerCount; i++) {
      const p = acquireWorker()
      .then((w) => {
          // terminate() can run while an asynchronously created worker is
          // still loading.  Return that worker to the pool instead of leaking
          // it or attaching it to a disposed runner.
          if (this._terminated) {
            releaseWorker(w)
            return null
          }
          const handleMessage = (e) => {
            if (e.data?.type === 'released') {
              idleWorkers.push(w)
              return
            }
            this._onWorkerMessage(e)
          }
          w.onmessage = handleMessage
          w.onerror = () => {
            const idx = idleWorkers.indexOf(w)
            if (idx >= 0) idleWorkers.splice(idx, 1)
            try { w.terminate() } catch (_e) {}
          }
          this.workers.push(w)
          return w
        })
        .catch((e) => {
          ErrorHelpers.warn(`BuddhaRunner: Worker ${i} Creation`, e)
          return null
        })
      creates.push(p)
    }
    // すべてのワーカー生成が終わったら解決される Promise を保持する
    this._workersReady = Promise.all(creates).then(() => this.workers)
  }

  /**
   * すべてのワーカーを終了する
   */
  terminate() {
    this._startToken++
    this._clearPresentationQueue()
    for (const w of this.workers) {
      releaseWorker(w)
    }
    this.workers = []
    this._terminated = true
    this.running = false
    this._currentJobId = null
  }

  /**
   * すべての密度バッファを 0 に戻す
   */
  resetDensity() {
    DensityHelpers.resetBuffers({
      densityMap: this.densityMap,
      densityR: this.densityR,
      densityG: this.densityG,
      densityB: this.densityB,
    })
  }

  /**
   * 各軌道の描画速度を制御する待ち時間を設定する
   * @param {number} delay - 軌道ステップごとの待ち時間（ミリ秒）
   */
  setRenderSpeed(delay) {
    this.renderDelay = Math.max(0, delay)
    if (this.renderDelay === 0) {
      this._schedulePresentation()
    }
    // 速度設定をすべてのワーカーへ通知する
    for (const w of this.workers) {
      try {
        w.postMessage({ cmd: 'setSpeed', renderDelay: this.renderDelay })
      } catch (e) {
        ErrorHelpers.warn('Worker SetSpeed', e)
      }
    }
  }

  /** 1 ワーカーへ割り当てられるサンプル数から同時描画数の上限を求める。 */
  get maxRenderPointBatchSize() {
    return getRenderPointBatchLimit(this.samples, this.workerCount)
  }

  /** CPU で同時に描画を進める座標（軌道）の数を更新する。 */
  setRenderPointBatchSize(size) {
    this.renderPointBatchSize = normalizeRenderPointBatchSize(size, this.maxRenderPointBatchSize)
    this._assignTrajectorySlots()
  }

  // Divide the global concurrency budget; completed workers release their slots
  // immediately instead of holding up the remaining workers.
  _assignTrajectorySlots(notify = true) {
    const slots = this.workers.map(() => 0)
    let remaining = this.renderPointBatchSize
    while (remaining > 0) {
      let assigned = false
      for (let i = 0; i < slots.length && remaining > 0; i++) {
        if (this._finishedWorkers?.has(this.workers[i])) continue
        const capacity = this._assignedSamples?.[i] ?? this.samples
        if (slots[i] >= capacity) continue
        slots[i]++
        remaining--
        assigned = true
      }
      if (!assigned) break
    }
    this._trajectorySlots = slots
    if (notify) {
      this.workers.forEach((worker, i) => {
        worker.postMessage({
          cmd: 'setPointBatchSize', renderPointBatchSize: this.renderPointBatchSize,
          trajectorySlots: slots[i],
        })
      })
    }
  }

  /**
   * 指定パラメータで描画を開始する
   * @param {Object} params - 描画パラメータ
   */

  async start(params = {}) {
    if (this.running) return
    this._clearPresentationQueue()
    // Do not let an earlier async start() send its job after a newer request
    // has already started on this runner.
    if (this._currentJobId !== null) this._startToken++
    const startToken = ++this._startToken
    this._terminated = false
    this.resetDensity()
    this.maxIter = params.maxIter ?? this.maxIter
    this.samples = params.samples ?? this.samples
    this.width = params.width || this.width
    this.height = params.height || this.height
    // 指定があれば待ち時間を更新する
    if (params.renderDelay !== undefined) {
      this.renderDelay = params.renderDelay
    }
    this.renderPointBatchSize = normalizeRenderPointBatchSize(
      params.renderPointBatchSize ?? this.renderPointBatchSize,
      this.maxRenderPointBatchSize,
    )
    // この描画用の新しい job ID を発行し、古いメッセージを除外する
    this._jobSequence = (this._jobSequence || 0) + 1
    this._currentJobId = `${this._runnerId}:${this._jobSequence}`
    this._pendingWorkers = 0

    // 内部バッファを現在の描画サイズに合わせて作り直す
    // これにより前回と異なるサイズのデータが混ざって
    // ずれや欠けが出るのを防ぐ
    try {
      const buffers = DensityHelpers.createBuffers(this.width, this.height)
      this.densityMap = buffers.densityMap
      this.densityR = buffers.densityR
      this.densityG = buffers.densityG
      this.densityB = buffers.densityB
    } catch (e) {
      ErrorHelpers.warn('Buffer Allocation', e)
      // 再確保に失敗した場合は既存バッファをリセットして続行する
      this.resetDensity()
    }
    this.center = params.center || {
      x: DEFAULT_CONFIG.CENTER_X,
      y: DEFAULT_CONFIG.CENTER_Y,
    }
    this.zoom = params.zoom || DEFAULT_CONFIG.ZOOM
    this.palette = params.palette || DEFAULT_PALETTE

    // サンプル数をワーカーへ分配する
    // samples が workerCount より少ない場合でも、先頭から順に 1 件ずつ割り当てる
    const base = Math.floor(this.samples / this.workerCount)
    let rem = this.samples % this.workerCount
    this._pendingWorkers = 0
    this._sent = 0
    // 進捗差分計算用に、各ワーカーの処理済みサンプル数を保持する
    this._workerSamplesDone = new Array(this.workers.length).fill(0)
    // メッセージ送信前にワーカー生成完了を待つ
    if (this._workersReady) await this._workersReady
    // stop()/terminate()/別の start() が Worker 初期化待ちの間に呼ばれたら、
    // 古い開始処理からジョブを送らない。
    if (startToken !== this._startToken || this._terminated || this._currentJobId == null) return
    this._pendingWorkers = this.workers.length
    if (this._pendingWorkers === 0) {
      this.running = false
      this.onComplete({
        densityMap: this.densityMap,
        width: this.width,
        height: this.height,
        error: new Error('No Buddhabrot workers are available'),
      })
      return
    }
    this._finishedWorkers = new Set()
    this._assignedSamples = this.workers.map((_, i) => base + (i < rem ? 1 : 0))
    this._assignTrajectorySlots(false)
    for (let i = 0; i < this.workers.length; i++) {
      const assign = base + (rem > 0 ? 1 : 0)
      if (rem > 0) rem--
      const msg = {
        cmd: 'start',
        id: i,
        jobId: this._currentJobId, // 古いメッセージを除外するための ID
        samples: assign,
        maxIter: this.maxIter,
        width: this.width,
        height: this.height,
        center: this.center,
        zoom: this.zoom,
        mode: params.mode || DEFAULT_CONFIG.MODE,
        paletteStops: this.palette,
        iterationFunction: params.iterationFunction || null,
        brightness: params.brightness ?? DEFAULT_CONFIG.BRIGHTNESS,
        gamma: params.gamma ?? DEFAULT_CONFIG.GAMMA,
        buddhaBandMode: params.buddhaBandMode || DEFAULT_CONFIG.BAND_MODE,
        renderDelay: this.renderDelay, // 速度設定をワーカーへ渡す
        renderPointBatchSize: this.renderPointBatchSize,
        trajectorySlots: this._trajectorySlots[i],
        waitForPresentation: true,
        fractalType: params.fractalType || null,
        juliaRe: params.juliaRe,
        juliaIm: params.juliaIm,
        // 必要なら初期 z0 を上書きする
        z0Real: params.z0Real,
        z0Imag: params.z0Imag,
        escapeRadius: params.escapeRadius !== undefined ? params.escapeRadius : 4.0,
      }
      // 描画側コールバックから参照できるよう保持する
      this.brightness = msg.brightness
      this.gamma = msg.gamma
      try {
        this.workers[i].postMessage(msg)
      } catch (e) {
        ErrorHelpers.warn(`Worker ${i} PostMessage`, e)
      }
      this._sent += msg.samples
    }

    // 送信完了後に running を有効化する
    await Promise.resolve()
    this.running = true
  }

  /**
   * すべてのワーカーを停止し、描画を終了する
   */
  stop() {
    this._startToken++
    this._clearPresentationQueue()
    this.running = false
    // 現在の job ID を無効化して、残っているメッセージを無視する
    this._jobSequence = (this._jobSequence || 0) + 1
    this._currentJobId = `${this._runnerId}:${this._jobSequence}`
    // 再開時に古い進捗が混ざらないように進捗情報をリセットする
    if (this._workerSamplesDone) {
      this._workerSamplesDone.fill(0)
    }
    for (const w of this.workers) w.postMessage({ cmd: 'stop' })
  }

  _clearPresentationQueue() {
    if (this._presentationFrame != null) cancelAnimationFrame(this._presentationFrame)
    if (this._presentationTimer != null) clearTimeout(this._presentationTimer)
    this._presentationFrame = null
    this._presentationTimer = null
    this._presentationQueue = []
    this._lastPresentationTimes = new Map()
  }

  // Every worker has its own presentation clock. A missing/long-running worker
  // must never prevent another ready worker from advancing its trajectories.
  _schedulePresentation() {
    if (this._presentationFrame != null || this._presentationTimer != null || this._presentationQueue.length === 0) return
    const present = () => {
      this._presentationFrame = null
      this._presentationTimer = null
      const now = Date.now()
      const ready = []
      this._presentationQueue = this._presentationQueue.filter((entry) => {
        const last = this._lastPresentationTimes.get(entry.worker)
        if (last !== undefined && now - last < this.renderDelay) return true
        ready.push(entry)
        return false
      })
      const presented = []
      for (const { data, worker } of ready) {
        if (!this.running || data.jobId !== this._currentJobId) continue
        this._mergeChunk(data.chunk)
        this._lastPresentationTimes.set(worker, Date.now())
        presented.push({ data, worker })
      }
      if (presented.length > 0) {
        try {
          // Every ready worker has already contributed to the density buffers.
          // Recolor the full canvas once, then release all of them together.
          // Calling onChunk once per worker made a larger Points per Batch value
          // spend most of its time repeatedly recoloring the same canvas.
          this.onChunk(presented[0].data.chunk)
        } finally {
          for (const { data, worker } of presented) {
            worker.postMessage({ cmd: 'presented', jobId: data.jobId, presentationId: data.chunk.presentationId })
          }
        }
      }
      this._schedulePresentation()
    }
    // requestAnimationFrame has a practical minimum interval of one display
    // frame.  Short delays must use a timer so that lowering the slider really
    // accelerates an orbit instead of being capped at roughly 60 steps/sec.
    if (this.renderDelay > 0 && this.renderDelay < 16) {
      let wait = this.renderDelay
      for (const entry of this._presentationQueue) {
        const last = this._lastPresentationTimes.get(entry.worker)
        if (last === undefined) {
          wait = 0
          break
        }
        wait = Math.min(wait, Math.max(0, last + this.renderDelay - Date.now()))
      }
      this._presentationTimer = setTimeout(present, wait)
      return
    }
    this._presentationFrame = requestAnimationFrame(present)
  }

  _onWorkerMessage(e) {
    const data = e.data
    // デバッグログは削除済み
    // まず job ID を確認し、現在の描画に対応するメッセージだけ処理する
    if (data.jobId !== this._currentJobId) {
      return // 古い、または無効なメッセージは無視する
    }
    // stop() 後に届いた残りのチャンクを処理しないようにする
    if (!this.running) return
    // worker 側のデバッグログは抑制している
    if (data.type === 'progress') {
      // worker 側のデバッグログは抑制している
      // worker 側のデバッグログは抑制している
      // 各ワーカーの進捗から差分を計算し、全体進捗として UI に渡す
      try {
        const widx = this.workers.indexOf(e.target)
        if (widx >= 0) {
          const prev = this._workerSamplesDone[widx] || 0
          const now = Number(data.done) || 0
          const delta = Math.max(0, now - prev)
          this._workerSamplesDone[widx] = now
          if (delta > 0) this.onProgress({ delta: delta, total: Number(data.total) || 0 })
        } else {
          // ワーカーを特定できない場合は元のデータをそのまま渡す
          this.onProgress(data)
        }
      } catch (_e) {
        // 例外時は元の挙動に戻す
        this.onProgress(data)
      }
    } else if (data.type === 'chunk') {
      if (data.chunk.presentationId !== undefined) {
        this._presentationQueue.push({ data, worker: e.target })
        this._schedulePresentation()
        return
      }
      // data.chunk は { x, y, w, h, r, g, b } 形式
      this._mergeChunk(data.chunk)
      this.onChunk(data.chunk)
    } else if (data.type === 'compile') {
    } else if (data.type === 'done') {
      if (this._finishedWorkers.has(e.target)) return
      this._finishedWorkers.add(e.target)
      this._assignTrajectorySlots()
      this._pendingWorkers--
      if (this._pendingWorkers <= 0) {
        this.running = false
        this.onComplete({
          densityMap: this.densityMap,
          width: this.width,
          height: this.height,
        })
      }
    }
  }

  /**
   * worker から届いた密度チャンクをメインバッファへ加算する
   * @param {Object} chunk - 位置情報と RGB 値を持つチャンク
   * @private
   */
  _mergeChunk(chunk) {
    const { x, y, w, h } = chunk
    // 疎なチャンク（indices + r/g/b）と通常の全面チャンクの両方に対応する
    if (chunk.indices && chunk.indices.length > 0) {
      const inds = chunk.indices
      const r = chunk.r
      const g = chunk.g
      const b = chunk.b
      for (let i = 0; i < inds.length; i++) {
        const srcIdx = inds[i]
        // srcIdx はチャンク内の一次元インデックスなので、描画先へ変換する
        const dstIdx = y * this.width + x + srcIdx
        if (r) this.densityR[dstIdx] += r[i]
        if (g) this.densityG[dstIdx] += g[i]
        if (b) this.densityB[dstIdx] += b[i]
      }
      return
    }

    const r = chunk.r
    const g = chunk.g
    const b = chunk.b
    const dstR = this.densityR
    const dstG = this.densityG
    const dstB = this.densityB
    const dstW = this.width
    for (let row = 0; row < h; row++) {
      const dstRowBase = (y + row) * dstW + x
      const srcRowBase = row * w
      for (let col = 0; col < w; col++) {
        const dstIdx = dstRowBase + col
        const srcIdx = srcRowBase + col
        if (r) dstR[dstIdx] += r[srcIdx]
        if (g) dstG[dstIdx] += g[srcIdx]
        if (b) dstB[dstIdx] += b[srcIdx]
      }
    }
  }
}

/**
 * Buddhabrot ランナーを生成するファクトリ関数
 * @param {Object} options - 設定
 * @param {boolean} [options.useGpu] - GPU を使うかどうか
 * @returns {Promise<BuddhabrotRunner|BuddhabrotWebGPU>}
 */
export async function createBuddhaRunner(options = {}) {
  if (options.useGpu) {
    try {
      // GPU 実装を必要時のみ読み込む
      const mod = await import('./buddhabrotWebGPU.mjs')
      if (mod?.BuddhabrotWebGPU) {
        return new mod.BuddhabrotWebGPU(options)
      }
    } catch (e) {
      ErrorHelpers.warn('Buddhabrot WebGPU Load', e)
      // 読み込み失敗時は CPU 実装へフォールバックする
    }
  }
  return new BuddhabrotRunner(options)
}

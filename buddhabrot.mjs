/**
 * Based on bertbaron/mandelbrot by Bert Baron
 * This file is part of the Mandelbrot Formula Explorer project.
 * Licensed under GPL-3.0.
 */

import { createWorkerFrom } from './workerLoader.mjs'

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
    // レンダリング速度の制御値（ミリ秒）。サンプル生成の間隔に使う
    this.renderDelay = options.renderDelay ?? 0
    // 描画セッション管理用の ID。古い worker メッセージを除外する
    this._currentJobId = 0
    this._initWorkers()
  }

  /**
   * ワーカープールを初期化する
   * @private
   */
  _initWorkers() {
    this.terminate()
    // キャッシュ済みの Blob URL からワーカーを作成し、
    // worker スクリプトの取得回数を抑える
    const creates = []
    for (let i = 0; i < this.workerCount; i++) {
      // createWorkerFrom は getWorkerBlobUrl を内部で使い Worker を返す
      const p = createWorkerFrom('buddhabrotWorker.mjs', { type: 'module' })
        .then((w) => {
          w.onmessage = (e) => this._onWorkerMessage(e)
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
    for (const w of this.workers) {
      try {
        w.terminate()
      } catch (e) {
        ErrorHelpers.warn('Worker Termination', e)
      }
    }
    this.workers = []
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
   * サンプル生成速度を制御する待ち時間を設定する
   * @param {number} delay - サンプルバッチごとの待ち時間（ミリ秒）
   */
  setRenderSpeed(delay) {
    this.renderDelay = Math.max(0, delay)
    // 速度設定をすべてのワーカーへ通知する
    for (const w of this.workers) {
      try {
        w.postMessage({ cmd: 'setSpeed', renderDelay: this.renderDelay })
      } catch (e) {
        ErrorHelpers.warn('Worker SetSpeed', e)
      }
    }
  }

  /**
   * 指定パラメータで描画を開始する
   * @param {Object} params - 描画パラメータ
   */

  async start(params = {}) {
    if (this.running) return
    this.resetDensity()
    this.maxIter = params.maxIter ?? this.maxIter
    this.samples = params.samples ?? this.samples
    this.width = params.width || this.width
    this.height = params.height || this.height
    // 指定があれば待ち時間を更新する
    if (params.renderDelay !== undefined) {
      this.renderDelay = params.renderDelay
    }
    // この描画用の新しい job ID を発行し、古いメッセージを除外する
    this._currentJobId = (this._currentJobId + 1) | 0

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
    this._pendingWorkers = this.workerCount
    this._sent = 0
    // 進捗差分計算用に、各ワーカーの処理済みサンプル数を保持する
    this._workerSamplesDone = new Array(this.workers.length).fill(0)
    // メッセージ送信前にワーカー生成完了を待つ
    if (this._workersReady) await this._workersReady
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
    this.running = false
    // 現在の job ID を無効化して、残っているメッセージを無視する
    this._currentJobId = (this._currentJobId + 1) | 0
    // 再開時に古い進捗が混ざらないように進捗情報をリセットする
    if (this._workerSamplesDone) {
      this._workerSamplesDone.fill(0)
    }
    for (const w of this.workers) w.postMessage({ cmd: 'stop' })
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
      // data.chunk は { x, y, w, h, r, g, b } 形式
      this._mergeChunk(data.chunk)
      this.onChunk(data.chunk)
    } else if (data.type === 'compile') {
    } else if (data.type === 'done') {
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

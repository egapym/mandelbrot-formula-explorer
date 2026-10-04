/**
 * @author Bert Baron
 * @modified
 */

// worker ごとに停止確認タイミングを少しずらし、同時集中を避ける
const STOP_CHECK_INTERVAL = 200 + Math.floor(Math.random() * 100)

export class WorkerContext {
  constructor() {
    this.currentJob = null
    this.lastStoppedJob = null
    this.nextStopCheck = 0
    this.resetStats()
  }

  initTask(jobToken) {
    this.currentJob = jobToken
    this.nextStopCheck = Date.now() + STOP_CHECK_INTERVAL
  }

  resetStats() {
    this.stats = {
      numberOfHighPrecisionPoints: 0,
      numberOfLowPrecisionPoints: 0,
      numberOfLowPrecisionMisses: 0,
      errorOffsetsPos: [],
      errorOffsetsNeg: [],
    }
  }

  shouldStop() {
    const currentJobToken = this.currentJob
    const ts = Date.now()
    let shouldStop = false
    if (ts > this.nextStopCheck) {
      shouldStop = this._shouldStop(currentJobToken)
      this.nextStopCheck = ts + STOP_CHECK_INTERVAL
    }
    return shouldStop
  }

  _shouldStop(jobToken) {
    if (!jobToken) {
      return false
    }
    const xhr = new XMLHttpRequest()
    xhr.open('GET', jobToken, /* async= */ false)
    try {
      xhr.send(null)
    } catch (_e) {
      return true // リクエストに失敗したため、URL は取り消されたと判断
    }
    return false // URL はまだ有効なので処理を続行できる
  }
}

// スムーズカラーリング計算で繰り返し使う log(2) を定数化（ピクセル毎の再計算を避ける）
const _LOG2_INV = 1 / Math.log(2)

// smooth バッファがあれば値を書き込み、補正後の iter を返す
export function smoothen(smooth, offset, iter, zq) {
  if (smooth && iter > 3) {
    const log_zn = Math.log(zq) / 2
    let nu = Math.log(log_zn * _LOG2_INV) * _LOG2_INV
    iter = Math.floor(iter + 1 - nu)
    nu = nu - Math.floor(nu)
    smooth[offset] = Math.floor(255 - 255 * nu)
  }
  return iter
}

/**
 * Orbit trap計算モジュール
 *
 * トラップ形状 (TRAP_SHAPE) とデータ収集モード (TRAP_MODE) を定義し、
 * 1ピクセル分の値を計算する汎用関数 calculatePixelOrbitTrap を提供する。
 *
 * パレット側が trapSpec (形状・モード・パラメータ) を持つだけで
 * 任意の組み合わせのOrbit trapを計算できる。
 */

// ============================================================
// 定数
// ============================================================

/**
 * トラップ形状定数
 *   CROSS    : クロス形状  min(|zr - tx|, |zi - ty|)
 *   RING     : リング形状  | |z - center| - radius |
 *   POINT    : 点 (原点またはtx,ty) への距離
 *   LINE     : 直線 (tx,ty を通り angle 方向) への距離
 *   PARABOLA : 放物線 (頂点 tx,ty, angle 方向) への近似距離
 *   CIRCLE   : 円 (中心 tx,ty, 半径 size) への距離 (RING の別名)
 *   TRIANGLE : 正三角形 SDF (中心 tx,ty, サイズ size, 回転 angle)
 *   SQUARE   : 正方形 SDF (中心 tx,ty, 半幅 size, 回転 angle)
 *   BITMAP   : ビットマップ画像 (trapSpec に bitmapData を格納)
 */
export const TRAP_SHAPE = {
  CROSS: 'cross',
  RING: 'ring',
  POINT: 'point',
  LINE: 'line',
  PARABOLA: 'parabola',
  CIRCLE: 'circle',
  TRIANGLE: 'triangle',
  SQUARE: 'square',
  BITMAP: 'bitmap',
}

/**
 * データ収集モード定数
 *   DISTANCE_CLOSEST  : 全反復を通じた最近接距離 → float
 *   DISTANCE_FARTHEST  : threshold 未満の中での最遠距離 → float
 *   DISTANCE_AVERAGE   : threshold 未満の距離の平均 → float
 *   CAPTURE_FIRST : threshold に初めて入った反復と強さをエンコード
 *                   → (capturedIter + strength) ただし未捕捉=0
 *   TIA           : 三角不等式平均 (形状不要, スムース補正付き) → float
 *   CAPTURE_STEP  : captureStep 番目の反復での距離を取得 → float
 */
export const TRAP_MODE = {
  DISTANCE_CLOSEST: 'distance_closest',
  DISTANCE_FARTHEST: 'distance_farthest',
  DISTANCE_AVERAGE: 'distance_average',
  CAPTURE_FIRST: 'capture_first',
  TIA: 'tia',
  CAPTURE_STEP: 'capture_step',
}

// ============================================================
// 公開 API
// ============================================================

/**
 * @typedef {Object} TrapSpec
 * トラップ計算の仕様を表すオブジェクト。palette.js 側で定義し
 * index.js 経由でタスクに付加、worker 側で使用する。
 *
 * @property {string}  mode           - TRAP_MODE の値 (必須)
 * @property {string}  [shape]        - TRAP_SHAPE の値 (TIA 以外で必須)
 * @property {number}  [tx=0]              - トラップ中心 x
 * @property {number}  [ty=0]              - トラップ中心 y
 * @property {number}  [size=1]            - 形状サイズ (半径／辺長など)。
 *                                     - **LINE の場合は線分の半長さ**。
 *                                     - **POINT の場合は無視される**。
 *                                     - **BITMAP の場合は画像の長辺サイズ**。
 * @property {number}  [angle=0]           - 形状の回転角 (ラジアン, line/parabola/triangle/square)
 * @property {number}  [threshold=Infinity] - 捕捉閾値
 * @property {number}  [startIter=0]       - 捕捉開始反復 (capture_first)
 * @property {number}  [captureStep=1]     - 何番目の反復を使うか (capture_step)
 * @property {Uint8ClampedArray} [bitmapData]   - ビットマップ画素データ (bitmap)
 * @property {number}  [bitmapWidth=0]     - ビットマップ幅
 * @property {number}  [bitmapHeight=0]    - ビットマップ高さ
 * @property {string}  [bitmapBackgroundColor="#002580"] - bitmap の背景色
 */

/**
 * 1ピクセル分のOrbit trap値を計算して返す。
 *
 * 返値の意味はモードによって異なる:
 *   DISTANCE_CLOSEST  → 最近接距離 (正の実数)
 *   DISTANCE_FARTHEST  → 閾値内での最大距離 (0 = 一度も入らなかった)
 *   DISTANCE_AVERAGE   → 閾値内距離の平均 (0 = 一度も入らなかった)
 *   CAPTURE_FIRST → capturedIter + strength (0 = 未捕捉)
 *   TIA           → スムース補正済み TIA 平均
 *
 * @param {number}   cr
 * @param {number}   ci
 * @param {number}   z0r
 * @param {number}   z0i
 * @param {Function} iterFn   - (zr, zi, cr, ci) => [zrNew, ziNew]
 * @param {number}   maxIter
 * @param {TrapSpec} trapSpec
 * @returns {number}
 */
export function calculatePixelOrbitTrap(cr, ci, z0r, z0i, iterFn, maxIter, trapSpec, escapeRadius = 4.0) {
  const mode = trapSpec.mode

  // helper: UV を負の float にエンコード (BITMAP 用)
  const encodeUV = (u, v) => {
    const uInt = Math.min(4095, Math.max(0, Math.round(u * 4095)))
    const vInt = Math.min(4095, Math.max(0, Math.round(v * 4095)))
    return -(uInt * 4096 + vInt + 1)
  }

  // TIA は形状不要のため個別処理
  if (mode === TRAP_MODE.TIA) {
    return _computeTIA(cr, ci, z0r, z0i, iterFn, maxIter)
  }

  // 形状・パラメータの読み出し
  const shape = trapSpec.shape
  const tx = trapSpec.tx ?? 0
  const ty = trapSpec.ty ?? 0
  const sz = trapSpec.size ?? 1 // 一般的に半径や半幅。LINE では半長さ。
  const threshold = trapSpec.threshold ?? Infinity
  const startIter = trapSpec.startIter ?? 0
  const bitmapW = trapSpec.bitmapWidth ?? 0
  const bitmapH = trapSpec.bitmapHeight ?? 0
  const bitmapSize = Math.abs(sz)
  let bitmapTrapWidth = bitmapSize
  let bitmapTrapHeight = bitmapSize
  if (shape === TRAP_SHAPE.BITMAP && bitmapW > 0 && bitmapH > 0 && bitmapSize > 0) {
    const bitmapAspect = bitmapW / bitmapH
    if (bitmapAspect >= 1) {
      bitmapTrapWidth = bitmapSize
      bitmapTrapHeight = bitmapSize / bitmapAspect
    } else {
      bitmapTrapWidth = bitmapSize * bitmapAspect
      bitmapTrapHeight = bitmapSize
    }
  }

  let zr = z0r
  let zi = z0i

  // 各モードの集計変数
  let dClosest = Infinity
  let dFarthest = 0
  let dFarthestBitmap = -Infinity
  let dSum = 0
  let dCount = 0
  let capturedResult = 0
  // BITMAP 用 UVサンプリング情報
  let closestBitmapU = null
  let closestBitmapV = null
  let farthestBitmapU = null
  let farthestBitmapV = null
  let avgBitmapUSum = 0
  let avgBitmapVSum = 0
  let avgBitmapCount = 0
  let firstBitmapU = null
  let firstBitmapV = null
  let stepBitmapU = null
  let stepBitmapV = null

  for (let i = 0; i < maxIter; i++) {
    const res = iterFn(zr, zi, cr, ci, i)
    let nr = res[0]
    let ni = res[1]
    // 数値が無限 / NaN になった場合はセンチネル値で置換
    if (!Number.isFinite(nr) || !Number.isFinite(ni)) {
      nr = 1e20
      ni = 0.0
    }
    zr = nr
    zi = ni

    // 形状への距離計算
    const dx = zr - tx
    const dy = zi - ty
    const ang = trapSpec.angle ?? 0
    let d
    // bitmap は後続のモードロジックでも u/v を使うため事前計算
    let u = 0,
      v = 0
    let bitmapSampleInBounds = false
    let bitmapSampleVisible = false
    if (shape === TRAP_SHAPE.BITMAP && bitmapTrapWidth > 0 && bitmapTrapHeight > 0) {
      const cosA = Math.cos(ang)
      const sinA = Math.sin(ang)
      const bitmapX = dx * cosA + dy * sinA
      const bitmapY = -dx * sinA + dy * cosA
      u = bitmapX / bitmapTrapWidth + 0.5
      v = -(bitmapY / bitmapTrapHeight) + 0.5
    }
    switch (shape) {
      case TRAP_SHAPE.CROSS:
        // クロス形状: x 軸と y 軸への最短距離を sz でスケーリング。
        // sz を大きくするとクロスの腕幅（引き込み範囲）が広がる。
        // sz=0 はゼロ除算を避けるため 1.0 にフォールバック。
        d = Math.min(Math.abs(dx), Math.abs(dy)) / (sz !== 0 ? sz : 1.0)
        break
      case TRAP_SHAPE.RING:
      case TRAP_SHAPE.CIRCLE:
        // リング/円形状: リングへの距離
        d = Math.abs(Math.sqrt(dx * dx + dy * dy) - sz)
        break
      case TRAP_SHAPE.POINT:
        // 点への距離
        d = Math.sqrt(dx * dx + dy * dy)
        break
      case TRAP_SHAPE.LINE: {
        // 直線または線分への距離
        // もともとは無限直線への距離のみを返していたため
        // size が無視されていた。size を半長さとして扱い、
        // 長さ 2*size の線分に変更する。
        //
        // 回転座標系 (angle) で (px,py) を計算し、px は線方向、py は
        // 法線方向の座標。px を [-size,size] にクランプした点との距離を返す。
        // size=0 では点トラップと同等になる。
        const cosA = Math.cos(ang)
        const sinA = Math.sin(ang)
        const px = dx * cosA + dy * sinA
        const py = -dx * sinA + dy * cosA
        let clamped = px
        if (sz !== 0) {
          clamped = Math.max(-sz, Math.min(sz, px))
        } else {
          // サイズ0なら拘束せず点距離にフォールバック
          clamped = 0
        }
        const dx2 = px - clamped
        d = Math.sqrt(dx2 * dx2 + py * py)
        break
      }
      case TRAP_SHAPE.PARABOLA: {
        // 放物線近似: angle 回転座標系での ly = lx^2 / sz
        const cosA = Math.cos(ang),
          sinA = Math.sin(ang)
        const lx = dx * cosA + dy * sinA
        const ly = -dx * sinA + dy * cosA
        d = Math.abs(ly - (lx * lx) / (sz !== 0 ? sz : 1.0))
        break
      }
      case TRAP_SHAPE.TRIANGLE: {
        // 正三角形のSDF (中心 tx,ty, 外接円半径 sz, 回転 angle)
        // sz=0 のときは点距離にフォールバック（ゼロ除算回避）
        if (sz === 0) {
          d = Math.sqrt(dx * dx + dy * dy)
          break
        }
        const cosA = Math.cos(ang),
          sinA = Math.sin(ang)
        let px = (dx * cosA + dy * sinA) / sz
        let py = (-dx * sinA + dy * cosA) / sz
        const k = Math.sqrt(3)
        px = Math.abs(px) - 1.0
        py = py + 1.0 / k
        if (px + k * py > 0.0) {
          const tmpX = 0.5 * (px - k * py)
          py = 0.5 * (k * px - py)
          px = tmpX
        }
        px -= Math.max(-2.0, Math.min(0.0, px))
        d = Math.sqrt(px * px + py * py) * Math.sign(py) * sz
        d = Math.abs(d)
        break
      }
      case TRAP_SHAPE.SQUARE: {
        // 正方形のSDF: 半幅 sz, 回転 angle
        const cosA = Math.cos(ang),
          sinA = Math.sin(ang)
        const qx = Math.abs(dx * cosA + dy * sinA) - sz
        const qy = Math.abs(-dx * sinA + dy * cosA) - sz
        d = Math.sqrt(Math.max(qx, 0) ** 2 + Math.max(qy, 0) ** 2) + Math.min(Math.max(qx, qy), 0)
        d = Math.abs(d)
        break
      }
      case TRAP_SHAPE.BITMAP: {
        // ビットマップトラップ: 画像のアスペクト比を保つ矩形にマッピングしてアルファ値から距離を算出
        const bitmapData = trapSpec.bitmapData
        // sz=0 またはデータ未設定のときは点距離にフォールバック（ゼロ除算回避）
        if (!bitmapData || bitmapW <= 0 || bitmapH <= 0 || bitmapTrapWidth <= 0 || bitmapTrapHeight <= 0) {
          d = Math.sqrt(dx * dx + dy * dy)
          break
        }
        // u,v はすでに計算済み
        if (u < 0 || u > 1 || v < 0 || v > 1) {
          // 画像外は矩形の輪郭をトラップ化せず、完全透明ピクセルと同じ扱いにする。
          d = 1.0
        } else {
          bitmapSampleInBounds = true
          // 最近傍ピクセルをサンプリングしアルファ値から距離を得る
          const px = Math.min(Math.floor(u * bitmapW), bitmapW - 1)
          const py = Math.min(Math.floor(v * bitmapH), bitmapH - 1)
          const idx = (py * bitmapW + px) * 4
          const alpha = bitmapData[idx + 3] / 255
          bitmapSampleVisible = bitmapData[idx + 3] >= 10
          // alpha=1 (不透明) → d=0, alpha=0 (透明) → d=1
          d = 1.0 - alpha
        }
        break
      }
      default:
        // フォールバック: 点距離
        d = Math.sqrt(dx * dx + dy * dy)
    }

    switch (mode) {
      case TRAP_MODE.DISTANCE_CLOSEST:
        if (d < dClosest) {
          dClosest = d
          if (shape === TRAP_SHAPE.BITMAP) {
            closestBitmapU = bitmapSampleInBounds ? u : null
            closestBitmapV = bitmapSampleInBounds ? v : null
          }
        }
        break

      case TRAP_MODE.DISTANCE_FARTHEST:
        // threshold 内に入った距離の中での最大値
        // 初期 dFarthest=0 のため等号付きで更新すると
        // 値0でもUVが記録される（真に最大値が0の場合に必要）。
        if (d < threshold && d >= dFarthest) {
          dFarthest = d
        }
        if (shape === TRAP_SHAPE.BITMAP && bitmapSampleVisible && d < threshold && d >= dFarthestBitmap) {
          dFarthestBitmap = d
          farthestBitmapU = u
          farthestBitmapV = v
        }
        break

      case TRAP_MODE.DISTANCE_AVERAGE:
        // threshold 内に入った距離の累計
        if (d < threshold) {
          dSum += d
          dCount++
          if (shape === TRAP_SHAPE.BITMAP && bitmapSampleInBounds) {
            avgBitmapUSum += u
            avgBitmapVSum += v
            avgBitmapCount++
          }
        }
        break

      case TRAP_MODE.CAPTURE_FIRST:
        // startIter 以降、最初に threshold に入った瞬間を記録して終了
        if (capturedResult === 0 && i >= startIter && d <= threshold) {
          const strength = 1.0 - d / threshold
          // 整数部 = 反復番号, 小数部 = 強さ (0.9999 に上限クランプ)
          capturedResult = i + 1 + Math.min(strength, 0.9999)
          if (shape === TRAP_SHAPE.BITMAP && bitmapSampleInBounds) {
            firstBitmapU = u
            firstBitmapV = v
          }
        }
        break

      case TRAP_MODE.CAPTURE_STEP: {
        // captureStep 番目の反復時の距離を記録 (1-indexed)
        const captureStep = trapSpec.captureStep ?? 1
        if (i + 1 === captureStep) {
          capturedResult = d
          if (shape === TRAP_SHAPE.BITMAP && bitmapSampleInBounds) {
            stepBitmapU = u
            stepBitmapV = v
          }
        }
        break
      }
    }

    // Escape R UIの設定値に従ってループを打ち切る。
    // z が escapeRadius を超えたら軌跡が発散したとみなし終了。
    if (zr * zr + zi * zi > escapeRadius * escapeRadius) break
  }

  // モード別に戻り値を決定
  switch (mode) {
    case TRAP_MODE.DISTANCE_CLOSEST:
      if (shape === TRAP_SHAPE.BITMAP && closestBitmapU !== null) {
        return encodeUV(closestBitmapU, closestBitmapV)
      }
      return Number.isFinite(dClosest) ? dClosest : 0.0
    case TRAP_MODE.DISTANCE_FARTHEST:
      if (shape === TRAP_SHAPE.BITMAP && farthestBitmapU !== null) {
        return encodeUV(farthestBitmapU, farthestBitmapV)
      }
      return dFarthest
    case TRAP_MODE.DISTANCE_AVERAGE:
      if (shape === TRAP_SHAPE.BITMAP) {
        if (avgBitmapCount > 0) {
          return encodeUV(avgBitmapUSum / avgBitmapCount, avgBitmapVSum / avgBitmapCount)
        }
        // fall through to numeric case so that 0 is returned when no samples
      }
      // for non-bitmap shapes we care about dCount, not avgBitmapCount
      return dCount > 0 ? dSum / dCount : 0.0
    case TRAP_MODE.CAPTURE_FIRST:
      if (shape === TRAP_SHAPE.BITMAP && firstBitmapU !== null) {
        return encodeUV(firstBitmapU, firstBitmapV)
      }
      return capturedResult
    case TRAP_MODE.CAPTURE_STEP:
      if (shape === TRAP_SHAPE.BITMAP && stepBitmapU !== null) {
        return encodeUV(stepBitmapU, stepBitmapV)
      }
      return capturedResult
    default:
      return 0.0
  }
}

// ============================================================
// 内部実装
// ============================================================

/**
 * TIA (Triangle Inequality Average) を計算する。
 * orbit trap 3.py の draw_image3 と同等のロジック。
 * スムース補正 (average + (average - average2) * f) 付き。
 */
function _computeTIA(cr, ci, z0r, z0i, iterFn, maxIter) {
  const OT3_BAILOUT = 1e20
  const LOG2 = Math.log(2)
  const lp2 = Math.log(Math.log(OT3_BAILOUT))
  const cDist = Math.sqrt(cr * cr + ci * ci)

  let zr = z0r
  let zi = z0i
  let dSum = 0.0
  let dSumPrev = 0.0
  let iterCount = 0
  let finalZq = 0.0
  let escaped = false

  for (let i = 0; i < maxIter; i++) {
    const res = iterFn(zr, zi, cr, ci, i)
    let nr = res[0]
    let ni = res[1]
    if (!Number.isFinite(nr) || !Number.isFinite(ni)) {
      nr = 1e20
      ni = 0.0
    }
    zr = nr
    zi = ni
    const zq = zr * zr + zi * zi

    if (zq > OT3_BAILOUT) {
      finalZq = zq
      escaped = true
      break
    }

    // 最初の反復 (i=0) はスキップ (Python: if iteration != 0)
    if (i > 0) {
      const txc = zr - cr
      const tyc = zi - ci
      const zDist = Math.sqrt(txc * txc + tyc * tyc)
      const radius = Math.sqrt(zq)
      const lowbound = Math.abs(zDist - cDist)
      const denom = zDist + cDist - lowbound
      dSumPrev = dSum
      if (denom > 1e-10) {
        dSum += (radius - lowbound) / denom
      }
      iterCount++
    }
  }

  // スムース補正付き平均を返す
  if (escaped && iterCount > 1 && finalZq > 1.0 && Number.isFinite(Math.log(finalZq))) {
    const average = dSum / iterCount
    const average2 = dSumPrev / (iterCount - 1)
    const f = (lp2 - Math.log(Math.log(finalZq))) / LOG2
    const corrected = average + (average - average2) * f
    return Number.isFinite(corrected) ? corrected : average
  }

  return iterCount > 0 ? dSum / iterCount : 0.0
}

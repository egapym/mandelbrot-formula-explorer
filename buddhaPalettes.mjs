/**
 * Based on bertbaron/mandelbrot by Bert Baron
 * This file is part of the Mandelbrot Formula Explorer project.
 * Licensed under GPL-3.0.
 */

/**
 * 定義済みの Buddhabrot カラーパレット。
 * 各パレットは色と比率を持つ band の配列で表す。
 */
export const BUDDHA_PALETTES = [
  {
    id: 'Preset-01',
    name: 'Preset-01',
    buddhaBandMode: 'perTrajectory',
    bands: [
      { color: [0, 67, 112], ratio: 0.04 },
      { color: [0, 179, 167], ratio: 0.04 },
      { color: [204, 197, 0], ratio: 0.04 },
      { color: [230, 58, 15], ratio: 0.88 },
    ],
  },
  {
    id: 'Preset-02',
    name: 'Preset-02',
    buddhaBandMode: 'perPoint',
    bands: [
      { color: [255, 255, 255], ratio: 0.02 },
      { color: [255, 200, 0], ratio: 0.18 },
      { color: [220, 0, 0], ratio: 0.8 },
    ],
  },
  {
    id: 'Preset-03',
    name: 'Preset-03',
    buddhaBandMode: 'perTrajectory',
    bands: [
      { color: [0, 0, 0], ratio: 0.01 },
      { color: [255, 255, 255], ratio: 0.19 },
      { color: [201, 50, 255], ratio: 0.8 },
    ],
  },
  {
    id: 'Preset-04',
    name: 'Preset-04',
    buddhaBandMode: 'perPoint',
    bands: [
      { color: [0, 0, 0], ratio: 0.05 },
      { color: [0, 255, 0], ratio: 0.65 },
      { color: [255, 255, 0], ratio: 0.3 },
    ],
  },
]

/**
 * ID から Buddhabrot パレットを取得する
 * @param {string} id - パレット ID
 * @returns {Object|undefined} パレット。見つからなければ undefined
 */
export function getBuddhaPalette(id) {
  return BUDDHA_PALETTES.find((p) => p.id === id)
}

/**
 * パレットから正規化済みの band stop を作る
 *
 * band の比率を合計 1 にそろえ、累積値を付けて返す。
 * @param {string} id - パレット ID
 * @returns {Array<{color: number[], ratio: number, cum: number}>|null} 正規化済み band。作れなければ null
 */
export function buildBuddhaStops(id) {
  const p = getBuddhaPalette(id)
  if (!p || !Array.isArray(p.bands) || p.bands.length === 0) return null
  // band を複製する。黒 [0,0,0] も有効な色なので削除しない
  const bands = p.bands.map((b) => ({
    color: b.color?.slice ? b.color.slice() : [0, 0, 0],
    ratio: Number(b.ratio) || 0,
  }))

  if (!bands || bands.length === 0)
    // 有効な band がなければ stop は作れない
    return null
  const sum = bands.reduce((a, b) => a + b.ratio, 0)
  if (sum <= 0) {
    const n = bands.length
    const r = 1 / n
    let acc = 0
    return bands.map((b) => {
      acc += r
      return { color: b.color, ratio: r, cum: acc }
    })
  }
  // 正規化して累積値を計算する
  let acc = 0
  return bands.map((b) => {
    const r = b.ratio / sum
    acc += r
    return { color: b.color, ratio: r, cum: acc }
  })
}

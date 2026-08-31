/**
 * @author Bert Baron
 * @modified
 */

import { TRAP_MODE, TRAP_SHAPE } from './orbitTrap.mjs'

export const DEFAULT_BITMAP_BACKGROUND_COLOR = '#002580'
const DEFAULT_BITMAP_BACKGROUND_RGB = [0, 37, 128]

function hexColorToRgb(hex, fallback = DEFAULT_BITMAP_BACKGROUND_RGB) {
  const color = String(hex ?? '').trim()
  const match = /^#?([0-9a-fA-F]{6})$/.exec(color)
  if (!match) return fallback

  const value = Number.parseInt(match[1], 16)
  return [(value >> 16) & 255, (value >> 8) & 255, value & 255]
}

function blendBitmapPixelOverBackground(bitmapData, idx, bgR, bgG, bgB) {
  const alpha = bitmapData[idx + 3]
  if (alpha < 10) return [bgR, bgG, bgB]
  if (alpha >= 255) return [bitmapData[idx], bitmapData[idx + 1], bitmapData[idx + 2]]

  const alphaRatio = alpha / 255
  const invAlpha = 1 - alphaRatio
  return [
    Math.round(bitmapData[idx] * alphaRatio + bgR * invAlpha),
    Math.round(bitmapData[idx + 1] * alphaRatio + bgG * invAlpha),
    Math.round(bitmapData[idx + 2] * alphaRatio + bgB * invAlpha),
  ]
}

export function getPalette(id) {
  const palette = PALETTES.find((p) => p.id === id)
  if (palette) return palette
  return BLUE_GOLD
}

export function initPallet(palette, density, rotate, _exp, max_iter) {
  const rgbaBuffer = new Uint8ClampedArray(max_iter * 4 + 20)
  // 0 and 1 = transparent (skipped)
  // 2 and 3 = color for points in the set (not diverged)
  // スムージング時に 1 つずらして参照するため、先頭側を重ねて確保する

  // セット内の点の色を設定する (index 2 と 3)
  const [inSetR, inSetG, inSetB] = palette.inSetColor || [0, 0, 0]
  rgbaBuffer[8] = inSetR // index 2 * 4
  rgbaBuffer[9] = inSetG
  rgbaBuffer[10] = inSetB
  rgbaBuffer[11] = 255
  rgbaBuffer[12] = inSetR // index 3 * 4
  rgbaBuffer[13] = inSetG
  rgbaBuffer[14] = inSetB
  rgbaBuffer[15] = 255

  // 密度は指数的に補正する
  density = 2 ** (density / 10)
  for (let i = 0; i <= max_iter; i++) {
    const v = density * i // Math.pow(i*2, 0.9)

    const [r, g, b] = palette.getColor(v, rotate)
    rgbaBuffer[(i + 4) * 4] = r
    rgbaBuffer[(i + 4) * 4 + 1] = g
    rgbaBuffer[(i + 4) * 4 + 2] = b
    rgbaBuffer[(i + 4) * 4 + 3] = 255
  }

  return rgbaBuffer
}

// 厳密さは未検証だが、発色は良い変換
function toSRGB(r, g, b) {
  function toSRGBComponent(c) {
    return c <= 0.04045 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4
  }

  return [toSRGBComponent(r), toSRGBComponent(g), toSRGBComponent(b)]
}

// 事前計算済み lookup buffer を使う通常パレットの基底クラス。
// 派生クラスは getColor(v, rotate) だけ実装すればよい。
class NormalPalette {
  constructor() {
    // スムーズ描画と skipTopLeft 最適化に対応するか。
    // 生データから各ピクセル色を計算する特殊パレットでは false にする。
    this.supportsSmooth = true
  }

  /**
   * 事前計算済み lookupBuffer から bufferData を埋める。
   * Offscreen.render() から呼ばれる。
   */
  renderPixels(bufferData, smoothData, values, smooth, _signs, lookupBuffer, withSmooth) {
    for (let i = 0; i < values.length; i++) {
      const iter = values[i]
      bufferData[i * 4] = lookupBuffer[iter * 4]
      bufferData[i * 4 + 1] = lookupBuffer[iter * 4 + 1]
      bufferData[i * 4 + 2] = lookupBuffer[iter * 4 + 2]
      bufferData[i * 4 + 3] = lookupBuffer[iter * 4 + 3]
      if (withSmooth) {
        smoothData[i * 4] = lookupBuffer[iter * 4 + 4]
        smoothData[i * 4 + 1] = lookupBuffer[iter * 4 + 5]
        smoothData[i * 4 + 2] = lookupBuffer[iter * 4 + 6]
        smoothData[i * 4 + 3] = smooth[i]
      }
    }
  }

  /**
   * 事前計算済み lookupBuffer を使ってプレビュー帯を描く。
   * renderPalette() から呼ばれる。
   */
  renderPreview(ctx, width, height, lookupBuffer) {
    const offset = 4
    const paletteSize = lookupBuffer.length / 4 - offset
    for (let i = 0; i < paletteSize; i++) {
      const colorIndex = i + offset
      const pos = Math.floor((i * width) / paletteSize)
      const w = Math.floor(((i + 1) * width) / paletteSize) - pos
      const r = lookupBuffer[colorIndex * 4]
      const g = lookupBuffer[colorIndex * 4 + 1]
      const b = lookupBuffer[colorIndex * 4 + 2]
      ctx.fillStyle = `rgb(${r},${g},${b})`
      ctx.fillRect(pos, 0, w, height)
    }
  }
}

class BlueGoldPalette extends NormalPalette {
  constructor() {
    super()
    this.id = 'blue_gold'
    this.name = 'Blue & Gold'
    this.wavelengths = [80, 81, 85]
    this.mirrorPosition = 565
    this.inSetColor = [0, 0, 0] // Black for points in the set
  }

  getColor(v, rotate) {
    let idx = (v * 2 + 590 + (rotate / 180) * this.mirrorPosition) % (this.mirrorPosition * 2)
    if (idx >= this.mirrorPosition) {
      idx = this.mirrorPosition - (idx - this.mirrorPosition)
    }

    const r = Math.cos((idx / this.wavelengths[0]) * Math.PI) * 0.5 + 0.5
    const g = Math.cos((idx / this.wavelengths[1]) * Math.PI) * 0.5 + 0.5
    const b = Math.cos((idx / this.wavelengths[2]) * Math.PI) * 0.5 + 0.5

    const [rr, gg, bb] = toSRGB(r, g, b)
    return [Math.round(rr * 255), Math.round(gg * 255), Math.round(bb * 255)]
  }
}

class GrayScalePalette extends NormalPalette {
  constructor(id, name, min, max) {
    super()
    this.id = id
    this.name = name
    this.min = min
    this.max = max
    this.inSetColor = [0, 0, 0] // Black for points in the set
  }

  getColor(v, rotate) {
    const idx = v * 1.6 + (rotate / 180) * 80
    const f = Math.sin((idx / 80) * Math.PI - Math.PI / 3) * 127 + 128
    return [f, f, f]
  }
}

class SingleColorPalette extends NormalPalette {
  constructor(id, name, color) {
    super()
    this.id = id
    this.name = name
    this.color = color
    this.inSetColor = [0, 0, 0] // Black for points in the set
  }

  getColor() {
    return this.color
  }
}

class IndexedPalette extends NormalPalette {
  constructor(id, name, colors, mirror, reverse, inSetColor = null) {
    super()
    this.id = id
    this.name = name
    this.colors = []
    this.colors = this.colors.concat(colors)
    reverse && this.colors.reverse()
    if (mirror) {
      this.colors = this.colors.concat(this.colors.slice(1, this.colors.length - 1).reverse())
    }
    this.inSetColor = inSetColor || [0, 0, 0] // Default to black if not specified
  }

  getColor(v, rotate) {
    const palette = this.colors
    const scaled = (v * palette.length) / 100 + (rotate / 360) * palette.length
    return this.getInterpolationFunctions().map((fn) => Math.round(fn(scaled)))
  }

  getInterpolationFunctions() {
    if (!this.interpolationFunctions) {
      this.interpolationFunctions = [0, 1, 2].map((i) => monotoneCubicInterpolationFN(this.colors.map((c) => c[i])))
    }
    return this.interpolationFunctions
  }
}

const BLUE_GOLD = new BlueGoldPalette()

// Ultra Fractal 風の配色。こちらは色間隔を等間隔にしている。
const MANDELBROT = new IndexedPalette(
  'mandelbrot',
  'Mandelbrot',
  [
    [0, 7, 100],
    [32, 107, 203],
    [237, 255, 255],
    [255, 170, 0],
    [0, 2, 0],
  ],
  false,
)

const LAVA = new IndexedPalette(
  'lava',
  'Lava',
  [
    [0, 0, 0],
    [10, 0, 0],
    [20, 0, 0], // [20, 0, 0],
    [40, 0, 0],
    [80, 0, 0],
    [160, 10, 0],
    [200, 40, 0],
    [240, 90, 0],
    [255, 160, 0],
    [255, 220, 10],
    [255, 255, 80],
    [255, 255, 160],
    [255, 255, 255],
  ],
  true,
)
const FALL = new IndexedPalette(
  'fall',
  'Fall',
  [
    [25, 25, 25],
    [128, 0, 0],
    [255, 69, 0],
    [255, 140, 0],
    [255, 215, 0],
    [255, 239, 184],
  ],
  false,
)
const OCEAN = new IndexedPalette(
  'ocean',
  'Ocean',
  [
    [0, 0, 51],
    [0, 0, 102],
    [0, 0, 153],
    [0, 51, 102],
    [0, 102, 204],
    [51, 153, 255],
    [102, 178, 255],
    [153, 204, 255],
    [204, 229, 255],
    [255, 255, 255],
  ],
  true,
)
const POP = new IndexedPalette(
  'pop',
  'Pop',
  [
    [255, 0, 0],
    [255, 165, 0],
    [255, 255, 0],
    [0, 128, 0],
    [0, 0, 255],
    [128, 0, 128],
    [255, 0, 255],
    [255, 192, 203],
    [255, 99, 71],
    [0, 255, 255],
    [0, 255, 0],
    [255, 0, 128],
  ],
  false,
)
const SKY_WATER = new IndexedPalette(
  'sky_water',
  'Sky & Water',
  [
    [0, 0, 51],
    [0, 51, 102],
    [0, 102, 153],
    [0, 153, 204],
    [51, 153, 204],
    [102, 178, 255],
    [153, 204, 255],
    [178, 223, 255],
    [204, 238, 255],
    [229, 255, 255],
    [255, 255, 255],
    [51, 153, 204],
    [0, 102, 153],
  ],
  false,
)
const JEWELLERY = new IndexedPalette(
  'jewellery',
  'Jewellery',
  [
    [0, 0, 51],
    [0, 0, 102],
    [0, 0, 153],
    [0, 102, 204],
    [51, 153, 255],
    [0, 102, 102],
    [0, 128, 128],
    [204, 204, 255],
    [255, 204, 0],
    [255, 0, 0],
    [255, 0, 255],
    [255, 255, 255],
    [51, 153, 255],
    [0, 0, 153],
  ],
  false,
)

// ============================================================
// 反復回数の偶奇や z の符号で色分けする特殊パレット。
// 標準の lookup buffer は使わず、renderPixels() / renderPreview() を直接実装する。
// ============================================================

class StripePalette {
  constructor() {
    this.id = 'stripe'
    this.name = 'Stripe'
    this.supportsSmooth = false
    this.inSetColor = [0, 0, 0]
    this.color1 = [255, 255, 0] // 奇数イテレーション
    this.color2 = [0, 0, 255] // 偶数イテレーション
  }

  // 描画では使わないが、initPallet() から呼ばれても壊れないように置く
  getColor() {
    return [0, 0, 0]
  }

  renderPixels(bufferData, _smoothData, values) {
    const c1 = this.color1
    const c2 = this.color2
    for (let i = 0; i < values.length; i++) {
      const iter = values[i]
      let r, g, b
      if (iter <= 3) {
        r = 0
        g = 0
        b = 0
      } else if ((iter & 1) === 1) {
        r = c1[0]
        g = c1[1]
        b = c1[2]
      } else {
        r = c2[0]
        g = c2[1]
        b = c2[2]
      }
      bufferData[i * 4] = r
      bufferData[i * 4 + 1] = g
      bufferData[i * 4 + 2] = b
      bufferData[i * 4 + 3] = 255
    }
  }

  renderPreview(ctx, width, height) {
    const c1 = this.color1
    const c2 = this.color2
    ctx.fillStyle = `rgb(${c1[0]},${c1[1]},${c1[2]})`
    ctx.fillRect(0, 0, Math.floor(width / 2), height)
    ctx.fillStyle = `rgb(${c2[0]},${c2[1]},${c2[2]})`
    ctx.fillRect(Math.floor(width / 2), 0, width - Math.floor(width / 2), height)
  }
}

class GridPalette {
  constructor() {
    this.id = 'grid'
    this.name = 'Grid'
    this.supportsSmooth = false
    this.inSetColor = [0, 0, 0]
    this.color1 = [255, 255, 0] // 同符号
    this.color2 = [0, 0, 255] // 異符号
  }

  // 描画では使わないが、initPallet() から呼ばれても壊れないように置く
  getColor() {
    return [0, 0, 0]
  }

  renderPixels(bufferData, _smoothData, values, _smooth, signs) {
    const c1 = this.color1
    const c2 = this.color2
    for (let i = 0; i < values.length; i++) {
      const s = signs[i]
      let r, g, b
      if (s === 1) {
        r = c1[0]
        g = c1[1]
        b = c1[2]
      } else if (s === 2) {
        r = c2[0]
        g = c2[1]
        b = c2[2]
      } else {
        r = 0
        g = 0
        b = 0
      }
      bufferData[i * 4] = r
      bufferData[i * 4 + 1] = g
      bufferData[i * 4 + 2] = b
      bufferData[i * 4 + 3] = 255
    }
  }

  renderPreview(ctx, width, height) {
    const c1 = this.color1
    const c2 = this.color2
    ctx.fillStyle = `rgb(${c1[0]},${c1[1]},${c1[2]})`
    ctx.fillRect(0, 0, Math.floor(width / 2), height)
    ctx.fillStyle = `rgb(${c2[0]},${c2[1]},${c2[2]})`
    ctx.fillRect(Math.floor(width / 2), 0, width - Math.floor(width / 2), height)
  }
}

// ============================================================
// Orbit trap パレット
//
// 汎用的な OrbitTrapPalette クラスを定義する。
// trapSpec (TRAP_SHAPE + TRAP_MODE + パラメータ) と着色関数 colorFn を
// コンストラクタに渡すだけで、任意の組み合わせのパレットを作成できる。
// ============================================================

/**
 * HSL → RGB 変換 (Pythonのget_color_hslと同等)
 * @param {number} h - 色相 0〜1
 * @param {number} s - 彩度 0〜1
 * @param {number} l - 輝度 0〜1
 * @returns {[number, number, number]} RGB 各0〜255
 */
function hslToRgb(h, s, l) {
	const coef = [0, 4, 2] // R, G, B チャンネルのオフセット
return coef.map((c) => {
  const val = Math.max(0, Math.min(1, Math.abs(((c + 6 * h) % 6) - 3) - 1))
  return Math.round((l + s * (val - 0.5) * (1 - Math.abs(2 * l - 1))) * 255)
})
}

/**
 * 汎用Orbit trapパレット。
 *
 * trapSpec でトラップの形状・データ収集モードを指定し、
 * colorFn で収集した値を色に変換する。
 * 新しいパレットを追加する際はこのクラスのインスタンスを PALETTES に追加するだけでよい。
 *
 * @example
 * // リングトラップ + DISTANCE_CLOSEST + コサインRGB
 * new OrbitTrapPalette(
 *   'my_ring', 'My Ring Trap',
 *   { mode: TRAP_MODE.DISTANCE_CLOSEST, shape: TRAP_SHAPE.RING, tx: 0, ty: 0, size: 0.5 },
 *   (dist) => hslToRgb(dist % 1.0, 1.0, 0.5)
 * )
 */
export class OrbitTrapPalette {
  /**
   * @param {string}   id       - パレットID (一意)
   * @param {string}   name     - 表示名
   * @param {import('./orbitTrap.mjs').TrapSpec} trapSpec - トラップ計算仕様
   * @param {Function} colorFn  - (value: number, iter: number) => [r, g, b] | null
   *                              null を返すと inSetColor を使用
   * @param {Object}   [options]
   * @param {number[]} [options.inSetColor=[0,0,0]] - セット内ピクセルの色
   */
  constructor(id, name, trapSpec, colorFn, options = {}) {
    this.id = id
    this.name = name
    // trapSpec はインスタンスプロパティとして保持。
    // index.js が task.trapSpec = palette.trapSpec でタスクに付加する。
    this.trapSpec = trapSpec
    this._colorFn = colorFn
    this.supportsSmooth = false
    this.inSetColor = options.inSetColor ?? [0, 0, 0]
    // Orbit trap 計算は CPU ワーカーで行うため、GPU パスを使用しない。
    // index.js はこのフラグが true の場合、useGpu がONでも CPU パスにフォールバックする。
    this.requiresCpu = true
  }

  // initPallet() から呼ばれることがあるためダミー実装が必要
  getColor() {
    return [0, 0, 0]
  }

  /**
   * @param {Uint8ClampedArray} bufferData
   * @param {Uint8ClampedArray} smoothData
   * @param {Int32Array}        values
   * @param {Uint8Array}        smooth
   * @param {Int8Array}         signs
   * @param {Uint8ClampedArray} lookupBuffer
   * @param {boolean}           withSmooth
   * @param {Float32Array}      zreal
   * @param {Float32Array}      zimag
   * @param {Float32Array|null} otData - calculatePixelOrbitTrap の結果 (1ピクセル1値)
   */
  renderPixels(bufferData, _smoothData, _values, _smooth, _signs, _lookupBuffer, _withSmooth, _zreal, _zimag, otData) {
    const values = _values
    const [inR, inG, inB] = this.inSetColor
    const isBitmapShape = this.trapSpec?.shape === 'bitmap'
    const [bgR, bgG, bgB] = hexColorToRgb(this.trapSpec?.bitmapBackgroundColor)

    // BITMAP でも画像未選択の間は orbitTrap.mjs 側の距離フォールバックで通常描画を維持する。
    // 画像があるときだけ bitmapData から RGB を直接参照する。
    const isBitmapDirect = isBitmapShape && this.trapSpec?.bitmapData

    for (let i = 0; i < values.length; i++) {
      const iter = values[i]
      // iter が小さすぎる場合 (セット内または非常に早い脱出) は inSetColor
      if (iter <= 3 || !otData) {
        bufferData[i * 4] = inR
        bufferData[i * 4 + 1] = inG
        bufferData[i * 4 + 2] = inB
        bufferData[i * 4 + 3] = 255
        continue
      }
      const val = otData[i]

      if (isBitmapDirect && val >= -0.5) {
        bufferData[i * 4] = bgR
        bufferData[i * 4 + 1] = bgG
        bufferData[i * 4 + 2] = bgB
        bufferData[i * 4 + 3] = 255
        continue
      }

      // BITMAPモード全般: 負のfloatにUV座標がエンコードされている
      if (isBitmapDirect && val < -0.5) {
        const packed = -val - 1 // = uInt * 4096 + vInt
        const uInt = Math.floor(packed / 4096)
        const vInt = Math.round(packed % 4096)
        const u = uInt / 4095
        const v = vInt / 4095
        const bitmapData = this.trapSpec.bitmapData
        const bitmapW = this.trapSpec.bitmapWidth
        const bitmapH = this.trapSpec.bitmapHeight
        const px = Math.min(Math.floor(u * bitmapW), bitmapW - 1)
        const py = Math.min(Math.floor(v * bitmapH), bitmapH - 1)
        const idx = (py * bitmapW + px) * 4
        const [outR, outG, outB] = blendBitmapPixelOverBackground(bitmapData, idx, bgR, bgG, bgB)
        bufferData[i * 4] = outR
        bufferData[i * 4 + 1] = outG
        bufferData[i * 4 + 2] = outB
        bufferData[i * 4 + 3] = 255
        continue
      }

      const rgb = this._colorFn(val, iter)
      if (!rgb) {
        bufferData[i * 4] = inR
        bufferData[i * 4 + 1] = inG
        bufferData[i * 4 + 2] = inB
      } else {
        bufferData[i * 4] = rgb[0]
        bufferData[i * 4 + 1] = rgb[1]
        bufferData[i * 4 + 2] = rgb[2]
      }
      bufferData[i * 4 + 3] = 255
    }
  }

  /**
   * colorFn を 0〜1 の値範囲で可視化するプレビュー。
   * CAPTURE_FIRST モードは 1〜20 の範囲でプレビューする。
   */
  renderPreview(ctx, width, height, _lookupBuffer) {
    const isCaptureFirst = this.trapSpec.mode === TRAP_MODE.CAPTURE_FIRST
    for (let i = 0; i < width; i++) {
      const t = i / width
      const val = isCaptureFirst ? 1 + t * 19 : t
      const rgb = this._colorFn(val, 10) ?? this.inSetColor
      ctx.fillStyle = 'rgb(' + rgb[0] + ',' + rgb[1] + ',' + rgb[2] + ')'
      ctx.fillRect(i, 0, 1, height)
    }
  }
}

// ============================================================
// Orbit trapパレット
// 新しいパレットを追加する場合は同様のパターンで追加する。
// ============================================================

/**
 * カスタムOrbit trapパレットのカラーパターン識別子。
 * makeOrbitTrapColorFn() に渡すと対応する着色関数を返す。
 */
export const COLOR_PATTERN = {
  COSINE_RGB: 'cosine_rgb', // コサインRGB (ring/cross系に最適)
  HSL_CAPTURE: 'hsl_capture', // HSL捕捉着色 (capture_first系に最適)
  TIA_TRIANGLE: 'tia_triangle', // TIA三角波HSL (TIA系に最適)
}

/**
 * カラーパターンIDに対応する着色関数を返す。
 * @param {string} patternId - COLOR_PATTERN の値
 * @returns {Function} (value: number, iter?: number) => [r, g, b] | null
 */
export function makeOrbitTrapColorFn(patternId) {
	switch (patternId) {
  case COLOR_PATTERN.COSINE_RGB:
    return (d) => {
      const DENSITY = 3.0
      const TRAP_POWER = 0.5
      const alpha = Math.max(d, 0) ** TRAP_POWER * DENSITY * 2 * Math.PI
      return [
        Math.round((Math.cos(alpha - Math.PI) + 1) * 0.5 * 255),
        Math.round((Math.cos(alpha - 0.75 * Math.PI) + 1) * 0.5 * 255),
        Math.round((Math.cos(alpha - 0.5 * Math.PI) + 1) * 0.5 * 255),
      ]
    }
  case COLOR_PATTERN.HSL_CAPTURE:
    return (encoded) => {
      // encoded=0 は未捕捉 → null で inSetColor を使用
      if (encoded === 0) return null
      const capturedIter = Math.floor(encoded)
      const strength = encoded - capturedIter
      const COLOR_DENSITY = 6.0
      const hue = (capturedIter / COLOR_DENSITY) % 1.0
      const lum = strength * 0.75
      return hslToRgb(hue, 1.0, lum)
    }
  case COLOR_PATTERN.TIA_TRIANGLE:
  default:
    return (tia) => {
      const COLOR_DENSITY = 4.0
      const COLOR_HUE = 1 / 3 // 緑系
      const COLOR_SAT = 0.6
      const scaledAlpha = tia * COLOR_DENSITY
      const frac = scaledAlpha - Math.floor(scaledAlpha)
      // 三角波: 0→0, 0.5→1, 1→0 の滑らかな輝度変化
      const lum = (0.5 - Math.abs(frac - 0.5)) * 2
      return hslToRgb(COLOR_HUE, COLOR_SAT, lum)
    }
}
}

/** orbit trap: クロストラップ + DISTANCE_CLOSEST + コサインRGB */
const OT_CROSS_COSINE = new OrbitTrapPalette(
  'orbit_trap_ring',
  'Orbit trap Cross (Cosine)',
  {
    mode: TRAP_MODE.DISTANCE_CLOSEST,
    shape: TRAP_SHAPE.CROSS,
    tx: 0.1,
    ty: -0.1,
  },
  makeOrbitTrapColorFn(COLOR_PATTERN.COSINE_RGB),
)
OT_CROSS_COSINE._colorPatternId = COLOR_PATTERN.COSINE_RGB

/** orbit trap: クロストラップ + CAPTURE_FIRST + HSL着色 */
const OT_CROSS_HSL = new OrbitTrapPalette(
  'orbit_trap_cross',
  'Orbit trap (HSL)',
  {
    mode: TRAP_MODE.CAPTURE_FIRST,
    shape: TRAP_SHAPE.CROSS,
    tx: 0.25,
    ty: 0.25,
    threshold: 0.05,
    startIter: 3,
  },
  makeOrbitTrapColorFn(COLOR_PATTERN.HSL_CAPTURE),
)
OT_CROSS_HSL._colorPatternId = COLOR_PATTERN.HSL_CAPTURE

/** orbit trap 3.py: TIA (三角不等式平均) + 三角波輝度 + HSL */
const OT_TIA = new OrbitTrapPalette(
  'orbit_trap_tia',
  'Orbit trap TIA',
  { mode: TRAP_MODE.TIA },
  makeOrbitTrapColorFn(COLOR_PATTERN.TIA_TRIANGLE),
)
OT_TIA._colorPatternId = COLOR_PATTERN.TIA_TRIANGLE

/**
 * UIから全パラメータをリアルタイムに変更できるカスタムOrbit trapパレット。
 * trapSpec と colorFn が可変で、UI変更のたびに index.js から更新される。
 */
export class CustomOrbitTrapPalette extends OrbitTrapPalette {
  constructor() {
    const defaultTrapSpec = {
      mode: TRAP_MODE.DISTANCE_CLOSEST,
      shape: TRAP_SHAPE.RING,
      tx: 0,
      ty: 0,
      size: 0.5,
      angle: 0,
      bitmapBackgroundColor: DEFAULT_BITMAP_BACKGROUND_COLOR,
      // HTMLのot-threshold初期値(0.5)と一致させる。
      // Infinityのままにするとユーザーがモードを変えた際にUIと内部状態が乖離する。
      threshold: 0.5,
      startIter: 0,
      captureStep: 1,
    }
    super('orbit_trap_custom', 'Orbit Trap (Custom)', defaultTrapSpec, makeOrbitTrapColorFn(COLOR_PATTERN.COSINE_RGB))
    this._colorPatternId = COLOR_PATTERN.COSINE_RGB
  }

  /**
   * カラーパターンを変更する。
   * @param {string} patternId - COLOR_PATTERN の値
   */
  setColorPattern(patternId) {
    this._colorPatternId = patternId
    this._colorFn = makeOrbitTrapColorFn(patternId)
  }

  /**
   * trapSpec の一部または全部を更新する。
   * @param {Partial<TrapSpec>} updates
   */
  updateTrapSpec(updates) {
    Object.assign(this.trapSpec, updates)
  }

  /**
   * トラップ設定と色パターンを既定値へ戻す。
   * UI のリセットボタンから使う。
   */
  resetTrapSpec() {
    const defaultTrapSpec = {
      mode: TRAP_MODE.DISTANCE_CLOSEST,
      shape: TRAP_SHAPE.RING,
      tx: 0,
      ty: 0,
      size: 0.5,
      angle: 0,
      bitmapBackgroundColor: DEFAULT_BITMAP_BACKGROUND_COLOR,
      threshold: 0.5,
      startIter: 0,
      captureStep: 1,
    }
    Object.assign(this.trapSpec, defaultTrapSpec)
    // 色パターンも既定値へ戻す
    this.setColorPattern(COLOR_PATTERN.COSINE_RGB)
  }
}

/** UI操作用シングルトン */
export const CUSTOM_ORBIT_TRAP_PALETTE = new CustomOrbitTrapPalette()

export const PALETTES = [
  MANDELBROT,
  BLUE_GOLD,
  LAVA,
  FALL,
  OCEAN,
  SKY_WATER,
  POP,
  JEWELLERY,
  new GrayScalePalette('gray_scale', 'Gray Scale', 0, 255),
  new SingleColorPalette('black_white', 'Pure B/W', [255, 255, 255]),
  new StripePalette(),
  new GridPalette(),
  OT_CROSS_COSINE,
  OT_CROSS_HSL,
  OT_TIA,
  CUSTOM_ORBIT_TRAP_PALETTE,
]

// 単調三次補間
function monotoneCubicInterpolationFN(values) {
  const N = values.length
  const delta = []
  for (let k = 0; k < N; k++) {
    delta.push(values[(k + 1) % N] - values[k])
  }

  const m = []
  for (let k = 1; k <= N; k++) {
    const dk = delta[k % N]
    const dk1 = delta[(k + 1) % N]
    m[(k + 1) % N] = dk * dk1 <= 0 ? 0 : (dk + dk1) / 2
    // 単純平均版の式は参考用に残す
  }

  for (let k = 0; k < N; k++) {
    if (delta[k] !== 0) {
      const alpha = m[k] / delta[k]
      const beta = m[(k + 1) % N] / delta[k]
      if (alpha < 0) {
        m[k] = 0
      }
      if (beta < 0) {
        m[(k + 1) % N] = 0
      }

      const sqRadius = alpha * alpha + beta * beta
      if (sqRadius > 9) {
        const tau = 3 / Math.sqrt(sqRadius)
        m[k] = tau * alpha * delta[k]
        m[(k + 1) % N] = tau * beta * delta[k]
      }
    }
  }

  return (x) => {
    const t = x - Math.floor(x)
    let k = Math.floor(x)
    if (k < 0) k += N

    const yk0 = values[k % N]
    const yk1 = values[(k + 1) % N]

    return yk0 * h00(t) + m[k % N] * h10(t) + yk1 * h01(t) + m[(k + 1) % N] * h11(t)
  }
}

function h00(t) {
	return (1 + 2 * t) * (1 - t) ** 2
}

function h10(t) {
	return t * (1 - t) ** 2
}

function h01(t) {
	return t * t * (3 - 2 * t)
}

function h11(t) {
	return t * t * (t - 1)
}

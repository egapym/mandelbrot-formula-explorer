/**
 * @author Bert Baron
 * @modified
 */

import { BuddhabrotRunner } from './buddhabrot.mjs'
import { BUDDHA_PALETTES, buildBuddhaStops, getBuddhaPalette } from './buddhaPalettes.mjs'
import { compileIterationFunction, getParsedExpression } from './customFunctionParser.mjs'
import * as favorites from './favorites.js'
import { functionPresets } from './functionPresets.mjs'
import * as fxp from './fxp.mjs'
import * as mcgpu from './mandelbrotCustomWebGPU.mjs'
import * as mgpu from './mandelbrotWebGPU.mjs'
import { OrbitTrapWebGPU } from './orbitTrapBitmapWebGPU.mjs'
import * as palette from './palette.js'
import { BAILOUT_SMOOTH, mandelbrot_high_precision } from './sharedCalculations.mjs'
import { jsExprToWGSL_safe } from './wgslCompiler.mjs'
import { WorkerContext } from './workerContext.mjs'
import { createWorkerFrom } from './workerLoader.mjs'

// ============================================================================
// ユーティリティ: trapSpecのキャッシュキー生成
// bitmapData (Uint8ClampedArray) は JSON 化が重いため除外し、
// bitmapVersion カウンタで変更を検知する。
// ============================================================================

/**
 * TrapSpec から比較用キャッシュキーを生成する。
 * bitmapData は大きすぎるため除外し、bitmapVersion を使って変更を検知する。
 * @param {Object|null} trapSpec
 * @returns {string}
 */
function _trapSpecKey(trapSpec) {
  if (!trapSpec) return 'null'
  const { bitmapData, ...rest } = trapSpec
  return JSON.stringify(rest)
}

function _orbitTrapColorPatternId(paletteObj) {
  if (paletteObj?._colorPatternId) return paletteObj._colorPatternId
  if (paletteObj?.id === 'orbit_trap_tia' || paletteObj?.trapSpec?.mode === 'tia') {
    return palette.COLOR_PATTERN.TIA_TRIANGLE
  }
  return null
}

function _orbitTrapGpuRenderKey(paletteObj) {
  if (!paletteObj?.trapSpec) return 'null'
  const colorPatternId = _orbitTrapColorPatternId(paletteObj) ?? ''
  return `${_trapSpecKey(paletteObj.trapSpec)}|color:${colorPatternId}`
}

/**
 * Orbit Trap の Bitmap Image プレビューをアスペクト比を保って描画する。
 * @param {HTMLCanvasElement} preview
 * @param {HTMLCanvasElement} sourceCanvas
 * @param {string|null} backgroundColor
 */
function _drawOrbitTrapBitmapPreview(preview, sourceCanvas, backgroundColor = null) {
  if (!preview || !sourceCanvas || sourceCanvas.width <= 0 || sourceCanvas.height <= 0) return

  const pCtx = preview.getContext('2d')
  if (!pCtx) return

  const previewW = preview.width
  const previewH = preview.height
  const scale = Math.min(previewW / sourceCanvas.width, previewH / sourceCanvas.height)
  const drawW = Math.max(1, Math.round(sourceCanvas.width * scale))
  const drawH = Math.max(1, Math.round(sourceCanvas.height * scale))
  const dx = Math.round((previewW - drawW) / 2)
  const dy = Math.round((previewH - drawH) / 2)

  pCtx.clearRect(0, 0, previewW, previewH)
  if (backgroundColor) {
    pCtx.fillStyle = backgroundColor
    pCtx.fillRect(0, 0, previewW, previewH)
  }
  pCtx.imageSmoothingEnabled = false
  pCtx.drawImage(sourceCanvas, 0, 0, sourceCanvas.width, sourceCanvas.height, dx, dy, drawW, drawH)
}

function _drawOrbitTrapBitmapPreviewFromImageData(
  preview,
  bitmapData,
  bitmapWidth,
  bitmapHeight,
  backgroundColor = null
) {
  if (!preview || !bitmapData || bitmapWidth <= 0 || bitmapHeight <= 0) return

  const tmpCanvas = document.createElement('canvas')
  tmpCanvas.width = bitmapWidth
  tmpCanvas.height = bitmapHeight
  const tmpCtx = tmpCanvas.getContext('2d')
  if (!tmpCtx) return

  const imgData = tmpCtx.createImageData(bitmapWidth, bitmapHeight)
  imgData.data.set(bitmapData)
  tmpCtx.putImageData(imgData, 0, 0)
  preview.classList.remove('d-none')
  _drawOrbitTrapBitmapPreview(preview, tmpCanvas, backgroundColor)
}

/**
 * #RRGGBB 形式の色コードへ正規化する。
 * @param {string} value
 * @returns {string|null}
 */
function _normalizeOrbitTrapColorCode(value) {
  const raw = String(value ?? '').trim()
  const color = /^[0-9a-fA-F]{6}$/.test(raw) ? `#${raw}` : raw
  if (!/^#[0-9a-fA-F]{6}$/.test(color)) return null
  return color.toUpperCase()
}

/**
 * Bitmap Image の背景色を trapSpec から取り出す。
 * @param {Object|null} spec
 * @returns {string}
 */
function _getOrbitTrapBitmapBackgroundColor(spec) {
  return _normalizeOrbitTrapColorCode(spec?.bitmapBackgroundColor) ?? palette.DEFAULT_BITMAP_BACKGROUND_COLOR
}

/**
 * Bitmap Image 背景色の picker 入力を同期する。
 * @param {string} color
 */
function _syncOrbitTrapBitmapBackgroundInputs(color) {
  const normalized = _normalizeOrbitTrapColorCode(color) ?? palette.DEFAULT_BITMAP_BACKGROUND_COLOR
  const picker = document.getElementById('ot-bitmap-bg-color')
  if (picker) picker.value = normalized.toLowerCase()
}

function _loadImageFromSrc(src) {
  return new Promise((resolve, reject) => {
    const img = new Image()
    img.onload = () => resolve(img)
    img.onerror = () => reject(new Error('画像の読み込みに失敗しました'))
    img.src = src
  })
}

function _buildOrbitTrapBitmapRecordFromImage(img) {
  const w = Math.max(1, img.naturalWidth || img.width)
  const h = Math.max(1, img.naturalHeight || img.height)
  const tmpCanvas = document.createElement('canvas')
  tmpCanvas.width = w
  tmpCanvas.height = h
  const ctx = tmpCanvas.getContext('2d')
  if (!ctx) {
    throw new Error('ビットマップ画像の描画コンテキスト取得に失敗しました')
  }

  // Bitmap Image は読み込み時に左右反転して保持する。
  ctx.save()
  ctx.translate(w, 0)
  ctx.scale(-1, 1)
  ctx.drawImage(img, 0, 0, w, h)
  ctx.restore()

  const imageData = ctx.getImageData(0, 0, w, h)
  return {
    bitmapData: new Uint8ClampedArray(imageData.data),
    bitmapWidth: w,
    bitmapHeight: h,
    bitmapVersion: Date.now(),
  }
}

async function _loadOrbitTrapBitmapRecordFromBlob(blob) {
  const url = URL.createObjectURL(blob)
  try {
    const img = await _loadImageFromSrc(url)
    return _buildOrbitTrapBitmapRecordFromImage(img)
  } finally {
    URL.revokeObjectURL(url)
  }
}

function _applyOrbitTrapBitmapRecord(customPalette, bitmapRecord) {
  if (!customPalette || !bitmapRecord?.bitmapData || bitmapRecord.bitmapWidth <= 0 || bitmapRecord.bitmapHeight <= 0) {
    return false
  }

  const appliedBitmapData = new Uint8ClampedArray(bitmapRecord.bitmapData)
  customPalette.updateTrapSpec({
    bitmapData: appliedBitmapData,
    bitmapWidth: bitmapRecord.bitmapWidth,
    bitmapHeight: bitmapRecord.bitmapHeight,
    bitmapVersion: bitmapRecord.bitmapVersion ?? Date.now(),
  })

  const preview = document.getElementById('ot-bitmap-preview')
  if (preview) {
    preview.classList.remove('d-none')
    _drawOrbitTrapBitmapPreviewFromImageData(
      preview,
      appliedBitmapData,
      bitmapRecord.bitmapWidth,
      bitmapRecord.bitmapHeight,
      _getOrbitTrapBitmapBackgroundColor(customPalette.trapSpec)
    )
  }

  return true
}

function _maybeAutoApplyQueryOrbitTrapBitmap() {
  if (queryOrbitTrapBitmapState.autoApplied || !queryOrbitTrapBitmapState.bitmapRecord) return false

  const customPalette = palette.CUSTOM_ORBIT_TRAP_PALETTE
  if (!customPalette || paletteComponent?.palette?.id !== 'orbit_trap_custom' || customPalette.trapSpec?.shape !== 'bitmap') {
    return false
  }

  if (!_applyOrbitTrapBitmapRecord(customPalette, queryOrbitTrapBitmapState.bitmapRecord)) return false

  queryOrbitTrapBitmapState.autoApplied = true
  if (typeof queryOrbitTrapBitmapState.applyChanges === 'function') {
    queryOrbitTrapBitmapState.applyChanges()
  } else {
    paletteComponent.notifyListeners()
    updatePermalink()
  }
  return true
}

async function _ensureQueryOrbitTrapBitmapLoaded() {
  if (!queryOrbitTrapBitmapState.url) return null
  if (queryOrbitTrapBitmapState.bitmapRecord) return queryOrbitTrapBitmapState.bitmapRecord
  if (queryOrbitTrapBitmapState.loadPromise) return queryOrbitTrapBitmapState.loadPromise
  if (queryOrbitTrapBitmapState.loadAttempted) return null

  queryOrbitTrapBitmapState.loadAttempted = true
  queryOrbitTrapBitmapState.loadPromise = (async () => {
    try {
      const response = await fetch(queryOrbitTrapBitmapState.url)
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}`)
      }
      const blob = await response.blob()
      queryOrbitTrapBitmapState.bitmapRecord = await _loadOrbitTrapBitmapRecordFromBlob(blob)
      _maybeAutoApplyQueryOrbitTrapBitmap()
      return queryOrbitTrapBitmapState.bitmapRecord
    } catch (e) {
      console.warn(
        `Orbit Trap: クエリパラメータ ${ORBIT_TRAP_BITMAP_URL_QUERY_KEY} の画像読み込みに失敗しました:`,
        e?.message ? e.message : e,
      )
      return null
    } finally {
      queryOrbitTrapBitmapState.loadPromise = null
    }
  })()

  return queryOrbitTrapBitmapState.loadPromise
}

// ============================================================================

/**
 * Orbit trap 設定を UI コントロールに反映するヘルパー。
 * パラメータはパーマリンクから復元したオブジェクトを想定し、
 * bitmapData は扱わない。colorPattern が渡されれば対応入力も切り替える。
 *
 * @param {Object} spec  - 保存された TrapSpec
 * @param {string} [colorPatternId] - COLOR_PATTERN 値
 */
function _applyOrbitTrapSpecToUI(spec, colorPatternId) {
  if (!spec) return
  const el = (id) => document.getElementById(id)
  const set = (id, v) => {
    const e = el(id)
    if (e) e.value = String(v)
  }

  if (el('ot-shape')) set('ot-shape', spec.shape || 'ring')
  if (el('ot-mode')) set('ot-mode', spec.mode || 'distance_closest')
  if (el('ot-size')) set('ot-size', spec.size != null ? spec.size : 0.5)
  if (el('ot-angle')) {
    let deg = spec.angle != null ? (spec.angle * 180) / Math.PI : 0
    // 1e-15 の誤差表示を避けるため丸める
    deg = Math.round(deg * 1e6) / 1e6
    set('ot-angle', deg)
  }
  if (el('ot-tx')) set('ot-tx', spec.tx != null ? spec.tx : 0)
  // 内部 ty はスクリーン Y 軸（下=正）なので UI 表示では符号を反転する（上=正の虚数）
  if (el('ot-ty')) set('ot-ty', spec.ty != null ? -spec.ty : 0)
  if (el('ot-threshold')) {
    const thr = spec.threshold == null || spec.threshold === Infinity ? 0 : spec.threshold
    set('ot-threshold', thr)
  }
  if (el('ot-start-iter')) set('ot-start-iter', spec.startIter != null ? spec.startIter : 0)
  if (el('ot-capture-step')) set('ot-capture-step', spec.captureStep != null ? spec.captureStep : 1)
  if (colorPatternId && el('ot-color-pattern')) set('ot-color-pattern', colorPatternId)
  _syncOrbitTrapBitmapBackgroundInputs(_getOrbitTrapBitmapBackgroundColor(spec))

  // 一部 UI は shape/mode 依存で隠されている可能性があるため再表示
  paletteComponent._updateOrbitTrapModeVisibility()
}

// ============================================================================
// 定数
// ============================================================================

const SQUARE_SIZE = 16 // 偶数である必要がある。全画面タスク時は -1
const DEFAULT_ITERATIONS = 1000
const DEFAULT_FRACTAL_TYPE = 'mandelbrot'
const DEFAULT_ITERATION_FUNCTION = 'z*z + c'
const DEFAULT_FRACTAL_PRESET_VALUE = 'preset:0'
const DEFAULT_WORKER_COUNT = navigator.hardwareConcurrency * 4 || 4

const MIN_PIXEL_SIZE = 1
const MAX_PIXEL_SIZE = 16
const MIN_ZOOM = fxp.fromNumber(0.1) // スクロールで 0.1（1/10）まで縮小できる
const NON_MANDELBROT_CPU_MAX_ZOOM = fxp.fromNumber(1.0e14)
const NON_MANDELBROT_GPU_MAX_ZOOM = fxp.fromNumber(1.0e5)
// The direct float32 GPU renderer is substantially faster at ordinary zooms.
// Beyond this point Mandelbrot needs the perturbation renderer to retain detail.
const MANDELBROT_DIRECT_GPU_MAX_ZOOM = NON_MANDELBROT_GPU_MAX_ZOOM
const ORBIT_TRAP_CPU_MAX_ZOOM = fxp.fromNumber(2.2e13)

// アニメーションと時間まわりの定数
const ANIMATION_CONSTANTS = {
  REDRAW_COOLDOWN: 500, // パーマリンク更新前の待機時間（ms）
  BUDDHA_REDRAW_DEBOUNCE: 16, // 待機時間（60fps で約 1 フレーム）
  DEFAULT_ANIMATION_SPEED: 0.15,
}

const PROGRESS_INDICATOR_CONSTANTS = {
  MIN_HIDDEN_INTERVAL_MS: 300,
  UPDATE_INTERVAL_MS: 100,
}

// 精度と計算まわりの定数
const PRECISION_CONSTANTS = {
  MIN_GPU_PRECISION: 64,
  MIN_CPU_PRECISION: 58,
  PRECISION_MARGIN: 5,
  DECIMAL_STRING_PRECISION: 30,
}

// リセットボタンで使う UI の既定値
const UI_DEFAULTS = {
  paletteDensity: 0,
  paletteRotate: 0,
  buddhaBrightness: 1.2,
  buddhaGamma: 4.8,
  buddhaRenderSpeed: 0,
}

const EMBEDDED_MODE = document.documentElement.classList.contains('embedded-mode')
const EMBEDDED_INITIAL_ZOOM_CORRECTION = {
  standardAspect: 4 / 3,
  standardFactor: 1,
  narrowAspect: 430 / 932,
  narrowFactor: 1.3,
  minFactor: 0.7,
  maxFactor: 2,
}
const INVERT_SCROLL_QUERY_KEY = 'invertScroll'
const ORBIT_TRAP_BITMAP_URL_QUERY_KEY = 'orbitTrapBitmapUrl'

function clampNumber(value, min, max) {
  return Math.max(min, Math.min(max, value))
}

function isOrbitTrapPalette(paletteObj) {
  return !!paletteObj?.trapSpec
}

function getZoomClampOptionsForPalette(paletteObj, options = {}) {
  const forceLimit = isOrbitTrapPalette(paletteObj)
  return {
    forceLimit,
    cpuMaxZoom: forceLimit ? ORBIT_TRAP_CPU_MAX_ZOOM : null,
    includeMinimum: !!options.includeMinimum || forceLimit,
  }
}

function getMainZoomLimitForState(fractalType, renderingWithGpu, scale = 60, options = {}) {
  if (!options.forceLimit && (!fractalType || fractalType === 'mandelbrot')) return null
  const limit = renderingWithGpu ? NON_MANDELBROT_GPU_MAX_ZOOM : options.cpuMaxZoom || NON_MANDELBROT_CPU_MAX_ZOOM
  return limit.withScale(scale)
}

function clampZoomForState(zoom, fractalType, renderingWithGpu, options = {}) {
  const scale = Number.isFinite(zoom?.scale) ? zoom.scale : 60
  const limit = getMainZoomLimitForState(fractalType, renderingWithGpu, scale, options)
  let normalizedZoom = zoom.withScale(scale)

  if (options.includeMinimum) {
    normalizedZoom = normalizedZoom.max(MIN_ZOOM.withScale(scale))
  }
  return limit && normalizedZoom.bigInt > limit.bigInt ? limit : normalizedZoom
}

function shouldUseDirectMandelbrotGpu(renderer) {
  if (renderer?.fractalType !== 'mandelbrot' || !renderer?.mandelbrotCustomGpu?.available) return false
  if (!renderer.zoom.withScale(MANDELBROT_DIRECT_GPU_MAX_ZOOM.scale).leq(MANDELBROT_DIRECT_GPU_MAX_ZOOM)) {
    return false
  }

  const zoom = renderer.zoom.toNumber()
  const width = renderer.width || renderer.canvas?.width || 0
  const height = renderer.height || renderer.canvas?.height || 0
  if (!Number.isFinite(zoom) || zoom <= 0 || width <= 0 || height <= 0) return false

  // The direct shader receives coordinates as float32. Use it only while one-pixel
  // steps remain distinguishable across the whole viewport; otherwise retain the
  // perturbation renderer so a faster frame never costs Mandelbrot detail.
  const pixelStep = 4 / (zoom * width)
  const halfWidth = 2 / zoom
  const halfHeight = (2 * height) / (zoom * width)
  const centerRe = renderer.center[0].toNumber()
  const centerIm = renderer.center[1].toNumber()
  const edges = [centerRe - halfWidth, centerRe + halfWidth, centerIm - halfHeight, centerIm + halfHeight]
  return edges.every((edge) => Math.fround(edge + pixelStep) !== Math.fround(edge))
}

function getBooleanQueryParam(key, defaultValue = false) {
  try {
    const raw = new URL(window.location.href).searchParams.get(key)
    if (raw == null) return defaultValue

    const normalized = raw.trim().toLowerCase()
    if (normalized === 'true') return true
    if (normalized === 'false') return false
  } catch (e) {
    console.warn(`クエリパラメータ ${key} の解析に失敗しました:`, e)
  }
  return defaultValue
}

const INVERT_SCROLL = getBooleanQueryParam(INVERT_SCROLL_QUERY_KEY)

function getEmbeddedViewportAspectRatio() {
  const width = window.innerWidth || document.documentElement?.clientWidth || 0
  const height = window.innerHeight || document.documentElement?.clientHeight || 0
  if (!(width > 0 && height > 0)) return EMBEDDED_INITIAL_ZOOM_CORRECTION.standardAspect
  return width / height
}

function getEmbeddedInitialZoomFactor() {
  if (!EMBEDDED_MODE) return 1

  const { standardAspect, standardFactor, narrowAspect, narrowFactor, minFactor, maxFactor } =
    EMBEDDED_INITIAL_ZOOM_CORRECTION
  const aspect = getEmbeddedViewportAspectRatio()
  const narrowness = (standardAspect - aspect) / (standardAspect - narrowAspect)
  const factor = standardFactor + narrowness * (narrowFactor - standardFactor)
  return clampNumber(factor, minFactor, maxFactor)
}

function applyEmbeddedInitialZoomCorrection(zoom) {
  if (!EMBEDDED_MODE) return zoom

  const scale = Number.isFinite(zoom?.scale) ? zoom.scale : 60
  const normalizedZoom = zoom.withScale(scale)
  return normalizedZoom.multiply(fxp.fromNumber(getEmbeddedInitialZoomFactor(), scale))
}

function getInitialFractalZoom(scale = 60) {
  return applyEmbeddedInitialZoomCorrection(fxp.fromNumber(1, scale))
}

function _getOrbitTrapBitmapUrlFromQuery() {
  try {
    const raw = new URL(window.location.href).searchParams.get(ORBIT_TRAP_BITMAP_URL_QUERY_KEY)?.trim()
    if (!raw) return ''
    return new URL(raw, window.location.href).href
  } catch (e) {
    console.warn(
      `Orbit Trap: クエリパラメータ ${ORBIT_TRAP_BITMAP_URL_QUERY_KEY} の解析に失敗しました:`,
      e?.message ? e.message : e,
    )
    return ''
  }
}

const queryOrbitTrapBitmapState = {
  url: _getOrbitTrapBitmapUrlFromQuery(),
  bitmapRecord: null,
  loadPromise: null,
  loadAttempted: false,
  autoApplied: false,
  applyChanges: null,
}

// ============================================================================
// DOM 要素キャッシュ
// ============================================================================

// DOM 要素をまとめて保持し、参照を安定させる
const DOM = {
  app: document.getElementById('app'),
  iterations: document.getElementById('max-iterations'),
  fullScreenButton: document.getElementById('fullscreen'),
  smoothToggle: document.getElementById('smooth'),
  resetButton: document.getElementById('reset'),
  gpuToggle: document.getElementById('gpu'),
  supersamplingToggle: document.getElementById('supersampling'),
  fractalTypeSelect: document.getElementById('fractalType'),
  iterationFunctionInput: document.getElementById('iterationFunction'),
  z0Real: document.getElementById('z0-real'),
  z0Imag: document.getElementById('z0-imag'),
  escapeRadius: document.getElementById('escape-radius'),
  buddha: {
    toggle: document.getElementById('buddha-toggle'),
    iterations: document.getElementById('buddha-iterations'),
    palette: document.getElementById('buddha-palette'),
    gpu: document.getElementById('buddha-gpu'),
    brightness: document.getElementById('buddha-brightness'),
    gamma: document.getElementById('buddha-gamma'),
    renderSpeed: document.getElementById('buddha-draw-speed'),
    render: document.getElementById('buddha-render'),
    stop: document.getElementById('buddha-stop'),
    scale: document.getElementById('buddha-scale'),
    download: document.getElementById('buddha-download'),
    mode: document.getElementById('buddhaMode'),
  },
  palette: {
    dropdown: document.getElementById('palette-dropdown'),
    density: document.getElementById('palette-density'),
    rotate: document.getElementById('palette-rotate'),
  },
  coords: {
    x: document.getElementById('coordX'),
    y: document.getElementById('coordY'),
    zoom: document.getElementById('coordZoom'),
    apply: document.getElementById('applyCoords'),
  },
}

// 反復関数に現在エラー表示が出ているかを保持する
let iterationFunctionHasError = false

// DOM 操作用ユーティリティ
const DOMHelpers = {
  /**
   * 反復関数入力のエラー表示を切り替える
   */
  setIterationFunctionError(hasError) {
    try {
      const label = document.querySelector("label[for='iterationFunction']")
      const input = DOM.iterationFunctionInput

      if (!label) return

      if (hasError) {
        label.classList.add('text-danger')
        input?.classList.add('is-invalid')
        iterationFunctionHasError = true
      } else {
        label.classList.remove('text-danger')
        input?.classList.remove('is-invalid')
        iterationFunctionHasError = false
      }
    } catch (e) {
      console.warn('Error toggling iteration function error state:', e)
    }
  },

  /**
   * 要素の値を安全に取得し、なければ既定値を返す
   */
  getElementValue(element, defaultValue = '') {
    return element?.value ?? defaultValue
  },

  /**
   * 要素の値を安全に設定する
   */
  setElementValue(element, value) {
    if (element) {
      element.value = String(value)
    }
  },

  /**
   * 要素の disabled 状態を切り替える
   */
  setDisabled(element, disabled) {
    if (element) {
      element.disabled = disabled
    }
  },
}

// 後方互換のために残している旧関数
function setIterationFunctionLabelError(on) {
  DOMHelpers.setIterationFunctionError(on)
}

/**
 * エラーメッセージ補助
 */
const ErrorHelpers = {
  /**
   * ログ出力用にエラーを整形する
   */
  format(error) {
    return error?.message ? error.message : String(error)
  },

  /**
   * 文脈付きで安全に console.warn する
   */
  warn(context, error) {
    console.warn(context, this.format(error))
  },

  /**
   * 文脈付きで安全に console.error する
   */
  error(context, error) {
    console.error(context, this.format(error))
  },
}

/**
 * コントロールを既定値へ戻す
 * @param {string} id - 要素 ID
 * @param {any} defaultValue - 設定する既定値
 */
function resetControlToDefault(id, defaultValue) {
  try {
    const el = document.getElementById(id)
    if (!el) return
    // 通常入力なら value を設定する
    el.value = String(defaultValue)
  } catch (e) {
    console.warn(`Error resetting control ${id}:`, e)
  }
}

/**
 * 既定値を適用し、必要な再描画を行う
 * @param {string} controlId - 対象コントロール ID
 * @param {any} defaultValue - 適用する既定値
 * @param {Object} options - 任意設定
 * @param {Function} options.onApply - 適用後のコールバック
 */
function applyDefaultAndRefresh(controlId, defaultValue, options = {}) {
  resetControlToDefault(controlId, defaultValue)

  try {
    // 必要なら追加処理を実行する
    options.onApply?.(defaultValue)

    // Buddhabrot の状態に応じて再描画方法を切り替える
    if (BuddhabrotState.active || BuddhabrotState.isViewEnabled()) {
      scheduleBuddhaRedraw()
    } else {
      redraw()
    }
  } catch (e) {
    console.warn(`Error applying default for ${controlId}:`, e)
  }
}

// ============================================================================
// ワーカーとフラクタル描画クラス
// ============================================================================

class MyWorker {
  constructor(taskqueue, resulthandler, errorHandler) {
    this.taskqueue = taskqueue
    this.resulthandler = resulthandler
    this.errorHandler = errorHandler
    // キャッシュ済み Blob URL からワーカーを作り、読み込みを使い回す。
    // 実体のワーカー準備が終わるまでは一時キューへ積んで待たせる。
    this.workerReady = false
    this._pendingMessages = []
    this.worker = null
    createWorkerFrom('worker.js', { type: 'module' })
      .then((w) => {
        this.worker = w
        this.worker.onmessage = (msg) => {
          // ワーカーのエラー通知は受けるが、結果処理自体は継続する
          if (msg.data.type === 'error') {
            if (this.errorHandler) {
              this.errorHandler(msg.data.message)
            }
          } else {
            this.onAnswer(msg.data)
          }
        }
        this.workerReady = true
        // 溜めていたメッセージを流す
        for (const m of this._pendingMessages) this.worker.postMessage(m)
        this._pendingMessages = []
      })
      .catch((e) => {
        console.error('Failed to create worker:', e)
        if (this.errorHandler) this.errorHandler(e?.message ? e.message : String(e))
      })
    this.busy = false
  }

  pickTask() {
    if (!this.busy && this.taskqueue.length > 0) {
      this.busy = true
      const msg = this.taskqueue.pop()
      if (this.workerReady && this.worker) {
        this.worker.postMessage(msg)
      } else {
        // ワーカー準備完了まではキューへ積む
        this._pendingMessages.push(msg)
      }
    }
  }

  onAnswer(answer) {
    this.busy = false
    this.resulthandler(answer)
    this.pickTask()
  }
}

class Mandelbrot {
  constructor(canvas, progress, paletteComponent) {
    this.canvas = canvas
    this.progress = progress
    this.taskqueue = []
    this.workers = []
    this.shownErrors = new Set() // 表示済みのエラーメッセージを記録する
    const workerCount = DEFAULT_WORKER_COUNT
    for (let i = 0; i < workerCount; i++) {
      const worker = new MyWorker(
        this.taskqueue,
        (result) => {
          this.onResult(result)
        },
        (errorMessage) => {
          this.handleWorkerError(errorMessage)
        },
      )
      this.workers.push(worker)
    }

    this.mandelbrotGpu = new mgpu.MandelbrotWebGPU(this, new WorkerContext(), (error) => this.gpuErrorCallback(error))

    this.mandelbrotCustomGpu = new mcgpu.MandelbrotCustomWebGPU(this, new WorkerContext(), (error) =>
      this.gpuErrorCallback(error),
    )
    this.orbitTrapGpu = new OrbitTrapWebGPU((error) => this.gpuErrorCallback(error))

    this.zoom = getInitialFractalZoom()
    this.center = [fxp.fromNumber(-0.5), fxp.fromNumber(0)]
    this.max_iter = DEFAULT_ITERATIONS
    this.smooth = true
    this.useGpu = true
    this.supersampling = 0 // 0=OFF, 2=2x2, 3=3x3, 4=4x4
    this.fractalType = 'mandelbrot' // 既定のフラクタル種別
    this.iterationFunction = 'z*z + c' // 既定の反復式
    this.escapeRadius = 4.0 // 脱出判定に使う半径（0-4）

    this.palette = []
    this.paletteComponent = paletteComponent
    this.initPallete(false)
    this.paletteComponent.addListener(() => {
      // Buddhabrot 表示中は通常フラクタルのパレット更新や再描画を行わない
      if (typeof buddhaActive !== 'undefined' && buddhaActive) return

      this.initPallete(true)
    })

    // 現在の描画ジョブ管理
    this.jobToken = null // 将来的には jobLevelToken のような名前が望ましい
    this.tasksLeft = 0
    this.jobId = 0
    this.jobLevel = 0
    this.viewRevision = 0
    this.activeGpuViewRevision = 0
    this.activeGpuScreen = null
    this.interactiveGpuRedraw = false
    this.currentInteractionGpuRedraw = false
    this._lastOrbitTrapGpuRenderKey = null
    this._skipOrbitTrapGpuOnce = false
  }

  restartWorkers() {
    // 既存ワーカーを終了する
    for (const worker of this.workers) {
      try {
        if (worker?.worker && typeof worker.worker.terminate === 'function') {
          worker.worker.terminate()
        }
      } catch (e) {
        console.warn('Error terminating worker during restart:', e)
      }
    }
    // 新しいワーカーを作り直す
    this.workers = []
    const workerCount = DEFAULT_WORKER_COUNT
    for (let i = 0; i < workerCount; i++) {
      const worker = new MyWorker(
        this.taskqueue,
        (result) => {
          this.onResult(result)
        },
        (errorMessage) => {
          this.handleWorkerError(errorMessage)
        },
      )
      this.workers.push(worker)
    }
  }

  handleWorkerError(errorMessage) {
    // 同じエラーは 1 回だけ表示する
    if (!this.shownErrors.has(errorMessage)) {
      this.shownErrors.add(errorMessage)
      this.logError('Custom Function Error', errorMessage)
      console.info('デフォルトのマンデルブロ集合 (z*z + c) を使用します。')
      DOMHelpers.setIterationFunctionError(true)
    }
  }

  logError(context, message) {
    console.error(`[${context}]`, message)
  }

  logWarning(context, message) {
    console.warn(`[${context}]`, message)
  }

  gpuErrorCallback(message) {
    console.log(`GPU error: ${message}`)
    // GPU トグルは強制的に切らず、利用不可であることだけツールチップで伝える
    DOM.gpuToggle.parentElement.setAttribute('title', 'WebGPU not supported')
    new bootstrap.Tooltip(DOM.gpuToggle.parentElement)

    // 必要なら CPU フォールバック表示へ切り替わるよう再描画する
    redraw()

    // Buddhabrot 描画中なら停止して消し、パンやズーム操作を戻す
    try {
      stopAndClearBuddha()
    } catch (e) {
      ErrorHelpers.warn('Error stopping Buddhabrot on GPU error:', e)
    }
  }

  resetStats() {
    this.stats = {
      time: 0,
      timeHighPrecision: 0,
      highPrecisionCalculations: 0,
      lowPrecisionMisses: 0,
    }
  }

  setCenter(center) {
    this.center = center
    this.viewRevision++
    this._updatePrecision()
  }

  setZoom(zoom) {
    const renderingWithGpu = this._willUseGpuForCurrentRender?.() ?? this.useGpu
    this.zoom = clampZoomForState(
      zoom,
      this.fractalType,
      renderingWithGpu,
      getZoomClampOptionsForPalette(this.paletteComponent?.palette),
    )
    this.viewRevision++
    this._updatePrecision()
  }

  _willUseGpuForCurrentRender() {
    return (
      this.useGpu &&
      (this._canUseOrbitTrapGpu?.() ||
        (!this.paletteComponent?.palette?.requiresCpu &&
          ((this.fractalType === 'mandelbrot' && this.mandelbrotGpu?.available) ||
            (this.fractalType === 'custom' && this.mandelbrotCustomGpu?.available))))
    )
  }

  applyZoomLimitForRenderMode(renderingWithGpu) {
    const clampedZoom = clampZoomForState(
      this.zoom,
      this.fractalType,
      renderingWithGpu,
      getZoomClampOptionsForPalette(this.paletteComponent?.palette),
    )
    const currentZoom = this.zoom.withScale(clampedZoom.scale)
    if (currentZoom.bigInt === clampedZoom.bigInt) return false

    this.zoom = clampedZoom
    this.viewRevision++
    this._updatePrecision()
    return true
  }

  applyZoomLimitForConfiguredMode() {
    return this.applyZoomLimitForRenderMode(this._willUseGpuForCurrentRender())
  }

  _updatePrecision() {
    this.requiredPrecision =
      this.zoom.multiply(fxp.fromNumber(this.width).withScale(this.zoom.scale)).bits() +
      PRECISION_CONSTANTS.PRECISION_MARGIN
    // 極端なズームでも 1px 分の移動量が 0 に丸められないよう、精度に少し余裕を持たせる
    if (this.useGpu) {
      this.precision = Math.max(PRECISION_CONSTANTS.MIN_GPU_PRECISION, Math.ceil(this.requiredPrecision / 8) * 8)
    } else {
      this.precision = Math.max(PRECISION_CONSTANTS.MIN_CPU_PRECISION, this.requiredPrecision)
    }
    // zoom と中心座標を新しい精度へ合わせる
    this.zoom = this.zoom.withScale(this.precision)
    this.center[0] = this.center[0].withScale(this.precision)
    this.center[1] = this.center[1].withScale(this.precision)
  }

  initOffscreens() {
    this.width = this.canvas.width
    this.height = this.canvas.height
    this.offscreens = []
    for (let scale = MAX_PIXEL_SIZE; scale >= MIN_PIXEL_SIZE; scale /= 2) {
      const offscreen = new Offscreen(this.canvas, scale, scale === MAX_PIXEL_SIZE, scale === MIN_PIXEL_SIZE)
      this.offscreens.push(offscreen)
    }
  }

  // キャンバスサイズ変更時に offscreen バッファと精度関連状態を更新する
  resized() {
    try {
      this.initOffscreens()
      this._updatePrecision()
    } catch (e) {
      ErrorHelpers.warn('Error in fractal.resized():', e)
    }
  }

  initPallete(redraw) {
    this.palette = palette.initPallet(
      this.paletteComponent.palette,
      this.paletteComponent.density,
      this.paletteComponent.rotate,
      this.paletteComponent.exp,
      this.max_iter,
    )
    renderPalette(this.palette, this.paletteComponent.palette)
    if (redraw) {
      // トラップ計算の仕様 (trapSpec) が変わった場合は再計算が必要。
      // OTパレットへの切り替え時、異なるOTパレット間の切り替え時、
      // あるいはOTパレットから通常パレットへ戻る場合にも再起動する。
      const paletteObj = this.paletteComponent?.palette
      const newTrapSpec = paletteObj?.trapSpec ?? null
      const newTrapKey = _trapSpecKey(newTrapSpec)
      const useOrbitTrapGpu = this._canUseOrbitTrapGpu()
      const newOrbitTrapGpuRenderKey = useOrbitTrapGpu ? _orbitTrapGpuRenderKey(paletteObj) : null
      if (
        newTrapKey !== this._lastTrapSpecKey ||
        (useOrbitTrapGpu && newOrbitTrapGpuRenderKey !== this._lastOrbitTrapGpuRenderKey)
      ) {
        // キーが異なればいずれの場合でもジョブをリセット
        this._lastTrapSpecKey = newTrapKey
        this._lastOrbitTrapGpuRenderKey = newOrbitTrapGpuRenderKey
        this.render(true)
        return
      }
      // Full GPU coloring does not keep CPU otData in sync, so avoid repainting
      // the current RGBA frame through the CPU palette path.
      if (useOrbitTrapGpu) return
      this._lastOrbitTrapGpuRenderKey = null
      const lastScreenNr = this.jobLevel < 1 ? this.offscreens.length : this.jobLevel - 1
      for (let screenNr = 0; screenNr <= lastScreenNr; screenNr++) {
        this.offscreens[screenNr]?.render(this.palette, this.max_iter, this.smooth, this.paletteComponent.palette)
      }
    }
  }

  _revokeJobToken() {
    if (this.jobToken) {
      URL.revokeObjectURL(this.jobToken)
      this.jobToken = null
    }
  }

  _createJobToken() {
    this.jobToken = URL.createObjectURL(new Blob())
  }

  _canUseOrbitTrapGpu(options = {}) {
    if (this._skipOrbitTrapGpuOnce) {
      if (options.consumeSkip) this._skipOrbitTrapGpuOnce = false
      return false
    }
    const paletteObj = this.paletteComponent?.palette
    const trapSpec = paletteObj?.trapSpec
    const colorPatternId = _orbitTrapColorPatternId(paletteObj)
    if (trapSpec?.mode === 'tia') return false
    const isSupportedOrbitTrapPalette = trapSpec && colorPatternId != null
    const hasSupportedTrap = trapSpec?.shape
    const supportedFractal = this.fractalType === 'mandelbrot' || this.fractalType === 'custom'
    // The dedicated Orbit Trap path uses direct float32 iteration. Keep very deep zooms
    // on the existing CPU fallback where precision handling is more conservative.
    const precisionOk = !this.requiredPrecision || this.requiredPrecision <= PRECISION_CONSTANTS.MIN_CPU_PRECISION
    return (
      this.useGpu &&
      isSupportedOrbitTrapPalette &&
      hasSupportedTrap &&
      supportedFractal &&
      precisionOk &&
      this.orbitTrapGpu?.available
    )
  }

  startNextJob(resetCaches) {
    // どの描画経路でも使えるよう、先に offscreen バッファを確保する
    if (!this.offscreens || this.offscreens.length === 0) {
      try {
        this.initOffscreens()
        this._updatePrecision()
      } catch (e) {
        console.warn('startNextJob: failed to initialize offscreens', e)
      }
    }

    if (this._canUseOrbitTrapGpu({ consumeSkip: true })) {
      this.startNextOrbitTrapGpuJob(resetCaches)
      return
    }

    // Mandelbrot の GPU 経路。
    // 通常ズームでは1回の2D dispatchで済む直接計算を使う。深いズームだけ、
    // 高精度を維持できる摂動法へ切り替える。
    // Orbit trap パレット (requiresCpu=true) はCPU計算が必要なため GPU パスをスキップする。
    if (
      this.useGpu &&
      !this.paletteComponent?.palette?.requiresCpu &&
      this.fractalType === 'mandelbrot' &&
      this.offscreens &&
      this.offscreens.length > 0
    ) {
      if (shouldUseDirectMandelbrotGpu(this)) {
        this.startNextCustomGpuJob(resetCaches)
        return
      }
      if (this.mandelbrotGpu?.available) {
        this.startNextGpuJob(resetCaches)
        return
      }
    }
    // カスタムフラクタルの GPU 経路
    // Orbit trap パレット (requiresCpu=true) はCPU計算が必要なため GPU パスをスキップする。
    if (
      this.useGpu &&
      !this.paletteComponent?.palette?.requiresCpu &&
      this.fractalType === 'custom' &&
      this.mandelbrotCustomGpu?.available &&
      this.offscreens &&
      this.offscreens.length > 0
    ) {
      this.startNextCustomGpuJob(resetCaches)
      return
    }
    this.jobLevel++
    if (
      !this.permalinkUpdated &&
      (this.jobLevel === this.offscreens.length ||
        performance.now() > this.jobStartTime + ANIMATION_CONSTANTS.REDRAW_COOLDOWN)
    ) {
      this.permalinkUpdated = true
      updatePermalink()
      // console.log(`Required precision: ${this.requiredPrecision} bits (zoom=${this.zoom.toNumber().toExponential(2)})`)
    }
    if (this.jobLevel === 0) {
      let totalTasks = 0
      for (const screen of this.offscreens) {
        const w = screen.buffer.width
        const h = screen.buffer.height
        const rowsPerTask = SQUARE_SIZE === -1 ? h : SQUARE_SIZE
        const colsPerTask = SQUARE_SIZE === -1 ? w : SQUARE_SIZE
        totalTasks += Math.ceil(h / rowsPerTask) * Math.ceil(w / colsPerTask)
      }
      this.progress.start(totalTasks)
    }

    this._revokeJobToken()
    let taskNumber = 0
    if (this.jobLevel < this.offscreens.length) {
      this._createJobToken()
      const screen = this.offscreens[this.jobLevel]
      const buffer = screen.buffer
      const w = buffer.width
      const h = buffer.height
      // z0 入力を先に読み、全 CPU タスクで使うハッシュへ反映する
      const z0Real = document.getElementById('z0-real') ? parseFloat(document.getElementById('z0-real').value) || 0 : 0
      const z0Imag = document.getElementById('z0-imag') ? parseFloat(document.getElementById('z0-imag').value) || 0 : 0
      const paramHash = `${this.max_iter}-${this.smooth}-${this.supersampling}-${this.fractalType}-${this.iterationFunction}-${this.escapeRadius}-${z0Real}-${z0Imag}`

      const frameTopLeft = this.canvas2complex(0, 0)
      // 幅や高さがピクセルサイズで割り切れない場合に備えて切り上げる
      const roundup = (value) => Math.ceil(value / screen.scale) * screen.scale
      const frameBottomRight = this.canvas2complex(roundup(this.width), roundup(this.height))

      // 低精度側は行単位でも速いが、まずは重い計算に効く正方タイルを優先する
      const rowsPerTask = SQUARE_SIZE === -1 ? h : SQUARE_SIZE
      const colsPerTask = SQUARE_SIZE === -1 ? w : SQUARE_SIZE
      // 先にタスクを集め、体感応答が良くなる順へ並べ替えられるようにする
      const tasksArr = []
      for (let i = 0; i < Math.ceil(h / rowsPerTask); i++) {
        const firstRow = i * rowsPerTask
        const lastRow = Math.min((i + 1) * rowsPerTask, h)
        for (let j = 0; j < Math.ceil(w / colsPerTask); j++) {
          const firstCol = j * colsPerTask
          const lastCol = Math.min((j + 1) * colsPerTask, w)
          const task = {
            type: 'task',
            jobId: this.jobId,
            jobToken: this.jobToken,
            pixelSize: screen.scale,
            taskNumber: taskNumber++,
            xOffset: firstCol,
            yOffset: firstRow,
            w: lastCol - firstCol,
            h: lastRow - firstRow,
            frameWidth: w,
            frameHeight: h,
            frameTopLeft: frameTopLeft,
            frameBottomRight: frameBottomRight,
            viewRevision: this.viewRevision,
            paramHash: paramHash,
            resetCaches: resetCaches,
            // skipTopLeft は前段結果を再利用する最適化。
            // ただし最終段で supersampling が有効な場合や特殊パレットでは無効にする。
            skipTopLeft:
              this.jobLevel > 0 &&
              !(screen.scale === 1 && this.supersampling > 0) &&
              this.paletteComponent.palette.supportsSmooth !== false,
            smooth: this.smooth,
            supersampling: screen.scale === 1 ? this.supersampling : 0, // Only on final pass
            maxIter: this.max_iter,
            precision: this.precision,
            requiredPrecision: this.requiredPrecision,
            fractalType: this.fractalType,
            iterationFunction: this.iterationFunction,
            // UI の初期 z0 があれば使う
            z0Real: z0Real,
            z0Imag: z0Imag,
            escapeRadius: this.escapeRadius,
            // アクティブなパレットが OrbitTrapPalette の場合のみ trapSpec を付加する
            // worker 側はこの値を見てOrbit trap計算の有無を判断する
            trapSpec: this.paletteComponent?.palette?.trapSpec ?? null,
          }
          // 並べ替え用にタイル中心も持たせる
          task._center = [task.xOffset + Math.floor(task.w / 2), task.yOffset + Math.floor(task.h / 2)]
          tasksArr.push(task)
        }
      }

      // アニメーション中はキャンバス中央付近のタイルを優先する
      if (animationState?.running) {
        const cx = Math.floor(w / 2)
        const cy = Math.floor(h / 2)
        tasksArr.sort((a, b) => {
          const da = (a._center[0] - cx) * (a._center[0] - cx) + (a._center[1] - cy) * (a._center[1] - cy)
          const db = (b._center[0] - cx) * (b._center[0] - cx) + (b._center[1] - cy) * (b._center[1] - cy)
          return da - db
        })
      }

      // 並べ替え済みなら近い順でキューへ積む
      for (const t of tasksArr) this.taskqueue.push(t)
      this.tasksLeft = this.taskqueue.length
      for (const worker of this.workers) {
        worker.pickTask()
      }
    } else {
      // すべての段階が終わったら進捗表示を閉じる
      this.progress.finish()
    }
  }

  onResult(answer) {
    // GPU が有効で Mandelbrot 描画中なら GPU 側の結果処理へ回す
    // Orbit trap パレット (requiresCpu=true) は GPU を使わないので CPU パスへ。
    if (
      this.useGpu &&
      !this.paletteComponent?.palette?.requiresCpu &&
      this.fractalType === 'mandelbrot' &&
      this.mandelbrotGpu?.available
    ) {
      this.onGpuResult(answer)
      return
    }
    // カスタムフラクタルでも GPU 有効時は GPU 結果処理へ回す
    if (
      this.useGpu &&
      !this.paletteComponent?.palette?.requiresCpu &&
      this.fractalType === 'custom' &&
      this.mandelbrotCustomGpu?.available
    ) {
      this.onGpuResult(answer)
      return
    }
    // console.log(`Received answer from worker`)
    const task = answer.task
    if (task.jobToken !== this.jobToken) {
      return // 古い描画ジョブの結果は無視する
    }
    if (task.viewRevision !== this.viewRevision) {
      return // 表示位置が変わった後に完了した古い結果は無視する
    }
    this.progress.update()
    if (answer.stats) {
      this.stats.time += answer.stats.time
      this.stats.timeHighPrecision += answer.stats.timeHighPrecision
      this.stats.highPrecisionCalculations += answer.stats.highPrecisionCalculations
      this.stats.lowPrecisionMisses += answer.stats.lowPrecisionMisses
    }

    // values がすべて 0 なら中断の可能性が高いので、回数制限付きで再投入する
    const MAX_RETRIES = 2
    try {
      if (answer.values) {
        let allZero = true
        for (let i = 0; i < answer.values.length; i++) {
          if (answer.values[i] !== 0) {
            allZero = false
            break
          }
        }
        if (allZero) {
          const taskObj = answer.task
          taskObj._retries = taskObj._retries ? taskObj._retries + 1 : 1
          if (taskObj._retries <= MAX_RETRIES) {
            // タスクを再投入し、tasksLeft は減らさない
            this.taskqueue.push(taskObj)
            // 再投入したタスクを拾えるようワーカーを起こす
            for (const worker of this.workers) {
              worker.pickTask()
            }
            return
          } else {
            // すべて 0 の結果は安全な代替値で埋めて描画可能にする
            if (answer.values) {
              answer.values.fill(2)
            }
            if (answer.smooth) {
              answer.smooth.fill(255)
            }
          }
        }
      }
    } catch (e) {
      console.warn('onResult: error while checking for all-zero values', e)
    }

    // 結果バッファを画面用バッファへ反映する
    // TODO: 全幅タスク向けの高速経路はまだ最適化の余地あり
    const offscreen = this.offscreens[this.jobLevel]
    for (let row = 0; row < task.h; row++) {
      const offset = (task.yOffset + row) * offscreen.buffer.width + task.xOffset
      offscreen.values.set(answer.values.subarray(row * task.w, (row + 1) * task.w), offset)
      if (this.smooth) {
        offscreen.smooth.set(answer.smooth.subarray(row * task.w, (row + 1) * task.w), offset)
      }
      if (answer.signs) {
        offscreen.signs.set(answer.signs.subarray(row * task.w, (row + 1) * task.w), offset)
      }
      if (answer.zreal) {
        offscreen.zreal.set(answer.zreal.subarray(row * task.w, (row + 1) * task.w), offset)
      }
      if (answer.zimag) {
        offscreen.zimag.set(answer.zimag.subarray(row * task.w, (row + 1) * task.w), offset)
      }
      // Orbit trapデータを格納
      if (answer.otData) {
        offscreen.otData.set(answer.otData.subarray(row * task.w, (row + 1) * task.w), offset)
      }
    }

    this.tasksLeft--
    if (this.tasksLeft === 0) {
      offscreen.render(this.palette, this.max_iter, this.smooth, this.paletteComponent.palette)
      // この段階の CPU 描画が成功したら、必要に応じてエラー表示を消す
      // ただしワーカー側でエラーが出ている間は勝手に消さない
      if (!iterationFunctionHasError) setIterationFunctionLabelError(false)
      this.startNextJob()
    }
  }

  startNextOrbitTrapGpuJob(resetCaches) {
    this._revokeJobToken()
    this._createJobToken()

    const screen = this.offscreens[this.offscreens.length - 1]
    this.activeGpuScreen = screen
    const w = screen.buffer.width
    const h = screen.buffer.height

    this.progress.start(w * h)
    const frameTopLeft = this.canvas2complex(0, 0)
    const roundup = (value) => Math.ceil(value / screen.scale) * screen.scale
    const frameBottomRight = this.canvas2complex(roundup(this.width), roundup(this.height))
    const z0Real = document.getElementById('z0-real') ? parseFloat(document.getElementById('z0-real').value) || 0 : 0
    const z0Imag = document.getElementById('z0-imag') ? parseFloat(document.getElementById('z0-imag').value) || 0 : 0
    const trapSpec = this.paletteComponent?.palette?.trapSpec ?? null
    const colorPatternId = _orbitTrapColorPatternId(this.paletteComponent?.palette)
    const paramHash = `${this.max_iter}-${this.smooth}-${this.supersampling}-${this.fractalType}-${this.iterationFunction}-${this.escapeRadius}-${z0Real}-${z0Imag}-${_trapSpecKey(trapSpec)}-${colorPatternId ?? ''}`

    const task = {
      type: 'task',
      jobId: this.jobId,
      jobToken: this.jobToken,
      viewRevision: this.viewRevision,
      pixelSize: screen.scale,
      taskNumber: 0,
      xOffset: 0,
      yOffset: 0,
      w,
      h,
      frameWidth: w,
      frameHeight: h,
      frameTopLeft,
      frameBottomRight,
      paramHash,
      resetCaches,
      skipTopLeft: false,
      smooth: this.smooth,
      supersampling: this.supersampling,
      maxIter: this.max_iter,
      precision: this.precision,
      requiredPrecision: this.requiredPrecision,
      fractalType: this.fractalType,
      iterationFunction: this.iterationFunction,
      z0: [z0Real, z0Imag],
      z0Real,
      z0Imag,
      escapeRadius: this.escapeRadius,
      animationQuick: this.currentInteractionGpuRedraw,
      trapSpec,
      colorPatternId,
      onUpdate: (answer) => this.onGpuUpdate(answer),
    }
    this.activeGpuViewRevision = task.viewRevision
    this.orbitTrapGpu.process(task)
  }

  startNextGpuJob(resetCaches) {
    this._revokeJobToken()
    this._createJobToken()

    const screen = getMainGpuRenderScreenForState(this)
    this.activeGpuScreen = screen
    const w = screen.buffer.width
    const h = screen.buffer.height

    this.progress.start(w * h)
    const paramHash = `${this.max_iter}-${this.smooth}-${this.supersampling}-${this.fractalType}-${this.iterationFunction}-${this.escapeRadius}`
    const frameTopLeft = this.canvas2complex(0, 0)
    // 幅や高さがピクセルサイズで割り切れない場合に備えて切り上げる
    const roundup = (value) => Math.ceil(value / screen.scale) * screen.scale
    const frameBottomRight = this.canvas2complex(roundup(this.width), roundup(this.height))

    const task = {
      type: 'task',
      jobId: this.jobId,
      jobToken: this.jobToken,
      viewRevision: this.viewRevision,
      pixelSize: screen.scale,
      taskNumber: 0,
      xOffset: 0,
      yOffset: 0,
      w: w,
      h: h,
      frameWidth: w,
      frameHeight: h,
      frameTopLeft: frameTopLeft,
      frameBottomRight: frameBottomRight,
      paramHash: paramHash,
      resetCaches: resetCaches,
      skipTopLeft: false,
      smooth: this.smooth,
      supersampling: this.supersampling,
      maxIter: this.max_iter,
      precision: this.precision,
      requiredPrecision: this.requiredPrecision,
      fractalType: this.fractalType,
      iterationFunction: this.iterationFunction,
      escapeRadius: this.escapeRadius,
      animationQuick: this.currentInteractionGpuRedraw,
    }
    this.activeGpuViewRevision = task.viewRevision
    this.mandelbrotGpu.process(task)
  }

  startNextCustomGpuJob(resetCaches) {
    this._revokeJobToken()
    this._createJobToken()

    // Standard Mandelbrot also uses this direct renderer at ordinary zooms.
    // Preserve its lightweight offscreen during fullscreen/Hi DPI interaction.
    const screen =
      this.fractalType === 'mandelbrot'
        ? getMainGpuRenderScreenForState(this)
        : this.offscreens[this.offscreens.length - 1]
    this.activeGpuScreen = screen
    const w = screen.buffer.width
    const h = screen.buffer.height

    this.progress.start(w * h)
    // 先行計算していた paramHash は廃止済み
    const frameTopLeft = this.canvas2complex(0, 0)
    const roundup = (value) => Math.ceil(value / screen.scale) * screen.scale
    const frameBottomRight = this.canvas2complex(roundup(this.width), roundup(this.height))

    // z0 を UI から取得する
    const z0Real = document.getElementById('z0-real') ? parseFloat(document.getElementById('z0-real').value) || 0 : 0
    const z0Imag = document.getElementById('z0-imag') ? parseFloat(document.getElementById('z0-imag').value) || 0 : 0
    const paramHash = `${this.max_iter}-${this.smooth}-${this.supersampling}-${this.fractalType}-${this.iterationFunction}-${this.escapeRadius}-${z0Real}-${z0Imag}`

    const task = {
      type: 'task',
      jobId: this.jobId,
      jobToken: this.jobToken,
      viewRevision: this.viewRevision,
      pixelSize: screen.scale,
      taskNumber: 0,
      xOffset: 0,
      yOffset: 0,
      w: w,
      h: h,
      frameWidth: w,
      frameHeight: h,
      frameTopLeft: frameTopLeft,
      frameBottomRight: frameBottomRight,
      paramHash: paramHash,
      resetCaches: resetCaches,
      skipTopLeft: false,
      smooth: this.smooth,
      supersampling: this.supersampling,
      maxIter: this.max_iter,
      precision: this.precision,
      requiredPrecision: this.requiredPrecision,
      fractalType: this.fractalType,
      iterationFunction: this.iterationFunction,
      z0: [z0Real, z0Imag],
      escapeRadius: this.escapeRadius,
      animationQuick: this.currentInteractionGpuRedraw,
    }
    this.activeGpuViewRevision = task.viewRevision
    this.mandelbrotCustomGpu.process(task)
  }

  onGpuResult(_answer) {
    console.log(`Received worker answer`)
  }

  onGpuUpdate(answer) {
    // 現在のジョブトークンと一致しない GPU 更新は無視する
    if (answer.jobToken !== this.jobToken) {
      return
    }
    if (this.activeGpuViewRevision !== this.viewRevision) {
      if (answer.isFinished) {
        this.progress.finish({ showRenderTime: false })
      }
      return
    }

    // GPU 描画エラーを確認する
    if (answer.error) {
      console.error('GPU rendering error:', answer.error)
      // 入力不正に気づけるようラベルへエラー表示を付ける
      setIterationFunctionLabelError(true)
      this.progress.finish()
      if (answer.fallbackToCpu && this.orbitTrapGpu) {
        this._skipOrbitTrapGpuOnce = true
        this.render(true)
      }
      return
    }

    // GPU 更新成功時も、ワーカー由来のエラー表示が残っているなら消さない
    if (!iterationFunctionHasError) setIterationFunctionLabelError(false)

    const screen = this.activeGpuScreen || this.offscreens[this.offscreens.length - 1]
    if (answer.rgba) {
      if (answer.isFinished) {
        this.progress.finish({ showRenderTime: !this.currentInteractionGpuRedraw })
      } else {
        const progress = Math.round((this.progress.tasks - this.progress.done) / 2)
        this.progress.update(progress)
      }
      screen.renderRgba(answer.rgba)
      if (
        !this.permalinkUpdated &&
        (answer.isFinished || performance.now() > this.jobStartTime + ANIMATION_CONSTANTS.REDRAW_COOLDOWN)
      ) {
        this.permalinkUpdated = true
        updatePermalink()
      }
      reapplyPendingInteractiveTransform()
      return
    }

    screen.values.set(answer.values)
    if (this.smooth) {
      screen.smooth.set(answer.smooth)
    }
    if (answer.signs) {
      screen.signs.set(answer.signs)
    }
    if (answer.zreal) {
      screen.zreal.set(answer.zreal)
    }
    if (answer.zimag) {
      screen.zimag.set(answer.zimag)
    }
    // Orbit trapデータを格納 (GPU結果)
    if (answer.otData) screen.otData.set(answer.otData)

    if (answer.isFinished) {
      // 完了時は進捗を確実に 100% にする
      this.progress.finish({ showRenderTime: !this.currentInteractionGpuRedraw })
    } else {
      // 中間更新では残り作業の半分を目安に進捗を進める
      const progress = Math.round((this.progress.tasks - this.progress.done) / 2)
      this.progress.update(progress)
    }

    screen.render(this.palette, this.max_iter, this.smooth, this.paletteComponent.palette)
    reapplyPendingInteractiveTransform()

    if (
      !this.permalinkUpdated &&
      (answer.isFinished || performance.now() > this.jobStartTime + ANIMATION_CONSTANTS.REDRAW_COOLDOWN)
    ) {
      this.permalinkUpdated = true
      updatePermalink()
    }
  }

  async render(resetCaches) {
    const zoomClamped = this.applyZoomLimitForRenderMode(this._willUseGpuForCurrentRender())
    if (zoomClamped && this === fractal) {
      updateCoordinateInputs()
    }
    this.taskqueue.length = 0
    this.jobId++
    this.jobLevel = -1
    this.jobStartTime = performance.now()
    this.permalinkUpdated = false
    this.currentInteractionGpuRedraw = !!this.interactiveGpuRedraw
    this.interactiveGpuRedraw = false
    this.resetStats()
    // console.log('Rendering...')
    this.startNextJob(resetCaches)
  }

  // キャンバス整数座標 x, y を固定小数点の複素数へ変換する
  canvas2complex(x, y) {
    // スケールが大きい場合に備え、x と y は整数へ丸めておく
    x = Math.round(x)
    y = Math.round(y)
    const w = fxp.fromNumber(this.width, this.precision)
    const h = fxp.fromNumber(this.height, this.precision)
    const scale = this.zoom.multiply(w).divide(fxp.fromNumber(4, this.precision))
    const center = this.center

    // scale が実質 0 なら、以降の 0 除算を避けるため中心へ寄せる
    try {
      // 0 以下の scale は不正として中心へ寄せる
      if (scale.bigIntValue() <= 0n) {
        return [center[0], center[1]]
      }
    } catch (_e) {
      // bigIntValue が取れない場合はそのまま後段に任せる
    }

    const r = fxp
      .fromNumber(x, this.precision)
      .subtract(w.divide(fxp.fromNumber(2, this.precision)))
      .divide(scale)
    const i = fxp
      .fromNumber(y, this.precision)
      .subtract(h.divide(fxp.fromNumber(2, this.precision)))
      .divide(scale)
    return [r.add(center[0]), i.add(center[1])]
  }
}

// ============================================================================
// Julia セット描画
// ============================================================================

/**
 * Julia セット用の軽量レンダラー。
 * 専用ワーカープールと offscreen バッファを使う。
 * 固定パラメータ c は setJuliaC() で設定する。
 * 表示は常に中心 (0,0)、zoom=1 の標準ビューとする。
 */
class JuliaRenderer {
  constructor(canvas, paletteComponent, progress = null) {
    this.canvas = canvas
    this.paletteComponent = paletteComponent
    this.progress = progress
    this.taskqueue = []
    this.workers = []
    this.jobToken = null
    this.jobId = 0
    this.jobLevel = 0
    this.tasksLeft = 0

    // Julia セットの固定パラメータ c
    this.juliaCRe = 0
    this.juliaCIm = 0

    // 描画設定はメイン側から毎回同期する
    this.max_iter = DEFAULT_ITERATIONS
    this.smooth = true
    this.supersampling = 0
    this.escapeRadius = 4.0
    this.useGpu = true
    this.precision = 64
    // フラクタル種別と反復関数もメイン側に合わせる
    this.fractalType = 'julia'
    this.iterationFunction = 'z*z + c'
    this._skipGpuOnce = false
    this._skipOrbitTrapGpuOnce = false
    this._lastOrbitTrapGpuRenderKey = null

    // 固定ビュー: center (0,0), zoom 1
    // 他の FxP 値と同じ scale を使い、乗除算の不整合を避ける
    this.zoom = fxp.fromNumber(1, 64)
    this.center = [fxp.fromNumber(0, 64), fxp.fromNumber(0, 64)]

    this.palette = []
    this.offscreens = null
    this.width = 0
    this.height = 0

    // Julia 側は極端な並列化が不要なので、メインより少ないワーカー数にする
    const workerCount = Math.max(2, navigator.hardwareConcurrency || 2)
    for (let i = 0; i < workerCount; i++) {
      this.workers.push(
        new MyWorker(
          this.taskqueue,
          (r) => this.onResult(r),
          () => {},
        ),
      )
    }
    this.juliaGpu = new mcgpu.MandelbrotCustomWebGPU(this, new WorkerContext(), (error) => this.gpuErrorCallback(error))
    this.orbitTrapGpu = new OrbitTrapWebGPU((error) => this.gpuErrorCallback(error))
    this.syncSettings()
  }

  /** 共有 PaletteComponent から最新設定を取り込む */
  syncSettings() {
    this.palette = palette.initPallet(
      this.paletteComponent.palette,
      this.paletteComponent.density,
      this.paletteComponent.rotate,
      this.paletteComponent.exp,
      this.max_iter,
    )
  }

  handleWorkerError(errorMessage) {
    console.error('[Julia Custom Function Error]', errorMessage)
    setIterationFunctionLabelError(true)
  }

  gpuErrorCallback(message) {
    console.log(`Julia GPU error: ${message}`)
  }

  /**
   * 再計算せずにパレットだけ更新し、既存 offscreen を再描画する。
   * Mandelbrot 側の initPallete(true) 相当。
   */
  applyPalette() {
    this.syncSettings()
    if (!this.offscreens) return
    // OTパレットへの切り替え時や異なるOTパレット間の切り替え時は再計算が必要。
    // 非OTパレットの場合は trapSpec=null なので再計算不要。
    const paletteObj = this.paletteComponent?.palette
    const newTrapSpec = paletteObj?.trapSpec ?? null
    const newTrapKey = _trapSpecKey(newTrapSpec)
    const useOrbitTrapGpu = this._canUseOrbitTrapGpu()
    const newOrbitTrapGpuRenderKey = useOrbitTrapGpu ? _orbitTrapGpuRenderKey(paletteObj) : null
    if (
      newTrapSpec !== null &&
      (newTrapKey !== this._lastTrapSpecKey ||
        (useOrbitTrapGpu && newOrbitTrapGpuRenderKey !== this._lastOrbitTrapGpuRenderKey))
    ) {
      this._lastTrapSpecKey = newTrapKey
      this._lastOrbitTrapGpuRenderKey = newOrbitTrapGpuRenderKey
      this.render()
      return
    }
    if (useOrbitTrapGpu) return
    this._lastOrbitTrapGpuRenderKey = null
    // GPU 描画では jobLevel が最終 offscreen を指したまま完了する。
    // CPU と同じ jobLevel - 1 にすると、表示中の最終フレームが再着色されない。
    const lastScreenNr = this._canUseGpu()
      ? this.jobLevel
      : this.jobLevel < 1
        ? this.offscreens.length - 1
        : this.jobLevel - 1
    for (let i = 0; i <= lastScreenNr; i++) {
      this.offscreens[i]?.render(this.palette, this.max_iter, this.smooth, this.paletteComponent.palette)
    }
  }

  /** キャンバス寸法変更後に呼ぶ */
  resized() {
    this.initOffscreens()
  }

  initOffscreens() {
    this.width = this.canvas.width
    this.height = this.canvas.height
    this.offscreens = []
    for (let scale = MAX_PIXEL_SIZE; scale >= MIN_PIXEL_SIZE; scale /= 2) {
      this.offscreens.push(new Offscreen(this.canvas, scale, scale === MAX_PIXEL_SIZE, scale === MIN_PIXEL_SIZE))
    }
  }

  /** 固定 Julia 定数 c を更新する */
  setJuliaC(re, im) {
    this.juliaCRe = re
    this.juliaCIm = im
  }

  setZoom(zoom) {
    const renderingWithGpu = this._willUseGpuForCurrentRender?.() ?? this.useGpu
    this.zoom = clampZoomForState(
      zoom,
      this.fractalType,
      renderingWithGpu,
      getZoomClampOptionsForPalette(this.paletteComponent?.palette, { includeMinimum: true }),
    )
  }

  _willUseGpuForCurrentRender() {
    return this.useGpu && (this._canUseOrbitTrapGpu?.() || this._canUseGpu?.())
  }

  applyZoomLimitForRenderMode(renderingWithGpu) {
    const clampedZoom = clampZoomForState(
      this.zoom,
      this.fractalType,
      renderingWithGpu,
      getZoomClampOptionsForPalette(this.paletteComponent?.palette, { includeMinimum: true }),
    )
    const currentZoom = this.zoom.withScale(clampedZoom.scale)
    if (currentZoom.bigInt === clampedZoom.bigInt) return false

    this.zoom = clampedZoom
    return true
  }

  applyZoomLimitForConfiguredMode() {
    return this.applyZoomLimitForRenderMode(this._willUseGpuForCurrentRender())
  }

  /** キャンバス座標を複素数 (FxP) へ変換する */
  canvas2complex(x, y) {
    x = Math.round(x)
    y = Math.round(y)
    const p = this.precision
    const w = fxp.fromNumber(this.width, p)
    const h = fxp.fromNumber(this.height, p)
    // 乗算前に zoom の scale を他の値へ合わせる
    const zoom = this.zoom.withScale(p)
    const scale = zoom.multiply(w).divide(fxp.fromNumber(4, p))
    try {
      if (scale.bigIntValue() <= 0n) return [this.center[0], this.center[1]]
    } catch (_e) {}
    const cx = this.center[0].withScale(p)
    const cy = this.center[1].withScale(p)
    const r = fxp
      .fromNumber(x, p)
      .subtract(w.divide(fxp.fromNumber(2, p)))
      .divide(scale)
    const i = fxp
      .fromNumber(y, p)
      .subtract(h.divide(fxp.fromNumber(2, p)))
      .divide(scale)
    return [r.add(cx), i.add(cy)]
  }

  _revokeJobToken() {
    if (this.jobToken) {
      URL.revokeObjectURL(this.jobToken)
      this.jobToken = null
    }
  }
  _createJobToken() {
    this.jobToken = URL.createObjectURL(new Blob())
  }

  _canUseGpu(options = {}) {
    if (this._skipGpuOnce) {
      if (options.consumeSkip) this._skipGpuOnce = false
      return false
    }
    return this.useGpu && !this.paletteComponent?.palette?.requiresCpu && this.juliaGpu?.available
  }

  _canUseOrbitTrapGpu(options = {}) {
    if (this._skipOrbitTrapGpuOnce) {
      if (options.consumeSkip) this._skipOrbitTrapGpuOnce = false
      return false
    }
    const paletteObj = this.paletteComponent?.palette
    const trapSpec = paletteObj?.trapSpec
    const colorPatternId = _orbitTrapColorPatternId(paletteObj)
    if (trapSpec?.mode === 'tia') return false
    return this.useGpu && !!trapSpec && colorPatternId != null && !!trapSpec?.shape && this.orbitTrapGpu?.available
  }

  startNextJob() {
    if (this._canUseOrbitTrapGpu({ consumeSkip: true })) {
      this.startNextOrbitTrapGpuJob()
      return
    }
    if (this._canUseGpu({ consumeSkip: true })) {
      this.startNextGpuJob()
      return
    }

    // CPU 描画が GPU と一致するよう、ユーザー指定の z0 を読む
    const z0Real = document.getElementById('z0-real') ? parseFloat(document.getElementById('z0-real').value) || 0 : 0
    const z0Imag = document.getElementById('z0-imag') ? parseFloat(document.getElementById('z0-imag').value) || 0 : 0

    this.jobLevel++
    if (this.jobLevel >= this.offscreens.length) {
      if (this.progress) this.progress.finish()
      return
    }
    if (this.jobLevel === 0 && this.progress) {
      // 全レベルぶんのタスク数を先に数える
      const _rowsPerTaskEst = SQUARE_SIZE === -1 ? 1 : SQUARE_SIZE
      const _colsPerTaskEst = SQUARE_SIZE === -1 ? 1 : SQUARE_SIZE
      let totalTasks = 0
      for (const s of this.offscreens) {
        const bw = s.buffer ? s.buffer.width : Math.ceil(this.canvas.width / s.scale)
        const bh = s.buffer ? s.buffer.height : Math.ceil(this.canvas.height / s.scale)
        const rpt = SQUARE_SIZE === -1 ? bh : SQUARE_SIZE
        const cpt = SQUARE_SIZE === -1 ? bw : SQUARE_SIZE
        totalTasks += Math.ceil(bh / rpt) * Math.ceil(bw / cpt)
      }
      this.progress.start(totalTasks)
    }
    this._revokeJobToken()
    this._createJobToken()

    const screen = this.offscreens[this.jobLevel]
    const buffer = screen.buffer
    const w = buffer.width
    const h = buffer.height
    const frameTopLeft = this.canvas2complex(0, 0)
    const roundup = (v) => Math.ceil(v / screen.scale) * screen.scale
    const frameBottomRight = this.canvas2complex(roundup(this.width), roundup(this.height))
    const rowsPerTask = SQUARE_SIZE === -1 ? h : SQUARE_SIZE
    const colsPerTask = SQUARE_SIZE === -1 ? w : SQUARE_SIZE

    let taskNumber = 0
    for (let i = 0; i < Math.ceil(h / rowsPerTask); i++) {
      const yOffset = i * rowsPerTask
      const taskH = Math.min(rowsPerTask, h - yOffset)
      for (let j = 0; j < Math.ceil(w / colsPerTask); j++) {
        const xOffset = j * colsPerTask
        const taskW = Math.min(colsPerTask, w - xOffset)
        this.taskqueue.push({
          type: 'task',
          jobId: this.jobId,
          jobToken: this.jobToken,
          pixelSize: screen.scale,
          taskNumber: taskNumber++,
          xOffset,
          yOffset,
          w: taskW,
          h: taskH,
          frameWidth: w,
          frameHeight: h,
          frameTopLeft,
          frameBottomRight,
          // CPU 側で使えるよう z0 をワーカーへ渡す
          z0Real: z0Real,
          z0Imag: z0Imag,
          paramHash: `julia-${this.max_iter}-${this.smooth}-${this.supersampling}-${this.fractalType}-${this.iterationFunction}-${this.juliaCRe}-${this.juliaCIm}-${this.escapeRadius}-${z0Real}-${z0Imag}`,
          resetCaches: false,
          skipTopLeft:
            this.jobLevel > 0 &&
            !(screen.scale === 1 && this.supersampling > 0) &&
            this.paletteComponent.palette.supportsSmooth !== false,
          smooth: this.smooth,
          supersampling: screen.scale === 1 ? this.supersampling : 0,
          maxIter: this.max_iter,
          precision: this.precision,
          requiredPrecision: 64,
          fractalType: this.fractalType,
          juliaRe: this.juliaCRe,
          juliaIm: this.juliaCIm,
          iterationFunction: this.iterationFunction,
          escapeRadius: this.escapeRadius,
          trapSpec: this.paletteComponent?.palette?.trapSpec ?? null,
        })
      }
    }
    this.tasksLeft = this.taskqueue.length
    for (const worker of this.workers) worker.pickTask()
  }

  startNextOrbitTrapGpuJob() {
    this._revokeJobToken()
    this._createJobToken()

    const screen = this.offscreens[this.offscreens.length - 1]
    const w = screen.buffer.width
    const h = screen.buffer.height
    this.jobLevel = this.offscreens.length - 1

    if (this.progress) this.progress.start(w * h)
    const frameTopLeft = this.canvas2complex(0, 0)
    const roundup = (value) => Math.ceil(value / screen.scale) * screen.scale
    const frameBottomRight = this.canvas2complex(roundup(this.width), roundup(this.height))
    const z0Real = document.getElementById('z0-real') ? parseFloat(document.getElementById('z0-real').value) || 0 : 0
    const z0Imag = document.getElementById('z0-imag') ? parseFloat(document.getElementById('z0-imag').value) || 0 : 0
    const trapSpec = this.paletteComponent?.palette?.trapSpec ?? null
    const colorPatternId = _orbitTrapColorPatternId(this.paletteComponent?.palette)
    const task = {
      type: 'task',
      jobId: this.jobId,
      jobToken: this.jobToken,
      pixelSize: screen.scale,
      taskNumber: 0,
      xOffset: 0,
      yOffset: 0,
      w,
      h,
      frameWidth: w,
      frameHeight: h,
      frameTopLeft,
      frameBottomRight,
      paramHash: `julia-orbit-gpu-${this.max_iter}-${this.smooth}-${this.supersampling}-${this.fractalType}-${this.iterationFunction}-${this.juliaCRe}-${this.juliaCIm}-${this.escapeRadius}-${z0Real}-${z0Imag}-${_trapSpecKey(trapSpec)}-${colorPatternId ?? ''}`,
      resetCaches: false,
      skipTopLeft: false,
      smooth: this.smooth,
      supersampling: this.supersampling,
      maxIter: this.max_iter,
      precision: this.precision,
      requiredPrecision: 64,
      fractalType: this.fractalType,
      iterationFunction: this.iterationFunction,
      z0: [z0Real, z0Imag],
      z0Real,
      z0Imag,
      juliaRe: this.juliaCRe,
      juliaIm: this.juliaCIm,
      escapeRadius: this.escapeRadius,
      trapSpec,
      colorPatternId,
      onUpdate: (answer) => this.onGpuUpdate(answer),
    }
    this.orbitTrapGpu.process(task)
  }

  startNextGpuJob() {
    this._revokeJobToken()
    this._createJobToken()

    const screen = this.offscreens[this.offscreens.length - 1]
    const w = screen.buffer.width
    const h = screen.buffer.height
    this.jobLevel = this.offscreens.length - 1

    if (this.progress) this.progress.start(w * h)

    const frameTopLeft = this.canvas2complex(0, 0)
    const roundup = (v) => Math.ceil(v / screen.scale) * screen.scale
    const frameBottomRight = this.canvas2complex(roundup(this.width), roundup(this.height))
    const z0Real = document.getElementById('z0-real') ? parseFloat(document.getElementById('z0-real').value) || 0 : 0
    const z0Imag = document.getElementById('z0-imag') ? parseFloat(document.getElementById('z0-imag').value) || 0 : 0
    const task = {
      type: 'task',
      jobId: this.jobId,
      jobToken: this.jobToken,
      pixelSize: screen.scale,
      taskNumber: 0,
      xOffset: 0,
      yOffset: 0,
      w,
      h,
      frameWidth: w,
      frameHeight: h,
      frameTopLeft,
      frameBottomRight,
      paramHash: `julia-gpu-${this.max_iter}-${this.smooth}-${this.supersampling}-${this.fractalType}-${this.iterationFunction}-${this.juliaCRe}-${this.juliaCIm}-${this.escapeRadius}-${z0Real}-${z0Imag}`,
      resetCaches: false,
      skipTopLeft: false,
      smooth: this.smooth,
      supersampling: this.supersampling,
      maxIter: this.max_iter,
      precision: this.precision,
      requiredPrecision: 64,
      fractalType: this.fractalType,
      juliaRe: this.juliaCRe,
      juliaIm: this.juliaCIm,
      iterationFunction: this.iterationFunction,
      z0: [z0Real, z0Imag],
      escapeRadius: this.escapeRadius,
    }
    this.juliaGpu.process(task)
  }

  onResult(answer) {
    const task = answer.task
    if (task.jobToken !== this.jobToken) return
    const offscreen = this.offscreens[this.jobLevel]
    if (!offscreen) return
    for (let row = 0; row < task.h; row++) {
      const offset = (task.yOffset + row) * offscreen.buffer.width + task.xOffset
      offscreen.values.set(answer.values.subarray(row * task.w, (row + 1) * task.w), offset)
      if (this.smooth && answer.smooth) {
        offscreen.smooth.set(answer.smooth.subarray(row * task.w, (row + 1) * task.w), offset)
      }
      if (answer.signs) {
        offscreen.signs.set(answer.signs.subarray(row * task.w, (row + 1) * task.w), offset)
      }
      if (answer.zreal) {
        offscreen.zreal.set(answer.zreal.subarray(row * task.w, (row + 1) * task.w), offset)
      }
      if (answer.zimag) {
        offscreen.zimag.set(answer.zimag.subarray(row * task.w, (row + 1) * task.w), offset)
      }
      // Orbit trapデータを格納 (Julia)
      if (answer.otData) {
        offscreen.otData.set(answer.otData.subarray(row * task.w, (row + 1) * task.w), offset)
      }
    }
    this.tasksLeft--
    if (this.progress) this.progress.update()
    if (this.tasksLeft === 0) {
      offscreen.render(this.palette, this.max_iter, this.smooth, this.paletteComponent.palette)
      this.startNextJob()
    }
  }

  onGpuUpdate(answer) {
    if (answer.jobToken !== this.jobToken) return

    if (answer.error) {
      console.error('Julia GPU rendering error:', answer.error)
      setIterationFunctionLabelError(true)
      if (this.progress) this.progress.finish()
      if (answer.fallbackToCpu) this._skipOrbitTrapGpuOnce = true
      else this._skipGpuOnce = true
      this.render()
      return
    }

    if (!iterationFunctionHasError) setIterationFunctionLabelError(false)

    const offscreen = this.offscreens[this.offscreens.length - 1]
    if (!offscreen) return

    if (answer.rgba) {
      if (this.progress) {
        if (answer.isFinished) this.progress.finish()
        else this.progress.update(Math.round((this.progress.tasks - this.progress.done) / 2))
      }
      offscreen.renderRgba(answer.rgba)
      return
    }

    offscreen.values.set(answer.values)
    if (this.smooth && answer.smooth) offscreen.smooth.set(answer.smooth)
    if (answer.signs) offscreen.signs.set(answer.signs)
    if (answer.zreal) offscreen.zreal.set(answer.zreal)
    if (answer.zimag) offscreen.zimag.set(answer.zimag)
    if (answer.otData) offscreen.otData.set(answer.otData)

    if (this.progress) {
      if (answer.isFinished) this.progress.finish()
      else this.progress.update(Math.round((this.progress.tasks - this.progress.done) / 2))
    }

    offscreen.render(this.palette, this.max_iter, this.smooth, this.paletteComponent.palette)
  }

  async render() {
    if (!this.canvas.width || !this.canvas.height) return
    if (!this.offscreens || this.offscreens.length === 0) this.initOffscreens()
    this.applyZoomLimitForRenderMode(this._willUseGpuForCurrentRender())
    // 実行中ジョブがあれば中断する
    this._revokeJobToken()
    this.taskqueue.length = 0
    this.jobId++
    this.jobLevel = -1
    this.startNextJob()
  }
}

class Offscreen {
  constructor(canvas, scale, first, last) {
    this.canvas = canvas
    this.scale = scale
    this.first = first
    this.last = last
    this.maincontext = canvas.getContext('2d')

    this.offscreen = document.createElement('canvas')
    this.offscreen.width = Math.ceil(this.canvas.width / scale)
    this.offscreen.height = Math.ceil(this.canvas.height / scale)
    this.offscreencontext = this.offscreen.getContext('2d')
    this.buffer = this.offscreencontext.createImageData(this.offscreen.width, this.offscreen.height)
    this.values = new Int32Array(this.buffer.width * this.buffer.height)
    this.smooth = new Uint8Array(this.buffer.width * this.buffer.height)
    this.signs = new Int8Array(this.buffer.width * this.buffer.height)
    this.zreal = new Float32Array(this.buffer.width * this.buffer.height)
    this.zimag = new Float32Array(this.buffer.width * this.buffer.height)
    // Orbit trapデータ用バッファ
    this.otData = new Float32Array(this.buffer.width * this.buffer.height)

    this.smoothscreen = document.createElement('canvas')
    this.smoothscreen.width = this.offscreen.width
    this.smoothscreen.height = this.offscreen.height
    this.smoothscreencontext = this.smoothscreen.getContext('2d')
    this.smoothbuffer = this.smoothscreencontext.createImageData(this.smoothscreen.width, this.smoothscreen.height)
  }

  renderRgba(rgba) {
    this.buffer.data.set(rgba)
    this.offscreencontext.putImageData(this.buffer, 0, 0)
    this.maincontext.imageSmoothingEnabled = false
    this.maincontext.drawImage(
      this.offscreen,
      0,
      0,
      this.offscreen.width * this.scale,
      this.offscreen.height * this.scale,
    )
  }

  render(palette, _max_iter, withSmooth, paletteObj = null) {
    const bufferData = this.buffer.data // Uint8ClampedArray
    const smoothData = this.smoothbuffer.data // Uint8ClampedArray
    const values = this.values // Int32Array
    const smooth = this.smooth // Uint8Array
    const signs = this.signs // Int8Array

    // ピクセル着色はパレットオブジェクト側へ委譲する
    if (paletteObj) {
      paletteObj.renderPixels(
        bufferData,
        smoothData,
        values,
        smooth,
        signs,
        palette,
        withSmooth,
        this.zreal,
        this.zimag,
        this.otData,
      )
    } else {
      // paletteObj がない場合は lookup buffer を直接使う
      for (let i = 0; i < values.length; i++) {
        const iter = values[i]
        bufferData[i * 4] = palette[iter * 4]
        bufferData[i * 4 + 1] = palette[iter * 4 + 1]
        bufferData[i * 4 + 2] = palette[iter * 4 + 2]
        bufferData[i * 4 + 3] = palette[iter * 4 + 3]
      }
    }

    this.offscreencontext.putImageData(this.buffer, 0, 0)
    this.maincontext.imageSmoothingEnabled = false
    this.maincontext.drawImage(
      this.offscreen,
      0,
      0,
      this.offscreen.width * this.scale,
      this.offscreen.height * this.scale,
    )
    if (withSmooth && paletteObj?.supportsSmooth) {
      this.smoothscreencontext.putImageData(this.smoothbuffer, 0, 0)
      this.maincontext.drawImage(
        this.smoothscreen,
        0,
        0,
        this.smoothscreen.width * this.scale,
        this.smoothscreen.height * this.scale,
      )
    }
  }
}

class ProgressMonitor {
  constructor(canvas, timeElementId = 'renderTimeValue') {
    this.canvas = canvas
    this.timeElementId = timeElementId
    this.ctx = canvas.getContext('2d')
    this.ctx.fillStyle = 'black'
    this.ctx.fillRect(0, 0, canvas.width, canvas.height)
    this.tasks = 0
    this.done = 0
    this.lastUpdate = 0
    this.startTime = 0
    this.completed = false
    this.visible = false
    this.lastHiddenAt = Number.NEGATIVE_INFINITY
    this.showTimer = null
    this.canvas.style.display = 'none'
  }

  start(tasks) {
    this.tasks = tasks
    this.done = 0
    this.completed = false
    this.lastUpdate = performance.now()
    this.startTime = this.lastUpdate
    this._cancelPendingShow()
    this._draw(0)
    this._showWithCooldown()
    // 描画中は render time を隠す
    if (this.timeElementId) {
      const el = document.getElementById(this.timeElementId)
      if (el) el.style.visibility = 'hidden'
    }
  }

  update(amount = 1) {
    this.done = Math.min(this.done + amount, this.tasks)
    const now = performance.now()
    if (now - this.lastUpdate > PROGRESS_INDICATOR_CONSTANTS.UPDATE_INTERVAL_MS) {
      const percent = (this.done / this.tasks) * 100
      // console.log(`Rendering ${percent.toFixed(0)}%`)
      this.lastUpdate = now
      this._draw(percent)
    }
    if (this.done === this.tasks && !this.completed) {
      this.completed = true
      this._draw(100)
    }
  }

  finish(options = {}) {
    const showRenderTime = options.showRenderTime !== false
    this.done = this.tasks
    this._cancelPendingShow()
    if (!this.completed) {
      this.completed = true
      this._draw(100)
    }
    // 完了後に描画時間を表示し、進捗表示を隠す
    const jobTime = performance.now() - this.startTime
    if (showRenderTime && this.timeElementId) {
      const renderTimeElement = document.getElementById(this.timeElementId)
      if (renderTimeElement) {
        renderTimeElement.innerText = `${jobTime.toFixed(0)}ms`
        renderTimeElement.style.visibility = 'visible'
      }
    }
    this._hide()
  }

  _draw(percentage) {
    // 透明背景の上に、白円と赤い進捗扇形を描く
    const ctx = this.ctx
    const width = this.canvas.width
    const height = this.canvas.height
    const radius = Math.min(width, height) / 2
    const centerX = width / 2
    const centerY = height / 2
    ctx.clearRect(0, 0, width, height)
    ctx.fillStyle = 'white'
    ctx.beginPath()
    ctx.arc(centerX, centerY, radius, 0, 2 * Math.PI)
    ctx.fill()
    ctx.fillStyle = 'red'
    ctx.beginPath()
    ctx.arc(centerX, centerY, radius, 0, (1 - percentage / 100) * 2 * Math.PI)
    ctx.lineTo(centerX, centerY)
    ctx.fill()
  }

  _showWithCooldown() {
    if (this.visible) {
      this._show()
      return
    }

    const elapsedSinceHide = performance.now() - this.lastHiddenAt
    if (elapsedSinceHide >= PROGRESS_INDICATOR_CONSTANTS.MIN_HIDDEN_INTERVAL_MS) {
      this._show()
      return
    }

    const wait = PROGRESS_INDICATOR_CONSTANTS.MIN_HIDDEN_INTERVAL_MS - elapsedSinceHide
    this.showTimer = window.setTimeout(() => {
      this.showTimer = null
      if (this.completed || this.done >= this.tasks) return
      this._show()
    }, wait)
  }

  _show() {
    const percent = this.tasks > 0 ? (this.done / this.tasks) * 100 : 0
    this._draw(percent)
    this.canvas.style.display = 'block'
    this.visible = true
  }

  _hide() {
    this.canvas.style.display = 'none'
    this.visible = false
    this.lastHiddenAt = performance.now()
  }

  _cancelPendingShow() {
    if (this.showTimer != null) {
      window.clearTimeout(this.showTimer)
      this.showTimer = null
    }
  }
}

function renderPalette(palette, paletteObj) {
  const ctx = paletteCanvasElement.getContext('2d')
  const width = paletteCanvasElement.offsetWidth
  const height = paletteCanvasElement.offsetHeight
  paletteCanvasElement.width = width
  paletteCanvasElement.height = height

  // プレビュー描画もパレットオブジェクト側へ委譲する
  if (paletteObj) {
    paletteObj.renderPreview(ctx, width, height, palette)
    return
  }

  // paletteObj がない場合は単純な lookup buffer 表示にする
  const offset = 4
  const paletteSize = palette.length / 4 - offset
  for (let i = 0; i < paletteSize; i++) {
    const colorIndex = i + offset
    const pos = Math.floor((i * width) / paletteSize)
    const w = Math.floor(((i + 1) * width) / paletteSize) - pos
    const r = palette[colorIndex * 4]
    const g = palette[colorIndex * 4 + 1]
    const b = palette[colorIndex * 4 + 2]
    ctx.fillStyle = `rgb(${r},${g},${b})`
    ctx.fillRect(pos, 0, w, height)
  }
}

function initMenu() {
  const menuToggle = document.getElementById('menu-toggle')
  menuToggle.addEventListener('click', (_e) => {
    const menu = document.getElementById('settings')
    menu.classList.toggle('hidden')
    menuToggle.classList.toggle('hidden')
  })
}

initMenu()

// functionPresets を fractalType セレクトへ追加する
try {
  if (DOM.fractalTypeSelect && Array.isArray(functionPresets) && functionPresets.length) {
    // まずは単純に option として追加する
    for (let i = 0; i < functionPresets.length; i++) {
      const opt = document.createElement('option')
      // 組み込み種別と衝突しない値にする
      opt.value = `preset:${i}`
      // 選択されたプリセットを特定できるよう index を保持する
      opt.dataset.presetIndex = String(i)
      // ラベルがあれば使い、なければ式を表示する
      const preset = functionPresets[i]
      const label = preset?.label ? preset.label : preset?.expr ? preset.expr : String(preset)
      opt.textContent = label
      DOM.fractalTypeSelect.appendChild(opt)
      // プリセット数が非常に多い場合は遅延追加も検討余地あり
    }
  }
} catch (e) {
  console.warn('Could not add functionPresets to fractalType select', e)
}

function hasFractalTypeOption(value) {
  if (!DOM.fractalTypeSelect || value == null) return false
  return Array.from(DOM.fractalTypeSelect.options).some((option) => option.value === value)
}

function getFractalPresetValueForType(fractalType) {
  const idx = functionPresets.findIndex((preset) => (preset?.fractalType || 'custom') === fractalType)
  return idx >= 0 ? `preset:${idx}` : null
}

function normalizeFractalTypeParams(params = {}) {
  const fallback = {
    fractalType: DEFAULT_FRACTAL_TYPE,
    iterationFunction: DEFAULT_ITERATION_FUNCTION,
    fractalDataPresetIndex: DEFAULT_FRACTAL_PRESET_VALUE,
  }
  const rawFractalType = typeof params.fractalType === 'string' ? params.fractalType : ''
  const rawPresetValue = typeof params.fractalDataPresetIndex === 'string' ? params.fractalDataPresetIndex : ''
  const presetMatch = rawPresetValue.match(/^preset:(-?\d+)$/)
  const presetIndex = presetMatch ? Number.parseInt(presetMatch[1], 10) : NaN
  const preset = presetIndex >= 0 ? functionPresets[presetIndex] : null
  const hasKnownFractalType = rawFractalType === DEFAULT_FRACTAL_TYPE || rawFractalType === 'custom'

  if (rawFractalType && !hasKnownFractalType) return fallback

  if (preset && hasFractalTypeOption(rawPresetValue)) {
    const fractalType = preset.fractalType || 'custom'
    if (rawFractalType && rawFractalType !== fractalType) return fallback
    return {
      fractalType,
      iterationFunction: preset.expr || params.iterationFunction || fallback.iterationFunction,
      fractalDataPresetIndex: rawPresetValue,
    }
  }

  if (rawFractalType === 'custom' && hasFractalTypeOption('preset:-1')) {
    return {
      fractalType: 'custom',
      iterationFunction: params.iterationFunction || fallback.iterationFunction,
      fractalDataPresetIndex: 'preset:-1',
    }
  }

  if (rawFractalType === DEFAULT_FRACTAL_TYPE) {
    const presetValue = getFractalPresetValueForType(DEFAULT_FRACTAL_TYPE) || fallback.fractalDataPresetIndex
    return {
      fractalType: DEFAULT_FRACTAL_TYPE,
      iterationFunction: params.iterationFunction || fallback.iterationFunction,
      fractalDataPresetIndex: presetValue,
    }
  }

  return fallback
}

const canvasElement = document.getElementById('mandelbrot-canvas')
const progressElement = document.getElementById('progress-canvas')
const paletteCanvasElement = document.getElementById('palette-canvas')
const fullResToggle = document.getElementById('fullres')

const tempCanvas = document.createElement('canvas')

class PaletteComponent {
  constructor() {
    this.listeners = []
    this.palette = palette.getPalette('mandelbrot')
    this.density = String(1)
    this.rotate = String(0)
    // this.exp = 0.9
  }

  init() {
    const paletteSelect = document.getElementById('palette-dropdown')

    palette.PALETTES.forEach((p) => {
      try {
        if (paletteSelect) {
          const opt = document.createElement('option')
          opt.value = p.id
          opt.textContent = p.name
          if (p.id === this.palette.id) opt.selected = true
          paletteSelect.appendChild(opt)
        }
      } catch (e) {
        // DOM 操作に失敗しても全体は続行する
        console.warn('Error populating palette list entry:', e?.message ? e.message : e)
      }
    })

    // セレクト変更時にパレット更新と再描画通知を行う
    if (paletteSelect) {
      paletteSelect.addEventListener('change', (e) => {
        try {
          const id = e.target.value
          const p = palette.getPalette(id)
          this.setPalette(p)
          this._updateOrbitTrapPanel(id)
          // カスタムパレットに切り替わったら、保存済みの仕様を UI に反映する
          // （resetTrapSpec は呼ばない → 前回設定を復元する）
          if (id === 'orbit_trap_custom' && palette.CUSTOM_ORBIT_TRAP_PALETTE) {
            _applyOrbitTrapSpecToUI(
              palette.CUSTOM_ORBIT_TRAP_PALETTE.trapSpec,
              palette.CUSTOM_ORBIT_TRAP_PALETTE._colorPatternId,
            )
          }
          if (!_maybeAutoApplyQueryOrbitTrapBitmap()) {
            this.notifyListeners()
            updatePermalink()
          }
        } catch (err) {
          console.warn('Error handling palette select change:', err)
        }
      })
    }

    // Buddhabrot 用パレットは専用セレクトだけに表示する

    let _permalinkDebounceTimer = null
    const _debouncedUpdatePermalink = () => {
      clearTimeout(_permalinkDebounceTimer)
      _permalinkDebounceTimer = setTimeout(() => updatePermalink(), 300)
    }

    this.densitySlider = document.getElementById('palette-density')
    this.densitySlider.addEventListener('input', () => {
      this.setDensity(this.densitySlider.value, true, false)
      _debouncedUpdatePermalink()
    })

    this.densityInput = document.getElementById('palette-density-input')
    if (this.densityInput) {
      this.densityInput.addEventListener('input', () => {
        const val = parseFloat(this.densityInput.value)
        if (!Number.isNaN(val)) {
          this.setDensity(val, false, true)
          _debouncedUpdatePermalink()
        }
      })
    }

    this.rotateSlider = document.getElementById('palette-rotate')
    this.rotateSlider.addEventListener('input', () => {
      this.setRotate(this.rotateSlider.value, true)
      _debouncedUpdatePermalink()
    })

    // buddha-supersampling の UI は廃止済み

    // Orbit Trap カスタムパネルを初期化する
    this._initOrbitTrapPanel()
  }

  setPalette(palette) {
    if (!palette) return
    this.palette = palette
    const paletteSelect = document.getElementById('palette-dropdown')

    if (paletteSelect) {
      // まずは value 直接設定で選択を試す
      try {
        paletteSelect.value = palette.id
      } catch (_e) {
        // だめなら option を順に探す
        for (let i = 0; i < paletteSelect.options.length; i++) {
          const opt = paletteSelect.options[i]
          if (opt.value === palette.id) {
            paletteSelect.selectedIndex = i
            break
          }
        }
      }
    }
  }

  setDensity(density, skipControl, skipInput) {
    // スライダー値との比較がぶれないよう文字列で保持する
    this.density = String(density)
    // document.getElementById("palette-density-label").innerText = "Density (" + density + ")"
    if (!skipControl) this.densitySlider.value = this.density
    if (!skipInput && this.densityInput) this.densityInput.value = this.density
    this.notifyListeners()
  }

  setRotate(rotate, skipControl) {
    // スライダー値との比較がぶれないよう文字列で保持する
    this.rotate = String(rotate)
    // document.getElementById("palette-rotate-label").innerText = "Rotate (" + rotate + ")"
    if (!skipControl) this.rotateSlider.value = this.rotate
    this.notifyListeners()
  }

  addListener(listener) {
    this.listeners.push(listener)
  }

  notifyListeners() {
    for (const listener of this.listeners) {
      listener(this.palette)
    }
  }

  /**
   * Orbit Trap カスタム設定パネルを初期化する。
   * `orbit_trap_custom` パレット選択時のみパネルを表示し、
   * UIの変更をリアルタイムにtrapSpecへ反映する。
   */
  _initOrbitTrapPanel() {
    const panel = document.getElementById('orbit-trap-panel')
    if (!panel) return

    // 現在のパレットに応じてパネルの初期表示を設定
    this._updateOrbitTrapPanel(this.palette?.id)

    // UIコントロールを現在の trapSpec に合わせて埋める（初回読み込みや
    // パレット切り替え後に仕様が異なる場合があるため）
    const customPalette = palette.CUSTOM_ORBIT_TRAP_PALETTE
    _applyOrbitTrapSpecToUI(customPalette.trapSpec, customPalette._colorPatternId)

    // UI変更時に設定を反映してレンダリングをトリガーする
    // bitmapのように非同期な場合を除いてsnapshot全体を再描画する
    const applyChanges = () => {
      this.notifyListeners()
      // パーマリンクにも反映
      updatePermalink()
    }
    queryOrbitTrapBitmapState.applyChanges = applyChanges

    // --- Shape ---
    const shapeSelect = document.getElementById('ot-shape')
    shapeSelect?.addEventListener('change', () => {
      customPalette.updateTrapSpec({ shape: shapeSelect.value })
      this._updateOrbitTrapModeVisibility()
      if (!_maybeAutoApplyQueryOrbitTrapBitmap()) {
        applyChanges()
      }
    })

    // --- Mode ---
    const modeSelect = document.getElementById('ot-mode')
    modeSelect?.addEventListener('change', () => {
      customPalette.updateTrapSpec({ mode: modeSelect.value })
      this._updateOrbitTrapModeVisibility()
      applyChanges()
    })

    // --- Size ---
    const sizeInput = document.getElementById('ot-size')
    sizeInput?.addEventListener('input', () => {
      const v = parseFloat(sizeInput.value)
      if (!Number.isNaN(v) && v >= 0) {
        customPalette.updateTrapSpec({ size: v })
        applyChanges()
      }
    })

    // --- Angle (度入力 → ラジアンに変換してtrapSpecへ) ---
    const angleInput = document.getElementById('ot-angle')
    angleInput?.addEventListener('input', () => {
      const v = parseFloat(angleInput.value)
      if (!Number.isNaN(v)) {
        customPalette.updateTrapSpec({ angle: (v * Math.PI) / 180 })
        applyChanges()
      }
    })

    // --- Center X ---
    const txInput = document.getElementById('ot-tx')
    txInput?.addEventListener('input', () => {
      const v = parseFloat(txInput.value)
      if (!Number.isNaN(v)) {
        customPalette.updateTrapSpec({ tx: v })
        applyChanges()
      }
    })

    // --- Center Y ---
    const tyInput = document.getElementById('ot-ty')
    tyInput?.addEventListener('input', () => {
      const v = parseFloat(tyInput.value)
      if (!Number.isNaN(v)) {
        // UI は「上=正の虚数」表示のため、内部座標（下=正）に変換して格納する
        customPalette.updateTrapSpec({ ty: -v })
        applyChanges()
      }
    })

    // --- Threshold (0入力時はInfinityに変換) ---
    const thresholdInput = document.getElementById('ot-threshold')
    thresholdInput?.addEventListener('input', () => {
      const v = parseFloat(thresholdInput.value)
      if (!Number.isNaN(v) && v >= 0) {
        customPalette.updateTrapSpec({ threshold: v === 0 ? Infinity : v })
        applyChanges()
      }
    })

    // --- Start Iter ---
    const startIterInput = document.getElementById('ot-start-iter')
    startIterInput?.addEventListener('input', () => {
      const v = parseInt(startIterInput.value, 10)
      if (!Number.isNaN(v) && v >= 0) {
        customPalette.updateTrapSpec({ startIter: v })
        applyChanges()
      }
    })

    // --- Capture Step (N番目の反復) ---
    const captureStepInput = document.getElementById('ot-capture-step')
    captureStepInput?.addEventListener('input', () => {
      const v = parseInt(captureStepInput.value, 10)
      if (!Number.isNaN(v) && v >= 1) {
        customPalette.updateTrapSpec({ captureStep: v })
        applyChanges()
      }
    })

    // --- Color Pattern ---
    const colorPatternSelect = document.getElementById('ot-color-pattern')
    colorPatternSelect?.addEventListener('change', () => {
      customPalette.setColorPattern(colorPatternSelect.value)
      applyChanges()
    })

    // --- Bitmap Background ---
    const bitmapBgColorInput = document.getElementById('ot-bitmap-bg-color')
    const applyBitmapBackgroundColor = (value) => {
      const color = _normalizeOrbitTrapColorCode(value)
      if (!color) return false
      _syncOrbitTrapBitmapBackgroundInputs(color)
      customPalette.updateTrapSpec({ bitmapBackgroundColor: color })
      const preview = document.getElementById('ot-bitmap-preview')
      if (preview) {
        _drawOrbitTrapBitmapPreviewFromImageData(
          preview,
          customPalette.trapSpec?.bitmapData,
          customPalette.trapSpec?.bitmapWidth ?? 0,
          customPalette.trapSpec?.bitmapHeight ?? 0,
          color
        )
      }
      applyChanges()
      return true
    }
    bitmapBgColorInput?.addEventListener('input', () => {
      applyBitmapBackgroundColor(bitmapBgColorInput.value)
    })

    // --- Orbit trap パネルのリセットボタン ---
    const otResetBtn = document.getElementById('ot-reset')
    otResetBtn?.addEventListener('click', () => {
      // 既定の trapSpec と色パターンへ戻す
      customPalette.resetTrapSpec()
      _applyOrbitTrapSpecToUI(customPalette.trapSpec, customPalette._colorPatternId)
      applyChanges()
    })

    // --- Bitmap ファイル選択 ---
    const bitmapFileInput = document.getElementById('ot-bitmap-file')
    bitmapFileInput?.addEventListener('change', async () => {
      const file = bitmapFileInput.files?.[0]
      if (!file) return
      try {
        const bitmapRecord = await _loadOrbitTrapBitmapRecordFromBlob(file)
        // 手動選択があれば、クエリ由来の自動適用で上書きしない。
        queryOrbitTrapBitmapState.autoApplied = true
        _applyOrbitTrapBitmapRecord(customPalette, bitmapRecord)
        applyChanges()
      } catch (e) {
        console.warn('Orbit Trap: ビットマップ画像の読み込みに失敗しました', e?.message ? e.message : e)
      }
    })
  }

  /**
   * パレットIDに応じてOrbit Trap設定パネルの表示/非表示を切り替える。
   * @param {string|null|undefined} paletteId
   */
  _updateOrbitTrapPanel(paletteId) {
    const panel = document.getElementById('orbit-trap-panel')
    if (!panel) return
    if (paletteId === 'orbit_trap_custom') {
      panel.classList.remove('d-none')
      this._updateOrbitTrapModeVisibility()
    } else {
      panel.classList.add('d-none')
    }
  }

  /**
   * Shape / Mode の選択値に応じてサブグループの表示を更新する。
   * 不要なUIを隠すことで操作ミスを防ぐ。
   */
  _updateOrbitTrapModeVisibility() {
    const shapeSelect = document.getElementById('ot-shape')
    const modeSelect = document.getElementById('ot-mode')
    if (!shapeSelect || !modeSelect) return

    const shape = shapeSelect.value
    const mode = modeSelect.value
    // TIA は形状を参照しないため、形状関連 UI をすべて非表示にする
    const isTia = mode === 'tia'
    const primaryRow = document.getElementById('ot-primary-controls-row')
    const secondaryRow = document.getElementById('ot-secondary-controls-row')
    const shapeGroup = document.getElementById('ot-shape-group')
    const colorPatternGroup = document.getElementById('ot-color-pattern-group')
    const sizeGroup = document.getElementById('ot-size-group')
    const angleGroup = document.getElementById('ot-angle-group')
    const isBitmap = shape === 'bitmap'

    // Shape / Size / Center: TIA や point は無関係なので非表示
    const hideSize = isTia || shape === 'point'
    shapeGroup?.classList.toggle('d-none', isTia)
    sizeGroup?.classList.toggle('d-none', hideSize)
    document.getElementById('ot-center-group')?.classList.toggle('d-none', isTia)

    // TIA のときは Color を Shape の位置へ移し、Mode と横並びにする。
    // 通常モードへ戻ったら元の行へ戻す。
    if (isTia) {
      if (primaryRow && colorPatternGroup && colorPatternGroup.parentElement !== primaryRow) {
        primaryRow.appendChild(colorPatternGroup)
      }
      colorPatternGroup?.classList.remove('col-4')
      colorPatternGroup?.classList.add('col-6')
      secondaryRow?.classList.add('d-none')
    } else {
      if (secondaryRow && colorPatternGroup && colorPatternGroup.parentElement !== secondaryRow) {
        secondaryRow.insertBefore(colorPatternGroup, sizeGroup ?? angleGroup ?? null)
      }
      colorPatternGroup?.classList.remove('col-6')
      colorPatternGroup?.classList.add('col-4')
      secondaryRow?.classList.remove('d-none')
    }

    // Bitmap では Color が隠れるため、Size と Angle を 2 カラムで横並びにする。
    if (isBitmap && !isTia) {
      sizeGroup?.classList.remove('col-4')
      sizeGroup?.classList.add('col-6')
      angleGroup?.classList.remove('col-4')
      angleGroup?.classList.add('col-6')
    } else {
      sizeGroup?.classList.remove('col-6')
      sizeGroup?.classList.add('col-4')
      angleGroup?.classList.remove('col-6')
      angleGroup?.classList.add('col-4')
    }

    // サイズラベルを調整（ラインは半長さとして扱う）。
    // point/​tia ではサイズコントロール自体が隠れているのでここでは強調不要。
    const sizeLabel = sizeGroup?.querySelector('label')
    const sizeInput = sizeGroup?.querySelector('input')
    if (sizeLabel && sizeInput) {
      if (shape === 'line') {
        sizeLabel.textContent = 'Half‑length'
        sizeInput.title = '線分の半長さ (全長は2×この値)'
      } else {
        sizeLabel.textContent = 'Size'
        sizeInput.title = '形状のサイズ (半径 / 辺長など)'
      }
    }

    // Angle: 回転が意味を持つ形状かつ TIA 以外のとき表示
    const shapesWithAngle = ['line', 'parabola', 'triangle', 'square', 'bitmap']
    angleGroup?.classList.toggle('d-none', isTia || !shapesWithAngle.includes(shape))

    // Bitmap group: BITMAP 選択かつ TIA 以外のとき表示
    document.getElementById('ot-bitmap-group')?.classList.toggle('d-none', isTia || shape !== 'bitmap')

    // bitmap 使用時は色が画像由来なので color は不要。
    // ただし TIA は shape を参照しないため、内部的に bitmap が残っていても表示する。
    document.getElementById('ot-color-pattern-group')?.classList.toggle('d-none', !isTia && shape === 'bitmap')

    // Threshold 行は capture_first / distance_farthest / distance_average のときだけ表示
    const modesWithThreshold = ['capture_first', 'distance_farthest', 'distance_average']
    document.getElementById('ot-threshold-group')?.classList.toggle('d-none', !modesWithThreshold.includes(mode))

    // Start Iter は capture_first のときだけ表示する
    document.getElementById('ot-start-iter-group')?.classList.toggle('d-none', mode !== 'capture_first')

    // Capture Step は capture_step のときだけ表示する
    document.getElementById('ot-step-group')?.classList.toggle('d-none', mode !== 'capture_step')
  }
}

const paletteComponent = new PaletteComponent()

// Buddhabrot 状態の集中管理
const BuddhabrotState = {
  runner: null,
  active: false,
  preservedDisplay: false,
  lockedByFractalChange: false,
  targetKind: null,

  isViewEnabled() {
    return DOM.buddha.toggle?.checked
  },

  reset() {
    this.runner = null
    this.active = false
    this.preservedDisplay = false
    this.targetKind = null
  },

  setActive(value) {
    this.active = value
  },

  lockByFractalChange() {
    this.lockedByFractalChange = true
    if (DOM.buddha.toggle) {
      DOM.buddha.toggle.disabled = true
      DOM.buddha.toggle.checked = false
    }
  },
}

// 後方互換のための旧グローバル変数
let buddhaRunner = null
let buddhaActive = false
let buddhaPreservedDisplay = false
let buddhaLockedByFractalChange = false

// 旧グローバル変数と BuddhabrotState を同期する
Object.defineProperty(window, 'buddhaRunner', {
  get: () => BuddhabrotState.runner,
  set: (v) => {
    BuddhabrotState.runner = v
  },
})
Object.defineProperty(window, 'buddhaActive', {
  get: () => BuddhabrotState.active,
  set: (v) => {
    BuddhabrotState.active = v
  },
})
Object.defineProperty(window, 'buddhaPreservedDisplay', {
  get: () => BuddhabrotState.preservedDisplay,
  set: (v) => {
    BuddhabrotState.preservedDisplay = v
  },
})
Object.defineProperty(window, 'buddhaLockedByFractalChange', {
  get: () => BuddhabrotState.lockedByFractalChange,
  set: (v) => {
    BuddhabrotState.lockedByFractalChange = v
  },
})

/**
 * Buddhabrot 用パレットの選択肢を追加する
 */
function populateBuddhaPaletteSelect() {
  const sel = document.getElementById('buddha-palette')
  if (!sel) return
  // Buddhabrot 専用パレット定義を使う
  BUDDHA_PALETTES.forEach((p) => {
    const opt = document.createElement('option')
    opt.value = p.id
    opt.textContent = p.name
    sel.appendChild(opt)
  })
}

// 起動時はトグルを無効・オフ状態でそろえる
try {
  const toggleInit = document.getElementById('buddha-toggle')
  if (toggleInit) {
    toggleInit.checked = false
    toggleInit.disabled = true
  }
} catch (e) {
  ErrorHelpers.warn('Error initializing buddha toggle state:', e)
}

// buddha-gpu トグルも初期状態をそろえる
try {
  const bgpuInit = document.getElementById('buddha-gpu')
  if (bgpuInit) {
    bgpuInit.checked = false
    // 有効・無効の判断は後段ロジックに任せる
  }
} catch (e) {
  ErrorHelpers.warn('Error initializing buddha-gpu toggle state:', e)
}

// Buddhabrot 表示と通常フラクタル表示の切り替えを結び付ける
try {
  const buddhaToggle = document.getElementById('buddha-toggle')
  if (buddhaToggle) {
    buddhaToggle.addEventListener('change', async (_e) => {
      const target = getBuddhabrotTargetSnapshot()
      // ビットマップ保存中なら少し待ち、停止済み Buddhabrot の復元を安定させる
      try {
        if (!savedBuddhaImageData && savedBuddhaImageDataPromise) {
          await savedBuddhaImageDataPromise
        }
      } catch (_err) {}
      // 無効状態のトグル変更は無視する
      if (buddhaToggle.disabled) return
      if (buddhaToggle.checked) {
        // フラクタル変更でロックされているときは有効化を許可しない
        if (buddhaLockedByFractalChange) {
          buddhaToggle.checked = false
          buddhaToggle.disabled = true
          return
        }
        // すでに Buddhabrot 表示中なら何もしない
        if (buddhaActive) return
        // 停止前の Buddhabrot 画像があれば先に復元する
        if (savedBuddhaImageData && savedBuddhaImageData.targetKind === target.kind) {
          try {
            if (target.kind === 'julia') {
              clearJuliaOrbitCanvas()
              if (detailPinnedState?.isJulia) {
                clearPinnedDetailPopup()
              } else if (detailHoverState?.isJulia) {
                _clearDetailHoverState()
              }
              if (juliaCanvasElement) juliaCanvasElement.style.cursor = ''
            }
            restoreSavedImageToCanvas(savedBuddhaImageData, target.canvas)
            BuddhabrotState.targetKind = target.kind
            buddhaActive = true
            buddhaToggle.disabled = false
            return
          } catch (err) {
            ErrorHelpers.warn('Error restoring saved Buddhabrot image:', err)
            // 復元できなければ下の再開処理へ進む
          }
        } else if (savedBuddhaImageData && savedBuddhaImageData.targetKind !== target.kind) {
          discardSavedBuddhaImageData()
        }

        // 以前の density バッファが残っていれば復元を試す
        if (buddhaRunner?.densityR && buddhaRunner.densityG && buddhaRunner.densityB) {
          // まずは軽い確認として、一部サンプルに非ゼロがあるか見る
          let found = false
          const dr = buddhaRunner.densityR
          const step = Math.max(1, Math.floor(dr.length / 1000))
          for (let i = 0; i < dr.length; i += step) {
            if (dr[i]) {
              found = true
              break
            }
          }
          if (found) {
            try {
              buddhaActive = true
              if (target.kind === 'julia') {
                clearJuliaOrbitCanvas()
                if (detailPinnedState?.isJulia) {
                  clearPinnedDetailPopup()
                } else if (detailHoverState?.isJulia) {
                  _clearDetailHoverState()
                }
                if (juliaCanvasElement) juliaCanvasElement.style.cursor = ''
              }
              // トグルは使える状態に戻す
              buddhaToggle.disabled = false
              // 保存済み density バッファを描画する
              const palNow = buddhaRunner.palette || buildPaletteFromId(document.getElementById('buddha-palette').value)
              drawBuddhaDensityChannels(
                buddhaRunner.densityR,
                buddhaRunner.densityG,
                buddhaRunner.densityB,
                buddhaRunner.width,
                buddhaRunner.height,
                palNow,
                buddhaRunner.brightness,
                buddhaRunner.gamma,
              )
              return
            } catch (err) {
              console.warn('Error restoring previous Buddhabrot display:', err?.message ? err.message : err)
              // だめなら新規描画へ進む
            }
          }
        }
        // 復元できる density がなければ新しく描画を始める
        try {
          startBuddhaRender()
        } catch (err) {
          console.warn('Error starting Buddhabrot from toggle:', err?.message ? err.message : err)
          buddhaToggle.checked = false
        }
      } else {
        // Buddhabrot 表示をオフにしたら、表示を残したままワーカーだけ止める
        if (buddhaActive || buddhaPreservedDisplay || savedBuddhaImageData) {
          // Stop 後の保持表示だけ残っている場合もある
          if (buddhaActive) stopBuddhaPreserveDisplay()
          // Buddhabrot 状態は保ちつつ、通常フラクタル表示へ戻す
          try {
            // 保存時の状態と現在状態を比較する
            const displayTargetKind = BuddhabrotState.targetKind || target.kind
            const displayCanvas = getBuddhabrotDisplayCanvas(displayTargetKind)
            const currentState = FractalStateHelpers.getCurrentState(displayTargetKind)
            const match = savedFractalImageData && FractalStateHelpers.stateMatches(savedFractalImageData, currentState)

            if (match) {
              restoreSavedImageToCanvas(savedFractalImageData, displayCanvas)
              // ビューが変わるまでは保存画像を残し、再オン時に再利用できるようにする
            } else {
              // Buddhabrot 表示中に通常フラクタル設定が変わっている可能性があるので、
              // パレット設定を反映し直してから正規の再描画を行う。
              fractal.initPallete(true)

              redrawBuddhabrotSourceView(displayTargetKind)
            }
          } catch (e) {
            ErrorHelpers.warn('Error restoring saved fractal image after turning off Buddha:', e)
            redrawBuddhabrotSourceView(BuddhabrotState.targetKind || target.kind)
          }
        }
      }
    })
  }
} catch (e) {
  ErrorHelpers.warn('Error wiring buddha-toggle handler:', e)
}

// Buddhabrot の GPU 利用設定トグルを接続する
const buddhaGpuToggle = document.getElementById('buddha-gpu')
if (buddhaGpuToggle) {
  buddhaGpuToggle.addEventListener('change', (_e) => {
    try {
      // GPU 版の runner が動作中に OFF へした場合は停止し、次回は CPU 版で作り直す
      if (!buddhaGpuToggle.checked && buddhaRunner) {
        const isGpuRunner =
          (buddhaRunner.constructor && buddhaRunner.constructor.name === 'BuddhabrotWebGPU') ||
          typeof buddhaRunner.devicePromise !== 'undefined'
        if (isGpuRunner) {
          buddhaRunner.terminate?.()
        }
      }
    } catch (e) {
      console.warn('Error handling buddha-gpu toggle change:', e)
    }
  })
}

/**
 * パレット ID から設定オブジェクトを作る
 * @param {string} id - パレット ID
 * @returns {Object} パレット設定オブジェクト
 */
function buildPaletteFromId(id) {
  // まず Buddhabrot 専用パレットを探す
  const bp = getBuddhaPalette(id)
  if (bp) {
    return buildBuddhaStops(id)
  }

  // なければ通常パレットで代用する
  const p = palette.getPalette(id)
  const colors = []
  try {
    const samples = [0, 50, 100]
    const weights = [0.9, 0.085, 0.015]
    for (let i = 0; i < samples.length; i++) {
      const v = samples[i]
      let c
      try {
        c = p.getColor(v, 0)
      } catch (_e) {
        c = [255, 255, 255]
      }
      colors.push({ color: c, weight: weights[i] })
    }
  } catch (_e) {
    colors.push({ color: [255, 255, 255], weight: 0.9 })
    colors.push({ color: [0, 0, 255], weight: 0.085 })
    colors.push({ color: [128, 0, 128], weight: 0.015 })
  }
  return colors
}

function getCurrentJuliaCFromMain() {
  return {
    re: fractal.center[0].toNumber ? fractal.center[0].toNumber() : 0,
    im: fractal.center[1].toNumber ? fractal.center[1].toNumber() : 0,
  }
}

function syncJuliaRendererSettingsFromMain() {
  if (!juliaState?.renderer) return null
  const renderer = juliaState.renderer
  renderer.max_iter = fractal.max_iter
  renderer.smooth = fractal.smooth
  renderer.supersampling = fractal.supersampling
  renderer.escapeRadius = fractal.escapeRadius !== undefined ? fractal.escapeRadius : 4.0
  renderer.useGpu = fractal.useGpu
  if (fractal.fractalType === 'custom') {
    renderer.fractalType = 'julia-custom'
    renderer.iterationFunction = fractal.iterationFunction
  } else {
    renderer.fractalType = 'julia'
    renderer.iterationFunction = 'z*z + c'
  }
  renderer.syncSettings()
  const juliaC = getCurrentJuliaCFromMain()
  renderer.setJuliaC(juliaC.re, juliaC.im)
  return { renderer, juliaC }
}

function getBuddhabrotTargetSnapshot() {
  if (juliaState?.active && juliaState.renderer && juliaCanvasElement) {
    const synced = syncJuliaRendererSettingsFromMain()
    const renderer = synced?.renderer || juliaState.renderer
    const juliaC = synced?.juliaC || getCurrentJuliaCFromMain()
    return {
      kind: 'julia',
      canvas: juliaCanvasElement,
      renderer,
      fractalType: renderer.fractalType || 'julia',
      iterationFunction: renderer.iterationFunction || 'z*z + c',
      juliaRe: juliaC.re,
      juliaIm: juliaC.im,
    }
  }
  return {
    kind: 'main',
    canvas: canvasElement,
    renderer: fractal,
    fractalType: fractal?.fractalType || 'mandelbrot',
    iterationFunction: fractal?.iterationFunction || '',
    juliaRe: undefined,
    juliaIm: undefined,
  }
}

function getBuddhabrotDisplayCanvas(targetKind = BuddhabrotState.targetKind) {
  if (targetKind === 'julia' && juliaCanvasElement) return juliaCanvasElement
  return canvasElement
}

function getBuddhabrotProgressMonitor(targetKind = BuddhabrotState.targetKind) {
  if (targetKind === 'julia') return juliaState?.renderer?.progress || null
  return fractal?.progress || null
}

function hideProgressMonitor(progressMonitor) {
  if (!progressMonitor) return
  try {
    progressMonitor._cancelPendingShow?.()
    if (typeof progressMonitor._hide === 'function') {
      progressMonitor._hide()
    } else if (progressMonitor.canvas) {
      progressMonitor.canvas.style.display = 'none'
    }
  } catch (e) {
    console.warn('Error hiding progress monitor:', e?.message ? e.message : e)
  }
}

function hideInactiveBuddhabrotProgress(targetKind = BuddhabrotState.targetKind) {
  hideProgressMonitor(getBuddhabrotProgressMonitor(targetKind === 'julia' ? 'main' : 'julia'))
}

function finishBuddhabrotProgress(targetKind = BuddhabrotState.targetKind) {
  const progressMonitor = getBuddhabrotProgressMonitor(targetKind)
  if (!progressMonitor) return
  try {
    progressMonitor.finish?.()
  } catch (e) {
    console.warn('Error finishing Buddhabrot progress:', e?.message ? e.message : e)
    hideProgressMonitor(progressMonitor)
  }
}

function hasVisibleOrRunningBuddhabrot() {
  return (
    buddhaActive ||
    buddhaPreservedDisplay ||
    !!buddhaRunner ||
    !!savedBuddhaImageData ||
    !!savedBuddhaImageDataPromise ||
    BuddhabrotState.isViewEnabled()
  )
}

function isJuliaCanvasInteractionBlockedByBuddhabrot() {
  return buddhaRenderPending || buddhaActive || BuddhabrotState.isViewEnabled()
}

function isBuddhabrotViewShownOnJulia() {
  return (
    BuddhabrotState.targetKind === 'julia' &&
    BuddhabrotState.isViewEnabled() &&
    (buddhaActive || buddhaPreservedDisplay || !!buddhaRunner || !!savedBuddhaImageData)
  )
}

function isMainCanvasInteractionBlockedByBuddhabrot() {
  return (buddhaActive || BuddhabrotState.isViewEnabled()) && !isBuddhabrotViewShownOnJulia()
}

function hasJuliaBuddhabrotState() {
  return BuddhabrotState.targetKind === 'julia' || savedBuddhaImageData?.targetKind === 'julia'
}

function lockBuddhabrotViewToggle() {
  buddhaLockedByFractalChange = true
  const toggle = DOM.buddha.toggle || document.getElementById('buddha-toggle')
  if (toggle) {
    toggle.checked = false
    toggle.disabled = true
  }
}

function invalidateJuliaBuddhabrotView({ clearVisible = false } = {}) {
  if (!hasJuliaBuddhabrotState()) return false
  if (clearVisible && (BuddhabrotState.isViewEnabled() || buddhaActive)) {
    stopAndClearBuddha()
  } else {
    if (BuddhabrotState.targetKind === 'julia' && buddhaRunner?.running) {
      buddhaRunner.stop()
    }
    finishBuddhabrotProgress(BuddhabrotState.targetKind)
    hideInactiveBuddhabrotProgress(BuddhabrotState.targetKind)
    buddhaActive = false
    buddhaPreservedDisplay = false
    discardSavedBuddhaImageData()
    savedFractalImageData = null
    BuddhabrotState.targetKind = null
    disableBuddhaDownload()
  }
  lockBuddhabrotViewToggle()
  return true
}

function restoreSavedImageToCanvas(savedImageData, canvas) {
  const ctx = canvas.getContext('2d')
  CanvasHelpers.restoreImage(ctx, savedImageData, canvas.width, canvas.height)
}

function restoreFractalDisplayAfterClearingBuddha(targetKind = savedFractalImageData?.targetKind) {
  if (!savedFractalImageData) return false
  if (targetKind !== savedFractalImageData.targetKind) return false
  const canvas = getBuddhabrotDisplayCanvas(targetKind)
  if (!canvas) return false
  restoreSavedImageToCanvas(savedFractalImageData, canvas)
  return true
}

function restoreOrRedrawFractalDisplayAfterClearingBuddha(targetKind = 'main') {
  const currentState = FractalStateHelpers.getCurrentState(targetKind)
  const canRestore =
    savedFractalImageData &&
    savedFractalImageData.targetKind === targetKind &&
    FractalStateHelpers.stateMatches(savedFractalImageData, currentState)
  if (canRestore && restoreFractalDisplayAfterClearingBuddha(targetKind)) return true
  redrawBuddhabrotSourceView(targetKind)
  return false
}

function redrawBuddhabrotSourceView(targetKind = BuddhabrotState.targetKind) {
  if (targetKind === 'julia') {
    redrawJulia()
    return
  }
  redraw()
}

function discardSavedBuddhaImageData() {
  savedBuddhaImageCaptureGeneration += 1
  if (savedBuddhaImageData?.type === 'bitmap' && typeof savedBuddhaImageData.bitmap?.close === 'function') {
    savedBuddhaImageData.bitmap.close()
  }
  savedBuddhaImageData = null
  savedBuddhaImageDataPromise = null
}

/**
 * Buddhabrot の描画ジョブを開始する
 * @async
 * @throws {Error} 描画を開始できない場合
 */
async function startBuddhaRender() {
  const target = getBuddhabrotTargetSnapshot()
  const targetCanvas = target.canvas
  const targetRenderer = target.renderer
  // 呼び出し時点で Buddhabrot 表示中だったかを覚えておく
  const wasBuddhaActive = !!buddhaActive
  const targetWasShowingBuddha =
    BuddhabrotState.targetKind === target.kind && (wasBuddhaActive || BuddhabrotState.isViewEnabled())
  const renderRequestGeneration = ++buddhaRenderRequestGeneration
  buddhaRenderPending = true
  const isBuddhaRenderRequestCurrent = () => renderRequestGeneration === buddhaRenderRequestGeneration
  const abortIfBuddhaRenderCanceled = () => {
    if (isBuddhaRenderRequestCurrent()) return false
    buddhaRenderPending = false
    return true
  }

  // 新しい Buddhabrot が完成するまでは、古い View 表示を切り替えられないようにする
  const viewToggle = DOM.buddha.toggle || document.getElementById('buddha-toggle')
  if (viewToggle) {
    viewToggle.checked = false
    viewToggle.disabled = true
  }

  // 通常フラクタル描画が進行中なら止め、Buddhabrot に計算資源を譲る
  try {
    if (typeof fractal !== 'undefined' && fractal) {
      fractal._revokeJobToken?.()
      // 古い CPU タスクが再実行されないようキューを空にする
      if (Array.isArray(fractal.taskqueue)) fractal.taskqueue.length = 0
      // GPU 側にも停止を促す
      if (fractal.mandelbrotGpu) fractal.mandelbrotGpu.newTask = null
      if (fractal.mandelbrotCustomGpu) fractal.mandelbrotCustomGpu.newTask = null
      if (fractal.orbitTrapGpu) fractal.orbitTrapGpu.newTask = null
    }
    if (target.kind === 'julia' && juliaState.renderer) {
      juliaState.renderer._revokeJobToken?.()
      if (Array.isArray(juliaState.renderer.taskqueue)) juliaState.renderer.taskqueue.length = 0
      if (juliaState.renderer.juliaGpu) juliaState.renderer.juliaGpu.newTask = null
      if (juliaState.renderer.orbitTrapGpu) juliaState.renderer.orbitTrapGpu.newTask = null
    }
  } catch (e) {
    console.warn('Error attempting to stop normal fractal render before Buddhabrot start:', e?.message ? e.message : e)
  }

  // すでに Buddhabrot 実行中なら一度止めてからやり直す
  try {
    if (buddhaActive) stopAndClearBuddha(true)
  } catch (e) {
    console.warn('Error stopping existing Buddhabrot before starting new one:', e?.message ? e.message : e)
  }
  let samples = parseInt(document.getElementById('buddha-iterations').value, 10)
  if (Number.isNaN(samples)) samples = 100000
  // 専用 supersampling UI は廃止済み。fractal 側の値があれば使う
  const ss = targetRenderer?.supersampling ? parseInt(targetRenderer.supersampling, 10) : 0
  if (ss > 1) samples = Math.floor(samples * ss * ss)

  const palId = document.getElementById('buddha-palette').value
  // Buddhabrot 専用パレットがあればそのまま渡し、なければ通常パレットから組み立てる
  // GPU と CPU で同じ累積バンド値になるよう、正規化済み stops は常に作る
  const rawPal = buildPaletteFromId(palId)
  // Buddhabrot パレットなら buddhaBandMode も保ったまま object 形式へそろえる
  const bp = (() => {
    try {
      return getBuddhaPalette(palId)
    } catch (_e) {
      return null
    }
  })()
  const pal = Array.isArray(rawPal)
    ? {
        bands: rawPal,
        buddhaBandMode: bp?.buddhaBandMode ? bp.buddhaBandMode : undefined,
      }
    : rawPal

  const brightness = parseFloat(document.getElementById('buddha-brightness')?.value) || 1.8
  const gamma = parseFloat(document.getElementById('buddha-gamma')?.value) || 0.8
  const _rawRenderDelay = parseFloat(document.getElementById('buddha-draw-speed')?.value) || 0
  const renderDelay = _rawRenderDelay === 1 ? 0.01 : _rawRenderDelay

  BuddhabrotState.targetKind = target.kind
  if (target.kind === 'julia') {
    clearJuliaOrbitCanvas()
    if (detailPinnedState?.isJulia) {
      clearPinnedDetailPopup()
    } else if (detailHoverState?.isJulia) {
      _clearDetailHoverState()
    }
    if (juliaCanvasElement) juliaCanvasElement.style.cursor = ''
  }

  // 開始前に対象キャンバス上の描画を整理する
  const ctx = targetCanvas.getContext('2d')
  // 後で戻せるよう、通常フラクタル側の画像を必要時だけ保存する
  if (!wasBuddhaActive) {
    try {
      const currentState = FractalStateHelpers.getCurrentState(target.kind)

      if (targetWasShowingBuddha) {
        // View ON の Buddhabrot 画像を通常フラクタルの復元元として保存しない。
        // 座標変更後の再描画では、復元ではなく正規レンダーへ戻す。
        if (!savedFractalImageData || !FractalStateHelpers.stateMatches(savedFractalImageData, currentState)) {
          savedFractalImageData = null
        }
      } else if (!savedFractalImageData || !FractalStateHelpers.stateMatches(savedFractalImageData, currentState)) {
        // getImageData の読み戻し警告を避けるため、まず createImageBitmap を優先する。
        // startBuddhaRender は async なのでここで await できる。
        let captured = false
        if (typeof createImageBitmap === 'function') {
          const bitmap = await createImageBitmap(targetCanvas)
          savedFractalImageData = {
            ...currentState,
            type: 'bitmap',
            bitmap: bitmap,
            width: targetCanvas.width,
            height: targetCanvas.height,
          }
          captured = true
        }

        if (!captured) {
          // だめなら同期読み戻しへ切り替える。
          // 警告を減らすため、一度一時キャンバスへ描いてから getImageData を呼ぶ。
          tempCanvas.width = targetCanvas.width
          tempCanvas.height = targetCanvas.height
          let tempCtx
          try {
            tempCtx = tempCanvas.getContext('2d', {
              willReadFrequently: true,
            })
          } catch (_e) {
            tempCtx = tempCanvas.getContext('2d')
          }
          // 対象キャンバスを一時キャンバスへ描き写す
          tempCtx.clearRect(0, 0, tempCanvas.width, tempCanvas.height)
          tempCtx.drawImage(targetCanvas, 0, 0)
          const img = tempCtx.getImageData(0, 0, tempCanvas.width, tempCanvas.height)
          savedFractalImageData = {
            ...currentState,
            type: 'imageData',
            imageData: img,
          }
        }
      }
    } catch (e) {
      // cross-origin などで getImageData が失敗しても、既存の保存画像は消さない。
      // 読み戻せないだけで通常フラクタルのキャッシュを失わないようにする。
      console.warn('Could not capture canvas ImageData for buddhabrot preservation:', e?.message ? e.message : e)
    }
  }
  if (abortIfBuddhaRenderCanceled()) return

  ctx.save()
  ctx.fillStyle = 'black'
  ctx.fillRect(0, 0, targetCanvas.width, targetCanvas.height)
  ctx.restore()

  // 必要なら runner を作る。
  // ただし GPU runner が残っていて GPU トグルが OFF なら停止し、CPU runner を作り直す。

  // 判定には Buddhabrot 専用 GPU トグルを使い、通常 GPU トグルとは独立させる。
  const gpuEl = document.getElementById('buddha-gpu') || document.getElementById('gpu')
  const wantGpu = gpuEl ? !!gpuEl.checked : false
  if (!wantGpu && buddhaRunner) {
    const isGpuRunner =
      (buddhaRunner.constructor && buddhaRunner.constructor.name === 'BuddhabrotWebGPU') ||
      typeof buddhaRunner.devicePromise !== 'undefined'
    if (isGpuRunner) {
      buddhaRunner.terminate?.()
      buddhaRunner = null
    }
  }

  if (!buddhaRunner) {
    // 細かい chunk 更新は 1 フレーム 1 回にまとめ、メインスレッド負荷を下げる
    let drawScheduled = false
    const runnerGeneration = ++buddhaRunnerGeneration
    const scheduleDraw = () => {
      if (drawScheduled) return
      drawScheduled = true
      requestAnimationFrame(() => {
        drawScheduled = false
        const runner = buddhaRunner
        if (!runner || buddhaRunnerGeneration !== runnerGeneration) return
        try {
          const palNow = runner.palette || pal
          drawBuddhaDensityChannels(
            runner.densityR,
            runner.densityG,
            runner.densityB,
            runner.width,
            runner.height,
            palNow,
            runner.brightness,
            runner.gamma,
          )
        } catch (e) {
          console.warn('Error drawing buddha density in scheduled rAF:', e?.message ? e.message : e)
        }
      })
    }

    // factory がまだ使えない場合は CPU runner をその場で作る
    try {
      buddhaRunner = new BuddhabrotRunner({
        workerCount: Math.max(1, DEFAULT_WORKER_COUNT),
        width: targetCanvas.width,
        height: targetCanvas.height,
        maxIter: targetRenderer.max_iter,
        samples: samples,
        renderDelay: renderDelay,
        onProgress: (data) => {
          // delta が来る場合はその差分で進捗を進める
          try {
            if (!buddhaActive) return
            const progressMonitor = getBuddhabrotProgressMonitor()
            if (!progressMonitor) return
            if (data && typeof data.delta === 'number') progressMonitor.update(data.delta)
            else progressMonitor.update(1)
          } catch (e) {
            console.warn('Error updating progress from BuddhabrotRunner onProgress:', e?.message ? e.message : e)
          }
        },
        onChunk: (_chunk) => {
          // 進捗更新は別で行うので、ここでは描画予約だけ行う
          try {
            scheduleDraw()
          } catch (e) {
            console.warn('Error scheduling buddha draw:', e?.message ? e.message : e)
          }
        },
        onComplete: (result) => {
          try {
            finishBuddhabrotProgress()
            hideInactiveBuddhabrotProgress()
          } catch (e) {
            console.warn('Error finishing progress after buddha complete:', e?.message ? e.message : e)
          }
          // 最終描画
          try {
            drawBuddhaDensityChannels(
              buddhaRunner.densityR,
              buddhaRunner.densityG,
              buddhaRunner.densityB,
              result.width,
              result.height,
              pal,
              buddhaRunner.brightness,
              buddhaRunner.gamma,
            )
          } catch (e) {
            console.warn('Error drawing final buddha density on complete:', e?.message ? e.message : e)
          }
          // 描画完了後に Buddha view トグルを有効化する
          const toggle = document.getElementById('buddha-toggle')
          if (toggle) {
            toggle.disabled = false
            toggle.checked = true
          }
          // 通常フラクタルへ戻せるよう、savedFractalImageData は保持したままにする

          // scale が 1 より大きいときだけ高解像度ダウンロードを有効化する
          const scaleSel = document.getElementById('buddha-scale')
          const downloadBtn = document.getElementById('buddha-download')
          if (scaleSel && downloadBtn) {
            const scale = parseInt(scaleSel.value, 10) || 1
            // density バッファがあり、scale > 1 のときだけ有効化する
            if (scale > 1 && buddhaRunner && buddhaRunner.densityR) {
              downloadBtn.disabled = false
            } else {
              downloadBtn.disabled = true
            }
          }
          // 途中停止後は古い画像を掴んでいることがあるため、ここで取り直しておく
          if (buddhaPreservedDisplay) {
            requestAnimationFrame(() => {
              try {
                const displayCanvas = getBuddhabrotDisplayCanvas(BuddhabrotState.targetKind)
                const captureTargetKind = BuddhabrotState.targetKind
                const captureGeneration = ++savedBuddhaImageCaptureGeneration
                if (typeof createImageBitmap === 'function') {
                  savedBuddhaImageDataPromise = createImageBitmap(displayCanvas).then((bitmap) => {
                    if (captureGeneration !== savedBuddhaImageCaptureGeneration) {
                      bitmap.close?.()
                      return
                    }
                    savedBuddhaImageData = {
                      type: 'bitmap',
                      bitmap,
                      width: displayCanvas.width,
                      height: displayCanvas.height,
                      targetKind: captureTargetKind,
                    }
                    savedBuddhaImageDataPromise = null
                  })
                } else {
                  tempCanvas.width = displayCanvas.width
                  tempCanvas.height = displayCanvas.height
                  let tCtx
                  try {
                    tCtx = tempCanvas.getContext('2d', {
                      willReadFrequently: true,
                    })
                  } catch (_) {
                    tCtx = tempCanvas.getContext('2d')
                  }
                  tCtx.clearRect(0, 0, tempCanvas.width, tempCanvas.height)
                  tCtx.drawImage(displayCanvas, 0, 0)
                  savedBuddhaImageData = {
                    type: 'imageData',
                    imageData: tCtx.getImageData(0, 0, tempCanvas.width, tempCanvas.height),
                    width: tempCanvas.width,
                    height: tempCanvas.height,
                    targetKind: captureTargetKind,
                  }
                }
              } catch (e) {
                console.warn('Error re-capturing Buddhabrot image after stop:', e?.message ? e.message : e)
              }
            })
          }
        },
      })
    } catch (e) {
      console.error('index.js: error creating buddhaRunner', e?.message ? e.message : e, e)
    }
    // GPU 優先設定で factory が使えるなら、GPU runner への切り替えを試す
    ;(async () => {
      try {
        // アプリ全体または Buddhabrot 専用の GPU トグルが ON のときだけ試す
        if (
          (() => {
            try {
              // Buddhabrot 専用 GPU トグルがあればそちらを優先する
              const bEl = document.getElementById('buddha-gpu')
              if (bEl) return !!bEl.checked
              const el = document.getElementById('gpu')
              return el ? !!el.checked : false
            } catch (_e) {
              return false
            }
          })()
        ) {
          // 非同期の別経路で差し替わっていないか確認できるよう、今の runner を控える
          const createdRunner = buddhaRunner
          const mod = await import('./buddhabrot.mjs')
          if (mod && typeof mod.createBuddhaRunner === 'function') {
            const gpuRunner = await mod.createBuddhaRunner({
              useGpu: true,
              workerCount: Math.max(1, DEFAULT_WORKER_COUNT),
              width: targetCanvas.width,
              height: targetCanvas.height,
              maxIter: targetRenderer.max_iter,
              samples: samples,
              renderDelay: buddhaRunner.renderDelay,
              onProgress: buddhaRunner.onProgress,
              onChunk: buddhaRunner.onChunk,
              onComplete: buddhaRunner.onComplete,
              brightness: buddhaRunner.brightness,
              gamma: buddhaRunner.gamma,
            })

            if (abortIfBuddhaRenderCanceled()) {
              gpuRunner?.terminate?.()
              return
            }
            if (!gpuRunner) return

            // GPU 初期化中に別 runner へ変わっていたら差し替えない
            if (buddhaRunner !== createdRunner) {
              return
            }

            // 途中状態を引き継ぎ、GPU 側で続行できるようにする
            const oldRunner = buddhaRunner
            const wasRunning = !!(oldRunner?.running && oldRunner._sent > 0)
            // 旧 runner を止める
            oldRunner?.terminate?.()
            // 差し替える
            buddhaRunner = gpuRunner

            // 旧 runner が動作中だったなら、同等パラメータで GPU runner を開始する
            if (wasRunning) {
              try {
                const params = {
                  samples: oldRunner.samples || 0,
                  maxIter: oldRunner.maxIter || targetRenderer.max_iter,
                  width: oldRunner.width || targetCanvas.width,
                  height: oldRunner.height || targetCanvas.height,
                  center: oldRunner.center || {
                    x: targetRenderer.center[0].toNumber ? targetRenderer.center[0].toNumber() : -0.5,
                    y: targetRenderer.center[1].toNumber ? targetRenderer.center[1].toNumber() : 0,
                  },
                  zoom: oldRunner.zoom || (targetRenderer.zoom.toNumber ? targetRenderer.zoom.toNumber() : 1),
                  supersampling: oldRunner.supersampling || 0,
                  palette: oldRunner.palette || pal,
                  mode: oldRunner.mode || mode,
                  brightness: oldRunner.brightness || brightness,
                  gamma: oldRunner.gamma || gamma,
                  iterationFunction: oldRunner.iterationFunction || target.iterationFunction,
                  fractalType: oldRunner.fractalType || target.fractalType,
                  juliaRe: oldRunner.juliaRe ?? target.juliaRe,
                  juliaIm: oldRunner.juliaIm ?? target.juliaIm,
                }
                await buddhaRunner.start(params)
              } catch (e) {
                console.warn('Error starting GPU buddhaRunner after replacement:', e?.message ? e.message : e)
              }
            }
          }
        }
      } catch (e) {
        console.warn('Could not initialize GPU buddha runner:', e?.message ? e.message : e)
      }
    })()
  }

  // runner の設定を更新して開始する
  // 要求 scale から最終描画サイズを決める
  let scale = 1
  try {
    const scaleSel = document.getElementById('buddha-scale')
    if (scaleSel) scale = Math.max(1, parseInt(scaleSel.value, 10) || 1)
  } catch (_e) {
    scale = 1
  }

  const targetWidth = Math.max(1, Math.round(targetCanvas.width * scale))
  const targetHeight = Math.max(1, Math.round(targetCanvas.height * scale))
  if (abortIfBuddhaRenderCanceled()) return

  if (scale === 1) {
    // 1x 描画では高解像度画像を作らないので、ダウンロードを無効化する
    disableBuddhaDownload()
  }

  // 以前の高解像度ダウンロード状態を消し、古い URL も破棄する
  try {
    const downloadBtn = document.getElementById('buddha-download')
    if (downloadBtn) {
      if (downloadBtn.dataset.hiresBlobUrl) {
        URL.revokeObjectURL(downloadBtn.dataset.hiresBlobUrl)
        delete downloadBtn.dataset.hiresBlobUrl
        delete downloadBtn.dataset.hiresFilename
      }
      downloadBtn.disabled = true
    }
  } catch (e) {
    console.warn('Error resetting buddha-download state before start:', e?.message ? e.message : e)
  }

  buddhaRunner.width = targetWidth
  buddhaRunner.height = targetHeight
  buddhaRunner.maxIter = targetRenderer.max_iter
  let mode = 'buddha'
  const modeEl = document.getElementById('buddhaMode')
  mode = modeEl && typeof modeEl.value !== 'undefined' ? modeEl.value || 'buddha' : 'buddha'
  // Buddhabrot 用進捗を開始する。タスク総数は samples ベースの近似値を使う
  try {
    if (samples > 0) {
      // 既存の逆側進捗表示は開始前に閉じる
      hideInactiveBuddhabrotProgress(target.kind)
      // ProgressMonitor はタスク数前提なので、sample 数をそのまま使う
      getBuddhabrotProgressMonitor(target.kind)?.start(samples)
    }
  } catch (e) {
    console.warn('Error starting progress for buddha samples:', e?.message ? e.message : e)
  }
  // pal は {color, weight} の stop 配列としてワーカーへ渡す
  buddhaRunner.palette = pal
  // アクティブ扱いにして、操作を制限する
  buddhaActive = true
  buddhaRenderPending = false
  // 手動開始時は、フラクタル変更由来のロックを解除する
  buddhaLockedByFractalChange = false
  // view トグルはここでは有効化せず、描画完了時に有効化する

  // GPU 優先なら、start() 前に GPU runner を作って差し替えられるか試す
  if (
    (() => {
      try {
        // Buddhabrot 専用 GPU トグルを優先し、なければ全体トグルを見る
        const bel = document.getElementById('buddha-gpu')
        if (bel) return !!bel.checked
        const el = document.getElementById('gpu')
        return el ? !!el.checked : false
      } catch (_e) {
        return false
      }
    })()
  ) {
    try {
      const mod = await import('./buddhabrot.mjs')
      if (mod && typeof mod.createBuddhaRunner === 'function') {
        const maybeGpu = await mod.createBuddhaRunner({
          useGpu: true,
          workerCount: Math.max(1, DEFAULT_WORKER_COUNT),
          width: targetWidth || targetCanvas.width,
          height: targetHeight || targetCanvas.height,
          maxIter: targetRenderer.max_iter,
          samples: samples,
          onProgress: buddhaRunner?.onProgress,
          onChunk: buddhaRunner?.onChunk,
          onComplete: buddhaRunner?.onComplete,
          brightness: buddhaRunner?.brightness,
          gamma: buddhaRunner?.gamma,
        })
        if (abortIfBuddhaRenderCanceled()) {
          maybeGpu?.terminate?.()
          return
        }
        if (maybeGpu) {
          buddhaRunner.terminate?.()
          buddhaRunner = maybeGpu
        }
      }
    } catch (e) {
      console.warn('Could not initialize GPU runner before start:', e?.message ? e.message : e)
    }
  }

  if (abortIfBuddhaRenderCanceled()) return
  const runnerToStart = buddhaRunner
  if (!runnerToStart || typeof runnerToStart.start !== 'function') {
    console.error('index.js: runnerToStart.start is not available', runnerToStart)
  }
  await runnerToStart.start({
    samples: samples,
    maxIter: targetRenderer.max_iter,
    width: buddhaRunner.width,
    height: buddhaRunner.height,
    center: {
      x: targetRenderer.center[0].toNumber ? targetRenderer.center[0].toNumber() : -0.5,
      y: targetRenderer.center[1].toNumber ? targetRenderer.center[1].toNumber() : 0,
    },
    zoom: targetRenderer.zoom.toNumber ? targetRenderer.zoom.toNumber() : 1,
    // Buddhabrot 専用 supersampling UI があれば読む
    supersampling: targetRenderer.supersampling,
    palette: pal,
    mode: mode,
    fractalType: target.fractalType,
    brightness: brightness,
    gamma: gamma,
    iterationFunction: target.iterationFunction,
    // UI 側で指定された初期 z0 があれば含める
    z0Real: document.getElementById('z0-real') ? parseFloat(document.getElementById('z0-real').value) : undefined,
    z0Imag: document.getElementById('z0-imag') ? parseFloat(document.getElementById('z0-imag').value) : undefined,
    escapeRadius: targetRenderer.escapeRadius !== undefined ? targetRenderer.escapeRadius : 4.0,
    juliaRe: target.juliaRe,
    juliaIm: target.juliaIm,
    gpu: (() => {
      try {
        const bel = document.getElementById('buddha-gpu')
        if (bel) return !!bel.checked
        const el = document.getElementById('gpu')
        return el ? !!el.checked : false
      } catch (_e) {
        return false
      }
    })(),
    // buddhaBandMode は UI 指定を最優先し、なければパレット設定、最後に既定値を使う
    buddhaBandMode: (() => {
      const uiEl = document.getElementById('buddha-bandmode')
      const uiVal = uiEl?.value ? uiEl.value : null
      if (uiVal) return uiVal
      if (pal?.buddhaBandMode) return pal.buddhaBandMode
      return 'perPoint'
    })(),
  })
}

// 実行中の Buddhabrot を停止し、バッファとキャンバスを消去する
/**
 * Buddhabrot の描画状態をすべて停止・消去する
 * @param {boolean} suppressToggleChange - true ならトグル状態を変えない
 */
function stopAndClearBuddha(suppressToggleChange = false) {
  const displayCanvas = getBuddhabrotDisplayCanvas()
  buddhaRunnerGeneration++
  buddhaRedrawScheduled = false
  if (buddhaRunner?.running) {
    buddhaRunner.stop()
  }
  buddhaActive = false
  // 完全消去時は保持表示モードも解除する
  buddhaPreservedDisplay = false

  if (buddhaRunner) {
    // density バッファを消す
    if (buddhaRunner.densityR) buddhaRunner.densityR.fill(0)
    if (buddhaRunner.densityG) buddhaRunner.densityG.fill(0)
    if (buddhaRunner.densityB) buddhaRunner.densityB.fill(0)
  }

  // 再起動トリガーを残さないよう runner を完全に破棄する
  if (buddhaRunner) {
    // terminate() を持たない実装もあるため、存在確認して呼ぶ
    buddhaRunner.terminate?.()
    buddhaRunner = null
  }

  if (lastHighResBuddhaBitmap && typeof lastHighResBuddhaBitmap.close === 'function') {
    lastHighResBuddhaBitmap.close()
  }
  lastHighResBuddhaBitmap = null
  discardSavedBuddhaImageData()
  _lastHighResBuddhaCanvas = null

  const ctx = displayCanvas.getContext('2d')
  ctx.save()
  ctx.fillStyle = 'black'
  ctx.fillRect(0, 0, displayCanvas.width, displayCanvas.height)
  ctx.restore()

  // 停止時のトグル状態を整える
  // 明示的に抑止されていない場合だけトグル更新を行う。
  if (!suppressToggleChange) {
    const toggle = document.getElementById('buddha-toggle')
    if (toggle) {
      // 無効状態で ON に見えないよう、先に OFF にしておく
      toggle.checked = false
      toggle.disabled = true
    }
  }

  // Buddhabrot 消去時は高解像度ダウンロードも無効化する
  disableBuddhaDownload()

  // 進捗が積み上がらないようリセットする
  try {
    finishBuddhabrotProgress(BuddhabrotState.targetKind)
    hideInactiveBuddhabrotProgress(BuddhabrotState.targetKind)
  } catch (e) {
    console.warn('Error finishing progress in stopAndClearBuddha():', e?.message ? e.message : e)
  }
  BuddhabrotState.targetKind = null
}

// Buddhabrot ワーカーだけ止め、表示中のキャンバスは残す
function stopBuddhaPreserveDisplay() {
  const displayCanvas = getBuddhabrotDisplayCanvas()
  if (buddhaRunner?.running) {
    buddhaRunner.stop()
  }

  buddhaActive = false

  // 最終描画を残したまま停止し、後から brightness/gamma だけ再適用できるようにする
  buddhaPreservedDisplay = true

  // キャプチャ前に CPU 側で同期描画し、最新の蓄積結果を画面へ反映させておく
  try {
    if (
      buddhaRunner?.densityR &&
      buddhaRunner.densityR.length > 0 &&
      buddhaRunner.width > 0 &&
      buddhaRunner.height > 0
    ) {
      const rBuf = buddhaRunner.densityR
      const gBuf = buddhaRunner.densityG
      const bBuf = buddhaRunner.densityB
      const width = buddhaRunner.width
      const height = buddhaRunner.height
      const brightness = buddhaRunner.brightness || 1.8
      const gamma = buddhaRunner.gamma || 0.8
      const len = width * height

      // 3 チャンネル合計で最大値を求める
      let max = 0
      for (let i = 0; i < len; i++) {
        const v = (rBuf[i] || 0) + (gBuf[i] || 0) + (bBuf[i] || 0)
        if (v > max) max = v
      }

      if (max > 0) {
        const off = document.createElement('canvas')
        off.width = width
        off.height = height
        const offCtx = off.getContext('2d')
        const img = offCtx.createImageData(width, height)
        const invLogDenom = (1 / Math.log10(1 + max)) * 2
        const denomScale = 1 + Math.log10(1 + max) / 4
        const data = img.data
        for (let i = 0, di = 0; i < len; i++, di += 4) {
          const rv = rBuf[i] || 0
          const gv = gBuf[i] || 0
          const bv = bBuf[i] || 0
          const lr = Math.log10(1 + rv) * invLogDenom
          const lg = Math.log10(1 + gv) * invLogDenom
          const lb = Math.log10(1 + bv) * invLogDenom
          const rn = Math.min(1, ((lr * brightness) / denomScale) ** gamma)
          const gn = Math.min(1, ((lg * brightness) / denomScale) ** gamma)
          const bn = Math.min(1, ((lb * brightness) / denomScale) ** gamma)
          data[di] = (255 * rn) | 0
          data[di + 1] = (255 * gn) | 0
          data[di + 2] = (255 * bn) | 0
          data[di + 3] = 255
        }
	        offCtx.putImageData(img, 0, 0)

	        const mainCtx = displayCanvas.getContext('2d')
	        const scale = Math.min(displayCanvas.width / width, displayCanvas.height / height)
	        const drawW = Math.max(1, Math.round(width * scale))
	        const drawH = Math.max(1, Math.round(height * scale))
	        const dx = Math.round((displayCanvas.width - drawW) / 2)
	        const dy = Math.round((displayCanvas.height - drawH) / 2)
	        mainCtx.clearRect(0, 0, displayCanvas.width, displayCanvas.height)
	        mainCtx.drawImage(off, 0, 0, width, height, dx, dy, drawW, drawH)
	      }
	    }
	  } catch (e) {
	    console.warn('Error flushing density to canvas in stopBuddhaPreserveDisplay():', e?.message ? e.message : e)
  }

  // 今見えている Buddhabrot キャンバスを保存し、後で view 切り替え時に戻せるようにする
  // まず createImageBitmap を優先し、だめなら同期読み戻しへ切り替える。
  const captureTargetKind = BuddhabrotState.targetKind
  const captureGeneration = ++savedBuddhaImageCaptureGeneration
  if (savedBuddhaImageData?.type === 'bitmap' && typeof savedBuddhaImageData.bitmap?.close === 'function') {
    savedBuddhaImageData.bitmap.close()
  }
  savedBuddhaImageData = null
  savedBuddhaImageDataPromise = null
  if (typeof createImageBitmap === 'function') {
    try {
      savedBuddhaImageDataPromise = createImageBitmap(displayCanvas).then((bitmap) => {
        if (captureGeneration !== savedBuddhaImageCaptureGeneration) {
          bitmap.close?.()
          return
        }
        savedBuddhaImageData = {
          type: 'bitmap',
          bitmap: bitmap,
          width: displayCanvas.width,
          height: displayCanvas.height,
          targetKind: captureTargetKind,
        }
        savedBuddhaImageDataPromise = null
      })
    } catch (_err) {
      // 下の同期読み戻しへ切り替える
      savedBuddhaImageDataPromise = null
    }
  }
  // createImageBitmap が使えない、または失敗した場合は同期読み戻しに切り替える
  if (!savedBuddhaImageData && !savedBuddhaImageDataPromise) {
    tempCanvas.width = displayCanvas.width
    tempCanvas.height = displayCanvas.height
    let tempCtx
    try {
      tempCtx = tempCanvas.getContext('2d', { willReadFrequently: true })
    } catch (_err) {
      tempCtx = tempCanvas.getContext('2d')
    }
    tempCtx.clearRect(0, 0, tempCanvas.width, tempCanvas.height)
    tempCtx.drawImage(displayCanvas, 0, 0)
    const img = tempCtx.getImageData(0, 0, tempCanvas.width, tempCanvas.height)
    savedBuddhaImageData = {
      type: 'imageData',
      imageData: img,
      width: tempCanvas.width,
      height: tempCanvas.height,
      targetKind: captureTargetKind,
    }
  }
  // 停止後の UI トグル状態を整える
  try {
    const toggle = document.getElementById('buddha-toggle')
    if (toggle) {
      // 現在の表示は残すがトグルは OFF のままにする。
      // フラクタル種類変更でロックされていない限り、再表示できるよう無効化はしない。
      toggle.checked = false
      if (typeof buddhaLockedByFractalChange !== 'undefined' && buddhaLockedByFractalChange) {
        toggle.disabled = true
      } else {
        toggle.disabled = false
      }
    }
  } catch (e) {
    console.warn('Error disabling buddha toggle in stopBuddhaPreserveDisplay():', e?.message ? e.message : e)
  }
  // 停止中・保持表示中は高解像度ダウンロードを無効化する
  disableBuddhaDownload()

  // 再開時に進捗が積み上がらないようリセットする
  try {
    finishBuddhabrotProgress(BuddhabrotState.targetKind)
    hideInactiveBuddhabrotProgress(BuddhabrotState.targetKind)
  } catch (e) {
    console.warn('Error finishing progress in stopBuddhaPreserveDisplay():', e?.message ? e.message : e)
  }
}

function drawBuddhaDensityChannels(rBuf, gBuf, bBuf, width, height, _pal, brightness = 1.8, gamma = 0.8) {
  // Buddhabrot が非アクティブなら早めに抜ける
  // ただし保持表示モード中は、既存 density バッファの再マッピングだけ許可する
  if (!buddhaActive && !buddhaPreservedDisplay) return
  const displayTargetKind = BuddhabrotState.targetKind || getBuddhabrotTargetSnapshot().kind
  const displayCanvas = getBuddhabrotDisplayCanvas(displayTargetKind)

  // 3 チャンネル合計の最大値を求める
  let max = 0
  const len = width * height
  for (let i = 0; i < len; i++) {
    const v = (rBuf[i] || 0) + (gBuf[i] || 0) + (bBuf[i] || 0)
    if (v > max) max = v
  }
  if (max === 0) return

  // CPU フォールバック描画用の offscreen キャンバスを用意する
  const off = document.createElement('canvas')
  off.width = width
  off.height = height
  const offCtx = off.getContext('2d')

  // 使える場合は GPU 側のカラーマッピングを試す
  let usedGpu = false
  try {
    if (typeof colorMapDensity === 'function') {
      usedGpu = true
      colorMapDensity({
        rBuf,
        gBuf,
        bBuf,
        width,
        height,
        brightness,
        gamma,
	      }).then((bitmap) => {
	        try {
	          _lastHighResBuddhaCanvas = null
	          if (lastHighResBuddhaBitmap && typeof lastHighResBuddhaBitmap.close === 'function') {
	            lastHighResBuddhaBitmap.close()
	          }
	          lastHighResBuddhaBitmap = bitmap
	          const mainCtx = displayCanvas.getContext('2d')
	          mainCtx.save()
	          mainCtx.clearRect(0, 0, displayCanvas.width, displayCanvas.height)
	          mainCtx.imageSmoothingEnabled = true
	          const scale = Math.min(displayCanvas.width / width, displayCanvas.height / height)
	          const drawW = Math.max(1, Math.round(width * scale))
	          const drawH = Math.max(1, Math.round(height * scale))
	          const dx = Math.round((displayCanvas.width - drawW) / 2)
	          const dy = Math.round((displayCanvas.height - drawH) / 2)
	          mainCtx.drawImage(bitmap, 0, 0, width, height, dx, dy, drawW, drawH)
	          mainCtx.restore()
	        } catch (err) {
          console.warn('Error drawing bitmap from colorMapDensity:', err?.message ? err.message : err)
        }
      })
    }
  } catch (e) {
    console.warn('colorMapDensity failed, falling back to CPU path:', e?.message ? e.message : e)
    usedGpu = false
  }

  // CPU フォールバック経路
  if (!usedGpu) {
    const img = offCtx.createImageData(width, height)
    const invLogDenom = (1 / Math.log10(1 + max)) * 2
    // 密度が大きいときは brightness の効きを少し抑え、サンプル増加で急に白飛びしないようにする
    const denomScale = 1 + Math.log10(1 + max) / 4 // tunable divisor
    const data = img.data
    for (let i = 0, di = 0; i < len; i++, di += 4) {
      const rv = rBuf[i] || 0
      const gv = gBuf[i] || 0
      const bv = bBuf[i] || 0
      const lr = Math.log10(1 + rv) * invLogDenom
      const lg = Math.log10(1 + gv) * invLogDenom
      const lb = Math.log10(1 + bv) * invLogDenom
      const rn = Math.min(1, ((lr * brightness) / denomScale) ** gamma)
      const gn = Math.min(1, ((lg * brightness) / denomScale) ** gamma)
      const bn = Math.min(1, ((lb * brightness) / denomScale) ** gamma)
      data[di] = (255 * rn) | 0
      data[di + 1] = (255 * gn) | 0
      data[di + 2] = (255 * bn) | 0
      data[di + 3] = 255
    }

	    offCtx.putImageData(img, 0, 0)
	    _lastHighResBuddhaCanvas = off

	    const mainCtx = displayCanvas.getContext('2d')
	    const srcW = width
	    const srcH = height
	    const dstW = displayCanvas.width
	    const dstH = displayCanvas.height
	    if (srcW === 0 || srcH === 0 || dstW === 0 || dstH === 0) return
    const scale = Math.min(dstW / srcW, dstH / srcH)
    const drawW = Math.max(1, Math.round(srcW * scale))
    const drawH = Math.max(1, Math.round(srcH * scale))
    const dx = Math.round((dstW - drawW) / 2)
    const dy = Math.round((dstH - drawH) / 2)

    try {
      mainCtx.save()
      mainCtx.clearRect(0, 0, dstW, dstH)
      mainCtx.imageSmoothingEnabled = true
      if (typeof createImageBitmap === 'function') {
        createImageBitmap(off).then((bitmap) => {
          if (lastHighResBuddhaBitmap && typeof lastHighResBuddhaBitmap.close === 'function') {
            lastHighResBuddhaBitmap.close()
          }
          lastHighResBuddhaBitmap = bitmap
          mainCtx.drawImage(bitmap, 0, 0, srcW, srcH, dx, dy, drawW, drawH)
          mainCtx.restore()
        })
      } else {
        mainCtx.drawImage(off, 0, 0, srcW, srcH, dx, dy, drawW, drawH)
        mainCtx.restore()
      }
    } catch (_e) {
      mainCtx.drawImage(off, 0, 0, srcW, srcH, dx, dy, drawW, drawH)
      mainCtx.restore()
    }
  }
}

// density バッファがなく、保存済み画像だけがある場合の簡易再マッピング処理。
// brightness / gamma の変更を、保存画像へ近似的に反映して描き直す。
function remapSavedBuddhaImage(brightness, gamma) {
  try {
    if (!savedBuddhaImageData) return
    const displayCanvas = getBuddhabrotDisplayCanvas(savedBuddhaImageData.targetKind)
    // 加工可能な ImageData を取り出す
    let imgData = null
    if (savedBuddhaImageData.type === 'imageData' && savedBuddhaImageData.imageData) {
      // キャッシュ済み画像を壊さないよう複製する
      const src = savedBuddhaImageData.imageData
      imgData = new ImageData(new Uint8ClampedArray(src.data), src.width, src.height)
    } else if (savedBuddhaImageData.type === 'bitmap' && savedBuddhaImageData.bitmap) {
      try {
        tempCanvas.width = savedBuddhaImageData.width
        tempCanvas.height = savedBuddhaImageData.height
        const tctx = tempCanvas.getContext('2d')
        tctx.clearRect(0, 0, tempCanvas.width, tempCanvas.height)
        tctx.drawImage(savedBuddhaImageData.bitmap, 0, 0)
        imgData = tctx.getImageData(0, 0, tempCanvas.width, tempCanvas.height)
      } catch (e) {
        console.warn('Could not extract ImageData from saved bitmap for remap:', e)
        return
      }
    } else {
      return
    }

    const data = imgData.data
    // 各チャンネルへ単純な brightness / gamma 補正をかける
    for (let i = 0; i < data.length; i += 4) {
      for (let c = 0; c < 3; c++) {
        const v = data[i + c] / 255 // 0..1
        // 近似式: pow(v * brightness, gamma)
        let nv = Math.min(1, v * brightness) ** gamma
        nv = Math.max(0, Math.min(1, nv))
        data[i + c] = Math.round(nv * 255)
      }
      // alpha はそのまま維持する
    }

	    // 補正後画像を中央寄せ・拡大縮小してメインキャンバスへ描く
	    try {
	      const ctx = displayCanvas.getContext('2d')
	      ctx.save()
	      ctx.clearRect(0, 0, displayCanvas.width, displayCanvas.height)
	      // 拡大縮小して描画する
	      const scale = Math.min(displayCanvas.width / imgData.width, displayCanvas.height / imgData.height)
	      const drawW = Math.max(1, Math.round(imgData.width * scale))
	      const drawH = Math.max(1, Math.round(imgData.height * scale))
	      const dx = Math.round((displayCanvas.width - drawW) / 2)
	      const dy = Math.round((displayCanvas.height - drawH) / 2)
      // 補正後画像は一時 offscreen キャンバスへ置いてから描く
      const off = document.createElement('canvas')
      off.width = imgData.width
      off.height = imgData.height
      const offCtx = off.getContext('2d')
      offCtx.putImageData(imgData, 0, 0)
      ctx.imageSmoothingEnabled = true
      ctx.drawImage(off, 0, 0, imgData.width, imgData.height, dx, dy, drawW, drawH)
      ctx.restore()
    } catch (e) {
      console.warn('Error drawing remapped saved buddha image:', e)
    }
  } catch (e) {
    console.warn('remapSavedBuddhaImage failed:', e)
  }
}

const fractal = new Mandelbrot(canvasElement, new ProgressMonitor(progressElement), paletteComponent)

// ---------- detail popup state -----------------------------------
let detailPopup = null
let detailEnabled = false
let detailHoverState = null
let detailPinnedState = null
const DETAIL_TOUCH_TOGGLE_DISTANCE = 20

// 少数桁をそろえて表示する簡易フォーマッタ
function _fmt(num) {
  return typeof num === 'number' ? num.toFixed(6) : String(num)
}

/**
 * メインの座標入力と同じ精度ロジックで FxP 座標を十進文字列へ変換する。
 */
function formatFxPCoord(fxp) {
  // updateCoordinateInputs と同じ考え方で zoom 指数を求める
  const zoomBigInt = fractal.zoom.bigIntValue()
  const zoomStr = zoomBigInt.toString()
  const zoomExp = zoomStr.length - 1
  const precisionCap = 6000
  const precision = Math.max(15, Math.min(fractal.precision, zoomExp + 10, precisionCap))
  return fxpToDecimalString(fxp, precision)
}

/**
 * 座標表示と同じ精度方針で実数値を文字列化する。
 * z には FxP 値がないため、計算した桁数で toFixed 相当の表示に寄せる。
 */
function formatFloatWithCoordPrecision(val) {
  const zoomBigInt = fractal.zoom.bigIntValue()
  const zoomStr = zoomBigInt.toString()
  const zoomExp = zoomStr.length - 1
  const precisionCap = 6000
  const precision = Math.max(15, Math.min(fractal.precision, zoomExp + 10, precisionCap))
  if (typeof val !== 'number' || !Number.isFinite(val)) return String(val)
  return val.toFixed(precision)
}

// ── Julia Set State ──────────────────────────────────────────────────────────
const juliaCanvasElement = document.getElementById('julia-canvas')
const juliaProgressElement = document.getElementById('julia-progress-canvas')

const juliaState = {
  active: false,
  /** @type {JuliaRenderer|null} */
  renderer: null,
}

if (juliaCanvasElement) {
  juliaState.renderer = new JuliaRenderer(
    juliaCanvasElement,
    paletteComponent,
    juliaProgressElement ? new ProgressMonitor(juliaProgressElement, null) : null,
  )
  // Julia 側のパレットもメイン側に追従させる。
  // 再計算はせず、既存 offscreen データへの再適用だけ行う。
  paletteComponent.addListener(() => {
    if (juliaState.active && juliaState.renderer) {
      juliaState.renderer.applyPalette()
    }
  })
}

// なめらかな移動用のアニメーション状態と補助関数
const animationState = {
  running: false,
  startCenter: null,
  startZoom: null,
  targetCenter: null,
  targetZoom: null,
  startTime: 0,
  duration: 1000,
  reqId: null,
}

function easeOutCubic(t) {
  return 1 - (1 - t) ** 3
}

// FxP 用の線形補間。補間前に scale をそろえる。
function fxpLerp(a, b, t) {
  // 各 scale と fractal.precision の最大値を採用する
  const scale = Math.max(a.scale, b.scale, fractal.precision || a.scale)
  const aa = a.withScale(scale)
  const bb = b.withScale(scale)
  const delta = bb.subtract(aa)
  const frac = fxp.fromNumber(t, scale)
  return aa.add(delta.multiply(frac))
}

// zoom 用の指数補間。対数空間で補間する。
function _fxpZoomInterp(a, b, t) {
  try {
    const aNum = a.toNumber()
    const bNum = b.toNumber()
    if (!Number.isFinite(aNum) || !Number.isFinite(bNum) || aNum <= 0 || bNum <= 0) throw new Error('bad')
    const ratio = bNum / aNum
    // NaN / Inf を避ける
    const interp = aNum * ratio ** t
    // 適切な scale の FxP として作る
    const scale = Math.max(a.scale, b.scale, fractal.precision || a.scale)
    return fxp.fromNumber(interp, scale)
  } catch (_e) {
    // 数値変換に失敗したら通常の FxP 線形補間へ戻す
    return fxpLerp(a, b, t)
  }
}

// 非常に大きい範囲向けの BigInt ベース FxP 補間
function _fxpLerpBig(a, b, t) {
  const scale = Math.max(a.scale, b.scale, fractal.precision || a.scale)
  const aa = a.withScale(scale)
  const bb = b.withScale(scale)
  const delta = bb.bigInt - aa.bigInt
  const denom = 1000000n // 1e6 resolution for interpolation
  const numer = BigInt(Math.floor(t * Number(denom)))
  const resBig = aa.bigInt + (delta * numer) / denom
  return new fxp.FxP(resBig, scale)
}

function startAnimation(targetCenter, targetZoom, durationMs) {
  // 既存のアニメーションがあれば止める
  stopAnimation()
  // 終了後に戻せるよう、直前の描画設定を保存する
  animationState.prev = {
    useGpu: fractal.useGpu,
    max_iter: fractal.max_iter,
    supersampling: fractal.supersampling,
    smooth: fractal.smooth,
  }

  // 開始値、目標値、時間情報を保存する
  animationState.startCenter = [fractal.center[0], fractal.center[1]]
  animationState.startZoom = fractal.zoom
  animationState.targetCenter = targetCenter
  animationState.targetZoom = targetZoom
  animationState.startTime = performance.now()
  animationState.duration = Math.max(10, durationMs || 1000)
  animationState.running = true

  // モード競合を避けるため、アニメーション中は Julia トグルを無効化する
  try {
    const juliaToggleEl = document.getElementById('julia-toggle')
    if (juliaToggleEl) juliaToggleEl.disabled = true
  } catch (_e) {}
  try {
    const origIter = animationState.prev?.max_iter ? animationState.prev.max_iter : fractal.max_iter
    // アニメーション中の反復回数は、元の半分以上かつ 300 以上、ただし元値は超えない
    const animIter = Math.min(origIter, Math.max(Math.floor(origIter * 0.5), 300))
    fractal.useGpu = false
    fractal.max_iter = animIter
    fractal.supersampling = 0
    fractal.smooth = false
    try {
      fractal.initPallete()
    } catch (_e) {}
  } catch (e) {
    console.warn('Error switching to low-quality mode for animation:', e)
  }

  // パン速度が一定になるよう、ピクセル距離からパン時間を決める
  const PAN_SPEED_PX_PER_SEC = 800 // pan speed in pixels per second (tunable)
  const minPanMs = 700 // minimum pan time
  try {
    const startNumX = animationState.startCenter[0].toNumber()
    const startNumY = animationState.startCenter[1].toNumber()
    const targetNumX = animationState.targetCenter[0].toNumber()
    const targetNumY = animationState.targetCenter[1].toNumber()
    const startZoomNum = animationState.startZoom.toNumber()
    // ピクセル差分は概ね coordDelta * zoom で見積もる
    const dx = (targetNumX - startNumX) * startZoomNum
    const dy = (targetNumY - startNumY) * startZoomNum
    const pixelDist = Math.sqrt(dx * dx + dy * dy)
    let panDuration = Math.max(minPanMs, (pixelDist / PAN_SPEED_PX_PER_SEC) * 1000)
    // ズーム時間を残すため、panDuration は全体の 90% までに抑える
    const maxPan = Math.max(0.1, animationState.duration * 0.9)
    if (panDuration > maxPan) panDuration = maxPan
    // 残り時間を zoomDuration に使う。最低 100ms は確保する
    let zoomDuration = Math.max(100, animationState.duration - panDuration)

    // 開始と終了を少し速めに感じさせるため、二次 easing を使う
    function easeInOutQuad(t) {
      return t < 0.5 ? 2 * t * t : 1 - (-2 * t + 2) ** 2 / 2
    }

    function step(now) {
      const elapsed = now - animationState.startTime
      let finishedEarly = false
      try {
        if (elapsed <= panDuration) {
          // PAN: 一定速度で進めるため、距離割合で線形補間する
          const localT = Math.max(0, Math.min(1, elapsed / panDuration))
          const nx = fxpLerp(animationState.startCenter[0], animationState.targetCenter[0], localT)
          const ny = fxpLerp(animationState.startCenter[1], animationState.targetCenter[1], localT)
          fractal.setCenter([nx, ny])
          fractal.setZoom(animationState.startZoom)
        } else {
          // ZOOM: 中心は目標座標へ固定し、zoom を進める
          fractal.setCenter([animationState.targetCenter[0], animationState.targetCenter[1]])
          const localElapsed = elapsed - panDuration
          const localT = Math.max(0, Math.min(1, localElapsed / zoomDuration))

          // 対数空間で台形プロファイルにし、ズーム深度に依らず最大速度をそろえる
          const sNum = animationState.startZoom.toNumber()
          const eNum = animationState.targetZoom.toNumber()
          if (Number.isFinite(sNum) && sNum > 0 && Number.isFinite(eNum) && eNum > 0) {
            const deltaLog = Math.log(eNum) - Math.log(sNum)
            const absDelta = Math.abs(deltaLog)
            // 対数空間での最大速度
            const MAX_LOG_RATE_PER_MS = 0.002 // ~2.0 per second
            // 加減速に使う短いランプ時間
            const RAMP_MS = 150

            // 最大速度を守るために必要な zoomDuration を計算する
            const computedZoomMs = Math.ceil(absDelta / MAX_LOG_RATE_PER_MS + RAMP_MS)
            // 計算結果の方が長ければ zoomDuration を延長する
            const finalZoomMs = Math.max(zoomDuration, computedZoomMs, 100)
            if (finalZoomMs !== zoomDuration) {
              zoomDuration = finalZoomMs
              animationState.duration = panDuration + zoomDuration
            }

            // ランプ区間と巡航区間を再計算する
            let rampMs = Math.min(RAMP_MS, finalZoomMs / 2)
            let cruiseMs = finalZoomMs - 2 * rampMs
            let maxRate = MAX_LOG_RATE_PER_MS

            // 巡航区間が負になるほど短い場合は、ランプを縮めて最大速度も下げる
            if (cruiseMs < 0) {
              rampMs = finalZoomMs / 2
              cruiseMs = 0
              // 総面積が absDelta になるよう maxRate を逆算する
              maxRate = absDelta / (cruiseMs + rampMs || 1)
            }

            // localElapsed 時点までの対数変化量を積分で求める
            let area = 0
            if (localElapsed <= 0) {
              area = 0
            } else if (localElapsed < rampMs) {
              // 立ち上がり区間
              const t = localElapsed
              area = (maxRate / (2 * rampMs)) * t * t
            } else if (localElapsed < rampMs + cruiseMs) {
              // 立ち上がり完了後の巡航区間
              area = 0.5 * maxRate * rampMs + maxRate * (localElapsed - rampMs)
            } else if (localElapsed < rampMs + cruiseMs + rampMs) {
              // 減速区間
              const t = localElapsed - (rampMs + cruiseMs) // 0..rampMs
              // 減速区間の積分
              area = 0.5 * maxRate * rampMs + maxRate * cruiseMs + maxRate * (t - (t * t) / (2 * rampMs))
            } else {
              // 完了済み
              area = 0.5 * maxRate * rampMs + maxRate * cruiseMs + 0.5 * maxRate * rampMs
            }

            // area を全体変化量に対する割合へ変換する
            const frac = absDelta > 0 ? Math.min(1, area / absDelta) : 1
            // 必要量に達したら早期完了扱いにする
            if (frac >= 1) {
              // 最終 zoom を厳密に合わせ、すぐ終了できるようにする
              fractal.setZoom(animationState.targetZoom)
              finishedEarly = true
            } else {
              const curLog = Math.log(sNum) + Math.sign(deltaLog) * frac * absDelta
              const cur = Math.exp(curLog)
              const nz = fxp.fromNumber(cur, fractal.precision || animationState.startZoom.scale)
              fractal.setZoom(nz)
            }
          } else {
            const le = easeInOutQuad(localT)
            const nz = fxpLerp(animationState.startZoom, animationState.targetZoom, le)
            fractal.setZoom(nz)
          }
        }

        redraw()
      } catch (e) {
        console.warn('Animation step error:', e)
        stopAnimation()
        return
      }

      // zoomDuration 延長の可能性を踏まえて全体進捗を再計算する
      const tGlobal = Math.min(1, (performance.now() - animationState.startTime) / (panDuration + zoomDuration))

      if (!finishedEarly && tGlobal < 1 && animationState.running) {
        animationState.reqId = requestAnimationFrame(step)
      } else {
        // 完了処理
        try {
          if (animationState.prev) {
            // アニメーション後は自動で GPU に戻さず、CPU モードを維持する
            fractal.useGpu = false
            // UI 上の GPU トグルも CPU モードに合わせる
            const gpuToggle = document.getElementById('gpu')
            if (gpuToggle) gpuToggle.checked = false
            fractal.max_iter = animationState.prev.max_iter
            fractal.supersampling = animationState.prev.supersampling
            fractal.smooth = animationState.prev.smooth
            // smooth の状態も UI へ反映する
            try {
              const smoothEl = document.getElementById('smooth')
              if (smoothEl) smoothEl.checked = !!fractal.smooth
            } catch (_e) {}
            fractal.initPallete()
          }
        } catch (e) {
          console.warn('Error restoring render state after animation:', e)
        }
        animationState.running = false
        animationState.reqId = null
        redraw()
        updatePermalink()
        _refreshPinnedOrbits()
      }
    }

    animationState.reqId = requestAnimationFrame(step)
  } catch (e) {
    // パン時間やズーム時間の計算に失敗したら単純な補間へ戻す
    console.warn('Error computing pan duration, falling back to default animation:', e)
    function step(now) {
      const elapsed = now - animationState.startTime
      const t = Math.min(1, elapsed / animationState.duration)
      const le = easeOutCubic(t)
      try {
        const nx = fxpLerp(animationState.startCenter[0], animationState.targetCenter[0], le)
        const ny = fxpLerp(animationState.startCenter[1], animationState.targetCenter[1], le)
        const nz = fxpLerp(animationState.startZoom, animationState.targetZoom, le)
        fractal.setCenter([nx, ny])
        fractal.setZoom(nz)
        redraw()
      } catch (_err) {
        stopAnimation()
        return
      }
      if (t < 1 && animationState.running) animationState.reqId = requestAnimationFrame(step)
      else {
        animationState.running = false
        animationState.reqId = null
        redraw()
        updatePermalink()
        _refreshPinnedOrbits()
      }
    }
    animationState.reqId = requestAnimationFrame(step)
  }
}

function stopAnimation() {
  if (animationState.reqId) {
    cancelAnimationFrame(animationState.reqId)
    animationState.reqId = null
  }
  // 保存しておいた描画状態があれば戻す
  try {
    if (animationState.prev) {
      // 手動停止時も CPU モードを維持する
      fractal.useGpu = false
      const gpuToggle = document.getElementById('gpu')
      if (gpuToggle) gpuToggle.checked = false
      fractal.max_iter = animationState.prev.max_iter
      fractal.supersampling = animationState.prev.supersampling
      fractal.smooth = animationState.prev.smooth
      try {
        const smoothEl = document.getElementById('smooth')
        if (smoothEl) smoothEl.checked = !!fractal.smooth
      } catch (_e) {}
      fractal.initPallete()
    }
  } catch (e) {
    console.warn('Error restoring render state on stopAnimation:', e)
  }
  animationState.running = false
  // Julia トグルを再度有効化する
  try {
    const juliaToggleEl = document.getElementById('julia-toggle')
    if (juliaToggleEl) juliaToggleEl.disabled = false
  } catch (_e) {}
  // 元の品質設定で再描画する
  redraw()
}

/**
 * キャンバス操作補助
 */
const CanvasHelpers = {
  /**
   * ImageData または Bitmap を安全にキャンバスへ戻す
   * @param {CanvasRenderingContext2D} ctx - キャンバスコンテキスト
   * @param {Object} imageData - 保存画像オブジェクト
   * @param {number} canvasWidth - キャンバス幅
   * @param {number} canvasHeight - キャンバス高
   */
  restoreImage(ctx, imageData, canvasWidth, canvasHeight) {
    try {
      if (imageData.type === 'imageData' && imageData.imageData) {
        ctx.putImageData(imageData.imageData, 0, 0)
      } else if (imageData.type === 'bitmap' && imageData.bitmap) {
        this.drawBitmapScaled(ctx, imageData, canvasWidth, canvasHeight)
      } else if (imageData.imageData) {
        ctx.putImageData(imageData.imageData, 0, 0)
      }
    } catch (e) {
      console.warn('Error restoring image to canvas:', e)
    }
  },

  /**
   * Bitmap をキャンバス内に収まるよう拡大縮小して描く
   */
  drawBitmapScaled(ctx, imageData, canvasWidth, canvasHeight) {
    ctx.save()
    ctx.clearRect(0, 0, canvasWidth, canvasHeight)
    ctx.imageSmoothingEnabled = true

    const scale = Math.min(canvasWidth / imageData.width, canvasHeight / imageData.height)

    const drawW = Math.max(1, Math.round(imageData.width * scale))
    const drawH = Math.max(1, Math.round(imageData.height * scale))
    const dx = Math.round((canvasWidth - drawW) / 2)
    const dy = Math.round((canvasHeight - drawH) / 2)

    ctx.drawImage(imageData.bitmap, 0, 0, imageData.width, imageData.height, dx, dy, drawW, drawH)

    ctx.restore()
  },
}

/**
 * 入力値検証ユーティリティ
 */
const _Validators = {
  /**
   * 数値が有限で有効かを確認する
   */
  isValidNumber(value) {
    return typeof value === 'number' && Number.isFinite(value)
  },

  /**
   * 文字列が妥当な十進数表現かを確認する
   */
  isValidDecimalString(str) {
    if (!str || typeof str !== 'string') return false
    const trimmed = str.trim()
    return /^-?\d*\.?\d+([eE][+-]?\d+)?$/.test(trimmed)
  },

  /**
   * 整数入力を解析して範囲チェックする
   */
  parseInteger(value, defaultValue = 0, min = -Infinity, max = Infinity) {
    const parsed = parseInt(value, 10)
    if (!Number.isFinite(parsed)) return defaultValue
    return Math.max(min, Math.min(max, parsed))
  },

  /**
   * 実数入力を解析して範囲チェックする
   */
  parseFloat(value, defaultValue = 0, min = -Infinity, max = Infinity) {
    const parsed = parseFloat(value)
    if (!Number.isFinite(parsed)) return defaultValue
    return Math.max(min, Math.min(max, parsed))
  },
}

/**
 * フラクタル状態比較ユーティリティ
 */
const FractalStateHelpers = {
  /**
   * 比較用に現在のフラクタル状態を取得する
   */
  getCurrentState(targetKind = 'main') {
    const isJuliaTarget = targetKind === 'julia' && juliaState?.renderer
    const renderer = isJuliaTarget ? juliaState.renderer : fractal
    const juliaC = isJuliaTarget ? getCurrentJuliaCFromMain() : null
    const juliaType = fractal.fractalType === 'custom' ? 'julia-custom' : 'julia'
    const juliaIterationFunction = fractal.fractalType === 'custom' ? fractal.iterationFunction : 'z*z + c'
    const curCenterX = fxpToDecimalString(renderer.center[0], 30)
    const curCenterY = fxpToDecimalString(renderer.center[1], 30)

    return {
      targetKind: isJuliaTarget ? 'julia' : 'main',
      iterationFunction: isJuliaTarget ? juliaIterationFunction : renderer.iterationFunction || '',
      fractalType: isJuliaTarget ? juliaType : renderer.fractalType || '',
      centerStr: `${curCenterX}|${curCenterY}`,
      zoomStr: fxpToDecimalString(renderer.zoom, 30),
      paletteId: paletteComponent?.palette?.id || null,
      paletteDensity: paletteComponent ? String(paletteComponent.density) : null,
      paletteRotate: paletteComponent ? String(paletteComponent.rotate) : null,
      supersampling: isJuliaTarget ? fractal.supersampling : renderer.supersampling,
      fullres: DOM.app?.querySelector('#fullres')?.checked || false,
      escapeRadius: isJuliaTarget ? fractal.escapeRadius : renderer.escapeRadius,
      juliaCStr: isJuliaTarget ? `${juliaC.re}|${juliaC.im}` : null,
    }
  },

  /**
   * 保存状態と現在状態が一致するか比較する
   */
  stateMatches(savedState, currentState) {
    return (
      savedState.targetKind === currentState.targetKind &&
      savedState.iterationFunction === currentState.iterationFunction &&
      savedState.fractalType === currentState.fractalType &&
      savedState.centerStr === currentState.centerStr &&
      savedState.zoomStr === currentState.zoomStr &&
      savedState.paletteId === currentState.paletteId &&
      savedState.paletteDensity === currentState.paletteDensity &&
      savedState.paletteRotate === currentState.paletteRotate &&
      savedState.supersampling === currentState.supersampling &&
      savedState.fullres === currentState.fullres &&
      savedState.escapeRadius === currentState.escapeRadius &&
      savedState.juliaCStr === currentState.juliaCStr
    )
  },
}

// 通常フラクタル表示の画像を保持し、Buddhabrot を閉じたときに
// 再描画せず元へ戻せるようにする。
// savedFractalImageData には次のオブジェクトを入れる:
// { imageData: ImageData, iterationFunction: string, fractalType: string }
let savedFractalImageData = null

// 最後に保持した Buddhabrot 表示を保存し、表示のオンオフ時に
// 再描画せず復元できるようにする。保持内容は
// { imageData: ImageData, width, height }。
let savedBuddhaImageData = null
// savedBuddhaImageData 用の createImageBitmap 完了待ち Promise
let savedBuddhaImageDataPromise = null
// 古い非同期キャプチャが後から完了して保存画像を巻き戻さないようにする
let savedBuddhaImageCaptureGeneration = 0

// ダウンロード用に直近の高解像度 Buddhabrot キャンバスを保持する
let _lastHighResBuddhaCanvas = null
// 描画に使った直近の ImageBitmap を保持し、差し替え時に close できるようにする
let lastHighResBuddhaBitmap = null
// 破棄済み runner から飛んでくる遅延描画を無効化するための世代番号
let buddhaRunnerGeneration = 0
// runner 生成前の非同期 Buddhabrot 開始リクエストをキャンセルするための世代番号
let buddhaRenderRequestGeneration = 0
let buddhaRenderPending = false

function disableBuddhaDownload() {
  const btn = document.getElementById('buddha-download')
  if (!btn) return
  if (btn.dataset.hiresBlobUrl) {
    URL.revokeObjectURL(btn.dataset.hiresBlobUrl)
    delete btn.dataset.hiresBlobUrl
    delete btn.dataset.hiresFilename
  }
  btn.disabled = true
}

// 明るさとガンマの連続更新をまとめ、高解像度での重い再描画を減らす
let buddhaRedrawScheduled = false
function scheduleBuddhaRedraw() {
  if (buddhaRedrawScheduled) return
  buddhaRedrawScheduled = true
  const generation = buddhaRunnerGeneration
  requestAnimationFrame(() => {
    buddhaRedrawScheduled = false
    // Buddhabrot 表示がオフなら再描画しない。
    const toggle = document.getElementById('buddha-toggle')
    if (toggle && !toggle.checked) return

    const runner = buddhaRunner
    if (!runner || buddhaRunnerGeneration !== generation) return
    drawBuddhaDensityChannels(
      runner.densityR,
      runner.densityG,
      runner.densityB,
      runner.width,
      runner.height,
      runner.palette,
      runner.brightness,
      runner.gamma,
    )
  })
}

let redrawTimeout = null

const COORDINATE_GRID_MAJOR_TARGET_PX = 240
const COORDINATE_GRID_MINOR_DIVISIONS = 10
const COORDINATE_GRID_MAX_LINES = 600

function _getCoordinateGridCanvas(isJulia = false) {
  return document.getElementById(isJulia ? 'julia-coordinate-grid-canvas' : 'coordinate-grid-canvas')
}

function _clearCoordinateGridCanvas(isJulia = false) {
  const overlayCanvas = _getCoordinateGridCanvas(isJulia)
  const ownerCanvas = isJulia ? document.getElementById('julia-canvas') : canvasElement
  const size = _syncOverlayCanvasToDisplay(overlayCanvas, ownerCanvas)
  if (size) overlayCanvas.getContext('2d').clearRect(0, 0, size[0], size[1])
}

function _chooseCoordinateGridMajorStep(scalePxPerUnit) {
  if (!Number.isFinite(scalePxPerUnit) || scalePxPerUnit <= 0) return 1
  const rawStep = COORDINATE_GRID_MAJOR_TARGET_PX / scalePxPerUnit
  const exponent = Math.floor(Math.log10(rawStep))
  const base = 10 ** exponent
  const normalized = rawStep / base
  const nice = normalized <= 1.5 ? 1 : normalized <= 3 ? 2 : normalized <= 7 ? 5 : 10
  return nice * base
}

function _formatCoordinateGridNumber(value, step, suffix = '') {
  if (!Number.isFinite(value)) return ''
  const threshold = Math.max(Math.abs(step) * 1e-8, 1e-12)
  let v = Math.abs(value) < threshold ? 0 : value
  if (Math.abs(v) >= 10000 || (Math.abs(v) > 0 && Math.abs(v) < 0.001)) {
    return `${v.toExponential(1).replace('+', '')}${suffix}`
  }
  const decimals = step >= 1 ? 0 : Math.min(12, Math.ceil(-Math.log10(step)) + 1)
  let text = v.toFixed(decimals)
  text = text.replace(/(\.\d*?[1-9])0+$/, '$1').replace(/\.0+$/, '')
  if (text === '-0') text = '0'
  return `${text}${suffix}`
}

function _drawCoordinateGridLabel(ctx, text, x, y) {
  if (!text) return
  ctx.save()
  ctx.lineWidth = 3
  ctx.lineJoin = 'round'
  ctx.strokeStyle = 'rgba(0, 0, 0, 0.85)'
  ctx.strokeText(text, x, y)
  ctx.fillStyle = 'rgba(235, 240, 255, 0.88)'
  ctx.fillText(text, x, y)
  ctx.restore()
}

function _strokeCoordinateGridPath(ctx, lineWidth, lineColor, outlineWidth = lineWidth + 0.75) {
  ctx.save()
  ctx.lineWidth = outlineWidth
  ctx.strokeStyle = 'rgba(0, 0, 0, 0.12)'
  ctx.stroke()
  ctx.lineWidth = lineWidth
  ctx.strokeStyle = lineColor
  ctx.stroke()
  ctx.restore()
}

function _forEachCoordinateGridLine(minValue, maxValue, step, callback) {
  if (!Number.isFinite(minValue) || !Number.isFinite(maxValue) || !Number.isFinite(step) || step <= 0) return
  const span = maxValue - minValue
  if (!Number.isFinite(span) || span <= 0) return
  let effectiveStep = step
  while (span / effectiveStep > COORDINATE_GRID_MAX_LINES) {
    effectiveStep *= 2
  }
  const first = Math.ceil((minValue - effectiveStep * 1e-8) / effectiveStep) * effectiveStep
  for (let value = first, count = 0; value <= maxValue + effectiveStep * 1e-8 && count < COORDINATE_GRID_MAX_LINES; value += effectiveStep, count++) {
    callback(Math.abs(value) < effectiveStep * 1e-8 ? 0 : value)
  }
}

function _syncOverlayCanvasToDisplay(overlayCanvas, ownerCanvas) {
  if (!overlayCanvas || !ownerCanvas) return null
  const width = Math.max(1, Math.round(overlayCanvas.offsetWidth || ownerCanvas.offsetWidth || ownerCanvas.width || 1))
  const height = Math.max(1, Math.round(overlayCanvas.offsetHeight || ownerCanvas.offsetHeight || ownerCanvas.height || 1))
  if (overlayCanvas.width !== width) overlayCanvas.width = width
  if (overlayCanvas.height !== height) overlayCanvas.height = height
  return [width, height]
}

function _drawCoordinateGridForView(overlayCanvas, ownerCanvas, center, zoom) {
  const size = _syncOverlayCanvasToDisplay(overlayCanvas, ownerCanvas)
  if (!size) return
  const [width, height] = size
  const ctx = overlayCanvas.getContext('2d')
  ctx.clearRect(0, 0, width, height)

  const zoomValue = zoom?.toNumber ? zoom.toNumber() : Number(zoom)
  if (!Number.isFinite(zoomValue) || zoomValue <= 0) return
  const centerX = center?.[0]?.toNumber ? center[0].toNumber() : Number(center?.[0] ?? 0)
  const centerY = center?.[1]?.toNumber ? center[1].toNumber() : Number(center?.[1] ?? 0)
  if (!Number.isFinite(centerX) || !Number.isFinite(centerY)) return

  const scale = (zoomValue * width) / 4
  const minX = centerX - width / (2 * scale)
  const maxX = centerX + width / (2 * scale)
  const minY = centerY - height / (2 * scale)
  const maxY = centerY + height / (2 * scale)
  const majorStep = _chooseCoordinateGridMajorStep(scale)
  const minorStep = majorStep / COORDINATE_GRID_MINOR_DIVISIONS
  const axisEpsilon = minorStep * 1e-6
  const toScreenX = (x) => width / 2 + (x - centerX) * scale
  const toScreenY = (y) => height / 2 + (y - centerY) * scale

  ctx.save()
  ctx.lineCap = 'butt'

  ctx.beginPath()
  _forEachCoordinateGridLine(minX, maxX, minorStep, (x) => {
    if (Math.abs(x / majorStep - Math.round(x / majorStep)) < 1e-6) return
    const sx = Math.round(toScreenX(x)) + 0.5
    ctx.moveTo(sx, 0)
    ctx.lineTo(sx, height)
  })
  _forEachCoordinateGridLine(minY, maxY, minorStep, (y) => {
    if (Math.abs(y / majorStep - Math.round(y / majorStep)) < 1e-6) return
    const sy = Math.round(toScreenY(y)) + 0.5
    ctx.moveTo(0, sy)
    ctx.lineTo(width, sy)
  })
  _strokeCoordinateGridPath(ctx, 1, 'rgba(255, 255, 255, 0.07)', 1.25)

  ctx.beginPath()
  _forEachCoordinateGridLine(minX, maxX, majorStep, (x) => {
    if (Math.abs(x) < axisEpsilon) return
    const sx = Math.round(toScreenX(x)) + 0.5
    ctx.moveTo(sx, 0)
    ctx.lineTo(sx, height)
  })
  _forEachCoordinateGridLine(minY, maxY, majorStep, (y) => {
    if (Math.abs(y) < axisEpsilon) return
    const sy = Math.round(toScreenY(y)) + 0.5
    ctx.moveTo(0, sy)
    ctx.lineTo(width, sy)
  })
  _strokeCoordinateGridPath(ctx, 0.9, 'rgba(255, 255, 255, 0.16)', 1.6)

  ctx.beginPath()
  const yAxisX = toScreenX(0)
  const xAxisY = toScreenY(0)
  if (yAxisX >= 0 && yAxisX <= width) {
    const sx = Math.round(yAxisX) + 0.5
    ctx.moveTo(sx, 0)
    ctx.lineTo(sx, height)
  }
  if (xAxisY >= 0 && xAxisY <= height) {
    const sy = Math.round(xAxisY) + 0.5
    ctx.moveTo(0, sy)
    ctx.lineTo(width, sy)
  }
  _strokeCoordinateGridPath(ctx, 1.2, 'rgba(255, 255, 255, 0.42)', 2)

  ctx.font = '11px ui-monospace, SFMono-Regular, Menlo, Consolas, monospace'
  ctx.textBaseline = 'middle'
  const labelY = Math.min(height - 12, Math.max(12, xAxisY + 12))
  const labelX = Math.min(width - 44, Math.max(4, yAxisX + 6))

  ctx.textAlign = 'center'
  _forEachCoordinateGridLine(minX, maxX, majorStep, (x) => {
    const sx = toScreenX(x)
    if (sx < -20 || sx > width + 20) return
    _drawCoordinateGridLabel(ctx, _formatCoordinateGridNumber(x, majorStep), sx, labelY)
  })

  ctx.textAlign = 'left'
  _forEachCoordinateGridLine(minY, maxY, majorStep, (y) => {
    if (Math.abs(y) < axisEpsilon) return
    const sy = toScreenY(y)
    if (sy < -12 || sy > height + 12) return
    _drawCoordinateGridLabel(ctx, _formatCoordinateGridNumber(-y, majorStep, 'i'), labelX, sy)
  })

  ctx.restore()
}

function drawMainCoordinateGrid() {
  if (!fractal) return
  if (!coordinateGridEnabled) {
    _clearCoordinateGridCanvas(false)
    return
  }
  const overlayCanvas = _getCoordinateGridCanvas(false)
  const view = typeof _getMainOrbitRenderView === 'function' ? _getMainOrbitRenderView() : { center: fractal.center, zoom: fractal.zoom }
  _drawCoordinateGridForView(overlayCanvas, canvasElement, view.center, view.zoom)
}

function drawJuliaCoordinateGrid() {
  const renderer = juliaState?.renderer
  const ownerCanvas = renderer?.canvas || document.getElementById('julia-canvas')
  const overlayCanvas = _getCoordinateGridCanvas(true)
  if (!coordinateGridEnabled) {
    _clearCoordinateGridCanvas(true)
    return
  }
  if (!juliaState?.active || !renderer) {
    const size = _syncOverlayCanvasToDisplay(overlayCanvas, ownerCanvas)
    if (size) overlayCanvas.getContext('2d').clearRect(0, 0, size[0], size[1])
    return
  }
  _drawCoordinateGridForView(overlayCanvas, ownerCanvas, renderer.center, renderer.zoom)
}

function refreshCoordinateGrid() {
  drawMainCoordinateGrid()
  drawJuliaCoordinateGrid()
}

/**
 * フラクタルを再描画する
 * @async
 * @param {boolean} resetCaches - アルゴリズムのキャッシュを初期化するか
 * @param {number} cooldown - 再描画前に待つ任意のミリ秒
 */
async function redraw(resetCaches, cooldown) {
  // 明示的に resetCaches が指定されたときだけ Buddhabrot を停止・消去する。
  try {
    if (resetCaches) stopAndClearBuddha()
  } catch (e) {
    ErrorHelpers.warn('Error stopping/clearing buddhabrot in redraw():', e)
  }
  showZoomFactor()
  if (redrawTimeout) {
    clearTimeout(redrawTimeout)
    redrawTimeout = null
  }

  // Buddhabrot 実行中で reset を要求していない場合は、進行中の処理や表示を
  // 壊さないよう通常の再描画を行わない。resetCaches が true なら、先に停止と
  // 消去を済ませてから通常描画へ進める。
  if (buddhaActive && !resetCaches && !isBuddhabrotViewShownOnJulia()) {
    return
  }

  if (cooldown) {
    redrawTimeout = setTimeout(() => {
      fractal.render(resetCaches)
      redrawTimeout = null
      // Julia が有効なら待機後にそちらも再描画する
      if (juliaState.active && !isJuliaCanvasInteractionBlockedByBuddhabrot()) redrawJulia()
    }, cooldown)
  } else {
    await fractal.render(resetCaches)
    // Julia が有効なら、現在の中心を c としてあわせて描画する
    if (juliaState.active && !isJuliaCanvasInteractionBlockedByBuddhabrot()) redrawJulia()
  }
}

function showZoomFactor() {
  // 座標入力欄を現在値に更新する
  updateCoordinateInputs()
  refreshCoordinateGrid()
}

/**
 * 現在の Mandelbrot 中心を c として Julia 集合を描画する。
 * 反復回数、スムーズ描画、パレット設定はメイン表示と同期する。
 */
function redrawJulia() {
  if (!juliaState.active || !juliaState.renderer) return
  if (isJuliaCanvasInteractionBlockedByBuddhabrot()) return
  const synced = syncJuliaRendererSettingsFromMain()
  if (!synced) return
  const renderer = synced.renderer
  // ピクセルサイズは resizeToCanvasSize() 済みなので、オフスクリーンだけ確認する
  const jCanvas = document.getElementById('julia-canvas')
  if (jCanvas && jCanvas.width > 0 && jCanvas.height > 0) {
    if (
      !renderer.offscreens ||
      renderer.offscreens.length === 0 ||
      renderer.width !== jCanvas.width ||
      renderer.height !== jCanvas.height
    ) {
      renderer.resized()
    }
  }
  // 以前はメイン表示とズームを同期していたが、現在は Julia 側で独立管理する
  drawJuliaCoordinateGrid()
  renderer.render()
}

/**
 * 十進数文字列をズーム入力欄で使う指数表記へ整形する。
 */
function formatDecimalToScientific(decimalStr) {
  decimalStr = (decimalStr || '').trim()
  if (!decimalStr) return '0.0e0'
  let sign = ''
  if (decimalStr[0] === '-') {
    sign = '-'
    decimalStr = decimalStr.substring(1)
  }

  if (/^0+(?:\.0*)?$/.test(decimalStr)) return `${sign}0.0e0`

  let intPart = decimalStr
  let fracPart = ''
  const dotIndex = decimalStr.indexOf('.')
  if (dotIndex !== -1) {
    intPart = decimalStr.substring(0, dotIndex)
    fracPart = decimalStr.substring(dotIndex + 1)
  }

  intPart = intPart.replace(/^0+/, '')

  if (intPart.length > 0) {
    const all = intPart + fracPart
    const exp = intPart.length - 1
    const first = all[0] || '0'
    const second = all[1] || '0'
    return `${sign}${first}.${second}e${exp}`
  }

  const nz = fracPart.match(/[^0]/)
  if (!nz) return `${sign}0.0e0`
  const idx = nz.index
  const exp = -(idx + 1)
  const first = fracPart[idx] || '0'
  const second = fracPart[idx + 1] || '0'
  return `${sign}${first}.${second}e${exp}`
}

function formatZoomValueForInput(zoom) {
  return formatDecimalToScientific(fxpToDecimalString(zoom, PRECISION_CONSTANTS.DECIMAL_STRING_PRECISION))
}

/**
 * 現在のフラクタル状態を座標入力欄へ反映する
 */
function updateCoordinateInputs() {
  // ズーム量に応じて必要な桁数を決める
  // 高精度のため、まず zoom を BigInt として取得する
  const zoomBigInt = fractal.zoom.bigIntValue()
  const zoomStr = zoomBigInt.toString()
  const zoomExp = zoomStr.length - 1

  // 整形済みの zoom 表記を使う
  const zoomFormatted = formatZoomValueForInput(fractal.zoom)

  // zoom の指数と現在の精度から、入力欄へ表示する桁数を決める。
  // 極端なズームでは内部の fractal.precision を優先しつつ、変換が重くなりすぎない
  // よう上限を設ける。これで深いズーム時の微小な座標差も入力欄に見える。
  const precisionCap = 6000 // 十進変換で安全な上限桁数
  // 中心座標入力欄に何桁の小数を表示するか決める。
  const precision = Math.max(15, Math.min(fractal.precision, zoomExp + 10, precisionCap))

  // FxP を高精度の十進文字列へ変換する
  const x = fxpToDecimalString(fractal.center[0], precision)
  const y = fxpToDecimalString(fractal.center[1], precision)

  // スクリーン Y 軸（下方向が正）と数学の虚数軸（上方向が正）は逆向きのため、
  // UI 表示では符号を反転して「画面上 = 正の虚数」になるよう変換する。
  const displayY = negateDecimalString(y)

  // 現在値で入力欄を更新する
  document.getElementById('coordX').value = x
  document.getElementById('coordY').value = displayY
  document.getElementById('coordZoom').value = zoomFormatted
}

/**
 * 十進数文字列の符号を反転して返す。
 * 例: '-1.23' → '1.23' / '0.5' → '-0.5' / '0' → '0'
 * スクリーン Y 軸（下が正）と数学的虚数軸（上が正）の変換に使用する。
 */
function negateDecimalString(s) {
  s = s.trim()
  if (!s || s === '0') return '0'
  return s.startsWith('-') ? s.slice(1) : `-${s}`
}

// FxP を指定精度の十進文字列へ変換する
function fxpToDecimalString(fxp, precision) {
  const bigInt = fxp.bigInt
  const scale = fxp.scale

  // 0 はそのまま返す
  if (bigInt === 0n) {
    return '0'
  }

  // 符号を分離して扱う
  const isNegative = bigInt < 0n
  const absBigInt = isNegative ? -bigInt : bigInt

  // 整数部: bigInt / 2^scale
  const intPart = absBigInt >> BigInt(scale)

  // 小数部: (bigInt % 2^scale) / 2^scale
  const fracMask = (1n << BigInt(scale)) - 1n
  let fracPart = absBigInt & fracMask

  // 小数部がなければ整数だけ返す
  if (fracPart === 0n) {
    return isNegative ? `-${intPart.toString()}` : intPart.toString()
  }

  // 小数部を十進桁へ変換する
  // 必要なら十分な精度まで伸ばして小さい値も拾う
  let fracStr = ''
  // maxDigits は小数部を文字列化する際のループ上限を制御する。
  // precision には表示すべき有効桁数が既に反映されているので、それを基準とする。
  const maxDigits = precision

  for (let i = 0; i < maxDigits; i++) {
    fracPart *= 10n
    const digit = fracPart >> BigInt(scale)
    fracStr += digit.toString()
    fracPart &= fracMask

    // 十分なら早めに打ち切る
    if (fracPart === 0n) {
      break
    }

    // 非ゼロの有効桁が十分なら止める
    if (i >= precision) {
      const nonZeroMatch = fracStr.match(/[1-9]/)
      if (nonZeroMatch) {
        const firstNonZeroPos = nonZeroMatch.index
        const significantDigits = fracStr.length - firstNonZeroPos
        if (significantDigits >= 15) {
          break // 必要な有効桁を確保できた
        }
      }
    }
  }

  // 末尾の 0 を落とす
  fracStr = fracStr.replace(/0+$/, '')

  // 結果を整形する
  let result = intPart.toString()
  if (fracStr.length > 0) {
    result += `.${fracStr}`
  }

  return isNegative ? `-${result}` : result
}

let lastX = canvasElement.width / 2
let lastY = canvasElement.height / 2
let dragStart = null
let juliaDragStart = null
let juliaLastTouchDistance = null
let juliaLastTouchCenter = null
let gpuInteractiveRedrawTimer = null
let gpuInteractiveFinalizeTimer = null
let gpuInteractiveRedrawPending = false
let gpuInteractiveNeedsFinalRedraw = false
let lastGpuInteractiveRedrawAt = 0
let pendingInteractivePanDx = 0
let pendingInteractivePanDy = 0
let pendingInteractivePinchView = null
let pendingInteractiveTransformScale = 1
let pendingInteractiveTransformTx = 0
let pendingInteractiveTransformTy = 0
let orbitDrawEnabled = false // 軌道表示が有効か
let orbitMode = 'lines+dots' // 'lines+dots' | 'lines' | 'dots'
let orbitGradientEnabled = false
let coordinateGridEnabled = false
// 軌道の固定表示。キャンバスをクリックすると、その点に軌道を固定する。
// 固定中は {clientX, clientY}、未固定なら null。
let pinnedOrbit = null
let juliaPinnedOrbit = null
let orbitTouchTapCandidate = false
let juliaOrbitTouchTapCandidate = false
let suppressOrbitClickUntil = 0
let suppressJuliaOrbitClickUntil = 0
// ドラッグ検出用。mousedown→mousemove のドラッグが起きたら true にし、
// ドラッグ直後の click で固定の切り替えが起きないようにする。
let _orbitPinDragged = false
let _juliaPinDragged = false
const TOUCH_CLICK_SUPPRESS_MS = 700
const GPU_INTERACTIVE_REDRAW_INTERVAL_MS = 50
const GPU_WHEEL_FINAL_REDRAW_DELAY_MS = 150
// let dragged = false

function isMainRenderGpuPath() {
  return !!fractal?._willUseGpuForCurrentRender?.()
}

function shouldRedrawMainGpuDuringInteraction() {
  return fractal?.fractalType !== 'mandelbrot' || shouldUseDirectMandelbrotGpu(fractal)
}

function getMainGpuRenderScreenForState(renderer) {
  const screens = renderer?.offscreens || []
  const finalScreen = screens[screens.length - 1]
  if (
    !renderer?.currentInteractionGpuRedraw ||
    renderer.fractalType !== 'mandelbrot' ||
    (!fullResUsesPhysicalPixels() && !document.fullscreenElement)
  ) {
    return finalScreen
  }

  const targetScale = Math.max(2, getWindowDevicePixelRatio())
  let selected = finalScreen
  let selectedScore = Number.POSITIVE_INFINITY
  for (const screen of screens) {
    if (!screen || screen.scale <= 1) continue
    const score = Math.abs(screen.scale - targetScale)
    if (score < selectedScore || (score === selectedScore && screen.scale < selected.scale)) {
      selected = screen
      selectedScore = score
    }
  }
  return selected || finalScreen
}

function cancelActiveMainRender() {
  try {
    const hadActiveJob = !!fractal.jobToken
    fractal._revokeJobToken?.()
    if (Array.isArray(fractal.taskqueue)) fractal.taskqueue.length = 0
    if (fractal.mandelbrotGpu) fractal.mandelbrotGpu.newTask = null
    if (fractal.mandelbrotCustomGpu) fractal.mandelbrotCustomGpu.newTask = null
    if (fractal.orbitTrapGpu) fractal.orbitTrapGpu.newTask = null
    if (hadActiveJob) fractal.progress?.finish?.({ showRenderTime: false })
  } catch (e) {
    console.warn('Error canceling active fractal render:', e?.message ? e.message : e)
  }
}

function cancelActiveJuliaRender() {
  try {
    const renderer = juliaState?.renderer
    if (!renderer) return
    const hadActiveJob = !!renderer.jobToken
    renderer._revokeJobToken?.()
    if (Array.isArray(renderer.taskqueue)) renderer.taskqueue.length = 0
    if (renderer.juliaGpu) renderer.juliaGpu.newTask = null
    if (renderer.orbitTrapGpu) renderer.orbitTrapGpu.newTask = null
    if (hadActiveJob) renderer.progress?.finish?.({ showRenderTime: false })
  } catch (e) {
    console.warn('Error canceling active Julia render:', e?.message ? e.message : e)
  }
}

function stopRenderingForJuliaToggleDuringBuddhabrot() {
  const shouldClearBuddhabrot = hasVisibleOrRunningBuddhabrot()
  const hadBuddhabrotWork = buddhaRenderPending || shouldClearBuddhabrot
  if (!hadBuddhabrotWork) return false

  // startBuddhaRender() が runner 生成前の await 中でも、続きで描画を始めないようにする。
  buddhaRenderRequestGeneration++
  buddhaRenderPending = false
  suppressResizeRedrawUntil = performance.now() + 500

  cancelActiveMainRender()
  cancelActiveJuliaRender()

  if (shouldClearBuddhabrot) {
    stopAndClearBuddha()
  } else {
    finishBuddhabrotProgress(BuddhabrotState.targetKind)
    hideInactiveBuddhabrotProgress(BuddhabrotState.targetKind)
    disableBuddhaDownload()
    const toggle = DOM.buddha.toggle || document.getElementById('buddha-toggle')
    if (toggle) {
      toggle.checked = false
      toggle.disabled = true
    }
  }

  return true
}

const scaleFactor = 1.02 // ズームを滑らかにするため 1 スクロールあたり約 2% に抑える

function zoomWithClicks(clicks, cooldown, options = {}) {
  zoomWithFactor(scaleFactor ** clicks, cooldown, options)
}

function getWheelZoomDelta(evt) {
  let delta = 0
  if (typeof evt.deltaY === 'number' && evt.deltaY !== 0) {
    let pixelDelta = evt.deltaY
    if (evt.deltaMode === 1) pixelDelta *= 16
    else if (evt.deltaMode === 2) pixelDelta *= window.innerHeight || 800
    delta = -pixelDelta / 40
  } else {
    delta = evt.wheelDelta ? evt.wheelDelta / 40 : evt.detail ? -evt.detail : 0
  }
  return INVERT_SCROLL ? delta : -delta
}

function hasPendingInteractivePan() {
  return pendingInteractivePanDx !== 0 || pendingInteractivePanDy !== 0
}

function hasPendingInteractiveTransform() {
  return (
    pendingInteractiveTransformScale !== 1 || pendingInteractiveTransformTx !== 0 || pendingInteractiveTransformTy !== 0
  )
}

function accumulateInteractivePan(dx, dy) {
  pendingInteractivePanDx += dx
  pendingInteractivePanDy += dy
}

function trackPendingInteractivePanTransform(dx, dy) {
  pendingInteractiveTransformTx += dx
  pendingInteractiveTransformTy += dy
}

function trackPendingInteractiveScaleTransform(factor, x, y) {
  pendingInteractiveTransformTx = factor * pendingInteractiveTransformTx + (1 - factor) * x
  pendingInteractiveTransformTy = factor * pendingInteractiveTransformTy + (1 - factor) * y
  pendingInteractiveTransformScale *= factor
}

function resetPendingInteractiveTransform() {
  pendingInteractiveTransformScale = 1
  pendingInteractiveTransformTx = 0
  pendingInteractiveTransformTy = 0
}

function clearPendingInteractiveRedrawState() {
  if (gpuInteractiveRedrawTimer != null) {
    clearTimeout(gpuInteractiveRedrawTimer)
    gpuInteractiveRedrawTimer = null
  }
  if (gpuInteractiveFinalizeTimer != null) {
    clearTimeout(gpuInteractiveFinalizeTimer)
    gpuInteractiveFinalizeTimer = null
  }
  gpuInteractiveRedrawPending = false
  gpuInteractiveNeedsFinalRedraw = false
  pendingInteractivePinchView = null
  pendingInteractivePanDx = 0
  pendingInteractivePanDy = 0
  resetPendingInteractiveTransform()
}

function reapplyPendingInteractiveTransform() {
  if (!hasPendingInteractiveTransform()) return
  if (tempCanvas.width !== canvasElement.width) tempCanvas.width = canvasElement.width
  if (tempCanvas.height !== canvasElement.height) tempCanvas.height = canvasElement.height

  const tempCtx = tempCanvas.getContext('2d')
  tempCtx.setTransform(1, 0, 0, 1, 0, 0)
  tempCtx.clearRect(0, 0, tempCanvas.width, tempCanvas.height)
  tempCtx.drawImage(canvasElement, 0, 0)

  const ctx = canvasElement.getContext('2d')
  ctx.save()
  ctx.setTransform(1, 0, 0, 1, 0, 0)
  ctx.clearRect(0, 0, canvasElement.width, canvasElement.height)
  ctx.setTransform(
    pendingInteractiveTransformScale,
    0,
    0,
    pendingInteractiveTransformScale,
    pendingInteractiveTransformTx,
    pendingInteractiveTransformTy,
  )
  ctx.drawImage(tempCanvas, 0, 0)
  ctx.restore()
}

function hasPendingInteractiveView() {
  return pendingInteractivePinchView != null || hasPendingInteractivePan()
}

function ensurePendingInteractivePinchView() {
  if (!pendingInteractivePinchView) {
    const p = fractal.precision
    let center = [fractal.center[0].withScale(p), fractal.center[1].withScale(p)]
    const zoom = fractal.zoom.withScale(p)
    if (hasPendingInteractivePan()) {
      center = _centerAfterPixelDelta(center, zoom, pendingInteractivePanDx, pendingInteractivePanDy, p)
      pendingInteractivePanDx = 0
      pendingInteractivePanDy = 0
    }
    pendingInteractivePinchView = {
      center,
      zoom,
    }
  }
  return pendingInteractivePinchView
}

function commitPendingInteractivePan() {
  if (!hasPendingInteractivePan()) return false
  const dx = pendingInteractivePanDx
  const dy = pendingInteractivePanDy
  pendingInteractivePanDx = 0
  pendingInteractivePanDy = 0
  applyPixelDeltaToCenter(dx, dy)
  return true
}

function startGpuInteractiveRedrawNow(options = {}) {
  if (!isMainRenderGpuPath()) {
    clearPendingInteractiveRedrawState()
    return
  }
  const finalRedraw = options.final === true
  if (pendingInteractivePinchView) {
    fractal.setZoom(pendingInteractivePinchView.zoom)
    fractal.setCenter(pendingInteractivePinchView.center)
    pendingInteractivePinchView = null
    pendingInteractivePanDx = 0
    pendingInteractivePanDy = 0
  } else {
    commitPendingInteractivePan()
  }
  resetPendingInteractiveTransform()
  lastGpuInteractiveRedrawAt = performance.now()
  fractal.interactiveGpuRedraw = !finalRedraw
  gpuInteractiveNeedsFinalRedraw = !finalRedraw
  cancelActiveMainRender()
  redraw(false, 0)
}

function scheduleGpuInteractiveRedraw() {
  // A new interaction supersedes a pending wheel-idle completion. Wheel input
  // schedules a fresh completion immediately after this function returns.
  if (gpuInteractiveFinalizeTimer != null) {
    clearTimeout(gpuInteractiveFinalizeTimer)
    gpuInteractiveFinalizeTimer = null
  }
  gpuInteractiveRedrawPending = true
  // Deep-zoom Mandelbrot uses the perturbation pipeline. Keep the transformed
  // previous frame responsive, but defer that expensive GPU recomputation until
  // the drag/pinch ends or wheel input becomes idle.
  if (!shouldRedrawMainGpuDuringInteraction()) return
  const now = performance.now()
  const wait = Math.max(0, GPU_INTERACTIVE_REDRAW_INTERVAL_MS - (now - lastGpuInteractiveRedrawAt))
  if (gpuInteractiveRedrawTimer != null) return
  gpuInteractiveRedrawTimer = setTimeout(() => {
    gpuInteractiveRedrawTimer = null
    if (!gpuInteractiveRedrawPending) return
    gpuInteractiveRedrawPending = false
    startGpuInteractiveRedrawNow()
  }, wait)
}

function scheduleGpuInteractiveFinalRedraw() {
  if (gpuInteractiveFinalizeTimer != null) clearTimeout(gpuInteractiveFinalizeTimer)
  gpuInteractiveFinalizeTimer = setTimeout(() => {
    gpuInteractiveFinalizeTimer = null
    flushGpuInteractiveRedraw()
  }, GPU_WHEEL_FINAL_REDRAW_DELAY_MS)
}

function flushGpuInteractiveRedraw() {
  if (!isMainRenderGpuPath()) {
    clearPendingInteractiveRedrawState()
    return
  }
  if (gpuInteractiveRedrawTimer != null) {
    clearTimeout(gpuInteractiveRedrawTimer)
    gpuInteractiveRedrawTimer = null
  }
  if (gpuInteractiveFinalizeTimer != null) {
    clearTimeout(gpuInteractiveFinalizeTimer)
    gpuInteractiveFinalizeTimer = null
  }
  if (
    !gpuInteractiveRedrawPending &&
    !gpuInteractiveNeedsFinalRedraw &&
    !hasPendingInteractiveView() &&
    !hasPendingInteractiveTransform()
  ) {
    return
  }
  gpuInteractiveRedrawPending = false
  startGpuInteractiveRedrawNow({ final: true })
}

function zoomWithFactor(factor, cooldown, options = {}) {
  const buddhaOnJuliaView = isBuddhabrotViewShownOnJulia()
  if (isMainCanvasInteractionBlockedByBuddhabrot()) return
  // stopBuddhaPreserveDisplay は runner が実行中のときだけ呼ぶ。
  // 停止済みで呼ぶと density バッファが同期描画され、ズームのたびに
  // Buddhabrot が一瞬フラクタルの上へ重なって見えてしまう。
  try {
    if (!buddhaOnJuliaView && buddhaRunner?.running) {
      stopBuddhaPreserveDisplay()
    }
  } catch (e) {
    console.warn('Error stopping buddha preserve display in zoomWithFactor():', e?.message ? e.message : e)
  }
  // ユーザーが明示的にズームしたら、Buddha トグルをロックして無効化する
  try {
    if (!buddhaOnJuliaView) {
      buddhaLockedByFractalChange = true
      buddhaPreservedDisplay = false
      const t = document.getElementById('buddha-toggle')
      if (t) {
        t.checked = false
        t.disabled = true
      }
    }
    // main 側を動かすと、元の Julia スナップショットは古くなる。
    savedFractalImageData = null
  } catch (e) {
    console.warn('Error disabling buddha toggle on zoom:', e?.message ? e.message : e)
  }
  const lowerBound = MIN_ZOOM.withScale(fractal.precision)
  if (fractal.zoom.leq(lowerBound) && factor < 1) return

  const deferGestureRedraw = options.gesture && isMainRenderGpuPath()
  if (deferGestureRedraw) {
    if (detailPinnedState) clearPinnedDetailPopup()
    const pendingView = ensurePendingInteractivePinchView()
    const zoomedPendingView = _zoomViewAroundCanvasPoint(
      pendingView.center,
      pendingView.zoom,
      factor,
      lastX,
      lastY,
      fractal.precision,
    )
    if (!zoomedPendingView.zoomChanged) return
    pendingInteractivePinchView = {
      center: zoomedPendingView.center,
      zoom: zoomedPendingView.zoom,
    }
    trackPendingInteractiveScaleTransform(zoomedPendingView.appliedFactor, lastX, lastY)
    scaleCanvas(zoomedPendingView.appliedFactor, lastX, lastY)
    scheduleGpuInteractiveRedraw()
  } else {
    const zoomedView = _zoomViewAroundCanvasPoint(fractal.center, fractal.zoom, factor, lastX, lastY, fractal.precision)
    if (!zoomedView.zoomChanged) return

    // 新しい中心座標とズームを反映する
    fractal.setCenter(zoomedView.center)
    fractal.setZoom(zoomedView.zoom)
    if (detailPinnedState) {
      clearPinnedDetailPopup()
    } else if (detailHoverState) {
      _renderDetailIndicator()
    }
    scaleCanvas(zoomedView.appliedFactor, lastX, lastY)
    redraw(false, cooldown)
  }
  refreshCoordinateGrid()
  _refreshPinnedOrbits()
  // 軌道表示が有効で未固定なら、現在のマウス位置で描き直す。
  // ズームで座標変換が変わっても、マウスを動かすまで mousemove は来ないため。
  if (orbitDrawEnabled && !pinnedOrbit) {
    try {
      const [cReFxp, cImFxp] = fractal.canvas2complex(lastX, lastY)
      const cReal = cReFxp.toNumber ? cReFxp.toNumber() : 0
      const cImag = cImFxp.toNumber ? cImFxp.toNumber() : 0
      drawOrbitOnCanvasAtComplex(cReal, cImag, lastX, lastY, cReFxp, cImFxp)
    } catch (_e) {}
  }
}

function handleScroll(evt) {
  // Buddhabrot 表示中、またはトグルがオンならホイールズームを無効化する
  if (isMainCanvasInteractionBlockedByBuddhabrot()) {
    // ページ全体のズームやスクロールを防ぐ
    if (evt && typeof evt.preventDefault === 'function') evt.preventDefault()
    return
  }

  updateMousePos(evt)
  const delta = getWheelZoomDelta(evt)
  if (delta) {
    zoomWithClicks(-delta, 0, { gesture: evt.type === 'wheel' })
    if (evt.type === 'wheel' && isMainRenderGpuPath()) scheduleGpuInteractiveFinalRedraw()
  }
  evt.preventDefault()
}

/** Julia キャンバス用のマウスホイールズーム処理。 */
function handleJuliaScroll(evt) {
  if (!juliaState.active || !juliaState.renderer) {
    evt.preventDefault()
    return
  }
  if (isJuliaCanvasInteractionBlockedByBuddhabrot()) {
    evt.preventDefault()
    return
  }
  const delta = getWheelZoomDelta(evt)
  if (!delta) {
    evt.preventDefault()
    return
  }

  const factor = scaleFactor ** -delta
  const renderer = juliaState.renderer
  const p = renderer.precision
  const lowerBound = MIN_ZOOM.withScale(p)
  if (renderer.zoom.withScale(p).leq(lowerBound) && factor < 1) {
    evt.preventDefault()
    return
  }
  invalidateJuliaBuddhabrotView()

  // Julia キャンバス基準のマウス位置を求める
  const rect = renderer.canvas.getBoundingClientRect()
  const mouseX = (evt.clientX - rect.left) * (renderer.canvas.width / rect.width)
  const mouseY = (evt.clientY - rect.top) * (renderer.canvas.height / rect.height)

  // カーソル下にある Julia 複素平面上の点
  const ptr = renderer.canvas2complex(mouseX, mouseY)

  // ズーム倍率を適用する
  const bigFactor = fxp.fromNumber(factor, p)
  const zoomFx = renderer.zoom.withScale(p)
  const renderingWithGpu = renderer._willUseGpuForCurrentRender?.() ?? renderer.useGpu
  const upperBound = getMainZoomLimitForState(
    renderer.fractalType,
    renderingWithGpu,
    p,
    getZoomClampOptionsForPalette(renderer.paletteComponent?.palette),
  )
  const rawZoom = zoomFx.multiply(bigFactor).max(lowerBound)
  const newZoom = upperBound ? rawZoom.min(upperBound) : rawZoom
  const zoomChanged = newZoom.bigInt !== zoomFx.bigInt
  if (!zoomChanged) {
    evt.preventDefault()
    return
  }

  // カーソル下の点が動かないよう中心を調整する
  const appliedFactor = newZoom.divide(zoomFx).toNumber()
  const invFactor = fxp.fromNumber(1.0 / appliedFactor, p)
  const offsetX = ptr[0].subtract(renderer.center[0].withScale(p)).multiply(invFactor)
  const offsetY = ptr[1].subtract(renderer.center[1].withScale(p)).multiply(invFactor)
  renderer.center = [ptr[0].subtract(offsetX), ptr[1].subtract(offsetY)]
  renderer.setZoom(newZoom)
  if (detailPinnedState) {
    clearPinnedDetailPopup()
  } else if (detailHoverState) {
    _renderDetailIndicator()
  }

  redrawJulia()
  _refreshPinnedOrbits()
  // Julia キャンバスで軌道表示が有効かつ未固定なら、現在のマウス位置で描き直す
  if (orbitDrawEnabled && !juliaPinnedOrbit && juliaState.active && juliaState.renderer) {
    try {
      drawOrbitOnJuliaCanvas(evt.clientX, evt.clientY)
    } catch (_e) {}
  }
  evt.preventDefault()
}

function _juliaCanvasCoords(evt) {
  const renderer = juliaState.renderer
  const rect = renderer.canvas.getBoundingClientRect()
  return [
    (evt.clientX - rect.left) * (renderer.canvas.width / rect.width),
    (evt.clientY - rect.top) * (renderer.canvas.height / rect.height),
  ]
}

function _juliaCanvasCoordsFromClient(clientX, clientY) {
  const renderer = juliaState.renderer
  const rect = renderer.canvas.getBoundingClientRect()
  return [
    (clientX - rect.left) * (renderer.canvas.width / rect.width),
    (clientY - rect.top) * (renderer.canvas.height / rect.height),
  ]
}

function _juliaDocMouseMove(evt) {
  if (!juliaState.active || !juliaState.renderer || !juliaDragStart) return
  if (isJuliaCanvasInteractionBlockedByBuddhabrot()) {
    _endJuliaDrag()
    if (evt?.cancelable) evt.preventDefault()
    return
  }
  if ((evt.buttons & 1) === 0) {
    _endJuliaDrag()
    return
  }

  const renderer = juliaState.renderer
  const [x, y] = _juliaCanvasCoords(evt)

  const dx = x - juliaDragStart[0]
  const dy = y - juliaDragStart[1]
  if (dx !== 0 || dy !== 0) invalidateJuliaBuddhabrotView()
  if (detailPinnedState) clearPinnedDetailPopup()

  const p = renderer.precision
  const w_fx = fxp.fromNumber(renderer.width, p)
  const scale_fx = renderer.zoom.withScale(p).multiply(w_fx).divide(fxp.fromNumber(4, p))
  const dxFx = fxp.fromNumber(-dx, p).divide(scale_fx)
  const dyFx = fxp.fromNumber(-dy, p).divide(scale_fx)

  renderer.center = [renderer.center[0].withScale(p).add(dxFx), renderer.center[1].withScale(p).add(dyFx)]

  // すぐに見た目が追従するよう、先にキャンバスだけ平行移動する
  const ctx = renderer.canvas.getContext('2d')
  ctx.save()
  ctx.translate(dx, dy)
  ctx.drawImage(renderer.canvas, 0, 0)
  ctx.restore()

  juliaDragStart = [x, y]
  _juliaPinDragged = true
  redrawJulia()
  _refreshPinnedOrbits()
}

function _endJuliaDrag() {
  juliaDragStart = null
  document.removeEventListener('mousemove', _juliaDocMouseMove)
  document.removeEventListener('mouseup', _endJuliaDrag)
}

function onJuliaMouseDown(evt) {
  if (!juliaState.active || !juliaState.renderer) return
  if (isJuliaCanvasInteractionBlockedByBuddhabrot()) {
    if (evt?.cancelable) evt.preventDefault()
    return
  }
  _juliaPinDragged = false
  juliaDragStart = _juliaCanvasCoords(evt)
  // ポインターが他要素へ乗ってもドラッグを続けられるよう、
  // document レベルで監視する。
  document.addEventListener('mousemove', _juliaDocMouseMove)
  document.addEventListener('mouseup', _endJuliaDrag)
}

function onJuliaMouseMove(evt) {
  if (isJuliaCanvasInteractionBlockedByBuddhabrot()) {
    if (juliaCanvasElement) juliaCanvasElement.style.cursor = ''
    return
  }
  // 固定した Julia 軌道の十字付近では、解除できることが分かるようポインター表示にする
  if (orbitDrawEnabled && juliaPinnedOrbit && juliaState.active && juliaState.renderer) {
    const renderer = juliaState.renderer
    const rect = renderer.canvas.getBoundingClientRect()
    const px_c = (evt.clientX - rect.left) * (renderer.canvas.width / rect.width)
    const py_c = (evt.clientY - rect.top) * (renderer.canvas.height / rect.height)
    const jZoom = renderer.zoom.toNumber ? renderer.zoom.toNumber() : 1
    const jW = renderer.canvas.width,
      jH = renderer.canvas.height
    const scale = (jZoom * jW) / 4
    const jCenterX = renderer.center[0].toNumber ? renderer.center[0].toNumber() : 0
    const jCenterY = renderer.center[1].toNumber ? renderer.center[1].toNumber() : 0
    const jOrbitCanvas = document.getElementById('julia-orbit-canvas')
    const ow = jOrbitCanvas ? jOrbitCanvas.offsetWidth : jW
    const oh = jOrbitCanvas ? jOrbitCanvas.offsetHeight : jH
    const ax = (px_c / jW) * ow
    const ay = (py_c / jH) * oh
    const bx = (((juliaPinnedOrbit.re - jCenterX) * scale + jW / 2) / jW) * ow
    const by = (((juliaPinnedOrbit.im - jCenterY) * scale + jH / 2) / jH) * oh
    if (juliaCanvasElement)
      juliaCanvasElement.style.cursor = Math.sqrt((ax - bx) ** 2 + (ay - by) ** 2) < 8 ? 'pointer' : ''
  } else {
    if (juliaCanvasElement) juliaCanvasElement.style.cursor = ''
  }
  // 軌道表示が有効なら Julia キャンバスへ描く。
  // 固定済みのときはホバーによる再描画を行わない。
  if (orbitDrawEnabled && juliaState.active && !juliaPinnedOrbit) {
    try {
      drawOrbitOnJuliaCanvas(evt.clientX, evt.clientY)
    } catch (_e) {}
  }
  // ドラッグ中の本処理は document 側の _juliaDocMouseMove が担当する。
  // ここでは非ドラッグ時のカーソル表示だけを素早く保つ。
}

function onJuliaMouseUp() {
  _endJuliaDrag()
}

function _updateIterations(delta) {
  setIterations(fractal.max_iter + delta)
}

/**
 * 最大反復回数を設定する
 * @param {number} value - 反復回数
 * @returns {boolean} 値が変わったとき true
 */
function setIterations(value) {
  // NaN や有限でない値を防ぐ（例: 空入力を parseInt すると NaN）
  if (!Number.isFinite(value) || value < 1) {
    // 表示は現在の有効な値へ戻す
    DOM.iterations.value = fractal.max_iter
    return false
  }
  const newIter = Math.max(1, Math.floor(value))
  if (newIter !== fractal.max_iter) {
    fractal.max_iter = newIter
    fractal.initPallete()
    DOM.iterations.value = fractal.max_iter
    // Buddhabrot の runner があれば maxIter だけ更新し、以後のサンプリングや
    // 再開時に新しい値を使う。Buddhabrot の実行中や表示中は描画を抑止しているため、
    // ここでは再描画を発火しない。
    if (typeof buddhaRunner !== 'undefined' && buddhaRunner && 'maxIter' in buddhaRunner) {
      buddhaRunner.maxIter = fractal.max_iter
    }
    // 反復回数の変更をパーマリンクへすぐ反映する
    updatePermalink()

    return true
  }
  return false
}

/**
 * 入力欄の座標をフラクタル表示へ適用する
 */
function applyCoordinates() {
  try {
    const xStr = document.getElementById('coordX').value.trim()
    const yStr = document.getElementById('coordY').value.trim()
    const zoomStr = document.getElementById('coordZoom').value.trim()

    if (!xStr || !yStr || !zoomStr) {
      alert('Please fill in all coordinate fields (X, Y, and Zoom)')
      return
    }

    // 高精度文字列パーサーで解釈し、現在の座標系スケールで検証する
    const currentScale = fractal.center[0].scale

    const x = parseDecimalString(xStr, currentScale)
    // UI は「上 = 正の虚数」表示のため、内部座標系（上 = 負の虚数）に戻すため符号を反転する。
    const y = parseDecimalString(negateDecimalString(yStr), currentScale)
    const zoom = parseDecimalString(zoomStr, currentScale)

    if (!x || !y || !zoom) {
      alert('Invalid coordinate values. Please enter valid numbers.')
      return
    }

    // zoom は正でなければならない
    if (zoom.bigInt <= 0n) {
      alert('Zoom must be greater than 0')
      return
    }

    // アニメーション用の目標値を準備する
    const targetCenter = [x, y]
    const targetRenderingWithGpu = fractal._willUseGpuForCurrentRender?.() ?? fractal.useGpu
    const targetZoom = clampZoomForState(
      zoom,
      fractal.fractalType,
      targetRenderingWithGpu,
      getZoomClampOptionsForPalette(fractal.paletteComponent?.palette),
    )
    const zoomWasClamped = zoom.withScale(targetZoom.scale).bigInt !== targetZoom.bigInt

    // Buddhabrot 実行中なら停止して消去し、パンやズームを再び有効にする
    try {
      stopAndClearBuddha()
    } catch (e) {
      console.warn('Error stopping/clearing buddha in applyCoordinates():', e?.message ? e.message : e)
    }
    // 明示的に表示を変えたので、Buddha トグルをロックして無効化する
    buddhaLockedByFractalChange = true
    const t = document.getElementById('buddha-toggle')
    if (t) {
      t.checked = false
      t.disabled = true
    }
    // 表示が変わるので保存済み画像を無効化する
    savedFractalImageData = null

    // 座標適用時は高解像度ダウンロードも無効にする
    disableBuddhaDownload()

    // 座標適用では、現在のフラクタル種別や反復式は上書きしない。
    // fractal.fractalType と関連 UI を維持する。

    // 再描画中に表示が丸まらないよう入力値を保持する
    const preservedInputs = {
      x: xStr,
      y: yStr,
      zoom: zoomWasClamped ? formatZoomValueForInput(targetZoom) : zoomStr,
    }

    // いったん初期位置へ戻してから目標位置へアニメーションする
    _clearPinnedOrbits()
    fractal.setCenter([fxp.fromNumber(-0.5), fxp.fromNumber(0)])
    fractal.setZoom(getInitialFractalZoom(fractal.precision))
    // 表示が変わるので保存済み画像を無効化する
    savedFractalImageData = null
    redraw()

    const animEnabled = document.getElementById('anim-enable')?.checked
    const SPEED = 0.15 // fixed animation speed (per user request)
    const baseMs = 2000
    const duration = Math.max(50, Math.floor(baseMs / SPEED))
    if (animEnabled) {
      startAnimation(targetCenter, targetZoom, duration)
    } else {
      // 先に zoom を入れて目標精度を決め、その後で center を入れる。
      // こうすると、内部精度へ変換する際の量子化や切り捨てを抑えられる。
      fractal.setZoom(targetZoom)
      fractal.setCenter(targetCenter)
      savedFractalImageData = null
      redraw()
      updatePermalink()
    }

    // 表示上の精度落ちを避けるため、再描画後に入力値を戻す
    setTimeout(() => {
      document.getElementById('coordX').value = preservedInputs.x
      document.getElementById('coordY').value = preservedInputs.y
      const appliedTargetZoom = fractal.zoom.withScale(targetZoom.scale).bigInt === targetZoom.bigInt
      document.getElementById('coordZoom').value = appliedTargetZoom
        ? preservedInputs.zoom
        : formatZoomValueForInput(fractal.zoom)
    }, 0)
  } catch (error) {
    console.error('Error applying coordinates:', error)
    alert(`Error applying coordinates: ${error.message}`)
  }
}

// 十進文字列（指数表記を含む）を FxP へ変換する
function parseDecimalString(str, scale = 60) {
  try {
    str = str.trim()
    if (!str) return null

    // 指数表記を処理する（例: "1.23e100"）
    let mantissa = str
    let exponent = 0

    const eIndex = str.toLowerCase().indexOf('e')
    if (eIndex !== -1) {
      mantissa = str.substring(0, eIndex)
      exponent = parseInt(str.substring(eIndex + 1), 10)
      if (Number.isNaN(exponent)) return null
    }

    // 仮数部を解釈する
    const isNegative = mantissa.startsWith('-')
    if (isNegative) mantissa = mantissa.substring(1)

    // 整数部と小数部へ分ける
    const parts = mantissa.split('.')
    let intPart = parts[0] || '0'
    let fracPart = parts[1] || ''

    // 指数を反映する
    if (exponent > 0) {
      // 小数点を右へ動かす
      if (exponent <= fracPart.length) {
        intPart += fracPart.substring(0, exponent)
        fracPart = fracPart.substring(exponent)
      } else {
        intPart += fracPart + '0'.repeat(exponent - fracPart.length)
        fracPart = ''
      }
    } else if (exponent < 0) {
      // 小数点を左へ動かす
      const shift = -exponent
      if (shift < intPart.length) {
        fracPart = intPart.substring(intPart.length - shift) + fracPart
        intPart = intPart.substring(0, intPart.length - shift)
      } else {
        fracPart = '0'.repeat(shift - intPart.length) + intPart + fracPart
        intPart = '0'
      }
    }

    // 指定スケールの BigInt へ変換する
    let bigInt = BigInt(intPart) << BigInt(scale)

    // 小数部を加える
    // これは fxpToDecimalString() の逆変換にあたる
    if (fracPart.length > 0) {
      // 先頭の 0 を数えて飛ばす
      let leadingZeros = 0
      while (leadingZeros < fracPart.length && fracPart[leadingZeros] === '0') {
        leadingZeros++
      }

      if (leadingZeros < fracPart.length) {
        // 有効な桁だけを取り出す
        const significantPart = fracPart.substring(leadingZeros)

        // スケールから使える十進桁数の上限を見積もる
        // 2^scale ≈ 10^(scale * 0.30103) なので、約 scale * 0.30103 桁を表せる
        const maxDecimalDigits = Math.floor(scale * 0.30103)

        // 計算可能な精度内に制限する
        const digitsToUse = Math.min(significantPart.length, maxDecimalDigits)

        // 有効桁を整数として積み上げる
        let fracValue = 0n
        for (let i = 0; i < digitsToUse; i++) {
          const digit = significantPart[i]
          if (digit < '0' || digit > '9') {
            console.error('Invalid digit:', digit)
            return null
          }
          fracValue = fracValue * 10n + BigInt(digit)
        }

        // fracValue は leadingZeros 個ぶん後ろにある digitsToUse 桁を表す。
        // したがって fracValue * 2^scale / 10^(leadingZeros + digitsToUse) で変換する。

        const totalDigits = leadingZeros + digitsToUse

        // scale ビット左へ寄せる
        const numerator = fracValue << BigInt(scale)

        // 10^totalDigits で分割しながら割る
        let result = numerator
        let remaining = totalDigits

        while (remaining > 0) {
          const chunk = Math.min(remaining, 100)
          const divisor = 10n ** BigInt(chunk)
          result = result / divisor
          remaining -= chunk
        }

        // 小数部だけを取り出して整数部へ合成する
        const fracMask = (1n << BigInt(scale)) - 1n
        const fracBigInt = result & fracMask

        bigInt = bigInt | fracBigInt
      }
    }

    if (isNegative) bigInt = -bigInt

    return fxp.fromJSON({ bigInt: bigInt.toString(), scale: scale })
  } catch (error) {
    console.error('Error parsing decimal string:', str, error)
    return null
  }
}

// ── Shared Orbit Helpers ────────────────────────────────────────────────

/** 現在の反復式をコンパイルし、失敗時は z^2+c に戻す。 */
function _getIterFn() {
  try {
    return compileIterationFunction(DOM.iterationFunctionInput?.value || 'z*z + c')
  } catch (_e) {
    return (zr, zi, cr, ci) => [zr * zr - zi * zi + cr, 2 * zr * zi + ci]
  }
}

/** z0 入力欄から [z0Real, z0Imag] を返す。 */
function _getZ0Inputs() {
  return [
    parseFloat(document.getElementById('z0-real')?.value) || 0,
    parseFloat(document.getElementById('z0-imag')?.value) || 0,
  ]
}

/**
 * 高精度 FxP 複素座標を orbit キャンバスの CSS ピクセル座標へ変換する。
 * 深いズームで Float64 の桁落ちを避けるため FxP 演算を使う。
 * @param {FxP} reFxp - 実部（FxP の桁数は任意）
 * @param {FxP} imFxp - 虚部（FxP の桁数は任意）
 * @param {number} ow  - orbit キャンバスの CSS 幅
 * @param {number} oh  - orbit キャンバスの CSS 高さ
 * @returns {[number, number]} [cssX, cssY]
 */
function _getMainOrbitRenderView() {
  const prec = fractal.precision
  let center = [fractal.center[0].withScale(prec), fractal.center[1].withScale(prec)]
  let zoom = fractal.zoom.withScale(prec)

  if (pendingInteractivePinchView) {
    center = [pendingInteractivePinchView.center[0].withScale(prec), pendingInteractivePinchView.center[1].withScale(prec)]
    zoom = pendingInteractivePinchView.zoom.withScale(prec)
  }
  if (hasPendingInteractivePan()) {
    center = _centerAfterPixelDelta(center, zoom, pendingInteractivePanDx, pendingInteractivePanDy, prec)
  }

  return { center, zoom }
}

function _fxpComplexToOrbitCss(reFxp, imFxp, ow, oh, view = null) {
  const prec = fractal.precision
  const orbitView = view || _getMainOrbitRenderView()
  const fw = fractal.width,
    fh = fractal.height
  // scale = zoom * fw / 4（複素平面 1 単位あたりのピクセル数）
  const scale_fxp = orbitView.zoom.withScale(prec).multiply(fxp.fromNumber(fw, prec)).divide(fxp.fromNumber(4, prec))
  const dre = reFxp.withScale(prec).subtract(orbitView.center[0])
  const dim = imFxp.withScale(prec).subtract(orbitView.center[1])
  const dre_px = dre.multiply(scale_fxp).toNumber()
  const dim_px = dim.multiply(scale_fxp).toNumber()
  return [((dre_px + fw / 2) / fw) * ow, ((dim_px + fh / 2) / fh) * oh]
}

function _getOrbitBailout(renderer = fractal) {
  if (renderer?.smooth) return BAILOUT_SMOOTH
  const escapeRadius = renderer?.escapeRadius ?? fractal.escapeRadius ?? 4
  return escapeRadius * escapeRadius
}

/**
 * 指定した高精度複素点での Mandelbrot 軌道を BigInt 演算で求める。
 * 深いズームで Float64 では区別できない近接ピクセルも正確に判定できる。
 * 標準 Mandelbrot（z0=0、iterFn=z*z+c）でのみ有効。
 * @param {FxP} cReFxp - c の実部
 * @param {FxP} cImFxp - c の虚部
 * @param {number} bailout - 描画側と同じ脱出判定値
 * @returns {{ orbit: Array<[number,number]>, escaped: boolean }}
 */
function _computeOrbitHighPrec(cReFxp, cImFxp, bailout = _getOrbitBailout()) {
  const prec = fractal.precision
  const bigScale = BigInt(prec)
  const re = cReFxp.withScale(prec).bigInt
  const im = cImFxp.withScale(prec).bigInt
  // mandelbrot_high_precision は z=0 から z=z^2+c を反復し、[itersMarker, zq, seq] を返す
  // seq は z1 以降の [[z_real, z_imag, zq], ...] で、z0=[0,0] は暗黙扱い
  // iterMarker=2 は max_iter 到達、そうでなければ脱出
  const [iterMarker, , seq] = mandelbrot_high_precision(re, im, fractal.max_iter, bailout, bigScale, prec)
  const escaped = iterMarker !== 2
  const orbit = [[0, 0], ...seq.map(([zr, zi]) => [zr, zi])]
  return { orbit, escaped }
}

/**
 * 開始値 z0 とパラメータ c から軌道点を求める。
 * 停止条件は描画処理と揃えている:
 *   ① |z|² > bailout なら escaped = true
 *   ② iter === maxIter なら escaped = false（集合内・黒）
 * @param {number} maxIter - 最大反復回数（fractal.max_iter を渡す）
 * @param {number} bailout - 描画側と同じ脱出判定値
 * @returns {{ orbit: Array<[number,number]>, escaped: boolean }}
 */
function _computeOrbitPoints(z0r, z0i, cr, ci, iterFn, _complexToScreen, maxIter, bailout = _getOrbitBailout()) {
  const orbit = [[z0r, z0i]]
  let zr = z0r,
    zi = z0i,
    escaped = false
  for (let iter = 0; iter < maxIter; iter++) {
    let nzr, nzi
    try {
      ;[nzr, nzi] = iterFn(zr, zi, cr, ci, iter)
    } catch (_e) {
      break
    }
    // Infinity/NaN は SAFE_SENTINEL と同様に脱出扱いにする
    if (!Number.isFinite(nzr) || !Number.isFinite(nzi)) {
      escaped = true
      break
    }
    zr = nzr
    zi = nzi
    // ① bailout 超過で脱出（描画側の while 終了条件と同じ）
    if (zr * zr + zi * zi > bailout) {
      escaped = true
      orbit.push([zr, zi])
      break
    }
    orbit.push([zr, zi])
  }
  // ループ完走時は escaped=false のままなので、集合内として扱う
  return { orbit, escaped }
}

function _getOrbitGradientColor(iterationIndex, totalIterations) {
  const ratio = Math.min(1, Math.max(0, iterationIndex / Math.max(1, totalIterations)))
  const hue = 58 - ratio * 54
  return `hsl(${hue}, 100%, 50%)`
}

/**
 * 事前計算した軌道をキャンバスへ描画する。
 * @param ctx             クリア済みの 2D コンテキスト
 * @param orbit           [[re,im], ...]
 * @param escaped         軌道が脱出したか
 * @param complexToScreen (re,im) → [px,py]
 * @param crosshairRe/Im  十字マーカー用の複素座標
 * @param directCx/Cy     任意の事前計算済み CSS 座標
 *                        （深いズームでの Float64 桁落ちを避けるため、指定時は
 *                         十字マーカーに complexToScreen を使わない）
 */
function _paintOrbitOnCtx(
  ctx,
  orbit,
  escaped,
  complexToScreen,
  crosshairRe,
  crosshairIm,
  directCx = null,
  directCy = null,
) {
  const lineColor = escaped ? 'rgba(255, 90, 90, 0.85)' : 'rgba(100, 180, 255, 0.85)'
  const dotColor = escaped ? 'rgba(255, 90, 90, 1)' : 'rgba(100, 180, 255, 1)'

  if (orbitMode !== 'dots') {
    ctx.lineWidth = 1.5
    if (orbitGradientEnabled) {
      const gradientTotalIterations = Math.max(1, orbit.length - 1)
      for (let k = 1; k < orbit.length; k++) {
        const [x0, y0] = complexToScreen(orbit[k - 1][0], orbit[k - 1][1])
        const [x1, y1] = complexToScreen(orbit[k][0], orbit[k][1])
        ctx.beginPath()
        ctx.strokeStyle = _getOrbitGradientColor(k, gradientTotalIterations)
        ctx.moveTo(x0, y0)
        ctx.lineTo(x1, y1)
        ctx.stroke()
      }
    } else {
      ctx.beginPath()
      ctx.strokeStyle = lineColor
      const [x0, y0] = complexToScreen(orbit[0][0], orbit[0][1])
      ctx.moveTo(x0, y0)
      for (let k = 1; k < orbit.length; k++) {
        const [xk, yk] = complexToScreen(orbit[k][0], orbit[k][1])
        ctx.lineTo(xk, yk)
      }
      ctx.stroke()
    }
  }

  if (orbitMode !== 'lines') {
    const gradientTotalIterations = Math.max(1, orbit.length - 1)
    for (let k = 0; k < orbit.length; k++) {
      const [xk, yk] = complexToScreen(orbit[k][0], orbit[k][1])
      const isFirst = k === 0
      const isLast = k === orbit.length - 1
      ctx.beginPath()
      ctx.arc(xk, yk, isFirst ? 4 : isLast ? 3.5 : 2.5, 0, Math.PI * 2)
      ctx.fillStyle = orbitGradientEnabled
        ? _getOrbitGradientColor(k, gradientTotalIterations)
        : isFirst
          ? 'rgba(255, 240, 60, 1)'
          : isLast
            ? 'rgba(255, 255, 255, 0.95)'
            : dotColor
      ctx.fill()
    }
  } else {
    // 線のみ表示でも、終点だけは白い点を描く
    const last = orbit[orbit.length - 1]
    const [xl, yl] = complexToScreen(last[0], last[1])
    ctx.beginPath()
    ctx.arc(xl, yl, 3.5, 0, Math.PI * 2)
    ctx.fillStyle = orbitGradientEnabled
      ? _getOrbitGradientColor(orbit.length - 1, Math.max(1, orbit.length - 1))
      : 'rgba(255, 255, 255, 0.95)'
    ctx.fill()
  }

  // 直接計算した CSS 座標があれば使い、深いズーム時の Float64 桁落ちを避ける
  const [cx, cy] = directCx !== null ? [directCx, directCy] : complexToScreen(crosshairRe, crosshairIm)
  _paintCrosshairOnCtx(ctx, cx, cy)
}

function _paintCrosshairOnCtx(ctx, cx, cy) {
  ctx.strokeStyle = 'rgba(255, 240, 60, 0.9)'
  ctx.lineWidth = 1.5
  const s = 5
  ctx.beginPath()
  ctx.moveTo(cx - s, cy)
  ctx.lineTo(cx + s, cy)
  ctx.moveTo(cx, cy - s)
  ctx.lineTo(cx, cy + s)
  ctx.stroke()
}

// ── Orbit Visualization ───────────────────────────────────────────────────
function clearOrbitCanvas() {
  const orbitCanvas = document.getElementById('orbit-canvas')
  if (!orbitCanvas) return
  orbitCanvas.getContext('2d').clearRect(0, 0, orbitCanvas.width, orbitCanvas.height)
  // 軌道の反復回数表示 UI は廃止されたため更新不要
}

/**
 * 現在の表示変換を使って、固定した複素点の Mandelbrot 軌道を描画する。
 * 表示のパンやズーム後に、固定した軌道を置き直すときに使う。
 *
 * @param cReal   c の実部（Float64。周期検出と相対位置の基準に使う）
 * @param cImag   c の虚部（Float64）
 * @param refPixX 任意: fractal キャンバスの物理ピクセル X（ホバー十字用）
 * @param refPixY 任意: fractal キャンバスの物理ピクセル Y
 * @param cReFxp  任意: FxP 実部。どのズームでも高精度な軌道と十字を描ける
 * @param cImFxp  任意: FxP 虚部
 */
function drawOrbitOnCanvasAtComplex(cReal, cImag, refPixX = null, refPixY = null, cReFxp = null, cImFxp = null) {
  const orbitCanvas = document.getElementById('orbit-canvas')
  if (!orbitCanvas) return
  const ow = orbitCanvas.offsetWidth,
    oh = orbitCanvas.offsetHeight
  if (ow < 1 || oh < 1) return
  if (orbitCanvas.width !== ow || orbitCanvas.height !== oh) {
    orbitCanvas.width = ow
    orbitCanvas.height = oh
  }
  const ctx = orbitCanvas.getContext('2d')
  ctx.clearRect(0, 0, ow, oh)

  const orbitView = _getMainOrbitRenderView()
  const zoomVal = orbitView.zoom.toNumber ? orbitView.zoom.toNumber() : 1
  const fw = fractal.width,
    fh = fractal.height
  const scale = (zoomVal * fw) / 4
  const centerX = orbitView.center[0].toNumber ? orbitView.center[0].toNumber() : 0
  const centerY = orbitView.center[1].toNumber ? orbitView.center[1].toNumber() : 0

  // 十字マーカーの位置は、使える中で最も精度の高い方法で求める:
  //   1. FxP 演算（どのズームでも Float64 の桁落ちを避ける）
  //   2. 物理ピクセルをそのまま使う（ホバー時、複素座標へ戻さない）
  //   3. Float64 計算式（低ズーム向け。zoom > 約 1e12 ではずれる可能性あり）
  let baseCssX, baseCssY
  if (cReFxp !== null && cImFxp !== null) {
    try {
      ;[baseCssX, baseCssY] = _fxpComplexToOrbitCss(cReFxp, cImFxp, ow, oh, orbitView)
    } catch (_) {
      baseCssX = refPixX !== null ? (refPixX * ow) / fw : (((cReal - centerX) * scale + fw / 2) / fw) * ow
      baseCssY = refPixY !== null ? (refPixY * oh) / fh : (((cImag - centerY) * scale + fh / 2) / fh) * oh
    }
  } else if (refPixX !== null) {
    baseCssX = (refPixX * ow) / fw
    baseCssY = (refPixY * oh) / fh
  } else {
    baseCssX = (((cReal - centerX) * scale + fw / 2) / fw) * ow
    baseCssY = (((cImag - centerY) * scale + fh / 2) / fh) * oh
  }

  // 軌道点 (re, im) をカーソル基準のキャンバス座標へ変換する。
  // (r - cReal) を使うことで、近い値どうしの Float64 桁落ちを抑える。
  const complexToOrbit = (r, i) => [
    baseCssX + (((r - cReal) * scale) / fw) * ow,
    baseCssY + (((i - cImag) * scale) / fh) * oh,
  ]

  // 標準 Mandelbrot かつ z0=[0,0] なら BigInt 経路で軌道を計算する。
  // これにより、どのズームでも正しい反復回数と脱出判定を得られる。
  const [z0Real, z0Imag] = _getZ0Inputs()
  let orbit, escaped
  const bailout = _getOrbitBailout(fractal)
  if (cReFxp !== null && cImFxp !== null && fractal.fractalType === 'mandelbrot' && z0Real === 0 && z0Imag === 0) {
    try {
      ;({ orbit, escaped } = _computeOrbitHighPrec(cReFxp, cImFxp, bailout))
    } catch (_) {
      // BigInt が使えない場合（例: 精度不足）は Float64 計算へ戻す
      const iterFn = _getIterFn()
      ;({ orbit, escaped } = _computeOrbitPoints(
        z0Real,
        z0Imag,
        cReal,
        cImag,
        iterFn,
        complexToOrbit,
        fractal.max_iter,
        bailout,
      ))
    }
  } else {
    const iterFn = _getIterFn()
    ;({ orbit, escaped } = _computeOrbitPoints(
      z0Real,
      z0Imag,
      cReal,
      cImag,
      iterFn,
      complexToOrbit,
      fractal.max_iter,
      bailout,
    ))
  }

  // 軌道の反復回数表示 UI は廃止されたため更新しない
  _paintOrbitOnCtx(ctx, orbit, escaped, complexToOrbit, cReal, cImag, baseCssX, baseCssY)
}

function drawOrbitOnCanvas(clientX, clientY) {
  const [px, py] = toGraphicsCoordinates(clientX, clientY)
  // fractal.canvas2complex は FxP 演算を使うため、深いズームでも
  // Float64 の桁落ちを避けた正確な複素座標を得られる。
  const [cReFxp, cImFxp] = fractal.canvas2complex(px, py)
  const cReal = cReFxp.toNumber ? cReFxp.toNumber() : 0
  const cImag = cImFxp.toNumber ? cImFxp.toNumber() : 0
  drawOrbitOnCanvasAtComplex(cReal, cImag, px, py, cReFxp, cImFxp)
}

// ── Julia Orbit Visualization ─────────────────────────────────────────────
function clearJuliaOrbitCanvas() {
  const jOrbitCanvas = document.getElementById('julia-orbit-canvas')
  if (!jOrbitCanvas) return
  jOrbitCanvas.getContext('2d').clearRect(0, 0, jOrbitCanvas.width, jOrbitCanvas.height)
  // 軌道の反復回数表示 UI は廃止されたため更新不要
}

/**
 * 現在の Julia 表示変換を使って、固定した複素点 z0 の軌道を描画する。
 * Julia 表示をパンやズームしたあと、固定軌道を置き直すときに使う。
 *
 * @param z0Real    初期 z0 点の実部（Float64）
 * @param z0Imag    初期 z0 点の虚部（Float64）
 * @param refPixX   任意: z0Real に対応する Julia キャンバスの物理ピクセル X
 * @param refPixY   任意: z0Imag に対応する Julia キャンバスの物理ピクセル Y
 */
function drawOrbitOnJuliaCanvasAtComplex(z0Real, z0Imag, refPixX = null, refPixY = null) {
  if (isJuliaCanvasInteractionBlockedByBuddhabrot()) return
  if (!juliaState.active || !juliaState.renderer) return
  const jOrbitCanvas = document.getElementById('julia-orbit-canvas')
  if (!jOrbitCanvas) return
  const ow = jOrbitCanvas.offsetWidth,
    oh = jOrbitCanvas.offsetHeight
  if (ow < 1 || oh < 1) return
  if (jOrbitCanvas.width !== ow || jOrbitCanvas.height !== oh) {
    jOrbitCanvas.width = ow
    jOrbitCanvas.height = oh
  }
  const ctx = jOrbitCanvas.getContext('2d')
  ctx.clearRect(0, 0, ow, oh)

  const renderer = juliaState.renderer
  const jZoom = renderer.zoom.toNumber ? renderer.zoom.toNumber() : 1
  const jW = renderer.canvas.width,
    jH = renderer.canvas.height
  const scale = (jZoom * jW) / 4
  const jCenterX = renderer.center[0].toNumber ? renderer.center[0].toNumber() : 0
  const jCenterY = renderer.center[1].toNumber ? renderer.center[1].toNumber() : 0
  const cReal = renderer.juliaCRe
  const cImag = renderer.juliaCIm

  const baseCssX = refPixX !== null ? (refPixX * ow) / jW : (((z0Real - jCenterX) * scale + jW / 2) / jW) * ow
  const baseCssY = refPixY !== null ? (refPixY * oh) / jH : (((z0Imag - jCenterY) * scale + jH / 2) / jH) * oh

  const complexToOrbit = (r, i) => [
    baseCssX + (((r - z0Real) * scale) / jW) * ow,
    baseCssY + (((i - z0Imag) * scale) / jH) * oh,
  ]

  const iterFn = _getIterFn()
  const maxIter = juliaState.renderer?.max_iter ?? fractal.max_iter
  const bailout = _getOrbitBailout(renderer)
  const { orbit, escaped } = _computeOrbitPoints(
    z0Real,
    z0Imag,
    cReal,
    cImag,
    iterFn,
    complexToOrbit,
    maxIter,
    bailout,
  )

  // 軌道の反復回数表示 UI は廃止されたため更新しない
  _paintOrbitOnCtx(ctx, orbit, escaped, complexToOrbit, z0Real, z0Imag, baseCssX, baseCssY)
}

function drawOrbitOnJuliaCanvas(clientX, clientY) {
  if (!juliaState.active || !juliaState.renderer) return
  const renderer = juliaState.renderer
  const rect = renderer.canvas.getBoundingClientRect()
  const px = (clientX - rect.left) * (renderer.canvas.width / rect.width)
  const py = (clientY - rect.top) * (renderer.canvas.height / rect.height)
  const jZoom = renderer.zoom.toNumber ? renderer.zoom.toNumber() : 1
  const jW = renderer.canvas.width,
    jH = renderer.canvas.height
  const scale = (jZoom * jW) / 4
  const jCenterX = renderer.center[0].toNumber ? renderer.center[0].toNumber() : 0
  const jCenterY = renderer.center[1].toNumber ? renderer.center[1].toNumber() : 0
  const z0Real = (px - jW / 2) / scale + jCenterX
  const z0Imag = (py - jH / 2) / scale + jCenterY
  // 十字マーカーを高精度に置けるよう、物理ピクセル座標も渡す
  drawOrbitOnJuliaCanvasAtComplex(z0Real, z0Imag, px, py)
}

/** 軌道の反復回数表示要素を更新する。 */
function _updateOrbitIterCount(_count) {
  // no-op: 反復回数表示 UI は廃止されたが、呼び出し互換のため残している
}

/**
 * 現在の表示変換で、固定済みの軌道をすべて描き直す。
 * パンやズームで座標変換が変わったあとに呼ぶ。
 */
function _refreshPinnedOrbits() {
  if (orbitDrawEnabled && pinnedOrbit) {
    try {
      drawOrbitOnCanvasAtComplex(
        pinnedOrbit.re,
        pinnedOrbit.im,
        null,
        null,
        pinnedOrbit.reFxp || null,
        pinnedOrbit.imFxp || null,
      )
    } catch (_) {}
  }
  if (orbitDrawEnabled && juliaPinnedOrbit && juliaState?.active) {
    try {
      drawOrbitOnJuliaCanvasAtComplex(juliaPinnedOrbit.re, juliaPinnedOrbit.im)
    } catch (_) {}
  }
}

/** 固定した軌道と対応する描画をすべて消す。 */
function _clearPinnedOrbits() {
  pinnedOrbit = null
  juliaPinnedOrbit = null
  clearOrbitCanvas()
  clearJuliaOrbitCanvas()
}

function _getClientCoordinatesFromEvent(evt) {
  if (evt.touches && evt.touches.length > 0) {
    return [evt.touches[0].clientX, evt.touches[0].clientY]
  }
  if (evt.changedTouches && evt.changedTouches.length > 0) {
    return [evt.changedTouches[0].clientX, evt.changedTouches[0].clientY]
  }
  if (typeof evt.clientX !== 'undefined' && typeof evt.clientY !== 'undefined') {
    return [evt.clientX, evt.clientY]
  }
  return [(evt.pageX || 0) - (window.scrollX || 0), (evt.pageY || 0) - (window.scrollY || 0)]
}

function _togglePinnedOrbitAtClient(clientX, clientY) {
  const [px, py] = toGraphicsCoordinates(clientX, clientY)
  const zoomVal = fractal.zoom.toNumber ? fractal.zoom.toNumber() : 1
  const fw = fractal.width,
    fh = fractal.height
  const scale = (zoomVal * fw) / 4
  const centerX = fractal.center[0].toNumber ? fractal.center[0].toNumber() : 0
  const centerY = fractal.center[1].toNumber ? fractal.center[1].toNumber() : 0
  const re = (px - fw / 2) / scale + centerX
  const im = (py - fh / 2) / scale + centerY
  // FxP の複素座標を求める。高精度な軌道描画と固定解除判定に使う。
  const [reFxp, imFxp] = fractal.canvas2complex ? fractal.canvas2complex(px, py) : [null, null]
  if (pinnedOrbit) {
    // 可能なら FxP を使い、オーバーレイ上のピクセル座標で比較する。
    const orbitCanvas = document.getElementById('orbit-canvas')
    const ow = orbitCanvas ? orbitCanvas.offsetWidth : fw
    const oh = orbitCanvas ? orbitCanvas.offsetHeight : fh
    // クリック位置は物理ピクセルから直接求める
    const ax = (px / fw) * ow
    const ay = (py / fh) * oh
    let bx, by
    if (pinnedOrbit.reFxp && pinnedOrbit.imFxp) {
      try {
        ;[bx, by] = _fxpComplexToOrbitCss(pinnedOrbit.reFxp, pinnedOrbit.imFxp, ow, oh)
      } catch (_) {
        bx = (((pinnedOrbit.re - centerX) * scale + fw / 2) / fw) * ow
        by = (((pinnedOrbit.im - centerY) * scale + fh / 2) / fh) * oh
      }
    } else {
      bx = (((pinnedOrbit.re - centerX) * scale + fw / 2) / fw) * ow
      by = (((pinnedOrbit.im - centerY) * scale + fh / 2) / fh) * oh
    }
    if (Math.sqrt((ax - bx) ** 2 + (ay - by) ** 2) < 8) {
      pinnedOrbit = null
      clearOrbitCanvas()
      return
    }
  }
  pinnedOrbit = { re, im, reFxp, imFxp }
  try {
    drawOrbitOnCanvasAtComplex(re, im, px, py, reFxp, imFxp)
  } catch (_) {}
}

function _togglePinnedJuliaOrbitAtClient(clientX, clientY) {
  const renderer = juliaState.renderer
  if (isJuliaCanvasInteractionBlockedByBuddhabrot()) return
  if (!juliaState.active || !renderer) return
  const [px, py] = toGraphicsCoordinatesOnCanvas(renderer.canvas, clientX, clientY)
  const jZoom = renderer.zoom.toNumber ? renderer.zoom.toNumber() : 1
  const jW = renderer.canvas.width,
    jH = renderer.canvas.height
  const scale = (jZoom * jW) / 4
  const jCenterX = renderer.center[0].toNumber ? renderer.center[0].toNumber() : 0
  const jCenterY = renderer.center[1].toNumber ? renderer.center[1].toNumber() : 0
  const re = (px - jW / 2) / scale + jCenterX
  const im = (py - jH / 2) / scale + jCenterY
  if (juliaPinnedOrbit) {
    const jOrbitCanvas = document.getElementById('julia-orbit-canvas')
    const ow = jOrbitCanvas ? jOrbitCanvas.offsetWidth : jW
    const oh = jOrbitCanvas ? jOrbitCanvas.offsetHeight : jH
    const toPx = (r, i) => [
      (((r - jCenterX) * scale + jW / 2) / jW) * ow,
      (((i - jCenterY) * scale + jH / 2) / jH) * oh,
    ]
    const [ax, ay] = toPx(re, im)
    const [bx, by] = toPx(juliaPinnedOrbit.re, juliaPinnedOrbit.im)
    if (Math.sqrt((ax - bx) ** 2 + (ay - by) ** 2) < 8) {
      juliaPinnedOrbit = null
      clearJuliaOrbitCanvas()
      return
    }
  }
  juliaPinnedOrbit = { re, im }
  try {
    drawOrbitOnJuliaCanvasAtComplex(re, im, px, py)
  } catch (_) {}
}

function onMouseDown(evt) {
  // Buddhabrot 表示中、またはトグルがオンの間は、表示を固定するため
  // パンやドラッグの開始を許可しない。
  if (isMainCanvasInteractionBlockedByBuddhabrot()) return
  _orbitPinDragged = false
  updateMousePos(evt)
  dragStart = [lastX, lastY]
}

function onMouseMove(evt) {
  const [clientX, clientY] = _getClientCoordinatesFromEvent(evt)
  const buddhaOnJuliaView = isBuddhabrotViewShownOnJulia()
  // 固定した軌道の十字付近では、解除できることが分かるようポインター表示にする
  if (orbitDrawEnabled && pinnedOrbit) {
    const [px_c, py_c] = toGraphicsCoordinates(clientX, clientY)
    const zoomVal = fractal.zoom.toNumber ? fractal.zoom.toNumber() : 1
    const fw = fractal.width,
      fh = fractal.height
    const scale = (zoomVal * fw) / 4
    const centerX = fractal.center[0].toNumber ? fractal.center[0].toNumber() : 0
    const centerY = fractal.center[1].toNumber ? fractal.center[1].toNumber() : 0
    const orbitCanvas = document.getElementById('orbit-canvas')
    const ow = orbitCanvas ? orbitCanvas.offsetWidth : fw
    const oh = orbitCanvas ? orbitCanvas.offsetHeight : fh
    const ax = (px_c / fw) * ow
    const ay = (py_c / fh) * oh
    let bx, by
    if (pinnedOrbit.reFxp && pinnedOrbit.imFxp) {
      try {
        ;[bx, by] = _fxpComplexToOrbitCss(pinnedOrbit.reFxp, pinnedOrbit.imFxp, ow, oh)
      } catch (_) {
        bx = (((pinnedOrbit.re - centerX) * scale + fw / 2) / fw) * ow
        by = (((pinnedOrbit.im - centerY) * scale + fh / 2) / fh) * oh
      }
    } else {
      bx = (((pinnedOrbit.re - centerX) * scale + fw / 2) / fw) * ow
      by = (((pinnedOrbit.im - centerY) * scale + fh / 2) / fh) * oh
    }
    canvasElement.style.cursor = Math.sqrt((ax - bx) ** 2 + (ay - by) ** 2) < 8 ? 'pointer' : ''
  } else {
    canvasElement.style.cursor = ''
  }
  // 軌道表示は早期 return より前に描く。
  // 固定済みのときはホバーによる再描画を行わず、その表示を保つ。
  if (orbitDrawEnabled && !pinnedOrbit) {
    try {
      drawOrbitOnCanvas(clientX, clientY)
    } catch (_e) {}
  }
  // Buddhabrot 表示中、またはトグルがオンの間はパンとドラッグを止める
  if (isMainCanvasInteractionBlockedByBuddhabrot()) return
  updateMousePos(evt)
  if (evt.type === 'mousemove' && (evt.buttons & 1) === 0) {
    // 外からキャンバスへ入ったときの意図しないドラッグ開始を防ぐ
    if (gpuInteractiveRedrawPending) {
      flushGpuInteractiveRedraw()
    }
    dragStart = null
    return
  }

  if (dragStart) {
    if (detailPinnedState) clearPinnedDetailPopup()
    // stopBuddhaPreserveDisplay は runner 実行中だけ呼ぶ。
    // 停止後に呼ぶと density がキャンバスへ描かれ、パンのたびに
    // Buddhabrot がフラクタルの上へ一瞬重なって見える。
    try {
      if (!buddhaOnJuliaView && buddhaRunner?.running) {
        stopBuddhaPreserveDisplay()
      }
    } catch (e) {
      console.warn('Error stopping buddha preserve display in onMouseMove():', e?.message ? e.message : e)
    }
    // ユーザー操作で表示を動かしたときは、ずれた表示で Buddhabrot を
    // 誤って再表示しないよう、Buddha トグルをロックして無効化する。
    if (!buddhaOnJuliaView) {
      buddhaLockedByFractalChange = true
      buddhaPreservedDisplay = false
      const t = document.getElementById('buddha-toggle')
      if (t) {
        t.checked = false
        t.disabled = true
      }
    }
    // 表示を動かしたので保存済み画像を無効化する
    savedFractalImageData = null
    // ピクセル差分を求め、高精度 FxP の複素差分として反映する
    const dx = lastX - dragStart[0]
    const dy = lastY - dragStart[1]

    // canvas2complex() と同じ scale = zoom * width / 4 を再計算する
    const w_fx = fxp.fromNumber(fractal.width, fractal.precision)
    const _scale_fx = fractal.zoom.multiply(w_fx).divide(fxp.fromNumber(4, fractal.precision))

    // 複素平面上の差分 = pixelDelta / scale
    // 非常に小さい差分も残せるよう BigInt 経路で反映する

    const deferDragRedraw = isMainRenderGpuPath()

    if (deferDragRedraw) {
      accumulateInteractivePan(dx, dy)
      trackPendingInteractivePanTransform(dx, dy)
      panCanvas(dx, dy)
      dragStart = [lastX, lastY]
      _orbitPinDragged = true
      scheduleGpuInteractiveRedraw()
      if (juliaState.active) redrawJulia()
    } else {
      applyPixelDeltaToCenter(dx, dy)

      panCanvas(dx, dy)
      dragStart = [lastX, lastY]
      _orbitPinDragged = true
      redraw()
    }
    _refreshPinnedOrbits()
    refreshCoordinateGrid()
  }
}

// 現在のキャンバス画像を指定点まわりで拡大縮小する
// 背景で再描画している間も、ユーザーにはすぐ見た目の変化を返せる
function scaleCanvas(factor, x, y) {
  // console.log(`Scaling canvas by ${factor} around (${x}, ${y})`)
  const tempCtx = tempCanvas.getContext('2d')
  tempCtx.drawImage(canvasElement, 0, 0)
  const ctx = canvasElement.getContext('2d')
  ctx.save()
  ctx.translate(x, y)
  ctx.scale(factor, factor)
  ctx.translate(-x, -y)
  ctx.imageSmoothingEnabled = true
  ctx.drawImage(canvasElement, 0, 0) //, -x, -y, canvasElement.width, canvasElement.height);
  ctx.restore()
}

function panCanvas(dx, dy) {
  const ctx = canvasElement.getContext('2d')
  ctx.save()
  ctx.translate(dx, dy)
  ctx.drawImage(canvasElement, 0, 0)
  ctx.restore()
}

function onMouseUp(evt) {
  updateMousePos(evt)
  dragStart = null
  flushGpuInteractiveRedraw()
}

function updateMousePos(evt) {
  // Use client coordinates and bounding rect for robust mapping across
  // scrolling, CSS transforms, and different device pixel ratios.
  const [clientX, clientY] = _getClientCoordinatesFromEvent(evt)
  ;[lastX, lastY] = toGraphicsCoordinates(clientX, clientY)
}

function toGraphicsCoordinates(x, y) {
  // x,y はビューポート基準の client 座標。CSS やレイアウト差分を吸収するため、
  // bounding client rect を使ってキャンバスのピクセル座標へ変換する。
  const rect = canvasElement.getBoundingClientRect()
  const relX = x - rect.left
  const relY = y - rect.top
  const cw = rect.width || canvasElement.offsetWidth || 1
  const ch = rect.height || canvasElement.offsetHeight || 1
  return [(relX / cw) * canvasElement.width, (relY / ch) * canvasElement.height]
}

// 任意のキャンバス要素に対して使える変換版
function toGraphicsCoordinatesOnCanvas(canvas, clientX, clientY) {
  const rect = canvas.getBoundingClientRect()
  const relX = clientX - rect.left
  const relY = clientY - rect.top
  const cw = rect.width || canvas.offsetWidth || 1
  const ch = rect.height || canvas.offsetHeight || 1
  return [(relX / cw) * canvas.width, (relY / ch) * canvas.height]
}

// ホバーポップアップに表示する情報を計算する
function computeMouseDetails(clientX, clientY, isJulia) {
  if (isJulia) {
    if (isJuliaCanvasInteractionBlockedByBuddhabrot()) return null
    if (!juliaState.active || !juliaState.renderer) return null
    const canvas = juliaCanvasElement
    const [px, py] = toGraphicsCoordinatesOnCanvas(canvas, clientX, clientY)
    const [reFxp, imFxp] = juliaState.renderer.canvas2complex(px, py)
    const re = reFxp.toNumber ? reFxp.toNumber() : 0
    const im = imFxp.toNumber ? imFxp.toNumber() : 0
    const cReal = juliaState.renderer.juliaCRe
    const cImag = juliaState.renderer.juliaCIm
    const iterFn = _getIterFn()
    const maxIter = juliaState.renderer.max_iter ?? fractal.max_iter
    const bailout = _getOrbitBailout(juliaState.renderer)
    const { orbit } = _computeOrbitPoints(re, im, cReal, cImag, iterFn, () => [], maxIter, bailout)
    const iter = orbit.length - 1
    const zFinal = orbit[orbit.length - 1] || [re, im]
    return { re, im, reFxp, imFxp, iter, zFinal }
  } else {
    const [px, py] = toGraphicsCoordinates(clientX, clientY)
    const [reFxp, imFxp] = fractal.canvas2complex(px, py)
    const re = reFxp.toNumber ? reFxp.toNumber() : 0
    const im = imFxp.toNumber ? imFxp.toNumber() : 0
    // z0 入力は軌道描画と同じ方法で読む（ユーザーが編集可能な値）
    const [z0Real, z0Imag] = _getZ0Inputs()
    const cReal = re
    const cImag = im
    let orbit
    const bailout = _getOrbitBailout(fractal)
    if (reFxp !== null && imFxp !== null && fractal.fractalType === 'mandelbrot' && z0Real === 0 && z0Imag === 0) {
      try {
        ;({ orbit } = _computeOrbitHighPrec(reFxp, imFxp, bailout))
      } catch (_) {
        const iterFn = _getIterFn()
        ;({ orbit } = _computeOrbitPoints(
          z0Real,
          z0Imag,
          cReal,
          cImag,
          iterFn,
          () => [],
          fractal.max_iter,
          bailout,
        ))
      }
    } else {
      const iterFn = _getIterFn()
      ;({ orbit } = _computeOrbitPoints(
        z0Real,
        z0Imag,
        cReal,
        cImag,
        iterFn,
        () => [],
        fractal.max_iter,
        bailout,
      ))
    }
    const iter = orbit.length - 1
    const zFinal = orbit[orbit.length - 1] || [z0Real, z0Imag]
    return { re, im, reFxp, imFxp, iter, zFinal }
  }
}

function formatDetailInfo(info) {
  function joinReIm(reStr, imStr) {
    // imStr may already include sign; normalize
    let sign = '+'
    let norm = imStr
    if (imStr.startsWith('-')) {
      sign = '-'
      norm = imStr.substring(1)
    }
    return `${reStr} ${sign} ${norm}i`
  }

  let coordStr
  if (info.reFxp) {
    const xStr = formatFxPCoord(info.reFxp)
    let yStr = formatFxPCoord(info.imFxp)
    yStr = negateDecimalString(yStr)
    coordStr = joinReIm(xStr, yStr)
  } else {
    coordStr = joinReIm(_fmt(info.re), _fmt(info.im))
  }

  const z = info.zFinal
  let zStr = ''
  if (z) {
    const zr = formatFloatWithCoordPrecision(z[0])
    const zi = formatFloatWithCoordPrecision(z[1])
    zStr = joinReIm(zr, zi)
  }

  return `coord: ${coordStr}\niter: ${info.iter}\nz: ${zStr}`
}

function showDetailPopup(text, clientX, clientY) {
  if (!detailPopup) return
  detailPopup.textContent = text
  detailPopup.style.left = `${clientX + 12}px`
  detailPopup.style.top = `${clientY + 12}px`
  detailPopup.hidden = false
}

function hideDetailPopup() {
  if (detailPopup) detailPopup.hidden = true
}

function _getDetailCrosshairCanvas(isJulia) {
  return document.getElementById(isJulia ? 'julia-detail-crosshair-canvas' : 'detail-crosshair-canvas')
}

function _getDetailOwnerCanvas(isJulia) {
  return isJulia ? juliaCanvasElement : canvasElement
}

function _clearDetailCrosshairCanvas(isJulia) {
  const canvas = _getDetailCrosshairCanvas(isJulia)
  if (!canvas) return
  const ctx = canvas.getContext('2d')
  ctx.clearRect(0, 0, canvas.width, canvas.height)
}

function _clearDetailCrosshairs() {
  _clearDetailCrosshairCanvas(false)
  _clearDetailCrosshairCanvas(true)
}

function _getActiveDetailState() {
  return detailPinnedState || detailHoverState
}

function _drawDetailCrosshairAtClient(clientX, clientY, isJulia) {
  const overlayCanvas = _getDetailCrosshairCanvas(isJulia)
  const ownerCanvas = _getDetailOwnerCanvas(isJulia)
  if (!overlayCanvas || !ownerCanvas) return false

  const ow = overlayCanvas.offsetWidth
  const oh = overlayCanvas.offsetHeight
  if (ow < 1 || oh < 1) return false
  if (overlayCanvas.width !== ow || overlayCanvas.height !== oh) {
    overlayCanvas.width = ow
    overlayCanvas.height = oh
  }

  const rect = ownerCanvas.getBoundingClientRect()
  const cw = rect.width || ownerCanvas.offsetWidth || 1
  const ch = rect.height || ownerCanvas.offsetHeight || 1
  const cssX = ((clientX - rect.left) / cw) * ow
  const cssY = ((clientY - rect.top) / ch) * oh

  const ctx = overlayCanvas.getContext('2d')
  ctx.clearRect(0, 0, ow, oh)
  _paintCrosshairOnCtx(ctx, cssX, cssY)
  return true
}

function _renderDetailIndicator() {
  _clearDetailCrosshairs()
  if (!detailEnabled) {
    hideDetailPopup()
    return
  }

  const state = _getActiveDetailState()
  if (!state) {
    hideDetailPopup()
    return
  }

  if (!_showComputedDetailPopupAt(state.clientX, state.clientY, state.isJulia)) {
    if (detailPinnedState === state) detailPinnedState = null
    if (detailHoverState === state) detailHoverState = null
    hideDetailPopup()
    return
  }

  if (detailPinnedState === state) {
    _drawDetailCrosshairAtClient(state.clientX, state.clientY, state.isJulia)
  }
}

function _setDetailHoverState(clientX, clientY, isJulia) {
  detailHoverState = { clientX, clientY, isJulia }
  _renderDetailIndicator()
}

function _clearDetailHoverState() {
  detailHoverState = null
  _renderDetailIndicator()
}

function clearAllDetailIndicators() {
  detailHoverState = null
  detailPinnedState = null
  _renderDetailIndicator()
}

function clearPinnedDetailPopup() {
  detailPinnedState = null
  _renderDetailIndicator()
}

function _showComputedDetailPopupAt(clientX, clientY, isJulia) {
  const info = computeMouseDetails(clientX, clientY, isJulia)
  if (!info) return false
  showDetailPopup(formatDetailInfo(info), clientX, clientY)
  return true
}

function _togglePinnedDetailPopupAt(clientX, clientY, isJulia) {
  if (
    detailPinnedState &&
    detailPinnedState.isJulia === isJulia &&
    Math.hypot(clientX - detailPinnedState.clientX, clientY - detailPinnedState.clientY) < DETAIL_TOUCH_TOGGLE_DISTANCE
  ) {
    clearPinnedDetailPopup()
    return
  }
  detailPinnedState = { clientX, clientY, isJulia }
  _renderDetailIndicator()
}

function initDetailsFeature() {
  detailPopup = document.getElementById('detail-popup') || null
  const detailToggle = document.getElementById('detail-toggle')
  if (detailToggle) {
    detailToggle.addEventListener('change', (e) => {
      detailEnabled = e.target.checked
      if (!detailEnabled) {
        clearAllDetailIndicators()
      } else {
        _renderDetailIndicator()
      }
    })
  }
  canvasElement.addEventListener('mousemove', (evt) => {
    if (!detailEnabled || detailPinnedState) return
    _setDetailHoverState(evt.clientX, evt.clientY, false)
  })
  canvasElement.addEventListener('mouseout', () => {
    if (!detailPinnedState) _clearDetailHoverState()
  })
  if (juliaCanvasElement) {
    juliaCanvasElement.addEventListener('mousemove', (evt) => {
      if (isJuliaCanvasInteractionBlockedByBuddhabrot()) {
        if (detailPinnedState?.isJulia) {
          clearPinnedDetailPopup()
        } else if (detailHoverState?.isJulia) {
          _clearDetailHoverState()
        }
        return
      }
      if (!detailEnabled || detailPinnedState) return
      _setDetailHoverState(evt.clientX, evt.clientY, true)
    })
    juliaCanvasElement.addEventListener('mouseout', () => {
      if (!detailPinnedState) _clearDetailHoverState()
    })
  }
}

// ピクセル差分（整数）を高精度 FxP の複素平面差分へ変換する旧説明。
// 現在は pixelDeltaToFxP を廃止し、applyPixelDeltaToCenter を使う。

function _canvas2complexForView(x, y, center, zoom, precision = fractal.precision) {
  x = Math.round(x)
  y = Math.round(y)
  const w = fxp.fromNumber(fractal.width, precision)
  const h = fxp.fromNumber(fractal.height, precision)
  const zoomFx = zoom.withScale(precision)
  const scale = zoomFx.multiply(w).divide(fxp.fromNumber(4, precision))
  const centerRe = center[0].withScale(precision)
  const centerIm = center[1].withScale(precision)
  try {
    if (scale.bigIntValue() <= 0n) {
      return [centerRe, centerIm]
    }
  } catch (_e) {}
  const r = fxp
    .fromNumber(x, precision)
    .subtract(w.divide(fxp.fromNumber(2, precision)))
    .divide(scale)
  const i = fxp
    .fromNumber(y, precision)
    .subtract(h.divide(fxp.fromNumber(2, precision)))
    .divide(scale)
  return [r.add(centerRe), i.add(centerIm)]
}

function _centerAfterPixelDelta(center, zoom, dx, dy, precision = fractal.precision) {
  // サブピクセル移動はまれで影響も小さいため、まず整数ピクセルへ丸める。
  // 必要なら将来は有理数演算へ拡張できる。
  const dxInt = BigInt(Math.round(dx))
  const dyInt = BigInt(Math.round(dy))

  // scale_fx は canvas2complex と同じ定義
  const w_fx = fxp.fromNumber(fractal.width, precision)
  const scale_fx = zoom.withScale(precision).multiply(w_fx).divide(fxp.fromNumber(4, precision))

  const denom = scale_fx.bigInt // integer representing scale_fx * 2^precision
  // 1 ピクセル差分を表現できる中心座標スケールを決める。
  // denom が 2^(S+P) より大きいと 1 ピクセルが 1 未満の中心単位になるため、
  // denom のビット長から必要な S' を見積もる。
  const denomBitLen = scale_fx.bigInt === 0n ? 1 : scale_fx.bigInt.toString(2).length
  const marginBits = 4 // small safety margin
  const minS = Math.max(center[0].scale, Math.max(0, denomBitLen - precision - marginBits))

  const S = minS // 差分計算に使う中心座標のスケール
  const P = precision
  const shift = BigInt(S + P)

  function computeDeltaBig(pxInt) {
    const sign = pxInt < 0n ? -1n : 1n
    const absPx = pxInt < 0n ? -pxInt : pxInt
    // numerator = absPx * 2^(S+P)
    const numerator = absPx << shift
    // rounding: (numerator + denom/2) / denom
    const rounded = (numerator + denom / 2n) / denom
    return rounded * sign
  }

  const deltaBigRe = computeDeltaBig(dxInt)
  const deltaBigIm = computeDeltaBig(dyInt)

  // キャンバスを右へ動かすと実部は減るため、中心の BigInt 値は減算で更新する
  // scale を S へ広げた場合は、現在の中心座標も先に S へそろえてから差し引く
  let curRe = center[0]
  let curIm = center[1]
  if (curRe.scale !== S) {
    curRe = curRe.withScale(S)
    curIm = curIm.withScale(S)
  }

  const newCenterRe = curRe.bigInt - deltaBigRe
  const newCenterIm = curIm.bigInt - deltaBigIm

  return [new fxp.FxP(newCenterRe, S, BigInt(S)), new fxp.FxP(newCenterIm, S, BigInt(S))]
}

function _zoomViewAroundCanvasPoint(center, zoom, factor, x, y, precision = fractal.precision) {
  const lowerBound = MIN_ZOOM.withScale(precision)
  const zoomFx = zoom.withScale(precision)
  if (zoomFx.leq(lowerBound) && factor < 1) {
    return {
      center: [center[0].withScale(precision), center[1].withScale(precision)],
      zoom: zoomFx,
      appliedFactor: 1,
      zoomChanged: false,
    }
  }

  const ptr = _canvas2complexForView(x, y, center, zoomFx, precision)
  const bigFactor = fxp.fromNumber(factor, precision)
  const rawZoom = zoomFx.multiply(bigFactor).max(lowerBound)
  const renderingWithGpu = fractal._willUseGpuForCurrentRender?.() ?? fractal.useGpu
  const upperBound = getMainZoomLimitForState(
    fractal.fractalType,
    renderingWithGpu,
    precision,
    getZoomClampOptionsForPalette(fractal.paletteComponent?.palette),
  )
  const newZoom = upperBound ? rawZoom.min(upperBound) : rawZoom
  const zoomChanged = newZoom.bigInt !== zoomFx.bigInt
  const centerRe = center[0].withScale(precision)
  const centerIm = center[1].withScale(precision)
  const offsetX = ptr[0].subtract(centerRe)
  const offsetY = ptr[1].subtract(centerIm)
  const appliedFactor = zoomChanged ? newZoom.divide(zoomFx).toNumber() : 1
  const invFactor = fxp.fromNumber(1.0 / appliedFactor, precision)
  const newOffsetX = offsetX.multiply(invFactor)
  const newOffsetY = offsetY.multiply(invFactor)

  return {
    center: [ptr[0].subtract(newOffsetX), ptr[1].subtract(newOffsetY)],
    zoom: newZoom,
    appliedFactor,
    zoomChanged,
  }
}

// BigInt 演算でピクセル差分を fractal.center へ直接反映する。
// 中心座標の整数表現の変化量を計算することで、ズームに対して非常に小さい
// 差分も失われにくくする。
function applyPixelDeltaToCenter(dx, dy) {
  fractal.setCenter(_centerAfterPixelDelta(fractal.center, fractal.zoom, dx, dy, fractal.precision))
}

let devicePixelBoxSize = null
let resizeRafId = null
let pendingResizeEntries = null
let lastAppliedCanvasSizeKey = ''
let lastKnownViewportWidth = typeof window !== 'undefined' ? window.innerWidth : 0
let lastKnownViewportHeight = typeof window !== 'undefined' ? window.innerHeight : 0
let lastKnownDevicePixelRatio = typeof window !== 'undefined' ? window.devicePixelRatio || 1 : 1
let lastKnownScrollX = typeof window !== 'undefined' ? window.scrollX || 0 : 0
let lastKnownScrollY = typeof window !== 'undefined' ? window.scrollY || 0 : 0
let lastViewportScrollAt = 0
let canvasPixelResizeObserver = null
let devicePixelRatioQuery = null
let devicePixelRatioQueryHandler = null
let suppressResizeRedrawUntil = 0
const SCROLL_RESIZE_SUPPRESSION_MS = 350

function shouldSuppressResizeRedraw() {
  return performance.now() < suppressResizeRedrawUntil
}

function noteViewportScroll() {
  lastKnownScrollX = window.scrollX || 0
  lastKnownScrollY = window.scrollY || 0
  lastViewportScrollAt = performance.now()
}

function didViewportScrollRecently() {
  const scrollX = window.scrollX || 0
  const scrollY = window.scrollY || 0
  if (scrollX !== lastKnownScrollX || scrollY !== lastKnownScrollY) {
    lastKnownScrollX = scrollX
    lastKnownScrollY = scrollY
    lastViewportScrollAt = performance.now()
    return true
  }
  return performance.now() - lastViewportScrollAt < SCROLL_RESIZE_SUPPRESSION_MS
}

function shouldIgnoreScrollOnlyResize(entries) {
  if (document.fullscreenElement) return false
  const currentWidth = window.innerWidth
  const currentHeight = window.innerHeight
  const currentDevicePixelRatio = getWindowDevicePixelRatio()
  const widthChanged = currentWidth !== lastKnownViewportWidth
  const heightChanged = currentHeight !== lastKnownViewportHeight
  const devicePixelRatioChanged = currentDevicePixelRatio !== lastKnownDevicePixelRatio
  const hasObservedCanvasResize = entries && entries.length > 0
  if ((heightChanged || hasObservedCanvasResize) && !widthChanged && !devicePixelRatioChanged && didViewportScrollRecently()) {
    lastKnownViewportHeight = currentHeight
    lastKnownDevicePixelRatio = currentDevicePixelRatio
    return true
  }
  lastKnownViewportWidth = currentWidth
  lastKnownViewportHeight = currentHeight
  lastKnownDevicePixelRatio = currentDevicePixelRatio
  return false
}

function getDisplayPixelCanvasSize(canvas = canvasElement) {
  try {
    const rect = canvas.getBoundingClientRect()
    const cssWidth = rect.width || canvas.offsetWidth || 1
    const cssHeight = rect.height || canvas.offsetHeight || 1
    return [Math.max(1, Math.round(cssWidth)), Math.max(1, Math.round(cssHeight))]
  } catch (_e) {
    return [Math.max(1, Math.round(canvas?.offsetWidth || 1)), Math.max(1, Math.round(canvas?.offsetHeight || 1))]
  }
}

function getWindowDevicePixelRatio() {
  const dpr = window.devicePixelRatio || 1
  return Number.isFinite(dpr) && dpr > 0 ? dpr : 1
}

function differsFromDisplayPixels(width, height, displayWidth, displayHeight) {
  return Math.round(width) !== Math.round(displayWidth) || Math.round(height) !== Math.round(displayHeight)
}

function getDevicePixelContentBoxCanvasSize(entry) {
  try {
    const size = Array.isArray(entry?.devicePixelContentBoxSize)
      ? entry.devicePixelContentBoxSize[0]
      : entry?.devicePixelContentBoxSize
    const rawWidth = Number(size?.inlineSize)
    const rawHeight = Number(size?.blockSize)
    if (!Number.isFinite(rawWidth) || !Number.isFinite(rawHeight) || rawWidth <= 0 || rawHeight <= 0) return null
    const width = Math.max(1, Math.round(rawWidth))
    const height = Math.max(1, Math.round(rawHeight))
    const [displayWidth, displayHeight] = getDisplayPixelCanvasSize(canvasElement)
    if (!differsFromDisplayPixels(width, height, displayWidth, displayHeight)) return null
    return [width, height]
  } catch (_e) {
    return null
  }
}

function getDevicePixelRatioCanvasSize(canvas = canvasElement) {
  try {
    const dpr = getWindowDevicePixelRatio()
    if (dpr <= 1) return null
    const [displayWidth, displayHeight] = getDisplayPixelCanvasSize(canvas)
    const width = Math.max(1, Math.round(displayWidth * dpr))
    const height = Math.max(1, Math.round(displayHeight * dpr))
    if (!differsFromDisplayPixels(width, height, displayWidth, displayHeight)) return null
    return [width, height]
  } catch (_e) {
    return null
  }
}

function updateFullResToggleAvailability() {
  if (typeof fullResToggle !== 'undefined' && fullResToggle !== null) {
    fullResToggle.disabled = devicePixelBoxSize == null
  }
}

function fullResUsesPhysicalPixels() {
  return fullResToggle?.checked && devicePixelBoxSize != null
}

function getCanvasBufferSizeForDisplaySize(displayWidth, displayHeight) {
  const width = Math.max(1, Math.round(displayWidth))
  const height = Math.max(1, Math.round(displayHeight))
  if (!fullResUsesPhysicalPixels()) return [width, height]
  const dpr = getWindowDevicePixelRatio()
  const physicalWidth = Math.max(1, Math.round(width * dpr))
  const physicalHeight = Math.max(1, Math.round(height * dpr))
  if (!differsFromDisplayPixels(physicalWidth, physicalHeight, width, height)) return [width, height]
  return [physicalWidth, physicalHeight]
}

function syncDevicePixelRatioWatcher() {
  if (typeof window === 'undefined' || typeof window.matchMedia !== 'function') return
  if (devicePixelRatioQuery && devicePixelRatioQueryHandler) {
    if (typeof devicePixelRatioQuery.removeEventListener === 'function') {
      devicePixelRatioQuery.removeEventListener('change', devicePixelRatioQueryHandler)
    } else if (typeof devicePixelRatioQuery.removeListener === 'function') {
      devicePixelRatioQuery.removeListener(devicePixelRatioQueryHandler)
    }
  }

  const dpr = getWindowDevicePixelRatio()
  lastKnownDevicePixelRatio = dpr
  devicePixelRatioQuery = window.matchMedia(`(resolution: ${dpr}dppx)`)
  devicePixelRatioQueryHandler = () => {
    syncDevicePixelRatioWatcher()
    scheduleResize()
  }

  if (typeof devicePixelRatioQuery.addEventListener === 'function') {
    devicePixelRatioQuery.addEventListener('change', devicePixelRatioQueryHandler)
  } else if (typeof devicePixelRatioQuery.addListener === 'function') {
    devicePixelRatioQuery.addListener(devicePixelRatioQueryHandler)
  }
}

function observeCanvasPixelSize() {
  if (typeof ResizeObserver === 'undefined' || canvasPixelResizeObserver) return
  canvasPixelResizeObserver = new ResizeObserver((entries) => {
    scheduleResize(entries)
  })
  try {
    canvasPixelResizeObserver.observe(canvasElement, { box: 'device-pixel-content-box' })
  } catch (_e) {
    canvasPixelResizeObserver.observe(canvasElement)
  }
}

function setStyleIfChanged(element, property, value) {
  if (!element) return false
  if (element.style[property] === value) return false
  element.style[property] = value
  return true
}

function setCanvasBufferSizeIfChanged(canvas, width, height) {
  if (!canvas) return false
  let changed = false
  if (canvas.width !== width) {
    canvas.width = width
    changed = true
  }
  if (canvas.height !== height) {
    canvas.height = height
    changed = true
  }
  return changed
}

function scheduleResize(entries) {
  pendingResizeEntries = entries || null
  if (resizeRafId != null) return
  resizeRafId = requestAnimationFrame(() => {
    resizeRafId = null
    const queuedEntries = pendingResizeEntries
    pendingResizeEntries = null
    onResize(queuedEntries)
  })
}

function onResize(entries) {
  // let debugText = `${canvasElement.offsetWidth}x${canvasElement.offsetHeight}`
  if (shouldIgnoreScrollOnlyResize(entries)) return false

  devicePixelBoxSize = null
  if (entries && entries.length > 0) {
    const entry = entries[0]
    devicePixelBoxSize = getDevicePixelContentBoxCanvasSize(entry)
  }
  if (devicePixelBoxSize == null) {
    devicePixelBoxSize = getDevicePixelRatioCanvasSize()
  }
  lastKnownDevicePixelRatio = getWindowDevicePixelRatio()
  updateFullResToggleAvailability()
  return resizeToCanvasSize()
}

function refreshDevicePixelBoxSize() {
  devicePixelBoxSize = getDevicePixelRatioCanvasSize()
  updateFullResToggleAvailability()
}

function resizeToCanvasSize() {
  // ── Julia Mode ──────────────────────────────────────────────────────────
  if (juliaState?.active) {
    // ── フルスクリーン + Julia: Julia は全画面、MB はプレビューサイズ ──
    if (document.fullscreenElement) {
      const vw = window.innerWidth
      const vh = window.innerHeight

      // Julia キャンバスはビューポート全体を埋める
      const juliaWrap = document.querySelector('.julia-canvas-wrap')
      if (juliaWrap) {
        setStyleIfChanged(juliaWrap, 'width', `${vw}px`)
        setStyleIfChanged(juliaWrap, 'height', `${vh}px`)
        setStyleIfChanged(juliaWrap, 'flex', '0 0 auto')
      }
      // Hi DPI: CSS表示サイズ(vw/vh)は変えず、バッファを物理ピクセルに拡大する
      const [juliaBufferWidth, juliaBufferHeight] = getCanvasBufferSizeForDisplaySize(vw, vh)
      const jCanvas = document.getElementById('julia-canvas')
      let juliaCanvasChanged = false
      if (jCanvas && juliaState.renderer) {
        juliaCanvasChanged = setCanvasBufferSizeIfChanged(jCanvas, juliaBufferWidth, juliaBufferHeight)
        if (jCanvasChanged) juliaState.renderer.resized()
      }

      // MB キャンバスは設定パネル内のプレビューなので、CSS 上の表示サイズを使う
      const mbW = Math.max(1, canvasElement.offsetWidth)
      const mbH = Math.max(1, canvasElement.offsetHeight)
      const mainCanvasChanged = setCanvasBufferSizeIfChanged(canvasElement, mbW, mbH)
      if (mainCanvasChanged) {
        resizeTmpCanvas()
        fractal.resized()
      }

      const sizeEl = document.getElementById('sizeValue')
      if (sizeEl) sizeEl.innerText = `${juliaBufferWidth}x${juliaBufferHeight}`
      showZoomFactor()
      const canvasSizeChanged = mainCanvasChanged || juliaCanvasChanged
      if (canvasSizeChanged && !shouldSuppressResizeRedraw()) redraw()
      return canvasSizeChanged
    }

    // ── 非フルスクリーン Julia: 2 枚のキャンバスを縦積みで収める ──
    const mandelbrotDiv = document.getElementById('mandelbrot')
    const titleEl = document.querySelector('.site-title')
    const titleH = EMBEDDED_MODE ? 0 : titleEl ? titleEl.offsetHeight + 8 : 50

    // 1412px 以上では #mandelbrot.julia-mode は CSS で高さ管理されるため、
    // offsetHeight が使える。未満では高さが内容依存なので、viewport 基準で計算する。
    const isWideLayout = !EMBEDDED_MODE && window.innerWidth >= 1412
    const availViewH =
      EMBEDDED_MODE
        ? window.innerHeight
        : isWideLayout && mandelbrotDiv && mandelbrotDiv.offsetHeight > 150
        ? mandelbrotDiv.offsetHeight
        : window.innerHeight - titleH - 40 // 28 = gaps + padding

    // 幅も同様に、広いレイアウトでは offsetWidth、それ以外は viewport 幅を使う。
    const availW =
      EMBEDDED_MODE
        ? window.innerWidth
        : isWideLayout && mandelbrotDiv && mandelbrotDiv.offsetWidth > 100
        ? mandelbrotDiv.offsetWidth
        : Math.min(window.innerWidth, 800)
    const canvasGap = 8 // 2 枚のキャンバスの間隔
    const naturalW = 800
    const naturalH = 600
    const naturalAspect = naturalW / naturalH

    const maxH = Math.max(100, Math.floor((availViewH - canvasGap) / 2))
    const maxW = availW

    // 縦横とも収まるよう等比で縮小する
    const scaleH = maxH / naturalH
    const scaleW = maxW / naturalW
    const scale = Math.min(1, scaleH, scaleW)
    const canvasH = Math.max(1, Math.round(naturalH * scale))
    const canvasW = Math.max(1, Math.round(naturalH * scale * naturalAspect))

    // 両方のキャンバス枠へ明示サイズを設定する。
    // flex:none にして、flex レイアウトによる上書きを防ぐ。
    const mbWrap = document.getElementById('mandelbrot-canvas-wrap')
    if (mbWrap) {
      setStyleIfChanged(mbWrap, 'width', `${canvasW}px`)
      setStyleIfChanged(mbWrap, 'height', `${canvasH}px`)
      setStyleIfChanged(mbWrap, 'flex', '0 0 auto')
    }
    const juliaWrap = document.querySelector('.julia-canvas-wrap')
    if (juliaWrap) {
      setStyleIfChanged(juliaWrap, 'width', `${canvasW}px`)
      setStyleIfChanged(juliaWrap, 'height', `${canvasH}px`)
      setStyleIfChanged(juliaWrap, 'flex', '0 0 auto')
    }

    // Hi DPI: CSS表示サイズ(canvasW/canvasH)は変えず、バッファを物理ピクセルに拡大する
    const [bufW, bufH] = getCanvasBufferSizeForDisplaySize(canvasW, canvasH)

    const sizeEl = document.getElementById('sizeValue')
    if (sizeEl) sizeEl.innerText = `${bufW}x${bufH}`

    // メインキャンバスの実ピクセルサイズを設定する
    const mainCanvasChanged = setCanvasBufferSizeIfChanged(canvasElement, bufW, bufH)
    if (mainCanvasChanged) resizeTmpCanvas()

    // Julia キャンバスの実ピクセルサイズを設定する
    const jCanvas = document.getElementById('julia-canvas')
    let juliaCanvasChanged = false
    if (jCanvas && juliaState.renderer) {
      juliaCanvasChanged = setCanvasBufferSizeIfChanged(jCanvas, bufW, bufH)
      if (juliaCanvasChanged) juliaState.renderer.resized()
    }

    if (mainCanvasChanged) fractal.resized()

    // Buddhabrot 表示中は、その結果を描き直して見た目を保つ
    if (buddhaActive && buddhaRunner) {
      try {
        const pal = buddhaRunner.palette || buildPaletteFromId(document.getElementById('buddha-palette')?.value)
        drawBuddhaDensityChannels(
          buddhaRunner.densityR,
          buddhaRunner.densityG,
          buddhaRunner.densityB,
          buddhaRunner.width,
          buddhaRunner.height,
          pal,
          buddhaRunner.brightness,
          buddhaRunner.gamma,
        )
      } catch (_e) {}
      showZoomFactor()
      return mainCanvasChanged || juliaCanvasChanged
    }

    showZoomFactor()
    const canvasSizeChanged = mainCanvasChanged || juliaCanvasChanged
    if (canvasSizeChanged && !shouldSuppressResizeRedraw()) redraw()
    return canvasSizeChanged
  }

  // ── 通常モード ──────────────────────────────────────────
  const mandelbrotDiv = document.getElementById('mandelbrot')
  const mbWrap = document.getElementById('mandelbrot-canvas-wrap')
  const isWideLayout = window.innerWidth >= 1412
  if (mandelbrotDiv) {
    if (isWideLayout) {
      setStyleIfChanged(mandelbrotDiv, 'height', '')
      setStyleIfChanged(mandelbrotDiv, 'width', '')
      setStyleIfChanged(mandelbrotDiv, 'flex', '')
    } else {
      const targetWidth = Math.max(1, Math.round(mandelbrotDiv.clientWidth || canvasElement.offsetWidth || 1))
      setStyleIfChanged(mandelbrotDiv, 'height', `${Math.round((targetWidth * 3) / 4)}px`)
      setStyleIfChanged(mandelbrotDiv, 'width', '')
      setStyleIfChanged(mandelbrotDiv, 'flex', '')
    }
  }
  if (mbWrap) {
    setStyleIfChanged(mbWrap, 'width', '100%')
    setStyleIfChanged(mbWrap, 'height', '100%')
    setStyleIfChanged(mbWrap, 'flex', '1 1 auto')
  }

  let width = Math.max(1, Math.round(mandelbrotDiv?.clientWidth || canvasElement.offsetWidth || 1))
  let height = Math.max(1, Math.round(mandelbrotDiv?.clientHeight || canvasElement.offsetHeight || 1))

  if (devicePixelBoxSize == null && fullResToggle?.checked) {
    refreshDevicePixelBoxSize()
  }

  if (fullResUsesPhysicalPixels()) {
    ;[width, height] = devicePixelBoxSize
  }

  const sizeEl = document.getElementById('sizeValue')
  if (sizeEl) sizeEl.innerText = `${width}x${height}`

  const sizeKey = `${width}x${height}|julia:${juliaState?.active ? 1 : 0}|fs:${document.fullscreenElement ? 1 : 0}|hidpi:${fullResUsesPhysicalPixels() ? 1 : 0}`
  if (lastAppliedCanvasSizeKey === sizeKey && canvasElement.width === width && canvasElement.height === height) {
    showZoomFactor()
    return false
  }
  lastAppliedCanvasSizeKey = sizeKey

  const mainCanvasChanged = setCanvasBufferSizeIfChanged(canvasElement, width, height)

  if (mainCanvasChanged) resizeTmpCanvas()
  // Buddhabrot 実行中は通常の Mandelbrot 再描画へ戻さず、既存結果を描き直す。
  // これにより、フルスクリーン切替時の描画切替や長時間ジョブの中断を防げる。
  if (buddhaActive && buddhaRunner) {
    // runner 内部バッファのサイズは変えず、既存 density を描き直して
    // アスペクト比を保ったままキャンバスへ合わせる。
    try {
      const pal = buddhaRunner.palette || buildPaletteFromId(document.getElementById('buddha-palette')?.value)
      drawBuddhaDensityChannels(
        buddhaRunner.densityR,
        buddhaRunner.densityG,
        buddhaRunner.densityB,
        // 元の内部バッファサイズをそのまま使う
        buddhaRunner.width,
        buddhaRunner.height,
        pal,
        buddhaRunner.brightness,
        buddhaRunner.gamma,
      )
    } catch (e) {
      console.warn('Error drawing buddha density in resizeToCanvasSize():', e?.message ? e.message : e)
    }
    // UI 表示だけ更新し、重い再描画は起こさない
    showZoomFactor()
    return mainCanvasChanged
  }

  if (mainCanvasChanged) fractal.resized()
  showZoomFactor()
  if (mainCanvasChanged && !shouldSuppressResizeRedraw()) redraw()
  return mainCanvasChanged
}

function toggleFullScreen() {
  if (document.fullscreenElement) {
    document.exitFullscreen()
  } else {
    document.getElementById('main').requestFullscreen()
  }
}

const ELEMENTS_WITH_FS_CLASS = ['mandelbrot', 'palette-canvas', 'settings', 'footer', 'menu-toggle']

function resizeTmpCanvas() {
  tempCanvas.width = canvasElement.width
  tempCanvas.height = canvasElement.height
}

// 選択中のフラクタル種別に応じて GPU オプション表示を調整する
function updateGpuVisibility(_selectedType) {
  const parent = DOM.gpuToggle?.parentElement
  if (!parent) return
  // 現在は 'mandelbrot' と 'custom' の両方で GPU を使える
  parent.setAttribute('title', 'Enable WebGPU acceleration')
  parent.style.display = ''
  new bootstrap.Tooltip(parent)
}

// 指定されたフラクタル種別に対して、メイン GPU トグルの状態を整える。
// 現在はすべてのフラクタル種別で GPU を有効にできる。
function enforceMainGpuState(_selectedType) {
  const mainGpu = document.getElementById('gpu')
  if (!mainGpu) return
  // WebGPU が使えるなら、すべてのフラクタル種別で GPU を有効にする
  if (typeof navigator !== 'undefined' && 'gpu' in navigator) {
    mainGpu.disabled = false
    mainGpu.parentElement?.setAttribute('title', 'Enable WebGPU acceleration')
  }
}

let lastTouchDistance = null
let lastTouchCenter = null

function initListeners() {
  addEventListener('fullscreenchange', (_event) => {
    if (document.fullscreenElement) {
      for (const element of ELEMENTS_WITH_FS_CLASS) {
        const el = document.getElementById(element)
        if (el) {
          el.classList.add('fullscreen')
        }
      }
      document.documentElement.setAttribute('data-bs-theme', 'dark')
      // 全画面ではメニューを自動で隠さない。
      // 非表示トグルに気づけない可能性があるため。
      // document.getElementById('menu-toggle').classList.add('hidden')
      // document.getElementById('settings').classList.add('hidden')

      // Julia モードで全画面へ入るときは、MB キャンバスをプレビューへ移し、
      // Julia キャンバスを全画面に広げる。
      if (juliaState?.active) {
        enterJuliaFullscreen()
      }
    } else {
      for (const element of ELEMENTS_WITH_FS_CLASS) {
        const el = document.getElementById(element)
        if (el) {
          el.classList.remove('fullscreen')
        }
      }
      document.documentElement.setAttribute('data-bs-theme', 'light')
      // Julia モードで全画面を抜けるときは、MB キャンバスを戻し、
      // 縦積みレイアウトへ復帰する。
      if (juliaState?.active) {
        exitJuliaFullscreen()
        requestAnimationFrame(() => {
          resizeToCanvasSize()
          redrawJulia()
        })
      }
    }
    // 全画面の出入り時は Buddhabrot を停止するが、フラクタル設定自体は保つ。
    // runner を止めて、不要な重い処理だけを止める。
    try {
      if (buddhaRunner) {
        buddhaRunner.running = false
        // terminate を持たない実装でも落ちないよう守る
        buddhaRunner.terminate?.()
        buddhaRunner = null
      }

      buddhaActive = false
      finishBuddhabrotProgress(BuddhabrotState.targetKind)
      hideInactiveBuddhabrotProgress(BuddhabrotState.targetKind)

      // 全画面切替でキャンバスサイズや scale が変わるため、保存画像を無効化する
      savedFractalImageData = null

      redraw(true)

      // 表示が変わったので、Buddha トグルをロックして無効化する
      buddhaLockedByFractalChange = true
      const t = document.getElementById('buddha-toggle')
      if (t) {
        t.checked = false
        t.disabled = true
      }
    } catch (e) {
      console.warn('Error handling fullscreenchange cleanup:', e?.message ? e.message : e)
    }
  })

  // Julia モードではメインキャンバスサイズを JS で明示設定するため、
  // window / visualViewport 由来の変化を拾う。
  let _juliaResizeTimer = null
  window.addEventListener('resize', () => {
    clearTimeout(_juliaResizeTimer)
    _juliaResizeTimer = setTimeout(() => {
      const currentWidth = window.innerWidth
      const currentHeight = window.innerHeight
      const currentDevicePixelRatio = getWindowDevicePixelRatio()
      const widthChanged = currentWidth !== lastKnownViewportWidth
      const heightChanged = currentHeight !== lastKnownViewportHeight
      const devicePixelRatioChanged = currentDevicePixelRatio !== lastKnownDevicePixelRatio
      if (
        !document.fullscreenElement &&
        !widthChanged &&
        !devicePixelRatioChanged &&
        heightChanged &&
        didViewportScrollRecently()
      ) {
        lastKnownViewportHeight = currentHeight
        return
      }
      lastKnownViewportWidth = currentWidth
      lastKnownViewportHeight = currentHeight
      lastKnownDevicePixelRatio = currentDevicePixelRatio
      if (!widthChanged && !devicePixelRatioChanged && !juliaState?.active && !document.fullscreenElement) return
      scheduleResize()
    }, 60)
  })
  window.addEventListener('orientationchange', () => {
    scheduleResize()
  })
  window.visualViewport?.addEventListener('resize', () => {
    if (!juliaState?.active && !document.fullscreenElement) return
    if (!document.fullscreenElement && didViewportScrollRecently()) return
    scheduleResize()
  })
  window.addEventListener('scroll', noteViewportScroll, { passive: true })
  window.visualViewport?.addEventListener('scroll', noteViewportScroll, { passive: true })
  observeCanvasPixelSize()
  syncDevicePixelRatioWatcher()

  canvasElement.addEventListener('mousedown', onMouseDown)
  canvasElement.addEventListener('mousemove', onMouseMove)
  canvasElement.addEventListener('mouseup', onMouseUp)
  canvasElement.addEventListener('mouseleave', () => {
    canvasElement.style.cursor = ''
    if (orbitDrawEnabled) {
      if (pinnedOrbit) {
        try {
          drawOrbitOnCanvasAtComplex(
            pinnedOrbit.re,
            pinnedOrbit.im,
            null,
            null,
            pinnedOrbit.reFxp || null,
            pinnedOrbit.imFxp || null,
          )
        } catch (_) {}
      } else {
        clearOrbitCanvas()
      }
    }
  })
  canvasElement.addEventListener('click', (evt) => {
    if (performance.now() < suppressOrbitClickUntil) return
    if (!orbitDrawEnabled || _orbitPinDragged) return
    _togglePinnedOrbitAtClient(evt.clientX, evt.clientY)
  })

  canvasElement.addEventListener('wheel', handleScroll, {
    passive: false,
  })

  // Julia キャンバス用の独立スクロールズーム
  if (juliaCanvasElement) {
    juliaCanvasElement.addEventListener('wheel', handleJuliaScroll, {
      passive: false,
    })
    juliaCanvasElement.addEventListener('mousedown', onJuliaMouseDown)
    juliaCanvasElement.addEventListener('mousemove', onJuliaMouseMove)
    juliaCanvasElement.addEventListener('mouseup', onJuliaMouseUp)
    juliaCanvasElement.addEventListener('mouseleave', () => {
      if (juliaCanvasElement) juliaCanvasElement.style.cursor = ''
      onJuliaMouseUp()
      if (orbitDrawEnabled) {
        if (juliaPinnedOrbit) {
          try {
            drawOrbitOnJuliaCanvasAtComplex(juliaPinnedOrbit.re, juliaPinnedOrbit.im)
          } catch (_) {}
        } else {
          clearJuliaOrbitCanvas()
        }
      }
    })
    juliaCanvasElement.addEventListener('click', (evt) => {
      if (performance.now() < suppressJuliaOrbitClickUntil) return
      if (isJuliaCanvasInteractionBlockedByBuddhabrot()) return
      if (!orbitDrawEnabled || _juliaPinDragged || !juliaState.active) return
      _togglePinnedJuliaOrbitAtClient(evt.clientX, evt.clientY)
    })
    juliaCanvasElement.addEventListener(
      'touchstart',
      (evt) => {
        if (!juliaState.active || !juliaState.renderer) return
        if (isJuliaCanvasInteractionBlockedByBuddhabrot()) {
          if (evt.cancelable) evt.preventDefault()
          onJuliaMouseUp()
          juliaLastTouchDistance = null
          juliaLastTouchCenter = null
          juliaOrbitTouchTapCandidate = false
          return
        }
        if (evt.cancelable) evt.preventDefault()
        if (evt.touches.length === 1) {
          juliaOrbitTouchTapCandidate = true
          _juliaPinDragged = false
          juliaDragStart = _juliaCanvasCoordsFromClient(evt.touches[0].clientX, evt.touches[0].clientY)
          juliaLastTouchDistance = null
          juliaLastTouchCenter = null
        }
        if (evt.touches.length === 2) {
          juliaOrbitTouchTapCandidate = false
          juliaDragStart = null
          juliaLastTouchDistance = Math.hypot(
            evt.touches[0].pageX - evt.touches[1].pageX,
            evt.touches[0].pageY - evt.touches[1].pageY,
          )
          juliaLastTouchCenter = [
            (evt.touches[0].clientX + evt.touches[1].clientX) / 2,
            (evt.touches[0].clientY + evt.touches[1].clientY) / 2,
          ]
        }
      },
      { passive: false },
    )
    juliaCanvasElement.addEventListener(
      'touchmove',
      (evt) => {
        if (!juliaState.active || !juliaState.renderer) return
        if (isJuliaCanvasInteractionBlockedByBuddhabrot()) {
          if (evt.cancelable) evt.preventDefault()
          onJuliaMouseUp()
          juliaLastTouchDistance = null
          juliaLastTouchCenter = null
          juliaOrbitTouchTapCandidate = false
          return
        }
        if (evt.cancelable) evt.preventDefault()

        if (evt.touches.length === 1 && juliaDragStart) {
          const renderer = juliaState.renderer
          const [x, y] = _juliaCanvasCoordsFromClient(evt.touches[0].clientX, evt.touches[0].clientY)
          const dx = x - juliaDragStart[0]
          const dy = y - juliaDragStart[1]
          if (dx !== 0 || dy !== 0) invalidateJuliaBuddhabrotView()
          if (detailPinnedState) clearPinnedDetailPopup()

          const p = renderer.precision
          const w_fx = fxp.fromNumber(renderer.width, p)
          const scale_fx = renderer.zoom.withScale(p).multiply(w_fx).divide(fxp.fromNumber(4, p))
          const dxFx = fxp.fromNumber(-dx, p).divide(scale_fx)
          const dyFx = fxp.fromNumber(-dy, p).divide(scale_fx)

          renderer.center = [renderer.center[0].withScale(p).add(dxFx), renderer.center[1].withScale(p).add(dyFx)]

          const ctx = renderer.canvas.getContext('2d')
          ctx.save()
          ctx.translate(dx, dy)
          ctx.drawImage(renderer.canvas, 0, 0)
          ctx.restore()

          juliaDragStart = [x, y]
          _juliaPinDragged = true
          redrawJulia()
          _refreshPinnedOrbits()
          return
        }

        if (evt.touches.length === 2) {
          juliaOrbitTouchTapCandidate = false
          const newTouchDistance = Math.hypot(
            evt.touches[0].pageX - evt.touches[1].pageX,
            evt.touches[0].pageY - evt.touches[1].pageY,
          )
          const newTouchCenter = [
            (evt.touches[0].clientX + evt.touches[1].clientX) / 2,
            (evt.touches[0].clientY + evt.touches[1].clientY) / 2,
          ]
          if (!juliaLastTouchDistance || !juliaLastTouchCenter) {
            juliaLastTouchDistance = newTouchDistance
            juliaLastTouchCenter = newTouchCenter
            return
          }

          const factor = newTouchDistance / juliaLastTouchDistance
          const renderer = juliaState.renderer
          const p = renderer.precision
          const [lastCenterX, lastCenterY] = _juliaCanvasCoordsFromClient(juliaLastTouchCenter[0], juliaLastTouchCenter[1])
          const [newCenterX, newCenterY] = _juliaCanvasCoordsFromClient(newTouchCenter[0], newTouchCenter[1])
          const dx = newCenterX - lastCenterX
          const dy = newCenterY - lastCenterY
          const hasPan = dx !== 0 || dy !== 0
          const w_fx = fxp.fromNumber(renderer.width, p)
          const scale_fx = renderer.zoom.withScale(p).multiply(w_fx).divide(fxp.fromNumber(4, p))
          const dxFx = fxp.fromNumber(-dx, p).divide(scale_fx)
          const dyFx = fxp.fromNumber(-dy, p).divide(scale_fx)
          renderer.center = [renderer.center[0].withScale(p).add(dxFx), renderer.center[1].withScale(p).add(dyFx)]

          const ptr = renderer.canvas2complex(newCenterX, newCenterY)
          const lowerBound = MIN_ZOOM.withScale(p)
          const bigFactor = fxp.fromNumber(factor, p)
          const zoomFx = renderer.zoom.withScale(p)
          const renderingWithGpu = renderer._willUseGpuForCurrentRender?.() ?? renderer.useGpu
          const upperBound = getMainZoomLimitForState(
            renderer.fractalType,
            renderingWithGpu,
            p,
            getZoomClampOptionsForPalette(renderer.paletteComponent?.palette),
          )
          const rawZoom = zoomFx.multiply(bigFactor).max(lowerBound)
          const newZoom = upperBound ? rawZoom.min(upperBound) : rawZoom
          const zoomChanged = newZoom.bigInt !== zoomFx.bigInt
          if (!zoomChanged && !hasPan) {
            juliaLastTouchDistance = newTouchDistance
            juliaLastTouchCenter = newTouchCenter
            return
          }
          invalidateJuliaBuddhabrotView()
          if (detailPinnedState) clearPinnedDetailPopup()
          const appliedFactor = zoomChanged ? newZoom.divide(zoomFx).toNumber() : 1
          const invFactor = fxp.fromNumber(1.0 / appliedFactor, p)
          const offsetX = ptr[0].subtract(renderer.center[0].withScale(p)).multiply(invFactor)
          const offsetY = ptr[1].subtract(renderer.center[1].withScale(p)).multiply(invFactor)
          renderer.center = [ptr[0].subtract(offsetX), ptr[1].subtract(offsetY)]
          renderer.setZoom(newZoom)

          redrawJulia()
          _refreshPinnedOrbits()
          juliaLastTouchDistance = newTouchDistance
          juliaLastTouchCenter = newTouchCenter
        }
      },
      { passive: false },
    )
    juliaCanvasElement.addEventListener('touchend', (evt) => {
      if (isJuliaCanvasInteractionBlockedByBuddhabrot()) {
        onJuliaMouseUp()
        juliaLastTouchDistance = null
        juliaLastTouchCenter = null
        juliaOrbitTouchTapCandidate = false
        return
      }
      const touch = juliaOrbitTouchTapCandidate && evt.changedTouches?.length === 1 ? evt.changedTouches[0] : null
      onJuliaMouseUp()
      juliaLastTouchDistance = null
      juliaLastTouchCenter = null
      juliaOrbitTouchTapCandidate = false
      if (orbitDrawEnabled && touch && !_juliaPinDragged && juliaState.active) {
        suppressJuliaOrbitClickUntil = performance.now() + TOUCH_CLICK_SUPPRESS_MS
        _togglePinnedJuliaOrbitAtClient(touch.clientX, touch.clientY)
      }
      if (detailEnabled && touch && !_juliaPinDragged && juliaState.active) {
        _togglePinnedDetailPopupAt(touch.clientX, touch.clientY, true)
      }
    })
    juliaCanvasElement.addEventListener('touchcancel', () => {
      onJuliaMouseUp()
      juliaLastTouchDistance = null
      juliaLastTouchCenter = null
      juliaOrbitTouchTapCandidate = false
    })
  }

  canvasElement.addEventListener(
    'touchstart',
    (evt) => {
      if (evt.cancelable && evt.touches.length >= 1) evt.preventDefault()
      if (evt.touches.length === 1) {
        orbitTouchTapCandidate = true
        onMouseDown(evt)
      }
      if (evt.touches.length === 2) {
        orbitTouchTapCandidate = false
        lastTouchDistance = Math.hypot(
          evt.touches[0].pageX - evt.touches[1].pageX,
          evt.touches[0].pageY - evt.touches[1].pageY,
        )
        lastTouchCenter = [
          (evt.touches[0].clientX + evt.touches[1].clientX) / 2,
          (evt.touches[0].clientY + evt.touches[1].clientY) / 2,
        ]
      }
    },
    { passive: false },
  )
  canvasElement.addEventListener(
    'touchmove',
    (evt) => {
      if (evt.touches.length === 1) {
        if (evt.cancelable && document.fullscreenElement == null) evt.preventDefault()
        onMouseMove(evt)
        if (document.fullscreenElement != null) {
          // no preventDefault in full-screen mode because this may be used to exit full-screen
        }
      }
      if (evt.touches.length === 2) {
        orbitTouchTapCandidate = false
        if (evt.cancelable && document.fullscreenElement == null) evt.preventDefault()
        // Buddhabrot 表示中、またはトグルがオンならピンチ操作を無視する
        if (isMainCanvasInteractionBlockedByBuddhabrot()) {
          // 次のジェスチャーが乱れないよう、基準距離と中心だけは更新する
          lastTouchDistance = Math.hypot(
            evt.touches[0].pageX - evt.touches[1].pageX,
            evt.touches[0].pageY - evt.touches[1].pageY,
          )
          lastTouchCenter = [
            (evt.touches[0].clientX + evt.touches[1].clientX) / 2,
            (evt.touches[0].clientY + evt.touches[1].clientY) / 2,
          ]
          return
        }
        const newTouchDistance = Math.hypot(
          evt.touches[0].pageX - evt.touches[1].pageX,
          evt.touches[0].pageY - evt.touches[1].pageY,
        )
        const newTouchCenter = [
          (evt.touches[0].pageX + evt.touches[1].pageX) / 2,
          (evt.touches[0].pageY + evt.touches[1].pageY) / 2,
        ]
        const factor = newTouchDistance / lastTouchDistance

        ;[lastX, lastY] = toGraphicsCoordinates(newTouchCenter[0], newTouchCenter[1])
        const [newX, newY] = toGraphicsCoordinates(lastTouchCenter[0], lastTouchCenter[1])

        // 2 本指の中心移動量に合わせてキャンバスをパンする
        const dx = lastX - newX
        const dy = lastY - newY

        const deferPinchRedraw = isMainRenderGpuPath()
        if (deferPinchRedraw) {
          if (detailPinnedState) clearPinnedDetailPopup()
          const pendingView = ensurePendingInteractivePinchView()
          pendingView.center = _centerAfterPixelDelta(pendingView.center, pendingView.zoom, dx, dy, fractal.precision)
          const zoomedPendingView = _zoomViewAroundCanvasPoint(
            pendingView.center,
            pendingView.zoom,
            factor,
            lastX,
            lastY,
            fractal.precision,
          )
          if (!zoomedPendingView.zoomChanged) {
            lastTouchDistance = newTouchDistance
            lastTouchCenter = newTouchCenter
            return
          }
          pendingInteractivePinchView = {
            center: zoomedPendingView.center,
            zoom: zoomedPendingView.zoom,
          }
          trackPendingInteractivePanTransform(dx, dy)
          panCanvas(dx, dy)
          trackPendingInteractiveScaleTransform(zoomedPendingView.appliedFactor, lastX, lastY)
          scaleCanvas(zoomedPendingView.appliedFactor, lastX, lastY)
          scheduleGpuInteractiveRedraw()
          _refreshPinnedOrbits()
        } else {
          applyPixelDeltaToCenter(dx, dy)
          panCanvas(dx, dy)
          zoomWithFactor(factor, 0, { gesture: true })
        }
        lastTouchDistance = newTouchDistance
        lastTouchCenter = newTouchCenter
      }
    },
    { passive: false },
  )
  canvasElement.addEventListener('touchend', (evt) => {
    const touch = orbitTouchTapCandidate && evt.changedTouches?.length === 1 ? evt.changedTouches[0] : null
    onMouseUp(evt)
    lastTouchDistance = null
    lastTouchCenter = null
    flushGpuInteractiveRedraw()
    orbitTouchTapCandidate = false
    if (orbitDrawEnabled && touch && !_orbitPinDragged) {
      suppressOrbitClickUntil = performance.now() + TOUCH_CLICK_SUPPRESS_MS
      _togglePinnedOrbitAtClient(touch.clientX, touch.clientY)
    }
    if (detailEnabled && touch && !_orbitPinDragged) {
      _togglePinnedDetailPopupAt(touch.clientX, touch.clientY, false)
    }
    // evt.preventDefault()
  })
  canvasElement.addEventListener('touchcancel', () => {
    dragStart = null
    lastTouchDistance = null
    lastTouchCenter = null
    orbitTouchTapCandidate = false
    flushGpuInteractiveRedraw()
  })

  DOM.iterations.addEventListener('change', (event) => {
    const val = parseInt(event.target.value, 10)
    const changed = setIterations(val)
    if (changed && !BuddhabrotState.isViewEnabled()) {
      redraw()
      if (juliaState.active && !isJuliaCanvasInteractionBlockedByBuddhabrot()) syncJuliaRendererSettingsFromMain()
      _refreshPinnedOrbits()
    }
  })
  DOM.iterations.addEventListener('keydown', (event) => {
    if (event.key === 'Enter') {
      event.preventDefault()
      event.stopPropagation()
      try {
        const val = parseInt(event.target.value, 10)
        const changed = setIterations(val)
        if (changed && !BuddhabrotState.isViewEnabled()) {
          redraw()
          if (juliaState.active && !isJuliaCanvasInteractionBlockedByBuddhabrot()) syncJuliaRendererSettingsFromMain()
          _refreshPinnedOrbits()
        }
      } catch (e) {
        console.warn('Error handling Enter on iterations input:', e)
      }
    } else {
      event.stopPropagation()
    }
  })
  // どの座標入力欄でも Enter で座標を適用できるようにする
  try {
    const coordIds = ['coordX', 'coordY', 'coordZoom']
    coordIds.forEach((id) => {
      const el = document.getElementById(id)
      if (el) {
        el.addEventListener('keydown', (ev) => {
          if (ev.key === 'Enter') {
            ev.preventDefault()
            ev.stopPropagation()
            applyCoordinates()
          }
        })
      }
    })
  } catch (e) {
    console.warn('Error wiring coord Enter handlers:', e)
  }

  // 詳細ポップアップ機能を初期化する（トグルとホバー追跡）
  initDetailsFeature()

  DOM.smoothToggle.addEventListener('change', (event) => {
    fractal.smooth = event.target.checked
    if (juliaState.renderer) juliaState.renderer.smooth = event.target.checked
    redraw()
    _refreshPinnedOrbits()
  })
  DOM.supersamplingToggle.addEventListener('change', (event) => {
    fractal.supersampling = parseInt(event.target.value, 10)
    // Buddhabrot 表示中は、通常描画へ戻さずそのまま見せ続ける
    if (!buddhaActive) redraw(true) // supersampling 切替時はキャッシュも更新する
  })
  // 不要な再描画を避けるため、最後に適用した反復式を覚えておく
  let lastIterationFunctionValue = DOM.iterationFunctionInput ? DOM.iterationFunctionInput.value : ''

  DOM.fractalTypeSelect.addEventListener('change', (event) => {
    // プリセット選択なら、その値を先に反映する
    // （反復式、z0、fractalType、座標）
    const p = functionPresets[parseInt(event.target.value.split(':')[1], 10)]
    const selectedType = p?.fractalType ? p.fractalType : 'custom'

    const sel = event.target.selectedOptions
      ? event.target.selectedOptions[0]
      : event.target.options[event.target.selectedIndex]
    if (sel && sel.dataset && sel.dataset.presetIndex != null) {
      const idx = parseInt(sel.dataset.presetIndex, 10)
      const preset = functionPresets[idx]
      if (preset) {
        // 反復式の UI と内部状態を反映する
        const expr = preset.expr || String(preset)
        const iterInput = document.getElementById('iterationFunction')
        if (iterInput) iterInput.value = expr
        fractal.fractalType = selectedType
        fractal.label = preset?.label ? preset.label : preset.expr
        fractal.iterationFunction = expr
        fractal.dataPresetIndex = event.target.value

        // custom 以外のプリセット、またはプリセットで反復式を置き換えたときは、
        // 反復式エラー表示を消す。
        if (selectedType !== 'custom') {
          setIterationFunctionLabelError(false)
        }

        // z0 があれば反映し、なければ 0 を使う
        const z0RealInput = document.getElementById('z0-real')
        const z0ImagInput = document.getElementById('z0-imag')
        const z0RealValue = preset.z0Real !== undefined ? preset.z0Real : 0
        const z0ImagValue = preset.z0Imag !== undefined ? preset.z0Imag : 0

        if (z0RealInput) z0RealInput.value = String(z0RealValue)
        if (z0ImagInput) z0ImagInput.value = String(z0ImagValue)

        try {
          fractal.z0Real = z0RealValue
          fractal.z0Imag = z0ImagValue
        } catch (e) {
          console.warn('Could not apply preset z0 to fractal', e)
        }

        const coordXInput = document.getElementById('coordX')
        const coordYInput = document.getElementById('coordY')
        coordXInput.value = String(preset.coordX) || '0'
        // UI 表示は「上=正の虚数」規則のため符号を反転して表示する
        coordYInput.value = String(-(preset.coordY || 0))
        fractal.setCenter([fxp.fromNumber(preset.coordX || 0), fxp.fromNumber(preset.coordY || 0)])
      }
    }

    // GPU オプションの見え方も更新する
    updateGpuVisibility(selectedType)

    // WebGPU が使えるなら、すべてのフラクタル種別で GPU トグルを有効にする
    try {
      const mainGpu = document.getElementById('gpu')
      if (mainGpu && typeof navigator !== 'undefined' && 'gpu' in navigator) {
        mainGpu.disabled = false
        mainGpu.parentElement?.setAttribute('title', 'Enable WebGPU acceleration')
      }
    } catch (e) {
      console.warn('Error updating main GPU toggle on fractalType change:', e?.message ? e.message : e)
    }

    // フラクタル種別変更時は既定の表示へ戻す
    fractal.setZoom(getInitialFractalZoom(fractal.precision))
    fractal.restartWorkers()
    _clearPinnedOrbits()
    redraw(true) // フラクタル種別変更時はキャッシュも更新する
    lastIterationFunctionValue = DOM.iterationFunctionInput ? DOM.iterationFunctionInput.value : ''
    // ユーザーが明示的に種別を変えたので、Buddha トグルをロックし、
    // ユーザーが再開するまで再有効化されないようにする。
    try {
      buddhaLockedByFractalChange = true
      const toggle = document.getElementById('buddha-toggle')
      if (toggle) {
        toggle.checked = false
        toggle.disabled = true
      }
    } catch (e) {
      console.warn('Error locking buddha toggle on fractalType change:', e?.message ? e.message : e)
    }
    // フラクタル種別が変わったら高解像度ダウンロードも無効化する
    disableBuddhaDownload()
    // z0 入力は常に編集可能だが、整合性のためヘルパーは呼んでおく
    setZ0Enabled(true)
    // custom 以外へ変わったら、以前の反復式エラー表示を消して UI を戻す
    if (selectedType !== 'custom') {
      setIterationFunctionLabelError(false)
    }
    // 現在の反復式を再検証し、有効ならエラー表示を消す。
    // 以前の検証結果や worker エラーが残っていても、現状が正しければ戻す。
    try {
      const currVal = DOM.iterationFunctionInput ? DOM.iterationFunctionInput.value : ''
      let validationError = null
      try {
        const jsExpr = getParsedExpression(currVal)
        jsExprToWGSL_safe(jsExpr)
      } catch (err) {
        validationError = err
      }
      if (!validationError) {
        setIterationFunctionLabelError(false)
        try {
          fractal?.shownErrors?.clear()
        } catch (_e) {
          // 致命的ではないので無視する
        }
      }
    } catch (e) {
      console.warn('Error validating iteration function on fractalType change:', e)
    }
  })

  // z0 入力欄の UI 状態を整えるヘルパー。
  // 現在は常に編集可能なので enabled 引数は実質使わず、
  // ツールチップ更新だけ整合性のため残している。
  function setZ0Enabled(_enabled) {
    const elr = document.getElementById('z0-real')
    const eli = document.getElementById('z0-imag')
    if (elr) elr.disabled = false
    if (eli) eli.disabled = false
    // 入力欄の目的が分かるようタイトルだけ更新する
    const container = elr?.parentElement?.parentElement
    if (container) {
      container.setAttribute('title', 'Initial z0 for iteration')
    }
  }

  // 初期状態でも z0 は編集可能にしておく
  const _current = DOM.fractalTypeSelect ? DOM.fractalTypeSelect.value : 'mandelbrot'
  setZ0Enabled(true)

  const applyIterationFunction = async (value) => {
    // 前後の空白を除いて同じ値なら何もしない
    try {
      const newNormalized = String(value ?? '').trim()
      const currNormalized = String(fractal.iterationFunction ?? '').trim()
      if (newNormalized === currNormalized) {
        // 連続発火を避けるため、記録値だけ同期して終える
        lastIterationFunctionValue = value
        return
      }
    } catch (_e) {
      // 正規化に失敗しても、そのまま検証処理へ進む
    }
    // blur/change の時点で即時フィードバックできるよう、
    // クライアント側で式を検証する。
    let validationError = null
    try {
      const jsExpr = getParsedExpression(value)
      jsExprToWGSL_safe(jsExpr)
    } catch (e) {
      validationError = e
    }

    fractal.iterationFunction = value
    fractal.fractalType = 'custom'
    DOM.fractalTypeSelect.value = 'preset:-1'
    fractal.dataPresetIndex = 'preset:-1'

    // custom 関数では z0 入力を使える状態にしておく
    setZ0Enabled(true)

    // 反復式を変えたのでエラー履歴を消す
    fractal.shownErrors.clear()
    // クライアント側検証で失敗したら、すぐにエラー表示へする
    if (validationError) {
      setIterationFunctionLabelError(true)
      // worker エラー表示と揃えるため、shownErrors にも積む
      fractal.shownErrors.add(validationError.message || String(validationError))
    } else {
      setIterationFunctionLabelError(false)
    }

    // 以前は custom 関数で GPU を強制オフにしていたが、現在は説明だけ出し、
    // トグル状態は変更しない。
    DOM.gpuToggle.parentElement.setAttribute('title', 'GPU disabled for non-Mandelbrot fractal types')
    new bootstrap.Tooltip(DOM.gpuToggle.parentElement)

    fractal.restartWorkers()
    _clearPinnedOrbits()
    redraw(true) // 反復式変更時はキャッシュも更新する

    // 次回の focus/blur/change で変化判定できるよう、適用済み値を記録する
    lastIterationFunctionValue = value
  }
  DOM.iterationFunctionInput.addEventListener('change', (event) => {
    const newVal = event.target.value
    if (newVal !== lastIterationFunctionValue) {
      applyIterationFunction(newVal)
      lastIterationFunctionValue = newVal
    }
  })

  DOM.iterationFunctionInput.addEventListener('blur', (event) => {
    const newVal = event.target.value
    if (newVal !== lastIterationFunctionValue) {
      applyIterationFunction(newVal)
      lastIterationFunctionValue = newVal
    }
  })

  DOM.iterationFunctionInput.addEventListener('keydown', (event) => {
    if (event.key === 'Enter') {
      event.preventDefault()
      const newVal = event.target.value
      if (newVal !== lastIterationFunctionValue) {
        applyIterationFunction(newVal)
        lastIterationFunctionValue = newVal
      }
    }
    event.stopPropagation()
  })

  DOM.gpuToggle.addEventListener('change', (event) => {
    fractal.useGpu = event.target.checked
    clearPendingInteractiveRedrawState()
    fractal.applyZoomLimitForConfiguredMode()
    redraw()
  })
  DOM.fullScreenButton.addEventListener('click', (_event) => {
    disableBuddhaDownload()
    toggleFullScreen()
  })
  fullResToggle.addEventListener('change', (_event) => {
    resizeToCanvasSize()
    if (isBuddhabrotViewShownOnJulia()) {
      restoreOrRedrawFractalDisplayAfterClearingBuddha('main')
    } else if (!buddhaActive) {
      redraw()
    }
  })

  // 各スライダーのリセットボタン
  try {
    const resetPaletteDensityBtn = document.getElementById('reset-palette-density')
    if (resetPaletteDensityBtn) {
      resetPaletteDensityBtn.addEventListener('click', () => {
        // スライダー input と同じ挙動にし、フラクタル全体の再計算は起こさない。
        resetControlToDefault('palette-density', UI_DEFAULTS.paletteDensity)
        try {
          paletteComponent.setDensity(UI_DEFAULTS.paletteDensity)
        } catch (e) {
          console.warn('Error setting palette density in reset handler:', e)
        }
        // パーマリンクだけ更新する
        updatePermalink()
      })
    }
  } catch (e) {
    console.warn('Error wiring reset-palette-density button:', e)
  }

  try {
    const resetPaletteRotateBtn = document.getElementById('reset-palette-rotate')
    if (resetPaletteRotateBtn) {
      resetPaletteRotateBtn.addEventListener('click', () => {
        // スライダー input 相当の軽い更新にして、重い再計算を避ける
        resetControlToDefault('palette-rotate', UI_DEFAULTS.paletteRotate)
        try {
          paletteComponent.setRotate(UI_DEFAULTS.paletteRotate)
        } catch (e) {
          console.warn('Error setting palette rotate in reset handler:', e)
        }
        // パーマリンクだけ更新する
        updatePermalink()
      })
    }
  } catch (e) {
    console.warn('Error wiring reset-palette-rotate button:', e)
  }

  // Buddhabrot の明るさとガンマのリセットボタン
  try {
    const resetBuddhaBrightness = document.getElementById('reset-buddha-brightness')
    if (resetBuddhaBrightness) {
      resetBuddhaBrightness.addEventListener('click', () => {
        applyDefaultAndRefresh('buddha-brightness', UI_DEFAULTS.buddhaBrightness, {
          onApply: (val) => {
            try {
              if (buddhaRunner) {
                buddhaRunner.brightness = parseFloat(val)
              }
              // 保持中の Buddhabrot 表示があれば、再マップと再描画を予約する
              const toggle = document.getElementById('buddha-toggle')
              const toggleChecked = toggle ? !!toggle.checked : false
              // 表示が有効なら、リセット値がすぐ反映されるよう更新する
              if (toggleChecked) {
                if (buddhaRunner) {
                  scheduleBuddhaRedraw()
                } else if (savedBuddhaImageData) {
                  remapSavedBuddhaImage(
                    parseFloat(val) || 1.0,
                    buddhaRunner
                      ? buddhaRunner.gamma
                      : parseFloat(document.getElementById('buddha-gamma')?.value) || UI_DEFAULTS.buddhaGamma,
                  )
                }
              }
            } catch (e) {
              console.warn('Error setting buddha brightness in onApply:', e)
            }
          },
        })
      })
    }
  } catch (e) {
    console.warn('Error wiring reset-buddha-brightness button:', e)
  }

  try {
    const resetBuddhaGamma = document.getElementById('reset-buddha-gamma')
    if (resetBuddhaGamma) {
      resetBuddhaGamma.addEventListener('click', () => {
        applyDefaultAndRefresh('buddha-gamma', UI_DEFAULTS.buddhaGamma, {
          onApply: (val) => {
            try {
              if (buddhaRunner) {
                buddhaRunner.gamma = parseFloat(val)
              }
              const toggle = document.getElementById('buddha-toggle')
              const toggleChecked = toggle ? !!toggle.checked : false
              if (toggleChecked) {
                if (buddhaRunner) {
                  scheduleBuddhaRedraw()
                } else if (savedBuddhaImageData) {
                  remapSavedBuddhaImage(
                    buddhaRunner
                      ? buddhaRunner.brightness
                      : parseFloat(document.getElementById('buddha-brightness')?.value) || UI_DEFAULTS.buddhaBrightness,
                    parseFloat(val) || 1.0,
                  )
                }
              }
            } catch (e) {
              console.warn('Error setting buddha gamma in onApply:', e)
            }
          },
        })
      })
    }
  } catch (e) {
    console.warn('Error wiring reset-buddha-gamma button:', e)
  }

  // 明るさとガンマの変更時に表示中の Buddhabrot を更新する
  try {
    const buddhaBrightnessEl = document.getElementById('buddha-brightness')
    if (buddhaBrightnessEl) {
      buddhaBrightnessEl.addEventListener('input', () => {
        const val = parseFloat(buddhaBrightnessEl.value)
        // Buddhabrot 表示中なら runner と表示を更新する
        if (buddhaRunner) {
          buddhaRunner.brightness = val
          // 連続入力をまとめるため再描画は予約で行う
          scheduleBuddhaRedraw()
        }
      })
    }
  } catch (e) {
    console.warn('Error wiring buddha-brightness input handler:', e)
  }

  try {
    const buddhaGammaEl = document.getElementById('buddha-gamma')
    if (buddhaGammaEl) {
      buddhaGammaEl.addEventListener('input', () => {
        try {
          const val = parseFloat(buddhaGammaEl.value)
          if (buddhaRunner) {
            buddhaRunner.gamma = val
            // 重い処理を即時実行せず、再描画を予約する
            scheduleBuddhaRedraw()
          }
        } catch (e) {
          console.warn('Error handling buddha-gamma input:', e)
        }
      })
    }
  } catch (e) {
    console.warn('Error wiring buddha-gamma input handler:', e)
  }

  // Buddhabrot 描画速度のリセットボタン
  try {
    const resetBuddhaDrawSpeed = document.getElementById('reset-buddha-draw-speed')
    if (resetBuddhaDrawSpeed) {
      resetBuddhaDrawSpeed.addEventListener('click', () => {
        applyDefaultAndRefresh('buddha-draw-speed', UI_DEFAULTS.buddhaRenderSpeed, {
          onApply: (val) => {
            try {
              // 表示値も更新する
              const buddhaDrawSpeedValueEl = document.getElementById('buddha-draw-speed-value')
              if (buddhaDrawSpeedValueEl) {
                buddhaDrawSpeedValueEl.textContent = val
              }

              if (buddhaRunner && typeof buddhaRunner.setRenderSpeed === 'function') {
                const _v = parseFloat(val)
                buddhaRunner.setRenderSpeed(_v === 1 ? 0.01 : _v)
              }
            } catch (e) {
              console.warn('Error setting buddha render speed in onApply:', e)
            }
          },
        })
      })
    }
  } catch (e) {
    console.warn('Error wiring reset-buddha-draw-speed button:', e)
  }

  // スライダー変更時に描画速度を更新する
  try {
    const buddhaDrawSpeedEl = document.getElementById('buddha-draw-speed')
    const buddhaDrawSpeedValueEl = document.getElementById('buddha-draw-speed-value')
    if (buddhaDrawSpeedEl) {
      // 表示値を更新する
      const updateValueDisplay = () => {
        if (buddhaDrawSpeedValueEl) {
          buddhaDrawSpeedValueEl.textContent = buddhaDrawSpeedEl.value
        }
      }

      buddhaDrawSpeedEl.addEventListener('input', () => {
        try {
          const val = parseFloat(buddhaDrawSpeedEl.value)
          updateValueDisplay()
          if (buddhaRunner && typeof buddhaRunner.setRenderSpeed === 'function') {
            buddhaRunner.setRenderSpeed(val === 1 ? 0.01 : val)
          }
        } catch (e) {
          console.warn('Error handling buddha-draw-speed input:', e)
        }
      })

      // 初期表示値を入れる
      updateValueDisplay()
    }
  } catch (e) {
    console.warn('Error wiring buddha-draw-speed input handler:', e)
  }

  DOM.resetButton.addEventListener('click', (_event) => {
    reset()
  })

  // Buddhabrot 実行ボタン
  try {
    const buddhaBtn = document.getElementById('buddha-render')
    if (buddhaBtn) {
      buddhaBtn.addEventListener('click', async () => {
        try {
          await startBuddhaRender()
        } catch (e) {
          buddhaRenderPending = false
          console.warn('Error handling buddha-render button click:', e)
        }
      })
    }
  } catch (e) {
    console.warn('Error wiring buddha-render button:', e)
  }

  // Buddhabrot 停止ボタン
  try {
    const stopBtn = document.getElementById('buddha-stop')
    if (stopBtn) {
      stopBtn.addEventListener('click', () => {
        if (buddhaActive && buddhaRunner?.running) {
          stopBuddhaPreserveDisplay()
          // 保持表示を見せ続けるため、トグルを有効にしてオンのまま保つ
          const t = document.getElementById('buddha-toggle')
          if (t) {
            t.disabled = false
            t.checked = true
          }
          buddhaPreservedDisplay = true
          buddhaActive = false
        }
      })
    }
  } catch (e) {
    console.warn('Error wiring buddha-stop button:', e)
  }

  // Buddhabrot 用の高解像度ダウンロードボタン
  try {
    const downloadBtn = document.getElementById('buddha-download')
    if (downloadBtn) {
      downloadBtn.disabled = true
      downloadBtn.addEventListener('click', async () => {
        try {
          if (!buddhaRunner?.densityR) return
          const scaleSel = document.getElementById('buddha-scale')
          const scale = scaleSel ? Math.max(1, parseInt(scaleSel.value, 10) || 1) : 1
          const w = buddhaRunner.width
          const h = buddhaRunner.height
          const off = document.createElement('canvas')
          off.width = w
          off.height = h
          const offCtx2 = off.getContext('2d')
          const img = offCtx2.createImageData(w, h)
          const len = w * h
          const brightness =
            parseFloat(document.getElementById('buddha-brightness')?.value) || buddhaRunner.brightness || 1.8
          const gamma = parseFloat(document.getElementById('buddha-gamma')?.value) || buddhaRunner.gamma || 0.8
          const rBuf = buddhaRunner.densityR
          const gBuf = buddhaRunner.densityG
          const bBuf = buddhaRunner.densityB
          let max = 0
          for (let i = 0; i < len; i++) {
            const v = (rBuf[i] || 0) + (gBuf[i] || 0) + (bBuf[i] || 0)
            if (v > max) max = v
          }
          if (max === 0) return
          for (let i = 0; i < len; i++) {
            const rv = rBuf[i] || 0
            const gv = gBuf[i] || 0
            const bv = bBuf[i] || 0
            const lr = Math.log10(1 + rv) / Math.log10(1 + max)
            const lg = Math.log10(1 + gv) / Math.log10(1 + max)
            const lb = Math.log10(1 + bv) / Math.log10(1 + max)
            const rn = Math.min(1, (lr * brightness) ** gamma)
            const gn = Math.min(1, (lg * brightness) ** gamma)
            const bn = Math.min(1, (lb * brightness) ** gamma)
            const idx = i * 4
            img.data[idx] = Math.round(255 * rn)
            img.data[idx + 1] = Math.round(255 * gn)
            img.data[idx + 2] = Math.round(255 * bn)
            img.data[idx + 3] = 255
          }
          offCtx2.putImageData(img, 0, 0)
          off.toBlob((blob) => {
            if (!blob) return
            const url = URL.createObjectURL(blob)
            const filename = `buddhabrot_${scale}x_${w}x${h}.png`
            const a = document.createElement('a')
            a.href = url
            a.download = filename
            document.body.appendChild(a)
            a.click()
            a.remove()
            setTimeout(() => URL.revokeObjectURL(url), 5000)
          })
        } catch (e) {
          console.warn('Error generating hi-res download:', e?.message ? e.message : e)
        }
      })
    }
  } catch (e) {
    console.warn('Error wiring buddha-download button:', e?.message ? e.message : e)
  }

  // 画像保存ボタン
  try {
    const saveImageBtn = document.getElementById('save-image')
    if (saveImageBtn) {
      saveImageBtn.addEventListener('click', () => {
        saveCanvasAsPNG()
      })
    }
  } catch (e) {
    console.warn('Error wiring save-image button:', e)
  }

  // 座標適用ボタン
  try {
    const applyBtn = document.getElementById('applyCoords')
    if (applyBtn) {
      applyBtn.addEventListener('click', () => {
        applyCoordinates()
      })
    }
  } catch (e) {
    console.warn('Error wiring applyCoords button:', e)
  }

  // 座標リセットボタン
  try {
    const resetPositionBtn = document.getElementById('resetPosition')
    if (resetPositionBtn) {
      resetPositionBtn.addEventListener('click', () => {
        resetPosition()
      })
    }
  } catch (e) {
    console.warn('Error wiring resetPosition button:', e)
  }

  // キーボードショートカット
  try {
    document.addEventListener('keydown', (ev) => {
      try {
        const e = ev || window.event
        const target = e.target || e.srcElement
        // 入力欄の編集中は無視する
        if (target && /^(INPUT|TEXTAREA|SELECT)$/.test(target.tagName)) return

        const key = e.key
        // 全画面切替: f
        if (key === 'f' || key === 'F') {
          const el = document.getElementById('fullscreen')
          if (el) el.click()
          return
        }
        // Buddhabrot 実行: b
        if (key === 'b' || key === 'B') {
          const el = document.getElementById('buddha-render')
          if (el) el.click()
          return
        }
        // Smooth 切替: s
        if (key === 's' || key === 'S') {
          const el = document.getElementById('smooth')
          if (el) {
            el.checked = !el.checked
            el.dispatchEvent(new Event('change', { bubbles: true }))
          }
          return
        }
        // 反復回数 +/- : + / = で増加、- で減少
        if (key === '+' || key === '=') {
          const it = document.getElementById('max-iterations')
          if (it) {
            const val = parseInt(it.value || '0', 10) || 0
            it.value = val + 100
            it.dispatchEvent(new Event('change', { bubbles: true }))
          }
          return
        }
        if (key === '-') {
          const it = document.getElementById('max-iterations')
          if (it) {
            const val = parseInt(it.value || '0', 10) || 0
            it.value = Math.max(0, val - 100)
            it.dispatchEvent(new Event('change', { bubbles: true }))
          }
          return
        }
        // 全設定リセット: Backspace
        if (key === 'Backspace') {
          const el = document.getElementById('reset')
          if (el) el.click()
          // ブラウザの戻る動作を防ぐ
          e.preventDefault()
          return
        }
      } catch (innerE) {
        console.warn('Error in keyboard shortcut handler:', innerE)
      }
    })
  } catch (e) {
    console.warn('Error wiring keyboard shortcuts:', e)
  }

  // z0 入力変更時は状態を更新して再描画する。
  // Mandelbrot で非 0 の z0 を入れた場合は、custom へ切り替えて値を反映する。
  try {
    const z0RealEl = document.getElementById('z0-real')
    const z0ImagEl = document.getElementById('z0-imag')
    const onZ0Change = () => {
      const r = parseFloat(z0RealEl ? z0RealEl.value : '0') || 0
      const i = parseFloat(z0ImagEl ? z0ImagEl.value : '0') || 0
      // Mandelbrot 選択中に既定値以外へ変えたら、先に custom へ切り替える
      if (fractal.fractalType === 'mandelbrot' && (r !== 0 || i !== 0)) {
        fractal.fractalType = 'custom'
        DOM.fractalTypeSelect.value = 'preset:-1'
        fractal.dataPresetIndex = 'preset:-1'
        DOM.gpuToggle.parentElement.setAttribute('title', 'GPU disabled for non-Mandelbrot fractal types')
        new bootstrap.Tooltip(DOM.gpuToggle.parentElement)
        fractal.restartWorkers()
        _clearPinnedOrbits()
      }
      fractal.z0Real = r
      fractal.z0Imag = i
      redraw(true)
    }
    if (z0RealEl) z0RealEl.addEventListener('change', onZ0Change)
    if (z0ImagEl) z0ImagEl.addEventListener('change', onZ0Change)
  } catch (e) {
    console.warn('Error wiring z0 inputs:', e)
  }

  // Escape radius 変更時は状態を更新し、smooth を切って再描画する
  try {
    const escapeRadiusEl = document.getElementById('escape-radius')
    if (escapeRadiusEl) {
      escapeRadiusEl.addEventListener('input', () => {
        const val = parseFloat(escapeRadiusEl.value)
        if (!Number.isNaN(val)) {
          fractal.escapeRadius = Math.max(0, val)
          // escape radius 変更時は smooth coloring を無効化する
          fractal.smooth = false
          if (juliaState.renderer) juliaState.renderer.smooth = false
          const smoothEl = document.getElementById('smooth')
          if (smoothEl) smoothEl.checked = false
          redraw(true)
        }
      })
    }
  } catch (e) {
    console.warn('Error wiring escape-radius input:', e)
  }

  // favorites から Jump To... を組み立て、変更処理を接続する
  try {
    const jumpSel = document.getElementById('jump-to')
    if (jumpSel) {
      const favs = favorites.getFavorites()
      // 選択肢を追加する
      favs.forEach((f, idx) => {
        const opt = document.createElement('option')
        opt.value = String(idx)
        opt.textContent = f.label || `Favorite ${idx + 1}`
        jumpSel.appendChild(opt)
      })

      jumpSel.addEventListener('change', (ev) => {
        try {
          const val = ev.target.value
          const i = parseInt(val, 10)
          const fav = favs[i]
          if (fav?.params) {
            // Buddhabrot が表示中なら停止して消去し、Jump To 後は通常表示へ戻す
            if (typeof buddhaActive !== 'undefined' && buddhaActive) {
              stopAndClearBuddha()
              // Jump To 後に Buddhabrot が復元されないよう、保存データも消す
              discardSavedBuddhaImageData()
            } else if (typeof buddhaPreservedDisplay !== 'undefined' && buddhaPreservedDisplay) {
              // Jump To が選ばれたので保持表示も消す
              stopAndClearBuddha()
            } else if (typeof savedBuddhaImageData !== 'undefined' && savedBuddhaImageData) {
              // 保存済みの Buddhabrot ビットマップがあれば消す
              stopAndClearBuddha()
              discardSavedBuddhaImageData()
            }

            // まず params を展開して目標の center と zoom を取り出す
            const p = JSON.parse(atob(decodeURIComponent(fav.params)))
            const targetZoom = applyEmbeddedInitialZoomCorrection(fxp.fromJSON(p.zoom))
            const targetCenter = p.center.map((c) => fxp.fromJSON(c))

            // favorite へ移動するときは Julia モードを解除する
            if (juliaState?.active) {
              juliaState.active = false
              const jtToggle = document.getElementById('julia-toggle')
              if (jtToggle) jtToggle.checked = false
              const jtMb = document.getElementById('mandelbrot')
              jtMb?.classList.remove('julia-mode')
              const jtSection = document.getElementById('julia-section')
              if (jtSection) {
                jtSection.setAttribute('hidden', '')
                jtSection.classList.remove('visible')
              }
              const jtCrosshair = document.getElementById('julia-crosshair')
              if (jtCrosshair) jtCrosshair.setAttribute('hidden', '')
              const jtResetBtn = document.getElementById('julia-reset')
              if (jtResetBtn) jtResetBtn.setAttribute('hidden', '')
              if (document.fullscreenElement) {
                exitJuliaFullscreen()
              }
              const jtMbWrap = document.getElementById('mandelbrot-canvas-wrap')
              if (jtMbWrap) {
                jtMbWrap.style.width = ''
                jtMbWrap.style.height = ''
                jtMbWrap.style.flex = ''
              }
              const jtJuliaWrap = document.querySelector('.julia-canvas-wrap')
              if (jtJuliaWrap) {
                jtJuliaWrap.style.width = ''
                jtJuliaWrap.style.height = ''
                jtJuliaWrap.style.flex = ''
              }
              resizeToCanvasSize()
            }

            // Julia 中に隠していた Buddhabrot UI を戻す
            const buddhabrotSection = document.getElementById('buddhabrot-section')
            if (buddhabrotSection) buddhabrotSection.removeAttribute('hidden')

            // ほかのパラメータも復元する
            initFromParams(fav.params)
            _clearPinnedOrbits()

            // Move to initial home position immediately, then animate to target
            fractal.setCenter([fxp.fromNumber(-0.5), fxp.fromNumber(0)])
            fractal.setZoom(getInitialFractalZoom(fractal.precision))
            fractal.initPallete()
            redraw()

            const animEnabled = document.getElementById('anim-enable')?.checked
            const SPEED = 0.15 // 固定のアニメーション速度
            const baseMs = 2000
            const duration = Math.max(50, Math.floor(baseMs / SPEED))
            if (animEnabled) {
              // アニメーション時に GPU が有効ならオフへ切り替える
              const gpuEl = document.getElementById('gpu')
              if (gpuEl?.checked) {
                gpuEl.checked = false
                // すぐ反映されるよう内部状態も直接更新する
                fractal.useGpu = false
              }
              startAnimation(targetCenter, targetZoom, duration)
            } else {
              // 即時適用時は、先に zoom を入れてから center を入れる。
              // 深い座標への変換で切り捨てが起きにくくなる。
              fractal.setZoom(targetZoom)
              fractal.setCenter(targetCenter)
              fractal.initPallete()
              redraw()
              updatePermalink()
            }
          }
          // セレクトをプレースホルダーへ戻す
          jumpSel.selectedIndex = 0
        } catch (e) {
          console.warn('Error handling Jump To change:', e)
        }
      })
    }
  } catch (e) {
    console.warn('Error wiring Jump To dropdown:', e?.message ? e.message : e)
  }

  // ユーザーが入力したら反復式エラー表示を消す
  if (DOM.iterationFunctionInput) {
    DOM.iterationFunctionInput.addEventListener('input', () => {
      setIterationFunctionLabelError(false)
    })
  }

  // アニメーション関連 UI
  try {
    const animStopBtn = document.getElementById('anim-stop')
    if (animStopBtn) {
      animStopBtn.addEventListener('click', () => {
        stopAnimation()
      })
    }
    // 速度スライダーは廃止済みで、速度はコード側で固定する
    const animEnableEl = document.getElementById('anim-enable')
    if (animEnableEl) {
      // 既定値はオフ
      animEnableEl.checked = false
      // アニメーショントグルは GPU トグルを直接は触らない
    }
  } catch (e) {
    console.warn('Error wiring animation controls:', e?.message ? e.message : e)
  }

  // Orbit 表示の操作
  try {
    const coordinateGridToggleEl = document.getElementById('coordinate-grid-toggle')
    if (coordinateGridToggleEl) {
      coordinateGridToggleEl.checked = false
      coordinateGridToggleEl.addEventListener('change', (e) => {
        coordinateGridEnabled = e.target.checked
        refreshCoordinateGrid()
      })
    }
    const orbitToggleEl = document.getElementById('orbit-toggle')
    if (orbitToggleEl) {
      orbitToggleEl.addEventListener('change', (e) => {
        orbitDrawEnabled = e.target.checked
        if (!orbitDrawEnabled) {
          pinnedOrbit = null
          juliaPinnedOrbit = null
          clearOrbitCanvas()
          clearJuliaOrbitCanvas()
        }
      })
    }
    const orbitModeEl = document.getElementById('orbit-mode')
    if (orbitModeEl) {
      orbitModeEl.addEventListener('change', (e) => {
        orbitMode = e.target.value
        // モード変更後に固定済み軌道を描き直す
        _refreshPinnedOrbits()
      })
    }
    const orbitGradientToggleEl = document.getElementById('orbit-gradient-toggle')
    if (orbitGradientToggleEl) {
      orbitGradientToggleEl.addEventListener('change', (e) => {
        orbitGradientEnabled = e.target.checked
        _refreshPinnedOrbits()
      })
    }
  } catch (e) {
    console.warn('Error wiring orbit controls:', e?.message ? e.message : e)
  }

  // ── Julia Set モード切替 ────────────────────────────────────────────────
  try {
    const juliaToggleEl = document.getElementById('julia-toggle')
    const juliaResetBtn = document.getElementById('julia-reset')
    if (juliaToggleEl) {
      juliaToggleEl.addEventListener('change', (e) => {
        const nextJuliaActive = e.target.checked
        const stoppedBuddhabrotRendering = stopRenderingForJuliaToggleDuringBuddhabrot()
        juliaState.active = nextJuliaActive
        const mandelbrotDiv = document.getElementById('mandelbrot')
        const juliaSection = document.getElementById('julia-section')
        const crosshair = document.getElementById('julia-crosshair')

        // Buddhabrot UI は Julia 中も表示したままにする
        const buddhabrotSection = document.getElementById('buddhabrot-section')

        if (juliaState.active) {
          // Julia モードを有効化する
          mandelbrotDiv?.classList.add('julia-mode')
          if (buddhabrotSection) buddhabrotSection.removeAttribute('hidden')
          if (juliaSection) {
            juliaSection.removeAttribute('hidden')
            juliaSection.classList.add('visible')
          }
          if (crosshair) crosshair.removeAttribute('hidden')
          if (juliaResetBtn) juliaResetBtn.removeAttribute('hidden')
          // ↺ ボタンが高さを作るので、toggle wrapper の py-1 は外す
          juliaToggleEl.closest('.d-flex.align-items-center')?.classList.remove('py-1')

          if (document.fullscreenElement) {
            enterJuliaFullscreen({ redraw: !stoppedBuddhabrotRendering })
            if (stoppedBuddhabrotRendering) {
              requestAnimationFrame(() => {
                restoreOrRedrawFractalDisplayAfterClearingBuddha('main')
                redrawJulia()
              })
            }
          } else {
            requestAnimationFrame(() => {
              resizeToCanvasSize()
              if (stoppedBuddhabrotRendering) {
                restoreOrRedrawFractalDisplayAfterClearingBuddha('main')
                redrawJulia()
              } else {
                redrawJulia()
              }
            })
          }
        } else {
          // Julia モードを解除する
          mandelbrotDiv?.classList.remove('julia-mode')
          if (buddhabrotSection) buddhabrotSection.removeAttribute('hidden')
          if (juliaSection) {
            juliaSection.setAttribute('hidden', '')
            juliaSection.classList.remove('visible')
            juliaSection.style.width = ''
            juliaSection.style.height = ''
            juliaSection.style.flex = ''
          }
          if (crosshair) crosshair.setAttribute('hidden', '')
          if (juliaResetBtn) juliaResetBtn.setAttribute('hidden', '')
          // ↺ ボタンがないので、toggle wrapper の py-1 を戻す
          juliaToggleEl.closest('.d-flex.align-items-center')?.classList.add('py-1')

          if (document.fullscreenElement) {
            exitJuliaFullscreen()
          }
          // 明示的な inline size を消し、CSS 管理へ戻す
          if (mandelbrotDiv) {
            mandelbrotDiv.style.width = ''
            mandelbrotDiv.style.height = ''
            mandelbrotDiv.style.flex = ''
          }
          const mbWrap = document.getElementById('mandelbrot-canvas-wrap')
          if (mbWrap) {
            mbWrap.style.width = ''
            mbWrap.style.height = ''
            mbWrap.style.flex = ''
          }
          const juliaWrap = document.querySelector('.julia-canvas-wrap')
          if (juliaWrap) {
            juliaWrap.style.width = ''
            juliaWrap.style.height = ''
            juliaWrap.style.flex = ''
          }
          requestAnimationFrame(() => {
            resizeToCanvasSize()
            if (stoppedBuddhabrotRendering) restoreOrRedrawFractalDisplayAfterClearingBuddha('main')
          })
        }
      })
    }
    // Julia リセットボタン
    if (juliaResetBtn) {
      juliaResetBtn.addEventListener('click', () => {
        if (!juliaState.renderer) return
        invalidateJuliaBuddhabrotView({ clearVisible: true })
        const p = juliaState.renderer.precision
        juliaState.renderer.setZoom(fxp.fromNumber(1, p))
        juliaState.renderer.center = [fxp.fromNumber(0, p), fxp.fromNumber(0, p)]
        redrawJulia()
      })
    }
  } catch (e) {
    console.warn('Error wiring julia-toggle handler:', e)
  }
}

function reset() {
  // 表示をリセットするときは高解像度ダウンロードを無効にする
  disableBuddhaDownload()
  // 実行中の Buddhabrot はすぐ停止する
  try {
    stopAndClearBuddha()
  } catch (e) {
    console.warn('Error stopping/clearing buddha in reset():', e?.message ? e.message : e)
  }
  buddhaActive = false

  fractal.setZoom(getInitialFractalZoom(fractal.precision))
  // リセット時の既定中心座標
  let resetCenterX = 0
  let resetCenterY = 0
  const fractalTypeSelect = document.getElementById('fractalType')
  const val = fractalTypeSelect?.value || ''
  // プリセット選択時の値は preset:<index> 形式
  const presetIndex = parseInt(val.split(':')[1], 10)
  const p = functionPresets[presetIndex]
  resetCenterX = p?.coordX || 0
  resetCenterY = p?.coordY || 0

  const coordXEl = document.getElementById('coordX')
  const coordYEl = document.getElementById('coordY')
  if (coordXEl) coordXEl.value = String(resetCenterX)
  // UI 表示は「上=正の虚数」規則のため符号を反転して表示する
  if (coordYEl) coordYEl.value = String(-resetCenterY)

  fractal.setCenter([fxp.fromNumber(resetCenterX), fxp.fromNumber(resetCenterY)])

  // z0 も既定値へ戻す（0 またはプリセット値）
  try {
    const z0RealValue = p?.z0Real !== undefined ? p.z0Real : 0
    const z0ImagValue = p?.z0Imag !== undefined ? p.z0Imag : 0

    const z0RealEl = document.getElementById('z0-real')
    const z0ImagEl = document.getElementById('z0-imag')
    if (z0RealEl) z0RealEl.value = String(z0RealValue)
    if (z0ImagEl) z0ImagEl.value = String(z0ImagValue)

    fractal.z0Real = z0RealValue
    fractal.z0Imag = z0ImagValue
  } catch (e) {
    console.warn('Error resetting z0 in reset():', e?.message ? e.message : e)
  }

  paletteComponent.setDensity(1)
  paletteComponent.setRotate(0)
  // Buddhabrot の UI も既定値へ戻す
  const buddhaModeEl = document.getElementById('buddhaMode')
  if (buddhaModeEl) buddhaModeEl.value = 'buddha'

  const buddhaIterEl = document.getElementById('buddha-iterations')
  if (buddhaIterEl) buddhaIterEl.value = '3000000'

  const buddhaPalEl = document.getElementById('buddha-palette')
  if (buddhaPalEl?.options && buddhaPalEl.options.length > 0) {
    buddhaPalEl.selectedIndex = 0
  }

  try {
    applyDefaultAndRefresh('buddha-brightness', UI_DEFAULTS.buddhaBrightness, {
      onApply: (val) => {
        const fval = parseFloat(val)
        const el = document.getElementById('buddha-brightness')
        if (el) el.value = String(val)
        if (buddhaRunner) {
          buddhaRunner.brightness = fval
        }
      },
    })
  } catch (e) {
    console.warn('Error resetting buddha-brightness in reset():', e?.message ? e.message : e)
  }
  try {
    applyDefaultAndRefresh('buddha-gamma', UI_DEFAULTS.buddhaGamma, {
      onApply: (val) => {
        const fval = parseFloat(val)
        const el = document.getElementById('buddha-gamma')
        if (el) el.value = String(val)
        if (buddhaRunner) {
          buddhaRunner.gamma = fval
          drawBuddhaDensityChannels(
            buddhaRunner.densityR,
            buddhaRunner.densityG,
            buddhaRunner.densityB,
            buddhaRunner.width,
            buddhaRunner.height,
            buddhaRunner.palette,
            buddhaRunner.brightness,
            buddhaRunner.gamma,
          )
        }
      },
    })
  } catch (e) {
    console.warn('Error resetting buddha-gamma in reset():', e?.message ? e.message : e)
  }

  // Render Speed Delay（buddha-draw-speed）を既定値へ戻す
  try {
    applyDefaultAndRefresh('buddha-draw-speed', UI_DEFAULTS.buddhaRenderSpeed, {
      onApply: (val) => {
        const el = document.getElementById('buddha-draw-speed')
        if (el) el.value = String(val)
        const valueDisplay = document.getElementById('buddha-draw-speed-value')
        if (valueDisplay) valueDisplay.textContent = String(val)
      },
    })
  } catch (e) {
    console.warn('Error resetting buddha-draw-speed in reset():', e?.message ? e.message : e)
  }

  // buddha-scale を 1x へ戻す
  const buddhaScaleEl = document.getElementById('buddha-scale')
  if (buddhaScaleEl) buddhaScaleEl.value = '1'

  // メインフラクタルのパレットを mandelbrot に戻す
  try {
    const mandelbrotPalette = palette.getPalette('mandelbrot')
    if (mandelbrotPalette) {
      paletteComponent.setPalette(mandelbrotPalette)
      const paletteDropdown = document.getElementById('palette-dropdown')
      if (paletteDropdown) paletteDropdown.value = 'mandelbrot'
      // custom パレットでなくなったので Orbit trap パネルは閉じる
      paletteComponent._updateOrbitTrapPanel('mandelbrot')
      // Bitmap Image はリセット対象外とする（ot-reset ボタンと同じ挙動）
      // resetTrapSpec() で shape 等を初期値に戻す前にビットマップ関連データを退避し、
      // リセット後に復元することでアップロード済み画像を維持する。
      if (palette.CUSTOM_ORBIT_TRAP_PALETTE) {
        const customOTP = palette.CUSTOM_ORBIT_TRAP_PALETTE
        const savedBitmapData = customOTP.trapSpec?.bitmapData
        const savedBitmapWidth = customOTP.trapSpec?.bitmapWidth
        const savedBitmapHeight = customOTP.trapSpec?.bitmapHeight
        const savedBitmapVersion = customOTP.trapSpec?.bitmapVersion

        customOTP.resetTrapSpec()

        // ビットマップデータを復元する
        if (savedBitmapData) {
          customOTP.updateTrapSpec({
            bitmapData: savedBitmapData,
            bitmapWidth: savedBitmapWidth,
            bitmapHeight: savedBitmapHeight,
            bitmapVersion: savedBitmapVersion,
          })
          // プレビューキャンバスも復元する
          try {
            const preview = document.getElementById('ot-bitmap-preview')
            if (preview && savedBitmapWidth > 0 && savedBitmapHeight > 0) {
              _drawOrbitTrapBitmapPreviewFromImageData(
                preview,
                savedBitmapData,
                savedBitmapWidth,
                savedBitmapHeight,
                _getOrbitTrapBitmapBackgroundColor(customOTP.trapSpec)
              )
            }
          } catch (previewErr) {
            console.warn('Error restoring bitmap preview in reset():', previewErr)
          }
        }
      }
    }
  } catch (e) {
    console.warn('Error resetting palette in reset():', e?.message ? e.message : e)
  }

  // Smooth を既定値 true へ戻す
  try {
    fractal.smooth = true
    const smoothToggle = document.getElementById('smooth')
    if (smoothToggle) smoothToggle.checked = true
  } catch (e) {
    console.warn('Error resetting smooth in reset():', e?.message ? e.message : e)
  }

  // Escape Radius を既定値 4.0 へ戻す
  try {
    fractal.escapeRadius = 4.0
    const escapeRadiusEl = document.getElementById('escape-radius')
    if (escapeRadiusEl) escapeRadiusEl.value = '4'
  } catch (e) {
    console.warn('Error resetting escape radius in reset():', e?.message ? e.message : e)
  }

  // Hi DPI（fullres）をオフへ戻す
  try {
    const fullresToggle = document.getElementById('fullres')
    if (fullresToggle) {
      fullresToggle.checked = false
      // fullres の変更を反映するため resize する
      resizeToCanvasSize()
    }
  } catch (e) {
    console.warn('Error resetting fullres in reset():', e?.message ? e.message : e)
  }

  // WebGPU が使えるなら Fractal GPU をオンへ戻す
  try {
    const gpuToggle = document.getElementById('gpu')
    if (gpuToggle && !gpuToggle.disabled) {
      gpuToggle.checked = true
      // リセット内容がすぐ反映されるよう内部状態も更新する
      fractal.useGpu = true
    }
  } catch (e) {
    console.warn('Error resetting gpu in reset():', e?.message ? e.message : e)
  }

  // Supersampling を 0（OFF）へ戻す
  try {
    const supersamplingSelect = document.getElementById('supersampling')
    if (supersamplingSelect) {
      supersamplingSelect.value = '0'
      // リセット内容がすぐ反映されるよう内部状態も更新する
      fractal.supersampling = 0
    }
  } catch (e) {
    console.warn('Error resetting supersampling in reset():', e?.message ? e.message : e)
  }

  // Buddhabrot GPU は、WebGPU が使えればオン、なければオフへ戻す
  try {
    const buddhaGpuToggle = document.getElementById('buddha-gpu')
    if (buddhaGpuToggle) {
      if (typeof navigator !== 'undefined' && 'gpu' in navigator) {
        buddhaGpuToggle.checked = true
      } else {
        buddhaGpuToggle.checked = false
      }
    }
  } catch (e) {
    console.warn('Error resetting buddha-gpu in reset():', e?.message ? e.message : e)
  }

  // Reset で表示が変わるので保存済みの通常フラクタル画像を無効化する
  savedFractalImageData = null

  // custom モードなら現在の反復式入力値を維持する
  if (fractal.fractalType === 'custom') {
    const iterationFunctionInput = document.getElementById('iterationFunction')
    if (iterationFunctionInput?.value) {
      fractal.iterationFunction = iterationFunctionInput.value
    }
  }

  // 反復回数が変わっていても Reset で座標と UI 状態を確実に戻せるよう、
  // 既定の反復回数を必ず適用して再描画する。
  setIterations(DEFAULT_ITERATIONS)

  // リセット後の値でパレットを再初期化し、表示へ確実に反映する
  try {
    fractal.initPallete(false)
  } catch (e) {
    console.warn('Error re-initializing palette in reset():', e?.message ? e.message : e)
  }

  // Julia キャンバスの位置とズームも初期値へ戻す
  if (juliaState.renderer) {
    juliaState.renderer.setZoom(fxp.fromNumber(1, juliaState.renderer.precision))
    juliaState.renderer.center = [
      fxp.fromNumber(0, juliaState.renderer.precision),
      fxp.fromNumber(0, juliaState.renderer.precision),
    ]
  }

  // すべての設定が反映されるよう resetCaches=true で再描画する
  redraw(true)
  _refreshPinnedOrbits()
}

/**
 * 位置だけを初期値へ戻す（center と zoom）
 */
function resetPosition() {
  // Buddhabrot の描画は停止する
  try {
    stopAndClearBuddha()
  } catch (e) {
    console.warn('Error stopping/clearing buddha in resetPosition():', e?.message ? e.message : e)
  }
  buddhaActive = false

  // zoom を初期値に戻す
  fractal.setZoom(getInitialFractalZoom(fractal.precision))

  // 現在のフラクタル種別に応じた初期中心座標を求める
  let resetCenterX = 0
  let resetCenterY = 0
  const fractalTypeSelect = document.getElementById('fractalType')
  const val = fractalTypeSelect?.value || ''
  const presetIndex = parseInt(val.split(':')[1], 10)
  const p = functionPresets[presetIndex]
  if (p) {
    resetCenterX = p.coordX || 0
    resetCenterY = p.coordY || 0
  }

  // 座標入力欄を初期値に更新する
  const coordXEl = document.getElementById('coordX')
  const coordYEl = document.getElementById('coordY')
  const coordZoomEl = document.getElementById('coordZoom')
  if (coordXEl) coordXEl.value = String(resetCenterX)
  // UI 表示は「上=正の虚数」規則のため符号を反転して表示する
  if (coordYEl) coordYEl.value = String(-resetCenterY)
  if (coordZoomEl) coordZoomEl.value = formatZoomValueForInput(getInitialFractalZoom(fractal.precision))

  // 新しい中心座標を反映する
  fractal.setCenter([fxp.fromNumber(resetCenterX), fxp.fromNumber(resetCenterY)])

  // 保存済み画像は現在の状態と合わないため破棄する
  savedFractalImageData = null

  // Julia キャンバスの位置とズームも初期値へ戻す
  if (juliaState.renderer) {
    juliaState.renderer.setZoom(fxp.fromNumber(1, juliaState.renderer.precision))
    juliaState.renderer.center = [
      fxp.fromNumber(0, juliaState.renderer.precision),
      fxp.fromNumber(0, juliaState.renderer.precision),
    ]
  }

  // 描き直す
  redraw(true)
  _refreshPinnedOrbits()
}

const PERMALINK_UPDATE_MIN_INTERVAL_MS = 250
const PERMALINK_SECURITY_ERROR_BACKOFF_MS = 10000
let pendingPermalinkHref = null
let pendingPermalinkTimer = null
let lastPermalinkHref = ''
let lastPermalinkUpdateAt = 0
let permalinkBackoffUntil = 0

function _schedulePendingPermalinkUpdate(delay) {
  if (pendingPermalinkTimer != null) {
    clearTimeout(pendingPermalinkTimer)
  }
  pendingPermalinkTimer = setTimeout(() => {
    pendingPermalinkTimer = null
    _flushPendingPermalinkUpdate()
  }, Math.max(0, delay))
}

function _flushPendingPermalinkUpdate() {
  if (!pendingPermalinkHref) return

  const now = performance.now()
  if (now < permalinkBackoffUntil) {
    _schedulePendingPermalinkUpdate(permalinkBackoffUntil - now)
    return
  }

  const href = pendingPermalinkHref
  if (href === lastPermalinkHref) {
    pendingPermalinkHref = null
    return
  }

  const wait = PERMALINK_UPDATE_MIN_INTERVAL_MS - (now - lastPermalinkUpdateAt)
  if (wait > 0) {
    _schedulePendingPermalinkUpdate(wait)
    return
  }

  try {
    window.history.replaceState({}, '', href)
    lastPermalinkHref = href
    lastPermalinkUpdateAt = performance.now()
    if (pendingPermalinkHref === href) {
      pendingPermalinkHref = null
    }
  } catch (error) {
    const message = error?.message ? error.message : String(error)
    if (error?.name === 'SecurityError') {
      permalinkBackoffUntil = performance.now() + PERMALINK_SECURITY_ERROR_BACKOFF_MS
    }
    console.warn('Error updating permalink:', message)
    _schedulePendingPermalinkUpdate(
      Math.max(
        PERMALINK_UPDATE_MIN_INTERVAL_MS,
        permalinkBackoffUntil > 0 ? permalinkBackoffUntil - performance.now() : 0,
      ),
    )
  }
}

function updatePermalink() {
  const url = new URL(window.location)
  const p = url.searchParams
  const params = {
    center: fractal.center,
    zoom: fractal.zoom,
    max_iter: fractal.max_iter,
    smooth: fractal.smooth,
    fractalType: fractal.fractalType,
    fractalLabel: fractal.label,
    fractalDataPresetIndex: fractal.dataPresetIndex,
    iterationFunction: fractal.iterationFunction,
    // パーマリンク作成時は UI の初期 z0 も含める
    z0: [
      document.getElementById('z0-real') ? parseFloat(document.getElementById('z0-real').value) : 0.0,
      document.getElementById('z0-imag') ? parseFloat(document.getElementById('z0-imag').value) : 0.0,
    ],
    palette: {
      id: paletteComponent.palette.id,
      density: paletteComponent.density,
      rotate: paletteComponent.rotate,
    },
    escapeRadius: fractal.escapeRadius,
  }

  // OrbitTrap の custom パレットなら、bitmapData を除いた trapSpec と color pattern を含める
  if (params.palette.id === 'orbit_trap_custom') {
    const custom = paletteComponent.palette
    const spec = Object.assign({}, custom.trapSpec)
    // UI 互換のため、Infinity の threshold は 0 に戻して保存する
    if (spec.threshold === Infinity) spec.threshold = 0
    delete spec.bitmapData
    delete spec.bitmapVersion
    params.palette.trapSpec = spec
    params.palette.colorPattern = custom._colorPatternId
  }

  p.set('params', btoa(JSON.stringify(params)))
  pendingPermalinkHref = url.toString()
  _flushPendingPermalinkUpdate()
}

function initUI() {
  paletteComponent.init()
  populateBuddhaPaletteSelect()
  try {
    // WebGPU が使えない環境では、Buddhabrot 側も GPU オプションを無効にする
    if (DOM.gpuToggle) {
      if (typeof navigator === 'undefined' || !('gpu' in navigator)) {
        DOM.gpuToggle.checked = false
        DOM.gpuToggle.disabled = true
        DOM.gpuToggle.parentElement.setAttribute('title', 'WebGPU not supported')
      }
    }
    // Buddhabrot 専用 GPU トグルにも同じ自動判定を反映する
    const bgpu = document.getElementById('buddha-gpu')
    if (bgpu) {
      if (typeof navigator === 'undefined' || !('gpu' in navigator)) {
        bgpu.checked = false
        bgpu.disabled = true
        bgpu.parentElement?.setAttribute('title', 'WebGPU not supported')
      } else {
        // WebGPU 対応ブラウザなら、既存の GPU 挙動に合わせて既定でオンにする
        bgpu.checked = true
      }
    }

    // 初期表示時に fractalType が mandelbrot / custom 以外なら、
    // メイン GPU トグルはオフかつ無効にする。
    const mainGpu = document.getElementById('gpu')
    if (mainGpu && fractal?.fractalType && fractal.fractalType !== 'mandelbrot' && fractal.fractalType !== 'custom') {
      mainGpu.checked = false
      mainGpu.disabled = true
      mainGpu.parentElement?.setAttribute('title', 'GPU only available for Mandelbrot and Custom fractal types')
    }
  } catch (e) {
    console.warn('Error initializing GPU state:', e?.message ? e.message : e)
  }
}

// 現在のキャンバスを PNG で保存し、ダウンロードを開始する

// ── Julia 全画面: 実際の #mandelbrot-canvas-wrap を設定プレビューへ移す ──

/**
 * #mandelbrot-canvas-wrap を設定プレビューへ移し、
 * Julia キャンバスをビューポート全体へ広げる。
 * Julia モードで全画面へ入るとき、または全画面中に Julia を有効化したときに使う。
 */
function enterJuliaFullscreen({ redraw = true } = {}) {
  const mbWrap = document.getElementById('mandelbrot-canvas-wrap')
  const previewCol = document.getElementById('julia-mb-preview-col')
  const previewWrap = document.getElementById('julia-mb-preview-wrap')
  if (mbWrap && previewCol && mbWrap.parentElement !== previewCol) {
    previewCol.appendChild(mbWrap)
  }
  if (previewWrap) previewWrap.removeAttribute('hidden')
  requestAnimationFrame(() => {
    resizeToCanvasSize()
    if (redraw) redrawJulia()
  })
}

/**
 * #mandelbrot-canvas-wrap を #mandelbrot へ戻し、設定プレビューを隠す。
 * Julia モードで全画面を抜けるとき、または全画面中に Julia を無効化したときに使う。
 */
function exitJuliaFullscreen() {
  const mbWrap = document.getElementById('mandelbrot-canvas-wrap')
  const mandelbrotDiv = document.getElementById('mandelbrot')
  const juliaSection = document.getElementById('julia-section')
  const previewWrap = document.getElementById('julia-mb-preview-wrap')

  if (mbWrap && mandelbrotDiv && mbWrap.parentElement !== mandelbrotDiv) {
    // 元の DOM 順へ戻すため julia-section の前へ挿入する
    if (juliaSection && juliaSection.parentElement === mandelbrotDiv) {
      mandelbrotDiv.insertBefore(mbWrap, juliaSection)
    } else {
      mandelbrotDiv.appendChild(mbWrap)
    }
  }
  // プレビュー用に付けた inline style を消す
  if (mbWrap) {
    mbWrap.style.width = ''
    mbWrap.style.height = ''
    mbWrap.style.flex = ''
  }
  if (previewWrap) previewWrap.setAttribute('hidden', '')
}

// 現在のキャンバスを PNG として保存し、ダウンロードを開始する
function saveCanvasAsPNG() {
  const ts = Date.now()

  /**
   * フラクタルのキャンバスと必要に応じたオーバーレイを合成し、PNG として保存する。
   */
  function downloadCanvasWithOverlay(cnv, overlayCanvas, filename) {
    if (!cnv) return
    try {
      // 表示中のオーバーレイがあれば、先に offscreen へ合成する
      let srcCanvas = cnv
      if (overlayCanvas) {
        const off = document.createElement('canvas')
        off.width = cnv.width
        off.height = cnv.height
        const offCtx = off.getContext('2d')
        offCtx.drawImage(cnv, 0, 0)
        // orbit-canvas の CSS サイズ差分を吸収するため、実キャンバスに合わせて描く
        offCtx.drawImage(overlayCanvas, 0, 0, cnv.width, cnv.height)
        srcCanvas = off
      }
      if (srcCanvas.toBlob) {
        srcCanvas.toBlob((blob) => {
          if (!blob) return
          const a = document.createElement('a')
          const url = URL.createObjectURL(blob)
          a.href = url
          a.download = filename
          document.body.appendChild(a)
          a.click()
          a.remove()
          setTimeout(() => URL.revokeObjectURL(url), 1000)
        }, 'image/png')
        return
      }
      // 使えない場合は toDataURL に切り替える
      const dataUrl = srcCanvas.toDataURL('image/png')
      const a = document.createElement('a')
      a.href = dataUrl
      a.download = filename
      document.body.appendChild(a)
      a.click()
      a.remove()
    } catch (e) {
      console.warn('Error saving canvas as PNG:', e)
    }
  }

  const mbCanvas = document.getElementById('mandelbrot-canvas')
  if (mbCanvas) {
    // 軌道表示が見えているときだけオーバーレイを含める
    const orbitCanvas = orbitDrawEnabled ? document.getElementById('orbit-canvas') : null
    const orbitHasContent = orbitCanvas && (pinnedOrbit || false)
    downloadCanvasWithOverlay(mbCanvas, orbitHasContent ? orbitCanvas : null, `fractal_${ts}.png`)
  }

  // Julia モード中は Julia キャンバスも保存する
  if (juliaState?.active) {
    const juliaCanvas = document.getElementById('julia-canvas')
    if (juliaCanvas) {
      const jOrbitCanvas = orbitDrawEnabled ? document.getElementById('julia-orbit-canvas') : null
      const jOrbitHasContent = jOrbitCanvas && (juliaPinnedOrbit || false)
      // ブラウザ側でダウンロードが重ならないよう少し待つ
      setTimeout(
        () => downloadCanvasWithOverlay(juliaCanvas, jOrbitHasContent ? jOrbitCanvas : null, `julia_${ts}.png`),
        200,
      )
    }
  }
}

// 読み込み時に URL のパーマリンクを確認する
function init() {
  initUI()
  _ensureQueryOrbitTrapBitmapLoaded()
  // resizeTmpCanvas()
  onResize()
  fractal.initPallete()
  // パラメータ復元より先にリスナーを初期化する
  initListeners()

  const url = new URL(window.location)
  const params = url.searchParams.get('params')
  if (params) {
    initFromParams(params)
  } else {
    // パーマリンクがない場合も z0 入力は常に編集可能にしておく
    const elr = document.getElementById('z0-real')
    const eli = document.getElementById('z0-imag')
    if (elr) elr.disabled = false
    if (eli) eli.disabled = false
    const container = elr?.parentElement?.parentElement
    if (container) {
      container.setAttribute('title', 'Initial z0 for iteration')
    }
  }
  DOM.iterations.value = fractal.max_iter
  DOM.smoothToggle.checked = fractal.smooth
  DOM.iterationFunctionInput.value = fractal.iterationFunction
  // 起動時点の fractalType に合わせて GPU 表示を整える
  updateGpuVisibility(fractal.fractalType)
  // 現在の fractalType に応じてメイン GPU トグル状態を整える
  enforceMainGpuState(fractal.fractalType)
  redraw()
}

function initFromParams(params) {
  try {
    const p = JSON.parse(atob(decodeURIComponent(params)))
    fractal.setZoom(applyEmbeddedInitialZoomCorrection(fxp.fromJSON(p.zoom)))
    fractal.setCenter(p.center.map(fxp.fromJSON))
    const normalizedFractalParams = normalizeFractalTypeParams(p)
    fractal.max_iter = p.max_iter
    fractal.smooth = p.smooth
    fractal.fractalType = normalizedFractalParams.fractalType
    fractal.iterationFunction = normalizedFractalParams.iterationFunction
    fractal.dataPresetIndex = normalizedFractalParams.fractalDataPresetIndex
    DOM.fractalTypeSelect.value = normalizedFractalParams.fractalDataPresetIndex
    DOM.iterationFunctionInput.value = normalizedFractalParams.iterationFunction

    updateGpuVisibility(normalizedFractalParams.fractalType)
    // パラメータ復元時もメイン GPU トグル状態を整える
    enforceMainGpuState(normalizedFractalParams.fractalType)

    const zr = p.z0[0]
    const zi = p.z0[1]
    const elr = document.getElementById('z0-real')
    const eli = document.getElementById('z0-imag')
    if (elr) elr.value = String(zr)
    if (eli) eli.value = String(zi)
    fractal.z0Real = Number.isNaN(Number(zr)) ? 0 : Number(zr)
    fractal.z0Imag = Number.isNaN(Number(zi)) ? 0 : Number(zi)

    // fractalType に関係なく z0 入力は常に有効にしておく
    if (elr) elr.disabled = false
    if (eli) eli.disabled = false
    const container = elr?.parentElement?.parentElement
    if (container) {
      container.setAttribute('title', 'Initial z0 for iteration')
    }
    // custom 以外の params を復元したときは反復式エラー表示を消す
    try {
      if (normalizedFractalParams.fractalType !== 'custom') setIterationFunctionLabelError(false)
    } catch (e) {
      console.warn('Error clearing iteration function label during params init:', e)
    }

    if (DOM.gpuToggle) {
      DOM.gpuToggle.parentElement.setAttribute('title', 'GPU disabled for non-Mandelbrot fractal types')
      new bootstrap.Tooltip(DOM.gpuToggle.parentElement)
    }

    if (p.palette) {
      paletteComponent.setPalette(palette.getPalette(p.palette.id))
      paletteComponent.setDensity(p.palette.density)
      paletteComponent.setRotate(p.palette.rotate)
      // パレット復元後に Orbit Trap パネルの表示状態も合わせる
      paletteComponent._updateOrbitTrapPanel(p.palette.id)

      // custom OrbitTrap パレットなら追加情報も復元する
      if (p.palette.id === 'orbit_trap_custom') {
        const custom = paletteComponent.palette
        // カラーパターン
        if (p.palette.colorPattern && typeof custom.setColorPattern === 'function') {
          custom.setColorPattern(p.palette.colorPattern)
          const cpSel = document.getElementById('ot-color-pattern')
          if (cpSel) cpSel.value = p.palette.colorPattern
        }
        // Trap spec
        if (p.palette.trapSpec) {
          const spec = { ...p.palette.trapSpec }
          // 保存時に 0 としていた threshold は、復元時に Infinity へ戻す
          if (spec.threshold === 0) spec.threshold = Infinity
          custom.updateTrapSpec(spec)
          _applyOrbitTrapSpecToUI(spec, p.palette.colorPattern)
          _maybeAutoApplyQueryOrbitTrapBitmap()
        }
      }
    }

    if (p.escapeRadius != null && Number.isFinite(Number(p.escapeRadius))) {
      fractal.escapeRadius = Math.max(0, Number(p.escapeRadius))
      const escapeRadiusEl = document.getElementById('escape-radius')
      if (escapeRadiusEl) escapeRadiusEl.value = String(fractal.escapeRadius)
    }

    DOM.iterations.value = fractal.max_iter
    DOM.smoothToggle.checked = fractal.smooth
  } catch (e) {
    console.warn('Error restoring z0 from params:', e?.message ? e.message : e)
  }
}

// 自動テストや外部利用向けに、主要レンダラークラスを公開する
export { JuliaRenderer, Mandelbrot }

window.onload = init

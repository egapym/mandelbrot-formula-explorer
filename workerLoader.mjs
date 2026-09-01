/**
 * Based on bertbaron/mandelbrot by Bert Baron
 * This file is part of the Mandelbrot Formula Explorer project.
 * Licensed under GPL-3.0.
 */

// ============================================================================
// 定数
// ============================================================================

const FETCH_CONFIG = {
  CACHE: 'no-store',
  CONTENT_TYPE: 'text/javascript',
}

const REGEX = {
  STATIC_IMPORT: /import\s+(?:[\s\S]*?from\s+)?["']([^"']+)["']/g,
  DYNAMIC_IMPORT: /import\(\s*["']([^"']+)["']\s*\)/g,
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
   * fetch 用のエラーを作る
   */
  fetchError(url, status) {
    return new Error(`Failed to fetch script: ${url} (${status})`)
  },
}

const _blobUrlCache = new Map() // absoluteUrl -> Promise<string>

// ページの基準 URL を使ってスクリプトパスを絶対 URL に変換する。
function _resolveAbsolute(scriptPath) {
  try {
    return new URL(scriptPath, location.href).href
  } catch (_e) {
    // URL 化に失敗した場合は元の文字列をそのまま使う
    return scriptPath
  }
}

/**
 * URL からスクリプト本文を取得する
 * @param {string} url - スクリプト URL
 * @returns {Promise<string>}
 * @private
 */
async function _fetchText(url) {
  const res = await fetch(url, { cache: FETCH_CONFIG.CACHE })
  if (!res.ok) throw ErrorHelpers.fetchError(url, res.status)
  return await res.text()
}

/**
 * モジュールを再帰的に取得し、ローカル import を Blob URL に書き換える
 * @param {string} scriptPath - worker スクリプトのパス
 * @returns {Promise<string>} Blob URL
 */
export async function getWorkerBlobUrl(scriptPath) {
  const abs = _resolveAbsolute(scriptPath)
  if (_blobUrlCache.has(abs)) return _blobUrlCache.get(abs)

  const p = (async () => {
    const src = await _fetchText(abs)

    // import 文を集めて、ローカルモジュールの指定先を抽出する
    const specs = new Set()

    let m
    while (true) {
      m = REGEX.STATIC_IMPORT.exec(src)
      if (m === null) break
      specs.add(m[1])
    }
    while (true) {
      m = REGEX.DYNAMIC_IMPORT.exec(src)
      if (m === null) break
      specs.add(m[1])
    }

    // 相対パスや同一オリジンの参照だけを解決し、外部パッケージは無視する
    const relSpecs = []
    for (const s of specs) {
      if (s.startsWith('.') || s.startsWith('/')) {
        const resolved = new URL(s, abs).href
        relSpecs.push({ spec: s, resolved })
      }
    }

    // ローカル依存ごとに Blob URL を再帰的に用意する
    const resolvedMap = Object.create(null)
    await Promise.all(
      relSpecs.map(async ({ spec, resolved }) => {
        const depBlob = await getWorkerBlobUrl(resolved)
        resolvedMap[spec] = depBlob
      }),
    )

    // ソース中のローカル参照を対応する Blob URL に置き換える
    let transformed = src
    for (const { spec } of relSpecs) {
      const blobUrl = resolvedMap[spec]
      if (!blobUrl) continue
      // 完全一致する文字列リテラルだけを置換する
      const esc = spec.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')
      const re = new RegExp(`(["'])${esc}\\1`, 'g')
      transformed = transformed.replace(re, `"${blobUrl}"`)
      // 念のため引用符なしの形も置換する
      const re2 = new RegExp(esc, 'g')
      transformed = transformed.replace(re2, blobUrl)
    }

    const blob = new Blob([transformed], { type: FETCH_CONFIG.CONTENT_TYPE })
    const url = URL.createObjectURL(blob)
    return url
  })()

  _blobUrlCache.set(abs, p)
  return p
}

/**
 * スクリプトパスから Blob URL 経由で Worker を作成する
 * @param {string} scriptPath - worker スクリプトのパス
 * @param {Object} [options] - Worker オプション
 * @param {string} [options.type] - Worker の種別（例: 'module'）
 * @returns {Promise<Worker>}
 */
export async function createWorkerFrom(scriptPath, options = { type: 'module' }) {
  // iOS Safari can create a module Worker from a Blob URL, but fail when that
  // Blob imports other Blob URLs.  The failure is reported asynchronously by
  // the Worker, leaving callers with a worker that never returns a result.
  // A same-origin module URL keeps the import base intact and works in both
  // Safari and the other supported browsers.
  return new Worker(_resolveAbsolute(scriptPath), options)
}

/**
 * キャッシュ済み Blob URL をすべて解放する
 */
export function revokeAll() {
  for (const v of _blobUrlCache.values()) {
    v.then((url) => {
      try {
        URL.revokeObjectURL(url)
      } catch (e) {
        // 解放失敗は無視する
      }
    }).catch(() => {
      // Promise 側の失敗も無視する
    })
  }
  _blobUrlCache.clear()
}

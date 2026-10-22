// Settings not already represented by the fractal/palette model. Keep transient
// actions (Jump To, file picker, playback) out of this list.
export const SETTING_CONTROL_IDS = [
  'gpu', 'fullres', 'supersampling',
  'orbit-toggle', 'orbit-mode', 'orbit-gradient-toggle', 'coordinate-grid-toggle', 'detail-toggle',
  'julia-toggle', 'buddha-gpu', 'buddha-iterations', 'buddha-palette', 'buddhaMode',
  'buddha-bandmode', 'buddha-sampling-range', 'buddha-brightness', 'buddha-gamma',
  'buddha-draw-speed', 'buddha-point-batch-size', 'buddha-continuous', 'buddha-fixed-samples',
  'anim-speed', 'anim-fps', 'anim-auto-adjust', 'anim-min-iterations', 'anim-enable',
]

export function readSettingsControls(document) {
  return Object.fromEntries(SETTING_CONTROL_IDS.flatMap((id) => {
    const el = document.getElementById(id)
    return el ? [[id, el.type === 'checkbox' ? el.checked : el.value]] : []
  }))
}

export function writeSettingsControls(document, values) {
  const changed = []
  for (const id of SETTING_CONTROL_IDS) {
    const el = document.getElementById(id)
    const value = values?.[id]
    if (!el || value === undefined) continue
    const key = el.type === 'checkbox' ? 'checked' : 'value'
    if (key === 'checked' ? typeof value !== 'boolean' : typeof value !== 'string') continue
    if (el.tagName === 'SELECT' && !Array.from(el.options).some((opt) => opt.value === value)) continue
    if (key === 'checked' && value && el.disabled && (id === 'gpu' || id === 'buddha-gpu')) continue
    if (el[key] !== value) changed.push(id)
    el[key] = value
  }
  return changed
}

export function encodeSettings(params) {
  const bytes = new TextEncoder().encode(JSON.stringify(params))
  return encodeBytes(bytes)
}

function encodeBytes(bytes) {
  let binary = ''
  for (let i = 0; i < bytes.length; i += 8192) {
    binary += String.fromCharCode(...bytes.subarray(i, i + 8192))
  }
  return btoa(binary)
}

export function decodeSettings(encoded) {
  const binary = atob(encoded)
  // Old permalinks used Latin-1 btoa(JSON.stringify(...)).
  try {
    return JSON.parse(new TextDecoder('utf-8', { fatal: true }).decode(
      Uint8Array.from(binary, (char) => char.charCodeAt(0)),
    ))
  } catch {
    return JSON.parse(binary)
  }
}

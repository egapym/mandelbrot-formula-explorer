import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import { SETTING_CONTROL_IDS, readSettingsControls, writeSettingsControls,
  encodeSettings, decodeSettings } from '../settingsState.mjs'

const settings = { label: '日本語 🌀', controls: { gpu: false }, orbitTrap: { threshold: 0 } }
assert.deepEqual(decodeSettings(encodeSettings(settings)), settings)
assert.deepEqual(decodeSettings(btoa(JSON.stringify({ label: 'Lava é' }))), { label: 'Lava é' })
const controls = new Map(SETTING_CONTROL_IDS.map((id) => [id, { value: '1', type: 'number' }]))
controls.set('gpu', { type: 'checkbox', checked: false, disabled: true })
controls.set('orbit-mode', { tagName: 'SELECT', type: 'select-one', value: 'lines', options: [{ value: 'lines' }, { value: 'dots' }] })
const document = { getElementById: (id) => controls.get(id) }
const initial = readSettingsControls(document)
writeSettingsControls(document, { gpu: true, 'orbit-mode': 'dots', 'buddha-iterations': '900', unknown: 'x' })
assert.equal(controls.get('gpu').checked, false, 'Unsupported GPU stays off')
assert.equal(controls.get('orbit-mode').value, 'dots')
writeSettingsControls(document, { 'orbit-mode': 'invalid', 'buddha-iterations': null })
assert.equal(controls.get('orbit-mode').value, 'dots')
assert.equal(controls.get('buddha-iterations').value, '900')
writeSettingsControls(document, initial)
assert.deepEqual(readSettingsControls(document), initial, 'All controls return to their captured defaults')

// Audit every HTML setting so additions cannot silently miss persistence/reset.
const html = readFileSync(new URL('../index.html', import.meta.url), 'utf8')
const modelControls = new Set(['fractalType', 'iterationFunction', 'max-iterations', 'smooth',
  'z0-real', 'z0-imag', 'escape-radius', 'palette-dropdown', 'palette-density',
  'palette-density-input', 'palette-rotate', 'coordX', 'coordY', 'coordZoom',
  'buddha-toggle', 'jump-to', 'ot-bitmap-file'])
for (const match of html.matchAll(/<(input|select)\b[^>]*\bid="([^"]+)"[^>]*>/gs)) {
  const id = match[2]
  assert.ok(SETTING_CONTROL_IDS.includes(id) || modelControls.has(id) || id.startsWith('ot-'), `Unaccounted control: ${id}`)
}
const appSource = readFileSync(new URL('../index.js', import.meta.url), 'utf8')
assert.match(appSource, /delete spec\.bitmapData[\s\S]*delete spec\.bitmapWidth[\s\S]*delete spec\.bitmapHeight/,
  'Permalinks must omit uploaded bitmap data')
assert.match(appSource, /const spec = \{ \.\.\.p\.orbitTrap\.trapSpec \}[\s\S]*delete spec\.bitmapData[\s\S]*custom\.updateTrapSpec\(spec\)/,
  'Legacy permalink bitmap data must be ignored')
assert.doesNotMatch(appSource, /encodeBytes\(spec\.bitmapData\)|decodeBytes\(spec\.bitmapData\)/,
  'Bitmap bytes must not be encoded into or decoded from permalinks')
assert.match(appSource, /const selectedPresetValue = DOM\.fractalTypeSelect\.value[\s\S]*const selectedPreset = functionPresets\[selectedPresetIndex\]/,
  'Reset All must derive its defaults from the selected Fractal Type')
const resetSource = appSource.slice(appSource.indexOf('function reset() {'), appSource.indexOf('function resetPosition() {'))
assert.doesNotMatch(resetSource, /normalizeFractalTypeParams\(/,
  'Reset All must not replace the selected Fractal Type with the global default')
console.log('PASS: all settings controls audited, reset preserves Fractal Type, legacy links, and bitmap exclusion')

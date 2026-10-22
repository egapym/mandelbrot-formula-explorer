// Generate the baseline with: python3 test/buildBuddhabrotGpuPrototypes.py
// ?reference=./fixtures/gpu-prototypes/before.mjs compares every preset against a
// saved renderer. Mode/Julia/band combinations exercise the same sample stream.
import { BuddhabrotWebGPU } from '../buddhabrotWebGPU.mjs'
import { functionPresets } from '../functionPresets.mjs'
import { getBuddhaPalette } from '../buddhaPalettes.mjs'
const result = document.querySelector('#result')
const query = new URLSearchParams(location.search)
const assert = (ok, message) => { if (!ok) throw new Error(message) }
try {
  const adapter = await navigator.gpu?.requestAdapter()
  assert(adapter, 'WebGPU unavailable')
  const device = await adapter.requestDevice()
  const reference = query.get('reference')
  assert(reference, 'Provide a local reference renderer module')
  const url = new URL(reference, location.href)
  assert(url.origin === location.origin, 'Reference must be a local module')
  const Before = (await import(url.href)).BuddhabrotWebGPU
  const runners = [new Before({ devicePromise: Promise.resolve(device) }), new BuddhabrotWebGPU({ devicePromise: Promise.resolve(device) })]
  const passed = [], skipped = []
  try {
    const presets = [...functionPresets, ...[31, 32, 33].flatMap(index =>
      ['zAt', 'zDelay'].map(kind => ({ expr: `z*z+c+0.1*${kind}(${index})` })))]
    for (const [index, preset] of presets.entries()) {
      for (const julia of [false, true]) for (const mode of ['buddha', 'antibuddha']) for (const band of ['perPoint', 'perTrajectory']) {
        const label = `${index}:${preset.label || preset.expr}/${julia ? 'Julia' : 'main'}/${mode}/${band}`
        let expected, skip = false
        for (const [variant, runner] of runners.entries()) {
          let error
          runner.onComplete = data => { if (data.error) error = data.error }
          runner.onProgress = () => {}
          device.pushErrorScope('validation')
          await runner.start({ width: 160, height: 90, samples: 32769, maxIter: 256,
            gpu: true, sampleSeed: 0x1badb002, center: { x: preset.coordX || 0, y: preset.coordY || 0 }, zoom: 1,
            escapeRadius: 4, z0Real: preset.z0Real || 0, z0Imag: preset.z0Imag || 0,
            iterationFunction: preset.expr, fractalType: julia ? 'julia-custom' : (preset.fractalType || 'custom'),
            juliaRe: -0.7, juliaIm: 0.2, mode, buddhaBandMode: band, palette: getBuddhaPalette('Preset-05') })
          const validation = await device.popErrorScope()
          if (runner._cpuRunner || error || validation) {
            runner._cpuRunner?.terminate()
            if (variant === 0) { skipped.push(label); skip = true; break }
            throw new Error(`New GPU failure: ${label}: ${error || validation?.message || 'CPU fallback'}`)
          }
          const channels = [runner.densityR, runner.densityG, runner.densityB]
          if (variant === 0) expected = channels.map(channel => channel.slice())
          else for (let ch = 0; ch < 3; ch++) assert(channels[ch].every((v, i) => v === expected[ch][i]), `${label}: density differs in channel ${ch}`)
        }
        if (!skip) passed.push(label)
        result.textContent = `Running… ${passed.length} passed; ${skipped.length} reference GPU exclusions; preset ${index + 1}/${presets.length}`
      }
    }
    result.textContent = JSON.stringify({ status: 'PASS: exact RGB density parity', passed: passed.length, skipped, presets: functionPresets.length, historyBoundaryExpressions: presets.length - functionPresets.length }, null, 2)
  } finally { for (const runner of runners) runner._cpuRunner?.terminate(); device.destroy() }
} catch (error) { result.textContent = error.stack; console.error(error) }

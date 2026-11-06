import { BuddhabrotWebGPU } from '../buddhabrotWebGPU.mjs'
import { getBuddhaPalette } from '../buddhaPalettes.mjs'
import { mapBuddhabrotFrame } from '../buddhabrotCpuColorMap.mjs'

const assert = (ok, message) => { if (!ok) throw new Error(message) }
const params = {
  gpu: true, width: 64, height: 64, samples: 2_000_000, maxIter: 1000,
  center: { x: -0.5, y: 0 }, zoom: 1, escapeRadius: 4,
  mode: 'buddha', fractalType: 'mandelbrot', sampleSeed: 12345,
  palette: getBuddhaPalette('Preset-01'), buddhaBandMode: 'perTrajectory',
}
const channels = runner => ['densityR', 'densityG', 'densityB'].map(key => runner[key])
const delay = ms => new Promise(resolve => setTimeout(resolve, ms))

// Pace completed GPU groups so even fast hardware exercises multiple real
// one-second previews with a small sample budget. Sampling/shaders stay intact.
function pacedDevice(device, fallback) {
  const queue = new Proxy(device.queue, { get(target, key) {
    if (key === 'onSubmittedWorkDone') return async () => {
      await target.onSubmittedWorkDone()
      await delay(50)
    }
    const value = Reflect.get(target, key, target)
    return typeof value === 'function' ? value.bind(target) : value
  } })
  const limits = new Proxy(device.limits, { get(target, key) {
    // Each histogram fits, but the combined Float32 readback does not. This
    // selects the alternative six-integer-buffer readback on the same GPU.
    return fallback && key === 'maxStorageBufferBindingSize'
      ? params.width * params.height * 4 : Reflect.get(target, key, target)
  } })
  return new Proxy(device, { get(target, key) {
    if (key === 'queue') return queue
    if (key === 'limits') return limits
    const value = Reflect.get(target, key, target)
    return typeof value === 'function' ? value.bind(target) : value
  } })
}

async function run() {
  const adapter = await navigator.gpu?.requestAdapter()
  assert(adapter, 'WebGPU unavailable')
  const device = await adapter.requestDevice()
  const reference = new BuddhabrotWebGPU({ devicePromise: Promise.resolve(device) })
  const passed = []
  try {
    for (const fallback of [false, true]) {
      for (const stop of [false, true]) {
        const runner = new BuddhabrotWebGPU({ devicePromise: Promise.resolve(pacedDevice(device, fallback)) })
        let processed = 0, previews = 0, previousTime, previousSum = 0, failure
        const canvas = document.createElement('canvas')
        canvas.width = params.width
        canvas.height = params.height
        document.body.append(canvas)
        runner.onProgress = data => { processed += data.delta || 0 }
        runner.onComplete = data => { failure ||= data.error }
        runner.onChunk = chunk => {
          const now = performance.now()
          assert(runner.running && processed < params.samples, 'Preview arrived only after completion')
          assert(chunk.continuous === false, 'Finite preview marked Continuous')
          if (previousTime !== undefined) assert(now - previousTime >= 990, 'Preview exceeded one-second cadence')
          previousTime = now
          const sum = channels(runner).reduce((total, values) => total + values.reduce((s, v) => s + v, 0), 0)
          assert(sum > previousSum, 'Preview density did not grow')
          previousSum = sum
          const image = canvas.getContext('2d').createImageData(params.width, params.height)
          mapBuddhabrotFrame(...channels(runner), image.data, 1, 1.5)
          assert(image.data.some((v, i) => i % 4 !== 3 && v > 0), 'Empty preview image')
          canvas.getContext('2d').putImageData(image, 0, 0)
          previews++
          if (stop) runner.stop()
        }
        device.pushErrorScope('validation')
        await runner.start(params)
        const error = await device.popErrorScope()
        assert(!error && !failure && !runner._cpuRunner, error?.message || failure || 'CPU fallback')
        assert(previews >= (stop ? 1 : 2), 'Missing finite previews')
        assert(stop ? processed < params.samples : processed === params.samples, 'Incorrect completed sample count')
        assert(!!runner._cachedBuffers.densityReadback === !fallback, 'Wrong readback path')
        // Animation-style completion-only readback is an independent reference:
        // preview and Stop must retain exactly the same sampled contributions.
        let referencePreviews = 0
        reference.onChunk = () => referencePreviews++
        await reference.start({ ...params, samples: processed, reusePipelines: true })
        assert(referencePreviews === 0, 'Animation started publishing previews')
        assert(channels(runner).every((values, ch) => values.every((v, i) => v === channels(reference)[ch][i])),
          'Preview/Stop changed final seeded density')
        passed.push(`${fallback ? 'integer' : 'Float32'} readback, ${stop ? 'Stop' : 'completion'}: ${previews} previews, exact density`)
        runner.stop()
      }
    }
    return `PASS: ${passed.join('; ')}`
  } finally {
    reference.stop()
    device.destroy()
  }
}

async function checkAppDisplay() {
  const frame = document.createElement('iframe')
  frame.style.cssText = 'width:1000px;height:800px;border:0;display:block'
  frame.src = '../index.html'
  const loaded = new Promise(resolve => frame.addEventListener('load', resolve, { once: true }))
  document.body.append(frame)
  await loaded
  const doc = frame.contentDocument
  const waitFor = async (check, message) => {
    const deadline = performance.now() + 30000
    while (!check()) {
      assert(performance.now() < deadline, message)
      await delay(25)
    }
  }
  await waitFor(() => doc.getElementById('renderTimeValue').textContent.includes('ms'), 'Initial app render missing')
  const toggle = doc.getElementById('buddha-gpu')
  toggle.checked = true
  toggle.dispatchEvent(new frame.contentWindow.Event('change', { bubbles: true }))
  doc.getElementById('buddha-iterations').value = '1000000000'
  const view = doc.getElementById('buddha-toggle')
  const canvas = doc.getElementById('mandelbrot-canvas')
  const context = canvas.getContext('2d')
  const draw = context.drawImage.bind(context)
  let draws = 0
  context.drawImage = (...args) => { draws++; return draw(...args) }
  try {
    for (const stop of [false, true]) {
      draws = 0
      doc.querySelector('button[title="Start Buddhabrot rendering. Shortcut: b"]').click()
      await waitFor(() => draws > 0, 'App did not display GPU preview')
      assert(view.disabled, 'App displayed GPU image only after completion')
      const previewDraws = draws
      if (stop) doc.querySelector('button[title="Stop Buddhabrot"]').click()
      await waitFor(() => view.checked && !view.disabled && draws > previewDraws,
        'Completion/Stop did not publish final image and restore View')
      const pixels = context.getImageData(0, 0, canvas.width, canvas.height).data
      assert(pixels.some((v, i) => i % 4 !== 3 && v > 0), 'Completion/Stop left app canvas blank')
    }
  } finally { context.drawImage = draw }
  return 'app canvas updates during finite GPU rendering; completion and Stop retain image and restore View'
}

try { document.querySelector('#result').textContent = `${await run()}; ${await checkAppDisplay()}` }
catch (error) { document.querySelector('#result').textContent = `FAIL: ${error.stack || error}` }

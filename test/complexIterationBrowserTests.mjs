import { checkComplexRotateCpu, checkComplexRotateGpu } from './complexRotateTests.mjs'
import { BuddhabrotWebGPU } from '../buddhabrotWebGPU.mjs'
import * as fxp from '../fxp.mjs'
import { MandelbrotCustom } from '../mandelbrotCustom.mjs'
import { MandelbrotCustomWebGPU } from '../mandelbrotCustomWebGPU.mjs'
import { OrbitTrapWebGPU } from '../orbitTrapBitmapWebGPU.mjs'
import { WorkerContext } from '../workerContext.mjs'

const assert = (condition, message) => {
  if (!condition) throw new Error(message)
}
const context = { shouldStop: () => false }

async function run() {
  const adapter = await navigator.gpu?.requestAdapter()
  const device = await adapter?.requestDevice()
  assert(device, 'WebGPU device unavailable')
  checkComplexRotateCpu()
  await checkComplexRotateGpu(device)
  const gpu = new MandelbrotCustomWebGPU(
    { onGpuUpdate() {} },
    new WorkerContext(),
    (error) => {
      throw new Error(error)
    },
    { devicePromise: Promise.resolve(device) },
  )
  class TestBuddhabrot extends BuddhabrotWebGPU {
    initGpu() {
      return Promise.resolve(device)
    }
  }
  const buddha = new TestBuddhabrot({ width: 32, height: 32 })
  class FixedSampleBuddhabrot extends TestBuddhabrot {
    async _startGpu(params, originalDevice) {
      // Force the same known c on the real shader, isolating projection from
      // random sampling. Formula evaluation and density readback are unchanged.
      const fixedDevice = new Proxy(originalDevice, {
        get(target, key) {
          if (key === 'createShaderModule')
            return (descriptor) =>
              target.createShaderModule({
                ...descriptor,
                code: descriptor.code.replace(
                  'let sample = vec2<f32>(rx, ry);',
                  `let sample = vec2<f32>(${params.fixedSample[0].toFixed(8)}, ${params.fixedSample[1].toFixed(8)});`,
                ),
              })
          const value = Reflect.get(target, key, target)
          return typeof value === 'function' ? value.bind(target) : value
        },
      })
      await super._startGpu(params, fixedDevice)
    }
  }
  const fixedBuddha = new FixedSampleBuddhabrot({ width: 32, height: 32 })
  const trap = new OrbitTrapWebGPU(null, { devicePromise: Promise.resolve(device) })
  // Read the real shader's iteration marker before palette coloring can hide
  // a classification error. Coordinate mapping and uniforms remain unchanged.
  const trapShader = trap._shaderCode.bind(trap)
  trap._shaderCode = (expr) =>
    trapShader(expr).replace(
      'outPixels[idx] = colorFromTrap(iterValue, trapValue);',
      'outPixels[idx] = u32(iterValue);',
    )
  let checks = 0
  device.pushErrorScope('validation')
  try {
    const expressions = [
      'z+i',
      'c*i',
      'c+i',
      'z*z+c*i',
      'z*z + rotate(c, i*|Im(z)|)',
      'z*z + rotate(c, i*|Re(z)|)',
      'z*z+c+i',
      'z*z+i+c/i',
      'exp(1000+Re(c))*i+c',
      'exp(1000+Re(c))',
      '1e21+i',
      '1e20+i',
      '0.5*i',
    ]
    for (const expr of expressions) {
      for (const maxIter of [0, 4, 5, 64]) {
        for (const escapeRadius of [0.25, 4, 8]) {
          for (const smooth of [false, true]) {
            for (const isJulia of [false, true]) {
              const task = {
                w: 1,
                h: 1,
                frameWidth: 1,
                frameHeight: 1,
                xOffset: 0,
                yOffset: 0,
                frameTopLeft: [fxp.fromNumber(0.25), fxp.fromNumber(0.5)],
                frameBottomRight: [fxp.fromNumber(0.25), fxp.fromNumber(0.5)],
                maxIter,
                smooth,
                supersampling: 0,
                fractalType: isJulia ? 'julia-custom' : 'custom',
                iterationFunction: expr,
                z0: [0, 0],
                z0Real: 0,
                z0Imag: 0,
                escapeRadius,
                juliaRe: 0.25,
                juliaIm: 0.5,
              }
              const cpu = await new MandelbrotCustom(context).process(task)
              gpu.iterationFunction = expr
              const actual = await gpu.calculate(1, 1, task)
              assert(gpu.pipeline.pipeline, 'Ordinary test did not run on GPU')
              assert(!actual.error, actual.error)
              assert(
                (actual.values[0] === 2) === (cpu.values[0] === 2),
                `CPU/GPU escape mismatch: ${expr}, N=${maxIter}, R=${escapeRadius}, smooth=${smooth}, Julia=${isJulia}`,
              )
              assert(actual.values[0] === 2 || actual.values[0] >= 4, `Invalid escaped marker: ${expr}`)
              checks++
            }
          }
        }
      }
    }

    for (const expr of ['z+i', 'c+i', 'z*z+c', '1e21+i', '0.5*i']) {
      for (const maxIter of [4, 5]) {
        for (const escapeRadius of [0.25, 4]) {
          for (const smooth of [false, true]) {
            for (const isJulia of [false, true]) {
              const task = {
                w: 1,
                h: 1,
                frameWidth: 1,
                frameHeight: 1,
                xOffset: 0,
                yOffset: 0,
                frameTopLeft: [fxp.fromNumber(0), fxp.fromNumber(isJulia ? 0 : 5)],
                frameBottomRight: [fxp.fromNumber(0), fxp.fromNumber(isJulia ? 0 : 5)],
                maxIter,
                smooth,
                supersampling: 0,
                escapeRadius,
                fractalType: isJulia ? 'julia-custom' : 'custom',
                iterationFunction: expr,
                z0Real: 0,
                z0Imag: 0,
                juliaRe: 0,
                juliaIm: 5,
                trapSpec: { mode: 'distance_closest', shape: 'ring', size: 0.5 },
              }
              const cpu = await new MandelbrotCustom(context).process(task)
              const actual = await trap.render(task)
              assert(
                (actual.rgba[0] === 2) === (cpu.values[0] === 2),
                `Orbit Trap escape mismatch: ${expr}, N=${maxIter}, R=${escapeRadius}, smooth=${smooth}, Julia=${isJulia}`,
              )
              checks++
            }
          }
        }
      }
    }

    // Test exact orbit locations, rather than comparing random density images.
    // All sampled c values share the same z+i trajectory.
    const base = {
      width: 32,
      height: 32,
      samples: 64,
      maxIter: 5,
      gpu: true,
      fractalType: 'custom',
      iterationFunction: 'z+i',
      center: { x: 0, y: 0 },
      zoom: 0.25,
      escapeRadius: 4,
      z0Real: 0,
      z0Imag: 0,
      palette: { bands: [{ color: [255, 255, 255], cum: 1, ratio: 1 }] },
    }
    async function density(options, expectedIndices, runner = buddha) {
      let completion
      runner.onComplete = (result) => {
        completion = result
      }
      await runner.start({ ...base, ...options })
      assert(!runner._cpuRunner, 'Buddhabrot unexpectedly fell back to CPU')
      assert(completion && !completion.error, completion?.error || 'Missing GPU completion')
      const actual = Array.from(runner.densityR.keys()).filter((i) => runner.densityR[i] > 0)
      assert(
        JSON.stringify(actual) === JSON.stringify(expectedIndices),
        `Buddhabrot orbit mismatch: ${JSON.stringify(options)}, got ${actual}, expected ${expectedIndices}`,
      )
      checks++
    }
    const vertical = [18, 20, 22, 24].map((y) => y * 32 + 16)
    await density({ mode: 'buddha' }, vertical)
    await density({ mode: 'antibuddha' }, [])
    await density({ maxIter: 4, mode: 'buddha' }, [])
    await density({ maxIter: 4, mode: 'antibuddha' }, vertical)
    for (const iterationFunction of ['c*i', 'c+i', 'exp(1000+Re(c))*i+c', 'exp(1000+Re(c))', '1e21+i', '1e20+i']) {
      await density({ iterationFunction, mode: 'buddha' }, [])
      if (!iterationFunction.startsWith('c')) await density({ iterationFunction, mode: 'antibuddha' }, [])
    }
    await density({ iterationFunction: '0', z0Imag: 5, mode: 'buddha' }, [])
    await density({ iterationFunction: '0', z0Imag: 5, mode: 'antibuddha' }, [])
    // Large radii overflow their f32 square, so the rare component-limit path
    // must still reject the sentinel while retaining finite diagonal points.
    for (const mode of ['buddha', 'antibuddha']) {
      await density(
        { iterationFunction: '1e20+i', escapeRadius: 2e20, center: { x: 1e20, y: 0 }, zoom: 1e-20, mode },
        [],
      )
      await density(
        { iterationFunction: '8e19+8e19*i', escapeRadius: 2e20, zoom: 1e-20, mode },
        mode === 'antibuddha' ? [22 * 32 + 22] : [],
      )
    }
    await density(
      {
        iterationFunction: 'z+60000*i',
        escapeRadius: 200000,
        center: { x: 0, y: 110000 },
        zoom: 0.00001,
        maxIter: 4,
        mode: 'buddha',
      },
      [12, 16, 21].map((y) => y * 32 + 16),
    )
    for (const [iterationFunction, fixedSample, points] of [
      [
        'z*z+c*i',
        [0, -1],
        [
          [0, -1],
          [0, -2],
        ],
      ],
      [
        'z*z+c/i',
        [0, 1],
        [
          [0, 1],
          [0, 2],
        ],
      ],
      [
        'z*z+c+i',
        [1, -1],
        [
          [1, -1],
          [2, -1],
        ],
      ],
      [
        'z*z+i+c/i',
        [1, 1],
        [
          [1, 1],
          [1, 2],
        ],
      ],
      [
        'z*z+2*c+1+i',
        [0, -0.5],
        [
          [0, -0.5],
          [0.5, -0.5],
        ],
      ],
      [
        'z*z+(c+i)*i',
        [0, -2],
        [
          [0, -2],
          [0, -3],
        ],
      ],
    ]) {
      const indices = points.map(([x, y]) => Math.floor(y * 2 + 16) * 32 + Math.floor(x * 2 + 16)).sort((a, b) => a - b)
      await density({ iterationFunction, fixedSample, mode: 'buddha' }, indices, fixedBuddha)
      await density({ iterationFunction, fixedSample, mode: 'antibuddha', maxIter: 2 }, indices, fixedBuddha)
    }
    await density(
      {
        iterationFunction: 'z*z+c*i',
        fractalType: 'julia-custom',
        fixedSample: [0, 0],
        juliaRe: 0,
        juliaIm: -1,
        mode: 'buddha',
      },
      [16, 18, 20].map((x) => 16 * 32 + x),
      fixedBuddha,
    )
    // An actual, equally framed CPU/GPU render makes orientation and clipping
    // reviewable in the browser, alongside the numerical projection checks.
    const width = 400
    const height = 300
    const preview = {
      width,
      height,
      maxIter: 256,
      samples: 500000,
      palette: {
        bands: [
          { color: [0, 0, 0], cum: 0.01 },
          { color: [255, 100, 25], cum: 0.1 },
          { color: [60, 170, 255], cum: 1 },
        ],
      },
      iterationFunction: 'z*z+c*i',
      fractalType: 'custom',
      mode: 'buddha',
      center: { x: 0, y: 0.5 },
      zoom: 0.8,
      escapeRadius: 4,
      z0Real: 0,
      z0Imag: 0,
      gpu: true,
    }
    const ordinary = await new MandelbrotCustom(context).process({
      w: width,
      h: height,
      frameWidth: width,
      frameHeight: height,
      xOffset: 0,
      yOffset: 0,
      frameTopLeft: [fxp.fromNumber(-2.5), fxp.fromNumber(-1.375)],
      frameBottomRight: [fxp.fromNumber(2.5), fxp.fromNumber(2.375)],
      maxIter: preview.maxIter,
      iterationFunction: preview.iterationFunction,
      fractalType: 'custom',
      escapeRadius: 4,
      smooth: false,
      supersampling: 0,
    })
    const normalImage = new ImageData(width, height)
    for (let i = 0; i < ordinary.values.length; i++) {
      const value = ordinary.values[i]
      normalImage.data[i * 4] = value === 2 ? 0 : Math.min(255, value * 3)
      normalImage.data[i * 4 + 1] = value === 2 ? 0 : Math.min(255, value * 5)
      normalImage.data[i * 4 + 2] = value === 2 ? 0 : Math.min(255, 70 + value * 6)
      normalImage.data[i * 4 + 3] = 255
    }
    document.querySelector('#ordinary').getContext('2d').putImageData(normalImage, 0, 0)
    await buddha.start(preview)
    assert(!buddha._cpuRunner, 'Preview must render on GPU')
    const channels = [buddha.densityR, buddha.densityG, buddha.densityB]
    const maxima = channels.map((channel) => channel.reduce((maximum, value) => Math.max(maximum, value), 1))
    const densityImage = new ImageData(width, height)
    for (let i = 0; i < width * height; i++) {
      for (let ch = 0; ch < 3; ch++) {
        densityImage.data[i * 4 + ch] = 255 * Math.sqrt(channels[ch][i] / maxima[ch])
      }
      densityImage.data[i * 4 + 3] = 255
    }
    document.querySelector('#buddha').getContext('2d').putImageData(densityImage, 0, 0)
    const error = await device.popErrorScope()
    assert(!error, error?.message)
    document.querySelector('#result').textContent = `PASS: complex-angle rotation (CPU/WebGPU) and ${checks} CPU/WebGPU escape and Buddhabrot orbit checks`
  } finally {
    gpu.pipeline.dispose()
    buddha.stop()
    fixedBuddha.stop()
    trap._destroyOutputResources()
    trap._destroyBitmapBuffer()
    device.destroy()
  }
}

try {
  await run()
} catch (error) {
  document.querySelector('#result').textContent = error.stack
  console.error(error)
}

/**
 * WebGPU renderer for Orbit Trap Custom.
 *
 * This path computes the fractal iteration, orbit-trap sampling, and final RGBA
 * color on the GPU. The existing CPU orbit-trap renderer remains the fallback
 * and source of truth.
 */

import { getParsedExpression } from './customFunctionParser.mjs'
import { TRAP_MODE, TRAP_SHAPE } from './orbitTrap.mjs'
import { COLOR_PATTERN } from './palette.js'
import { BAILOUT_SMOOTH } from './sharedCalculations.mjs'
import { CUSTOM_FUNCTION_WGSL_HELPERS, jsExprToWGSL_safe } from './wgslCompiler.mjs'

const WORKGROUP_SIZE_X = 16
const WORKGROUP_SIZE_Y = 8
const SPEC_SIZE = 160
const DEFAULT_BACKGROUND = '#002580'
const MAX_VALID_VALUE_WGSL = '1e20'

const MODE_TO_ID = {
  [TRAP_MODE.DISTANCE_CLOSEST]: 0,
  [TRAP_MODE.DISTANCE_FARTHEST]: 1,
  [TRAP_MODE.DISTANCE_AVERAGE]: 2,
  [TRAP_MODE.CAPTURE_FIRST]: 3,
  [TRAP_MODE.CAPTURE_STEP]: 4,
  [TRAP_MODE.TIA]: 5,
}

const SHAPE_TO_ID = {
  [TRAP_SHAPE.CROSS]: 0,
  [TRAP_SHAPE.RING]: 1,
  [TRAP_SHAPE.CIRCLE]: 1,
  [TRAP_SHAPE.POINT]: 2,
  [TRAP_SHAPE.LINE]: 3,
  [TRAP_SHAPE.PARABOLA]: 4,
  [TRAP_SHAPE.TRIANGLE]: 5,
  [TRAP_SHAPE.SQUARE]: 6,
  [TRAP_SHAPE.BITMAP]: 7,
}

const COLOR_PATTERN_TO_ID = {
  [COLOR_PATTERN.COSINE_RGB]: 0,
  [COLOR_PATTERN.HSL_CAPTURE]: 1,
  [COLOR_PATTERN.TIA_TRIANGLE]: 2,
}

function _formatError(error) {
  return error?.message ? error.message : String(error)
}

function _normalizeColor(value, fallback = DEFAULT_BACKGROUND) {
  const raw = String(value ?? '').trim()
  const withHash = /^[0-9a-fA-F]{6}$/.test(raw) ? `#${raw}` : raw
  const color = /^#[0-9a-fA-F]{6}$/.test(withHash) ? withHash : fallback
  const hex = color.slice(1)
  return [
    Number.parseInt(hex.slice(0, 2), 16),
    Number.parseInt(hex.slice(2, 4), 16),
    Number.parseInt(hex.slice(4, 6), 16),
  ]
}

function _packRgba(r, g, b, a = 255) {
  return ((a & 255) << 24) | ((b & 255) << 16) | ((g & 255) << 8) | (r & 255)
}

function _splitF32(value) {
  const hi = Math.fround(value)
  return [hi, Math.fround(value - hi)]
}

function _isDefaultMandelbrotIteration(iterationFunction) {
  const expr = String(iterationFunction || 'z*z + c')
    .replace(/\s+/g, '')
    .toLowerCase()
  return expr === 'z*z+c' || expr === 'c+z*z'
}

function _createPackedBitmap(bitmapData, width, height) {
  const packed = new Uint32Array(width * height)
  for (let i = 0; i < packed.length; i++) {
    const src = i * 4
    packed[i] = _packRgba(bitmapData[src], bitmapData[src + 1], bitmapData[src + 2], bitmapData[src + 3])
  }
  return packed
}

function _customIterationToWGSL(iterationFunction) {
  let wgsl = jsExprToWGSL_safe(getParsedExpression(iterationFunction || 'z*z + c'))
  if (/\b__\w+/.test(wgsl)) {
    wgsl = wgsl.replace(/\b__([A-Za-z0-9_]+)/g, '_$1')
  }
  wgsl = wgsl.replace(/\+\s*-/g, '-')
  wgsl = wgsl.replace(/-\s*-/g, '+')
  return wgsl
}

export class OrbitTrapWebGPU {
  constructor(errorCallback = null) {
    this.errorCallback = errorCallback
    this.available = true
    this.devicePromise = this._initGpu()
    this.pipeline = null
    this.pipelineKey = null
    this.bindGroupLayout = null
    this.bitmapBuffer = null
    this.bitmapKey = null
    this.outputResources = null
    this.bindGroup = null
    this.bindGroupKey = null
    this.specArrayBuffer = new ArrayBuffer(SPEC_SIZE)
    this.specDataView = new DataView(this.specArrayBuffer)
    this.running = Promise.resolve()
    this.currentTask = null
    this.newTask = null
  }

  async _initGpu() {
    try {
      const adapter = await navigator.gpu?.requestAdapter({ powerPreference: 'high-performance' })
      const device = await adapter?.requestDevice()
      if (!device) {
        this.available = false
        this.errorCallback?.('need a browser that supports WebGPU')
        return null
      }

      this.available = true
      device.lost.then(() => {
        this.available = false
        this.pipeline = null
        this.pipelineKey = null
        this.bindGroupLayout = null
        this._destroyBitmapBuffer()
        this._destroyOutputResources()
        this.devicePromise = this._initGpu()
      })
      return device
    } catch (error) {
      this.available = false
      this.errorCallback?.(_formatError(error))
      return null
    }
  }

  shouldStop() {
    return this.currentTask !== this.newTask
  }

  async process(task) {
    this.newTask = task.jobToken
    await this.running
    if (task.jobToken !== this.newTask) return

    this.currentTask = task.jobToken
    this.running = this._process(task)
    await this.running
  }

  async _process(task) {
    try {
      const result = await this.render(task)
      if (this.shouldStop()) {
        task.onUpdate?.({
          jobToken: task.jobToken,
          renderedPixels: task.w * task.h,
          isFinished: true,
        })
        return result
      }

      task.onUpdate?.({
        jobToken: task.jobToken,
        rgba: result.rgba,
        width: task.w,
        height: task.h,
        renderedPixels: task.w * task.h,
        isFinished: true,
        gpuColored: true,
      })
      return result
    } catch (error) {
      const message = _formatError(error)
      console.warn('OrbitTrapBitmapWebGPU failed:', message)
      this.errorCallback?.(message)
      task.onUpdate?.({
        jobToken: task.jobToken,
        renderedPixels: 0,
        isFinished: true,
        error: message,
        fallbackToCpu: true,
      })
      return { error: message }
    }
  }

  async render(task) {
    const device = await this.devicePromise
    if (!device) throw new Error('WebGPU device is not available')

    const trapSpec = task.trapSpec

    const iterationExpr =
      task.fractalType === 'custom' || task.fractalType === 'julia-custom'
        ? _customIterationToWGSL(task.iterationFunction)
        : 'vec2<f32>(z.x * z.x - z.y * z.y + c.x, 2.0 * z.x * z.y + c.y)'
    const pipeline = await this._getPipeline(device, iterationExpr)
    const bitmapBuffer = this._ensureBitmapBuffer(device, trapSpec)
    const resources = this._ensureOutputResources(device, task.w, task.h, bitmapBuffer)

    device.queue.writeBuffer(resources.specBuffer, 0, this._buildSpec(task, trapSpec))

    const encoder = device.createCommandEncoder({ label: 'orbit trap bitmap encoder' })
    const pass = encoder.beginComputePass({ label: 'orbit trap bitmap color pass' })
    pass.setPipeline(pipeline)
    pass.setBindGroup(0, resources.bindGroup)
    pass.dispatchWorkgroups(Math.ceil(task.w / WORKGROUP_SIZE_X), Math.ceil(task.h / WORKGROUP_SIZE_Y), 1)
    pass.end()
    encoder.copyBufferToBuffer(resources.outBuffer, 0, resources.readBuffer, 0, resources.byteSize)
    device.queue.submit([encoder.finish()])

    await resources.readBuffer.mapAsync(GPUMapMode.READ)
    const rgba = new Uint8ClampedArray(resources.readBuffer.getMappedRange().slice(0, resources.byteSize))
    resources.readBuffer.unmap()

    return { rgba }
  }

  _buildSpec(task, trapSpec) {
    const data = this.specArrayBuffer
    const view = this.specDataView
    const frameTopLeft = task.frameTopLeft.map((v) => v.toNumber())
    const frameBottomRight = task.frameBottomRight.map((v) => v.toNumber())
    const cWidth = frameBottomRight[0] - frameTopLeft[0]
    const cHeight = frameBottomRight[1] - frameTopLeft[1]
    const ddr = cWidth / task.frameWidth
    const ddi = cHeight / task.frameHeight
    const modeId = MODE_TO_ID[trapSpec.mode] ?? MODE_TO_ID[TRAP_MODE.DISTANCE_CLOSEST]
    const shapeId = SHAPE_TO_ID[trapSpec.shape] ?? SHAPE_TO_ID[TRAP_SHAPE.RING]
    const colorPatternId = COLOR_PATTERN_TO_ID[task.colorPatternId] ?? COLOR_PATTERN_TO_ID[COLOR_PATTERN.COSINE_RGB]
    const ssScale = task.supersampling > 0 ? task.supersampling : 1
    const valueBailout = task.smooth ? BAILOUT_SMOOTH : (task.escapeRadius ?? 4.0) ** 2
    const trapEscapeBailout = (task.escapeRadius ?? 4.0) ** 2
    const threshold = Number.isFinite(trapSpec.threshold) ? trapSpec.threshold : 1e20
    const [bgR, bgG, bgB] = _normalizeColor(trapSpec.bitmapBackgroundColor, DEFAULT_BACKGROUND)
    const isJulia = task.fractalType === 'julia' || task.fractalType === 'julia-custom'
    const usesCustomIteration =
      (task.fractalType === 'custom' || task.fractalType === 'julia-custom') &&
      !_isDefaultMandelbrotIteration(task.iterationFunction)
    const frameOffsetR = task.xOffset * ddr
    const frameOffsetI = task.yOffset * ddi
    const fixedValueR = isJulia ? (task.juliaRe ?? 0) : (task.z0?.[0] ?? task.z0Real ?? 0)
    const fixedValueI = isJulia ? (task.juliaIm ?? 0) : (task.z0?.[1] ?? task.z0Imag ?? 0)
    const [, frameTopLeftRLo] = _splitF32(frameTopLeft[0])
    const [, frameTopLeftILo] = _splitF32(frameTopLeft[1])
    const [, frameOffsetRLo] = _splitF32(frameOffsetR)
    const [, frameOffsetILo] = _splitF32(frameOffsetI)
    const [, ddrLo] = _splitF32(ddr)
    const [, ddiLo] = _splitF32(ddi)
    const [, fixedValueRLo] = _splitF32(fixedValueR)
    const [, fixedValueILo] = _splitF32(fixedValueI)

    let offset = 0
    view.setUint32(offset, task.w, true)
    view.setUint32(offset + 4, task.h, true)
    view.setUint32(offset + 8, task.maxIter, true)
    view.setUint32(offset + 12, modeId, true)
    offset += 16

    view.setUint32(offset, task.smooth ? 1 : 0, true)
    view.setUint32(offset + 4, ssScale, true)
    view.setUint32(offset + 8, usesCustomIteration ? 1 : 0, true)
    view.setUint32(offset + 12, Math.max(1, trapSpec.captureStep ?? 1), true)
    offset += 16

    view.setFloat32(offset, frameTopLeft[0], true)
    view.setFloat32(offset + 4, frameTopLeft[1], true)
    view.setFloat32(offset + 8, frameOffsetR, true)
    view.setFloat32(offset + 12, frameOffsetI, true)
    offset += 16

    view.setFloat32(offset, ddr, true)
    view.setFloat32(offset + 4, ddi, true)
    view.setFloat32(offset + 8, fixedValueR, true)
    view.setFloat32(offset + 12, fixedValueI, true)
    offset += 16

    view.setFloat32(offset, trapEscapeBailout, true)
    view.setFloat32(offset + 4, valueBailout, true)
    view.setFloat32(offset + 8, trapSpec.tx ?? 0, true)
    view.setFloat32(offset + 12, trapSpec.ty ?? 0, true)
    offset += 16

    view.setFloat32(offset, trapSpec.size ?? 1, true)
    view.setFloat32(offset + 4, trapSpec.angle ?? 0, true)
    view.setFloat32(offset + 8, threshold, true)
    view.setFloat32(offset + 12, trapSpec.startIter ?? 0, true)
    offset += 16

    view.setUint32(offset, trapSpec.bitmapWidth ?? 0, true)
    view.setUint32(offset + 4, trapSpec.bitmapHeight ?? 0, true)
    view.setUint32(offset + 8, _packRgba(bgR, bgG, bgB), true)
    view.setUint32(offset + 12, _packRgba(0, 0, 0), true)
    offset += 16

    view.setUint32(offset, shapeId, true)
    view.setUint32(offset + 4, colorPatternId, true)
    view.setUint32(offset + 8, isJulia ? 1 : 0, true)
    view.setUint32(offset + 12, 0, true)
    offset += 16

    view.setFloat32(offset, frameTopLeftRLo, true)
    view.setFloat32(offset + 4, frameTopLeftILo, true)
    view.setFloat32(offset + 8, frameOffsetRLo, true)
    view.setFloat32(offset + 12, frameOffsetILo, true)
    offset += 16

    view.setFloat32(offset, ddrLo, true)
    view.setFloat32(offset + 4, ddiLo, true)
    view.setFloat32(offset + 8, fixedValueRLo, true)
    view.setFloat32(offset + 12, fixedValueILo, true)

    return data
  }

  _ensureOutputResources(device, width, height, bitmapBuffer) {
    const byteSize = width * height * 4
    const needsNewBuffers = !this.outputResources || this.outputResources.byteSize !== byteSize

    if (needsNewBuffers) {
      this._destroyOutputResources()
      this.outputResources = {
        byteSize,
        specBuffer: device.createBuffer({
          size: SPEC_SIZE,
          usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        }),
        outBuffer: device.createBuffer({
          size: byteSize,
          usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
        }),
        readBuffer: device.createBuffer({
          size: byteSize,
          usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
        }),
      }
    }

    const bindGroupKey = `${byteSize}:${this.bitmapKey ?? 'none'}`
    if (!this.bindGroup || this.bindGroupKey !== bindGroupKey) {
      this.bindGroup = device.createBindGroup({
        layout: this.bindGroupLayout,
        entries: [
          { binding: 0, resource: { buffer: this.outputResources.specBuffer } },
          { binding: 1, resource: { buffer: this.outputResources.outBuffer } },
          { binding: 2, resource: { buffer: bitmapBuffer } },
        ],
      })
      this.bindGroupKey = bindGroupKey
    }

    return {
      ...this.outputResources,
      bindGroup: this.bindGroup,
    }
  }

  _ensureBitmapBuffer(device, trapSpec) {
    const hasBitmapImage =
      trapSpec.shape === TRAP_SHAPE.BITMAP &&
      trapSpec.bitmapData &&
      (trapSpec.bitmapWidth ?? 0) > 0 &&
      (trapSpec.bitmapHeight ?? 0) > 0

    if (!hasBitmapImage) {
      const key = 'dummy'
      if (this.bitmapBuffer && this.bitmapKey === key) return this.bitmapBuffer

      this._destroyBitmapBuffer()
      const packed = new Uint32Array([0])
      this.bitmapBuffer = device.createBuffer({
        size: packed.byteLength,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      })
      device.queue.writeBuffer(this.bitmapBuffer, 0, packed)
      this.bitmapKey = key
      return this.bitmapBuffer
    }

    const key = `${trapSpec.bitmapVersion ?? 0}:${trapSpec.bitmapWidth}x${trapSpec.bitmapHeight}:${trapSpec.bitmapData?.byteLength ?? 0}`
    if (this.bitmapBuffer && this.bitmapKey === key) return this.bitmapBuffer

    this._destroyBitmapBuffer()
    const packed = _createPackedBitmap(trapSpec.bitmapData, trapSpec.bitmapWidth, trapSpec.bitmapHeight)
    this.bitmapBuffer = device.createBuffer({
      size: packed.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    })
    device.queue.writeBuffer(this.bitmapBuffer, 0, packed)
    this.bitmapKey = key
    return this.bitmapBuffer
  }

  _destroyBitmapBuffer() {
    if (this.bitmapBuffer) {
      try {
        this.bitmapBuffer.destroy()
      } catch (_e) {}
    }
    this.bitmapBuffer = null
    this.bitmapKey = null
    this.bindGroup = null
    this.bindGroupKey = null
  }

  _destroyOutputResources() {
    for (const buffer of [
      this.outputResources?.specBuffer,
      this.outputResources?.outBuffer,
      this.outputResources?.readBuffer,
    ]) {
      if (!buffer) continue
      try {
        if (buffer.mapState === 'mapped') buffer.unmap()
      } catch (_e) {}
      try {
        buffer.destroy()
      } catch (_e) {}
    }
    this.outputResources = null
    this.bindGroup = null
    this.bindGroupKey = null
  }

  async _getPipeline(device, iterationExpr) {
    const key = iterationExpr
    if (this.pipeline && this.pipelineKey === key) return this.pipeline

    const module = device.createShaderModule({
      label: 'orbit trap bitmap color shader',
      code: this._shaderCode(iterationExpr),
    })

    try {
      const info = await module.getCompilationInfo?.()
      if (info?.messages?.length) {
        for (const msg of info.messages) {
          const prefix = `OrbitTrapBitmapWebGPU shader ${msg.type}:`
          if (msg.type === 'error') console.error(prefix, msg.message)
          else console.warn(prefix, msg.message)
        }
      }
    } catch (_e) {}

    this.bindGroupLayout = device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
        { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
      ],
    })
    this.bindGroup = null
    this.bindGroupKey = null
    const layout = device.createPipelineLayout({ bindGroupLayouts: [this.bindGroupLayout] })
    this.pipeline = await device.createComputePipelineAsync({
      label: 'orbit trap bitmap color pipeline',
      layout,
      compute: { module, entryPoint: 'main' },
    })
    this.pipelineKey = key
    return this.pipeline
  }

  _shaderCode(iterationExpr) {
    return `
struct Spec {
  dims: vec4<u32>,
  flags: vec4<u32>,
  frame0: vec4<f32>,
  frame1: vec4<f32>,
  trap0: vec4<f32>,
  trap1: vec4<f32>,
  bitmap: vec4<u32>,
  extra: vec4<u32>,
  coord0: vec4<f32>,
  coord1: vec4<f32>,
};

@group(0) @binding(0) var<uniform> spec: Spec;
@group(0) @binding(1) var<storage, read_write> outPixels: array<u32>;
@group(0) @binding(2) var<storage, read> bitmapPixels: array<u32>;

${CUSTOM_FUNCTION_WGSL_HELPERS}

fn packRgba(r: u32, g: u32, b: u32, a: u32) -> u32 {
  return (a << 24u) | (b << 16u) | (g << 8u) | r;
}

fn channelR(pixel: u32) -> u32 { return pixel & 255u; }
fn channelG(pixel: u32) -> u32 { return (pixel >> 8u) & 255u; }
fn channelB(pixel: u32) -> u32 { return (pixel >> 16u) & 255u; }
fn channelA(pixel: u32) -> u32 { return (pixel >> 24u) & 255u; }

fn toByte(value: f32) -> u32 {
  return u32(round(clamp(value, 0.0, 255.0)));
}

fn blendBitmapPixelOverBackground(pixel: u32, background: u32) -> u32 {
  let alphaU = channelA(pixel);
  if (alphaU < 10u) {
    return background;
  }
  if (alphaU >= 255u) {
    return packRgba(channelR(pixel), channelG(pixel), channelB(pixel), 255u);
  }

  let alpha = f32(alphaU) / 255.0;
  let invAlpha = 1.0 - alpha;
  let r = f32(channelR(pixel)) * alpha + f32(channelR(background)) * invAlpha;
  let g = f32(channelG(pixel)) * alpha + f32(channelG(background)) * invAlpha;
  let b = f32(channelB(pixel)) * alpha + f32(channelB(background)) * invAlpha;
  return packRgba(toByte(r), toByte(g), toByte(b), 255u);
}

fn hslToRgb(h: f32, s: f32, l: f32) -> u32 {
  let rBase = fract((0.0 + 6.0 * h) / 6.0) * 6.0;
  let gBase = fract((4.0 + 6.0 * h) / 6.0) * 6.0;
  let bBase = fract((2.0 + 6.0 * h) / 6.0) * 6.0;
  let chromaScale = 1.0 - abs(2.0 * l - 1.0);
  let rVal = clamp(abs(rBase - 3.0) - 1.0, 0.0, 1.0);
  let gVal = clamp(abs(gBase - 3.0) - 1.0, 0.0, 1.0);
  let bVal = clamp(abs(bBase - 3.0) - 1.0, 0.0, 1.0);
  let r = (l + s * (rVal - 0.5) * chromaScale) * 255.0;
  let g = (l + s * (gVal - 0.5) * chromaScale) * 255.0;
  let b = (l + s * (bVal - 0.5) * chromaScale) * 255.0;
  return packRgba(toByte(r), toByte(g), toByte(b), 255u);
}

fn safeNext(z: vec2<f32>, c: vec2<f32>, n: f32) -> vec2<f32> {
  let z_next = ${iterationExpr};
  let valid = (z_next.x == z_next.x) && (z_next.y == z_next.y) &&
    abs(z_next.x) < ${MAX_VALID_VALUE_WGSL} && abs(z_next.y) < ${MAX_VALID_VALUE_WGSL};
  return select(vec2<f32>(${MAX_VALID_VALUE_WGSL}, 0.0), z_next, valid);
}

struct Ds {
  hi: f32,
  lo: f32,
};

struct Ds2 {
  x: Ds,
  y: Ds,
};

fn dsMake(value: f32) -> Ds {
  return Ds(value, 0.0);
}

fn dsFromHiLo(hi: f32, lo: f32) -> Ds {
  return dsNormalize(Ds(hi, lo));
}

fn dsToF32(value: Ds) -> f32 {
  return value.hi + value.lo;
}

fn dsNormalize(value: Ds) -> Ds {
  let s = value.hi + value.lo;
  let e = value.lo - (s - value.hi);
  return Ds(s, e);
}

fn dsAdd(a: Ds, b: Ds) -> Ds {
  let s = a.hi + b.hi;
  let bb = s - a.hi;
  let err = (a.hi - (s - bb)) + (b.hi - bb);
  return dsNormalize(Ds(s, err + a.lo + b.lo));
}

fn dsSub(a: Ds, b: Ds) -> Ds {
  return dsAdd(a, Ds(-b.hi, -b.lo));
}

fn dsMul(a: Ds, b: Ds) -> Ds {
  let p = a.hi * b.hi;
  let err = fma(a.hi, b.hi, -p) + (a.hi * b.lo + a.lo * b.hi);
  return dsNormalize(Ds(p, err));
}

fn dsAbsWithinLimit(value: Ds) -> bool {
  return abs(dsToF32(value)) < ${MAX_VALID_VALUE_WGSL};
}

fn safeNextMandelbrotDs(z: Ds2, c: Ds2) -> Ds2 {
  let zr2 = dsMul(z.x, z.x);
  let zi2 = dsMul(z.y, z.y);
  let zri = dsMul(z.x, z.y);
  let nextX = dsAdd(dsSub(zr2, zi2), c.x);
  let nextY = dsAdd(dsAdd(zri, zri), c.y);
  let x = dsToF32(nextX);
  let y = dsToF32(nextY);
  let valid = (x == x) && (y == y) && dsAbsWithinLimit(nextX) && dsAbsWithinLimit(nextY);
  if (valid) {
    return Ds2(nextX, nextY);
  }
  return Ds2(dsMake(${MAX_VALID_VALUE_WGSL}), dsMake(0.0));
}

fn ds2ToVec2(value: Ds2) -> vec2<f32> {
  return vec2<f32>(dsToF32(value.x), dsToF32(value.y));
}

fn dsNormSq(value: Ds2) -> f32 {
  let z = ds2ToVec2(value);
  return dot(z, z);
}

fn pixelToComplexDs(px: f32, py: f32) -> Ds2 {
  let realBase = dsAdd(
    dsFromHiLo(spec.frame0.x, spec.coord0.x),
    dsFromHiLo(spec.frame0.z, spec.coord0.z)
  );
  let imagBase = dsAdd(
    dsFromHiLo(spec.frame0.y, spec.coord0.y),
    dsFromHiLo(spec.frame0.w, spec.coord0.w)
  );
  let real = dsAdd(realBase, dsMul(dsMake(px), dsFromHiLo(spec.frame1.x, spec.coord1.x)));
  let imag = dsAdd(imagBase, dsMul(dsMake(py), dsFromHiLo(spec.frame1.y, spec.coord1.y)));
  return Ds2(real, imag);
}

fn z0FromSpecDs() -> Ds2 {
  return Ds2(
    dsFromHiLo(spec.frame1.z, spec.coord1.z),
    dsFromHiLo(spec.frame1.w, spec.coord1.w)
  );
}

fn encodeUV(u: f32, v: f32) -> f32 {
  let uInt = u32(clamp(round(u * 4095.0), 0.0, 4095.0));
  let vInt = u32(clamp(round(v * 4095.0), 0.0, 4095.0));
  return -f32(uInt * 4096u + vInt + 1u);
}

fn sampleBitmap(u: f32, v: f32) -> u32 {
  let bw = spec.bitmap.x;
  let bh = spec.bitmap.y;
  let px = min(u32(floor(clamp(u, 0.0, 1.0) * f32(bw))), bw - 1u);
  let py = min(u32(floor(clamp(v, 0.0, 1.0) * f32(bh))), bh - 1u);
  return bitmapPixels[py * bw + px];
}

fn computeIterValue(zInit: vec2<f32>, c: vec2<f32>) -> f32 {
  var z = zInit;
  var zq = dot(z, z);
  var iter = -1;

  loop {
    if (zq > spec.trap0.y) {
      break;
    }

    z = safeNext(z, c, f32(iter + 1));
    let oldIter = iter;
    iter = iter + 1;
    if (oldIter == i32(spec.dims.z)) {
      return 2.0;
    }

    zq = dot(z, z);
  }

  var iterValue = f32(iter + 4);
  if (spec.flags.x == 1u && iterValue > 3.0) {
    let logZn = log(zq) / 2.0;
    let nu = log(logZn / log(2.0)) / log(2.0);
    iterValue = floor(iterValue + 1.0 - nu);
  }
  return iterValue;
}

fn computeIterValueMandelbrotDs(zInit: Ds2, c: Ds2) -> f32 {
  var z = zInit;
  var zq = dsNormSq(z);
  var iter = -1;

  loop {
    if (zq > spec.trap0.y) {
      break;
    }

    z = safeNextMandelbrotDs(z, c);
    let oldIter = iter;
    iter = iter + 1;
    if (oldIter == i32(spec.dims.z)) {
      return 2.0;
    }

    zq = dsNormSq(z);
  }

  var iterValue = f32(iter + 4);
  if (spec.flags.x == 1u && iterValue > 3.0) {
    let logZn = log(zq) / 2.0;
    let nu = log(logZn / log(2.0)) / log(2.0);
    iterValue = floor(iterValue + 1.0 - nu);
  }
  return iterValue;
}

fn computeTrapValue(zInit: vec2<f32>, c: vec2<f32>) -> f32 {
  let shapeId = spec.extra.x;
  let trapSize = spec.trap1.x;
  var trapSizeOrOne = 1.0;
  if (trapSize != 0.0) {
    trapSizeOrOne = trapSize;
  }
  var cosA = 1.0;
  var sinA = 0.0;
  if (shapeId >= 3u && shapeId <= 7u) {
    cosA = cos(spec.trap1.y);
    sinA = sin(spec.trap1.y);
  }
  var bitmapTrapWidth = abs(spec.trap1.x);
  var bitmapTrapHeight = bitmapTrapWidth;
  if (shapeId == 7u && spec.bitmap.x > 0u && spec.bitmap.y > 0u && bitmapTrapWidth > 0.0) {
    let bitmapAspect = f32(spec.bitmap.x) / f32(spec.bitmap.y);
    if (bitmapAspect >= 1.0) {
      bitmapTrapHeight = bitmapTrapWidth / bitmapAspect;
    } else {
      bitmapTrapWidth = bitmapTrapWidth * bitmapAspect;
    }
  }
  var bitmapInvTrapWidth = 0.0;
  var bitmapInvTrapHeight = 0.0;
  if (bitmapTrapWidth > 0.0) {
    bitmapInvTrapWidth = 1.0 / bitmapTrapWidth;
  }
  if (bitmapTrapHeight > 0.0) {
    bitmapInvTrapHeight = 1.0 / bitmapTrapHeight;
  }

  var z = zInit;
  var dClosest = 1e20;
  var dFarthest = 0.0;
  var dFarthestBitmap = -1.0;
  var dSum = 0.0;
  var dCount = 0u;
  var captured = false;
  var closestU = 0.0;
  var closestV = 0.0;
  var closestHasUv = false;
  var farthestU = 0.0;
  var farthestV = 0.0;
  var farthestHasUv = false;
  var avgUSum = 0.0;
  var avgVSum = 0.0;
  var avgCount = 0u;
  var firstU = 0.0;
  var firstV = 0.0;
  var firstHasUv = false;
  var stepU = 0.0;
  var stepV = 0.0;
  var stepHasUv = false;
  var capturedResult = 0.0;

  for (var i = 0u; i < spec.dims.z; i = i + 1u) {
    z = safeNext(z, c, f32(i));

    let dx = z.x - spec.trap0.z;
    let dy = z.y - spec.trap0.w;
    var d = 0.0;
    var u = 0.0;
    var v = 0.0;
    var inBounds = false;
    var bitmapVisible = false;

    if (shapeId == 0u) {
      d = min(abs(dx), abs(dy)) / trapSizeOrOne;
    } else if (shapeId == 1u) {
      d = abs(sqrt(dx * dx + dy * dy) - spec.trap1.x);
    } else if (shapeId == 2u) {
      d = sqrt(dx * dx + dy * dy);
    } else if (shapeId == 3u) {
      let px = dx * cosA + dy * sinA;
      let py = -dx * sinA + dy * cosA;
      let clamped = select(0.0, clamp(px, -trapSize, trapSize), trapSize != 0.0);
      let dx2 = px - clamped;
      d = sqrt(dx2 * dx2 + py * py);
    } else if (shapeId == 4u) {
      let lx = dx * cosA + dy * sinA;
      let ly = -dx * sinA + dy * cosA;
      d = abs(ly - (lx * lx) / trapSizeOrOne);
    } else if (shapeId == 5u) {
      if (trapSize == 0.0) {
        d = sqrt(dx * dx + dy * dy);
      } else {
        var px = (dx * cosA + dy * sinA) / trapSize;
        var py = (-dx * sinA + dy * cosA) / trapSize;
        let k = sqrt(3.0);
        px = abs(px) - 1.0;
        py = py + 1.0 / k;
        if (px + k * py > 0.0) {
          let tmpX = 0.5 * (px - k * py);
          py = 0.5 * (k * px - py);
          px = tmpX;
        }
        px = px - clamp(px, -2.0, 0.0);
        d = abs(sqrt(px * px + py * py) * sign(py) * trapSize);
      }
    } else if (shapeId == 6u) {
      let qx = abs(dx * cosA + dy * sinA) - trapSize;
      let qy = abs(-dx * sinA + dy * cosA) - trapSize;
      d = sqrt(max(qx, 0.0) * max(qx, 0.0) + max(qy, 0.0) * max(qy, 0.0)) + min(max(qx, qy), 0.0);
      d = abs(d);
    } else if (shapeId == 7u && spec.bitmap.x > 0u && spec.bitmap.y > 0u && bitmapTrapWidth > 0.0 && bitmapTrapHeight > 0.0) {
      let bitmapX = dx * cosA + dy * sinA;
      let bitmapY = -dx * sinA + dy * cosA;
      u = bitmapX * bitmapInvTrapWidth + 0.5;
      v = -(bitmapY * bitmapInvTrapHeight) + 0.5;

      if (u < 0.0 || u > 1.0 || v < 0.0 || v > 1.0) {
        d = 1.0;
      } else {
        inBounds = true;
        let pixel = sampleBitmap(u, v);
        let alphaU = channelA(pixel);
        bitmapVisible = alphaU >= 10u;
        let alpha = f32(alphaU) / 255.0;
        d = 1.0 - alpha;
      }
    }

    if (spec.dims.w == 0u) {
      if (d < dClosest) {
        dClosest = d;
        closestHasUv = inBounds;
        closestU = u;
        closestV = v;
      }
    } else if (spec.dims.w == 1u) {
      if (d < spec.trap1.z && d >= dFarthest) {
        dFarthest = d;
      }
      if (shapeId == 7u && bitmapVisible && d < spec.trap1.z && d >= dFarthestBitmap) {
        dFarthestBitmap = d;
        farthestHasUv = true;
        farthestU = u;
        farthestV = v;
      }
    } else if (spec.dims.w == 2u) {
      if (d < spec.trap1.z) {
        dSum = dSum + d;
        dCount = dCount + 1u;
        if (inBounds) {
          avgUSum = avgUSum + u;
          avgVSum = avgVSum + v;
          avgCount = avgCount + 1u;
        }
      }
    } else if (spec.dims.w == 3u) {
      if (!captured && f32(i) >= spec.trap1.w && d <= spec.trap1.z) {
        captured = true;
        let strength = 1.0 - d / spec.trap1.z;
        capturedResult = f32(i + 1u) + min(strength, 0.9999);
        firstHasUv = inBounds;
        firstU = u;
        firstV = v;
      }
    } else if (spec.dims.w == 4u) {
      if (i + 1u == spec.flags.w) {
        capturedResult = d;
        stepHasUv = inBounds;
        stepU = u;
        stepV = v;
      }
    }

    if (dot(z, z) > spec.trap0.x) {
      break;
    }
  }

  if (spec.dims.w == 0u) {
    if (closestHasUv) { return encodeUV(closestU, closestV); }
    return select(dClosest, 0.0, dClosest >= 1e19);
  }
  if (spec.dims.w == 1u) {
    if (farthestHasUv) { return encodeUV(farthestU, farthestV); }
    return dFarthest;
  }
  if (spec.dims.w == 2u) {
    if (avgCount > 0u) { return encodeUV(avgUSum / f32(avgCount), avgVSum / f32(avgCount)); }
    if (dCount > 0u) { return dSum / f32(dCount); }
    return 0.0;
  }
  if (spec.dims.w == 3u) {
    if (firstHasUv) { return encodeUV(firstU, firstV); }
    return capturedResult;
  }
  if (spec.dims.w == 4u) {
    if (stepHasUv) { return encodeUV(stepU, stepV); }
    return capturedResult;
  }
  return 0.0;
}

fn computeTrapValueMandelbrotDs(zInit: Ds2, c: Ds2) -> f32 {
  let shapeId = spec.extra.x;
  let trapSize = spec.trap1.x;
  var trapSizeOrOne = 1.0;
  if (trapSize != 0.0) {
    trapSizeOrOne = trapSize;
  }
  var cosA = 1.0;
  var sinA = 0.0;
  if (shapeId >= 3u && shapeId <= 7u) {
    cosA = cos(spec.trap1.y);
    sinA = sin(spec.trap1.y);
  }
  var bitmapTrapWidth = abs(spec.trap1.x);
  var bitmapTrapHeight = bitmapTrapWidth;
  if (shapeId == 7u && spec.bitmap.x > 0u && spec.bitmap.y > 0u && bitmapTrapWidth > 0.0) {
    let bitmapAspect = f32(spec.bitmap.x) / f32(spec.bitmap.y);
    if (bitmapAspect >= 1.0) {
      bitmapTrapHeight = bitmapTrapWidth / bitmapAspect;
    } else {
      bitmapTrapWidth = bitmapTrapWidth * bitmapAspect;
    }
  }
  var bitmapInvTrapWidth = 0.0;
  var bitmapInvTrapHeight = 0.0;
  if (bitmapTrapWidth > 0.0) {
    bitmapInvTrapWidth = 1.0 / bitmapTrapWidth;
  }
  if (bitmapTrapHeight > 0.0) {
    bitmapInvTrapHeight = 1.0 / bitmapTrapHeight;
  }

  var z = zInit;
  var dClosest = 1e20;
  var dFarthest = 0.0;
  var dFarthestBitmap = -1.0;
  var dSum = 0.0;
  var dCount = 0u;
  var captured = false;
  var closestU = 0.0;
  var closestV = 0.0;
  var closestHasUv = false;
  var farthestU = 0.0;
  var farthestV = 0.0;
  var farthestHasUv = false;
  var avgUSum = 0.0;
  var avgVSum = 0.0;
  var avgCount = 0u;
  var firstU = 0.0;
  var firstV = 0.0;
  var firstHasUv = false;
  var stepU = 0.0;
  var stepV = 0.0;
  var stepHasUv = false;
  var capturedResult = 0.0;

  for (var i = 0u; i < spec.dims.z; i = i + 1u) {
    z = safeNextMandelbrotDs(z, c);
    let zVec = ds2ToVec2(z);

    let dx = zVec.x - spec.trap0.z;
    let dy = zVec.y - spec.trap0.w;
    var d = 0.0;
    var u = 0.0;
    var v = 0.0;
    var inBounds = false;
    var bitmapVisible = false;

    if (shapeId == 0u) {
      d = min(abs(dx), abs(dy)) / trapSizeOrOne;
    } else if (shapeId == 1u) {
      d = abs(sqrt(dx * dx + dy * dy) - spec.trap1.x);
    } else if (shapeId == 2u) {
      d = sqrt(dx * dx + dy * dy);
    } else if (shapeId == 3u) {
      let px = dx * cosA + dy * sinA;
      let py = -dx * sinA + dy * cosA;
      let clamped = select(0.0, clamp(px, -trapSize, trapSize), trapSize != 0.0);
      let dx2 = px - clamped;
      d = sqrt(dx2 * dx2 + py * py);
    } else if (shapeId == 4u) {
      let lx = dx * cosA + dy * sinA;
      let ly = -dx * sinA + dy * cosA;
      d = abs(ly - (lx * lx) / trapSizeOrOne);
    } else if (shapeId == 5u) {
      if (trapSize == 0.0) {
        d = sqrt(dx * dx + dy * dy);
      } else {
        var px = (dx * cosA + dy * sinA) / trapSize;
        var py = (-dx * sinA + dy * cosA) / trapSize;
        let k = sqrt(3.0);
        px = abs(px) - 1.0;
        py = py + 1.0 / k;
        if (px + k * py > 0.0) {
          let tmpX = 0.5 * (px - k * py);
          py = 0.5 * (k * px - py);
          px = tmpX;
        }
        px = px - clamp(px, -2.0, 0.0);
        d = abs(sqrt(px * px + py * py) * sign(py) * trapSize);
      }
    } else if (shapeId == 6u) {
      let qx = abs(dx * cosA + dy * sinA) - trapSize;
      let qy = abs(-dx * sinA + dy * cosA) - trapSize;
      d = sqrt(max(qx, 0.0) * max(qx, 0.0) + max(qy, 0.0) * max(qy, 0.0)) + min(max(qx, qy), 0.0);
      d = abs(d);
    } else if (shapeId == 7u && spec.bitmap.x > 0u && spec.bitmap.y > 0u && bitmapTrapWidth > 0.0 && bitmapTrapHeight > 0.0) {
      let bitmapX = dx * cosA + dy * sinA;
      let bitmapY = -dx * sinA + dy * cosA;
      u = bitmapX * bitmapInvTrapWidth + 0.5;
      v = -(bitmapY * bitmapInvTrapHeight) + 0.5;

      if (u < 0.0 || u > 1.0 || v < 0.0 || v > 1.0) {
        d = 1.0;
      } else {
        inBounds = true;
        let pixel = sampleBitmap(u, v);
        let alphaU = channelA(pixel);
        bitmapVisible = alphaU >= 10u;
        let alpha = f32(alphaU) / 255.0;
        d = 1.0 - alpha;
      }
    }

    if (spec.dims.w == 0u) {
      if (d < dClosest) {
        dClosest = d;
        closestHasUv = inBounds;
        closestU = u;
        closestV = v;
      }
    } else if (spec.dims.w == 1u) {
      if (d < spec.trap1.z && d >= dFarthest) {
        dFarthest = d;
      }
      if (shapeId == 7u && bitmapVisible && d < spec.trap1.z && d >= dFarthestBitmap) {
        dFarthestBitmap = d;
        farthestHasUv = true;
        farthestU = u;
        farthestV = v;
      }
    } else if (spec.dims.w == 2u) {
      if (d < spec.trap1.z) {
        dSum = dSum + d;
        dCount = dCount + 1u;
        if (inBounds) {
          avgUSum = avgUSum + u;
          avgVSum = avgVSum + v;
          avgCount = avgCount + 1u;
        }
      }
    } else if (spec.dims.w == 3u) {
      if (!captured && f32(i) >= spec.trap1.w && d <= spec.trap1.z) {
        captured = true;
        let strength = 1.0 - d / spec.trap1.z;
        capturedResult = f32(i + 1u) + min(strength, 0.9999);
        firstHasUv = inBounds;
        firstU = u;
        firstV = v;
      }
    } else if (spec.dims.w == 4u) {
      if (i + 1u == spec.flags.w) {
        capturedResult = d;
        stepHasUv = inBounds;
        stepU = u;
        stepV = v;
      }
    }

    if (dot(zVec, zVec) > spec.trap0.x) {
      break;
    }
  }

  if (spec.dims.w == 0u) {
    if (closestHasUv) { return encodeUV(closestU, closestV); }
    return select(dClosest, 0.0, dClosest >= 1e19);
  }
  if (spec.dims.w == 1u) {
    if (farthestHasUv) { return encodeUV(farthestU, farthestV); }
    return dFarthest;
  }
  if (spec.dims.w == 2u) {
    if (avgCount > 0u) { return encodeUV(avgUSum / f32(avgCount), avgVSum / f32(avgCount)); }
    if (dCount > 0u) { return dSum / f32(dCount); }
    return 0.0;
  }
  if (spec.dims.w == 3u) {
    if (firstHasUv) { return encodeUV(firstU, firstV); }
    return capturedResult;
  }
  if (spec.dims.w == 4u) {
    if (stepHasUv) { return encodeUV(stepU, stepV); }
    return capturedResult;
  }
  return 0.0;
}

fn computeTiaValue(zInit: vec2<f32>, c: vec2<f32>) -> f32 {
  let bailout = 1e20;
  let log2Value = log(2.0);
  let lp2 = log(log(bailout));
  let cDist = sqrt(dot(c, c));

  var z = zInit;
  var dSum = 0.0;
  var dSumPrev = 0.0;
  var iterCount = 0u;
  var finalZq = 0.0;
  var escaped = false;

  for (var i = 0u; i < spec.dims.z; i = i + 1u) {
    z = safeNext(z, c, f32(i));
    var zq = dot(z, z);
    if (!(zq == zq) || zq > 3e38) {
      zq = 3e38;
    }

    if (zq > bailout) {
      finalZq = zq;
      escaped = true;
      break;
    }

    if (i > 0u) {
      let delta = z - c;
      let zDist = sqrt(dot(delta, delta));
      let radius = sqrt(zq);
      let lowbound = abs(zDist - cDist);
      let denom = 2.0 * min(zDist, cDist);
      dSumPrev = dSum;
      if (denom > 1e-10) {
        dSum = dSum + clamp((radius - lowbound) / denom, 0.0, 1.0);
      }
      iterCount = iterCount + 1u;
    }
  }

  if (escaped && iterCount > 1u && finalZq > 1.0) {
    let average = dSum / f32(iterCount);
    let average2 = dSumPrev / f32(iterCount - 1u);
    let f = (lp2 - log(log(finalZq))) / log2Value;
    let corrected = average + (average - average2) * f;
    if ((corrected == corrected) && abs(corrected) < ${MAX_VALID_VALUE_WGSL}) {
      return clamp(corrected, 0.0, 1.0);
    }
    return clamp(average, 0.0, 1.0);
  }

  if (iterCount > 0u) {
    return clamp(dSum / f32(iterCount), 0.0, 1.0);
  }
  return 0.0;
}

fn computeTiaValueMandelbrotDs(zInit: Ds2, c: Ds2) -> f32 {
  let bailout = 1e20;
  let log2Value = log(2.0);
  let lp2 = log(log(bailout));
  let cVec = ds2ToVec2(c);
  let cDist = sqrt(dot(cVec, cVec));

  var z = zInit;
  var dSum = 0.0;
  var dSumPrev = 0.0;
  var iterCount = 0u;
  var finalZq = 0.0;
  var escaped = false;

  for (var i = 0u; i < spec.dims.z; i = i + 1u) {
    z = safeNextMandelbrotDs(z, c);
    let zVec = ds2ToVec2(z);
    var zq = dot(zVec, zVec);
    if (!(zq == zq) || zq > 3e38) {
      zq = 3e38;
    }

    if (zq > bailout) {
      finalZq = zq;
      escaped = true;
      break;
    }

    if (i > 0u) {
      let delta = zVec - cVec;
      let zDist = sqrt(dot(delta, delta));
      let radius = sqrt(zq);
      let lowbound = abs(zDist - cDist);
      let denom = 2.0 * min(zDist, cDist);
      dSumPrev = dSum;
      if (denom > 1e-10) {
        dSum = dSum + clamp((radius - lowbound) / denom, 0.0, 1.0);
      }
      iterCount = iterCount + 1u;
    }
  }

  if (escaped && iterCount > 1u && finalZq > 1.0) {
    let average = dSum / f32(iterCount);
    let average2 = dSumPrev / f32(iterCount - 1u);
    let f = (lp2 - log(log(finalZq))) / log2Value;
    let corrected = average + (average - average2) * f;
    if ((corrected == corrected) && abs(corrected) < ${MAX_VALID_VALUE_WGSL}) {
      return clamp(corrected, 0.0, 1.0);
    }
    return clamp(average, 0.0, 1.0);
  }

  if (iterCount > 0u) {
    return clamp(dSum / f32(iterCount), 0.0, 1.0);
  }
  return 0.0;
}

fn colorFromTrap(iterValue: f32, trapValue: f32) -> u32 {
  if (iterValue <= 3.0) {
    return spec.bitmap.w;
  }

  if (spec.extra.x == 7u && spec.bitmap.x > 0u && spec.bitmap.y > 0u) {
    if (trapValue >= -0.5) {
      return spec.bitmap.z;
    }

    let packed = -trapValue - 1.0;
    let uInt = floor(packed / 4096.0);
    let vInt = round(packed - uInt * 4096.0);
    let u = uInt / 4095.0;
    let v = vInt / 4095.0;
    let bitmapColor = sampleBitmap(u, v);
    return blendBitmapPixelOverBackground(bitmapColor, spec.bitmap.z);
  }

  if (spec.extra.y == 1u) {
    if (abs(trapValue) < 0.000001) {
      return spec.bitmap.w;
    }
    let capturedIter = floor(trapValue);
    let strength = trapValue - capturedIter;
    let hue = fract(capturedIter / 6.0);
    return hslToRgb(hue, 1.0, strength * 0.75);
  }

  if (spec.extra.y == 2u) {
    let scaledAlpha = trapValue * 4.0;
    let frac = scaledAlpha - floor(scaledAlpha);
    let lum = (0.5 - abs(frac - 0.5)) * 2.0;
    return hslToRgb(1.0 / 3.0, 0.6, lum);
  }

  let alpha = pow(max(trapValue, 0.0), 0.5) * 3.0 * 2.0 * 3.141592653589793;
  let r = (cos(alpha - 3.141592653589793) + 1.0) * 0.5 * 255.0;
  let g = (cos(alpha - 0.75 * 3.141592653589793) + 1.0) * 0.5 * 255.0;
  let b = (cos(alpha - 0.5 * 3.141592653589793) + 1.0) * 0.5 * 255.0;
  return packRgba(toByte(r), toByte(g), toByte(b), 255u);
}

@compute @workgroup_size(${WORKGROUP_SIZE_X}, ${WORKGROUP_SIZE_Y}, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x >= spec.dims.x || gid.y >= spec.dims.y) {
    return;
  }

  let idx = gid.y * spec.dims.x + gid.x;
  let ssScale = max(spec.flags.y, 1u);
  let sampleCount = f32(ssScale * ssScale);
  var totalIter = 0.0;
  var totalTrap = 0.0;

  for (var sy = 0u; sy < ssScale; sy = sy + 1u) {
    for (var sx = 0u; sx < ssScale; sx = sx + 1u) {
      var ox = 0.0;
      var oy = 0.0;
      if (ssScale > 1u) {
        ox = (f32(sx) + 0.5) / f32(ssScale);
        oy = (f32(sy) + 0.5) / f32(ssScale);
      }

      let pixelDs = pixelToComplexDs(f32(gid.x) + ox, f32(gid.y) + oy);
      let fixedDs = z0FromSpecDs();
      var cDs = pixelDs;
      var z0Ds = fixedDs;
      if (spec.extra.z != 0u) {
        cDs = fixedDs;
        z0Ds = pixelDs;
      }
      let c = ds2ToVec2(cDs);
      let z0 = ds2ToVec2(z0Ds);

      if (spec.flags.z == 0u) {
        totalIter = totalIter + computeIterValueMandelbrotDs(z0Ds, cDs);
        if (spec.dims.w == 5u) {
          totalTrap = totalTrap + computeTiaValueMandelbrotDs(z0Ds, cDs);
        } else {
          totalTrap = totalTrap + computeTrapValueMandelbrotDs(z0Ds, cDs);
        }
      } else {
        totalIter = totalIter + computeIterValue(z0, c);
        if (spec.dims.w == 5u) {
          totalTrap = totalTrap + computeTiaValue(z0, c);
        } else {
          totalTrap = totalTrap + computeTrapValue(z0, c);
        }
      }
    }
  }

  let avgIter = totalIter / sampleCount;
  let iterValue = select(floor(avgIter + 0.5), floor(avgIter), spec.flags.z == 1u);
  let trapValue = totalTrap / sampleCount;
  outPixels[idx] = colorFromTrap(iterValue, trapValue);
}
`
  }
}

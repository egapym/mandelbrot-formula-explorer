/**
 * Based on bertbaron/mandelbrot by Bert Baron
 * This file is part of the Mandelbrot Explorer project.
 * Licensed under GPL-3.0.
 */

import { getParsedExpression } from './customFunctionParser.mjs'
import { BAILOUT_MIN, BAILOUT_SMOOTH } from './sharedCalculations.mjs'
import { CUSTOM_FUNCTION_WGSL_HELPERS, jsExprToWGSL_safe } from './wgslCompiler.mjs'
import { WorkerContext } from './workerContext.mjs'

// ============================================================================
// 定数
// ============================================================================

const USE_GPU = true

// GPU の workgroup 設定
const WORKGROUP_CONFIG = {
  SIZE_X: 16, // workgroup の幅
  SIZE_Y: 8, // workgroup の高さ
}

// シェーダー用定数
const SHADER_CONSTANTS = {
  SPEC_SIZE: 16 * 4, // Uniforms 構造体サイズ（byte）
  DEFAULT_BAILOUT: BAILOUT_SMOOTH, // スムーズカラー用の脱出半径
  MIN_BAILOUT: BAILOUT_MIN, // 通常カラー用の最小 bailout
  MAX_VALID_VALUE: 1e20, // 座標の上限値（NaN / Inf 判定用）
  MAX_VALID_VALUE_WGSL: '1e20', // WGSL 向け表記
  IN_SET_INDEX: 2, // 集合内を表す値
  ESCAPE_OFFSET: 4, // 反復回数に加えるオフセット
  SMOOTH_SCALE: 255.0, // smooth 値のスケール
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
   * 文脈付きのエラーを出力する
   */
  logError(context, error) {
    console.error(`[${context}]`, this.format(error))
  },

  /**
   * 文脈付きの警告を出力する
   */
  logWarn(context, error) {
    console.warn(`[${context}]`, this.format(error))
  },

  /**
   * 親オブジェクトへエラーを通知する
   */
  notifyParent(parent, error) {
    if (parent && typeof parent.handleWorkerError === 'function') {
      parent.handleWorkerError(this.format(error))
    }
  },
}

// ============================================================================
// バッファ管理ユーティリティ
// ============================================================================

const BufferHelpers = {
  /**
   * 標準設定で GPU バッファを作る
   */
  createBuffer(device, size, usage) {
    return device.createBuffer({ size, usage })
  },

  /**
   * 読み書き用の storage buffer を作る
   */
  createStorageBuffer(device, size) {
    return this.createBuffer(device, size, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC)
  },

  /**
   * 結果読み出し用の buffer を作る
   */
  createReadBuffer(device, size) {
    return this.createBuffer(device, size, GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ)
  },

  /**
   * buffer を安全に破棄する
   */
  destroyBuffer(buffer) {
    if (buffer && typeof buffer.destroy === 'function') {
      buffer.destroy()
    }
  },
}

const USE_GPU_LEGACY = USE_GPU // 後方互換用

/**
 * WebGPU を使うカスタムフラクタルレンダラー
 *
 * ユーザー定義の反復関数を WGSL に変換し、GPU 上で実行します。
 */
export class MandelbrotCustomWebGPU {
  /**
   * @param {Object} p - 親の mandelbrot インスタンス
   * @param {WorkerContext} ctx - worker の実行コンテキスト
   * @param {Function} errorCallback - エラー通知関数
   */
  constructor(p, ctx, errorCallback) {
    this.p = p
    this.ctx = ctx
    this.errorCallback = errorCallback
    this.iterationFunction = 'z*z + c' // 既定値
    // 実際に使える GPU / device があるかどうかを保持する
    this.available = true
    this.devicePromise = this.initGpu()
    this.pipeline = this.createPipeline()
    this.running = Promise.resolve()
    this.currentTask = null
    this.newTask = null
  }

  async initGpu() {
    const adapter = await navigator.gpu?.requestAdapter({
      powerPreference: 'high-performance',
    })
    const device = await adapter?.requestDevice()
    if (!device) {
      // CPU へ安全にフォールバックできるよう利用不可にする
      this.available = false
      this.errorCallback('need a browser that supports WebGPU')
      return null
    }

    // GPU が利用可能
    this.available = true

    try {
      const info = await device?.adapterInfo
      console.log(`Custom Fractal GPU Adapter: ${info.vendor}:${info.architecture}:${info.device} ${info.description}`)
    } catch (error) {
      ErrorHelpers.logWarn('GPU Adapter Info', error)
    }

    device.lost.then(() => {
      console.log('GPU lost, reloading')
      this.devicePromise = this.initGpu()
      this.available = false
      this.pipeline = this.createPipeline()
    })
    return device
  }

  /**
   * 新しい描画パイプラインを作る
   * @returns {CustomFractalPipeline|null}
   */
  createPipeline() {
    return USE_GPU_LEGACY ? new CustomFractalPipeline(this, this.devicePromise) : null
  }

  shouldStop() {
    return this.currentTask !== this.newTask
  }

  async process(task) {
    this.newTask = task.jobToken
    await this.running
    if (task.jobToken !== this.newTask) {
      return
    }
    this.currentTask = task.jobToken

    this.iterationFunction = task.iterationFunction || 'z*z + c'
    this.max_iter = task.maxIter
    const w = task.w
    const h = task.h

    // スーパーサンプリング設定が変わったらパイプラインを作り直す
    if (this.lastSupersampling !== task.supersampling) {
      this.lastSupersampling = task.supersampling
      this.pipeline = this.createPipeline()
    }

    this.running = this.calculate(w, h, task)
    await this.running
  }

  async calculate(w, h, task) {
    // フレーム境界は task 側ですでに fxp 形式になっている
    const rmin = task.frameTopLeft[0]
    const rmax = task.frameBottomRight[0]
    const imin = task.frameTopLeft[1]
    const imax = task.frameBottomRight[1]

    // GPU へ渡すため、ピクセル差分を number に変換する
    const cWidth = rmax.subtract(rmin).toNumber()
    const cHeight = imax.subtract(imin).toNumber()

    const ddr = cWidth / task.frameWidth
    const ddi = cHeight / task.frameHeight
    const ddr0 = task.xOffset * ddr
    const ddi0 = task.yOffset * ddi

    // 左上座標は f64 で保持する
    const refr = rmin.toNumber()
    const refi = imin.toNumber()

    const bailout = task.smooth
      ? SHADER_CONSTANTS.DEFAULT_BAILOUT
      : task.escapeRadius !== undefined
        ? task.escapeRadius ** 2
        : 16

    const result = await this.renderDirect({
      w,
      h,
      max_iter: task.maxIter,
      refr,
      refi,
      ddr0,
      ddi0,
      ddr,
      ddi,
      doSmooth: task.smooth,
      bailout,
      supersampling: task.supersampling,
      iterationFunction: this.iterationFunction,
      z0: task.z0 || [0, 0],
      isJulia: task.fractalType === 'julia' || task.fractalType === 'julia-custom',
      juliaC: [task.juliaRe ?? 0.0, task.juliaIm ?? 0.0],
      escapeRadius: task.escapeRadius !== undefined ? task.escapeRadius : 4.0,
    })

    const values = result.values
    const smooth = result.smooth
    const signs = result.signs
    const zreal = result.zreal || null
    const zimag = result.zimag || null

    // パイプラインや描画でエラーが出たら、前段へ渡して UI に反映させる
    if (result?.error) {
      try {
        this.p.onGpuUpdate({
          jobToken: task.jobToken,
          values,
          smooth,
          signs,
          zreal,
          zimag,
          renderedPixels: values ? values.length : 0,
          isFinished: true,
          error: result.error,
        })
      } catch (e) {
        console.warn('Error forwarding GPU pipeline error to frontend:', e)
      }
      return { values, smooth, signs, zreal, zimag, error: result.error }
    }

    if (this.shouldStop()) {
      this.p.onGpuUpdate({
        jobToken: task.jobToken,
        values,
        smooth,
        signs,
        zreal,
        zimag,
        renderedPixels: values.length,
        isFinished: true,
      })
      return {
        values,
        smooth,
        signs,
        zreal,
        zimag,
        error: 'Stopped',
      }
    }

    this.p.onGpuUpdate({
      jobToken: task.jobToken,
      values,
      smooth,
      signs,
      zreal,
      zimag,
      renderedPixels: values.length,
      isFinished: true,
    })
    return { values, smooth, signs, zreal, zimag }
  }

  async renderDirect(params) {
    const device = await this.devicePromise
    const pipeline = await this.pipeline.getPipeline(
      device,
      params.doSmooth,
      params.bailout,
      params.supersampling,
      params.iterationFunction,
    )

    // パイプライン生成に失敗したらエラーを返す
    if (!pipeline) {
      const errorMsg = this.pipeline.lastError || 'Invalid iteration function - cannot compile to GPU shader'
      ErrorHelpers.logError('Pipeline Creation', errorMsg)
      return {
        values: new Int32Array(params.w * params.h).fill(SHADER_CONSTANTS.IN_SET_INDEX),
        smooth: params.doSmooth ? new Uint8ClampedArray(params.w * params.h) : null,
        error: errorMsg,
      }
    }

    const result = await this.pipeline.run(params)
    await this.pipeline.finish()
    return result
  }
}

/**
 * カスタムフラクタル描画用の GPU compute パイプライン
 */
class CustomFractalPipeline {
  constructor(ctx, devicePromise) {
    this.ctx = ctx
    this.devicePromise = devicePromise
    this.pipeline = null
    this.pipelineKey = null
    // GPU の占有率を上げるため 2D workgroup を使う
    // (16, 8) は画像処理向けの無難なバランス設定
    this.workgroupSizeX = WORKGROUP_CONFIG.SIZE_X
    this.workgroupSizeY = WORKGROUP_CONFIG.SIZE_Y
  }

  async getPipeline(device, doSmooth, bailout, supersampling, iterationFunction) {
    const key = `${doSmooth}:${bailout}:${supersampling}:${iterationFunction}`
    if (this.pipelineKey === key && this.pipeline) {
      return this.pipeline
    }

    this.pipelineKey = key

    // カスタム反復関数を WGSL へ変換する
    let wgslIterationExpr
    try {
      const jsExpr = getParsedExpression(iterationFunction)
      wgslIterationExpr = jsExprToWGSL_safe(jsExpr)

      // WGSL で扱えない二重アンダースコア識別子を置き換える
      if (/\b__\w+/.test(wgslIterationExpr)) {
        console.warn('Replacing double-underscore identifiers in WGSL expression')
        wgslIterationExpr = wgslIterationExpr.replace(/\b__([A-Za-z0-9_]+)/g, '_$1')
      }

      // "+ -" や "- -" のような崩れた演算子列を整える
      wgslIterationExpr = wgslIterationExpr.replace(/\+\s*-/g, '-')
      wgslIterationExpr = wgslIterationExpr.replace(/-\s*-/g, '+')
    } catch (e) {
      ErrorHelpers.logError('WGSL Compilation', e)
      if (e.stack) console.error('Stack:', e.stack)
      // 失敗時は標準 Mandelbrot の式に戻す
      wgslIterationExpr = 'vec2<f32>(z.x * z.x - z.y * z.y + c.x, 2.0 * z.x * z.y + c.y)'
      ErrorHelpers.logWarn('WGSL Fallback', 'Using standard Mandelbrot iteration')
      // コンパイル失敗を親へ通知する
      try {
        const parent = this.ctx?.p ? this.ctx.p : null
        ErrorHelpers.notifyParent(parent, e)
      } catch (notifyErr) {
        ErrorHelpers.logWarn('Error Notification Failed', notifyErr)
      }
    }

    const shaderCode = this.generateShader(wgslIterationExpr, doSmooth, bailout, supersampling)

    try {
      const shaderModule = device.createShaderModule({
        label: 'custom fractal shader',
        code: shaderCode,
      })

      // シェーダーコンパイル情報を記録する
      if (shaderModule.getCompilationInfo) {
        shaderModule.getCompilationInfo().then((info) => {
          if (info.messages.length > 0) {
            ErrorHelpers.logWarn('Shader Compilation', info.messages)
          }
        })
      }

      const bindGroupLayoutEntries = [
        {
          binding: 0,
          visibility: GPUShaderStage.COMPUTE,
          buffer: { type: 'uniform' },
        },
        {
          binding: 1,
          visibility: GPUShaderStage.COMPUTE,
          buffer: { type: 'storage' },
        },
      ]

      // Add smooth buffer binding only if smooth is enabled
      if (doSmooth) {
        bindGroupLayoutEntries.push({
          binding: 2,
          visibility: GPUShaderStage.COMPUTE,
          buffer: { type: 'storage' },
        })
      }

      // Signs buffer (always present for grid palette support)
      bindGroupLayoutEntries.push({
        binding: 3,
        visibility: GPUShaderStage.COMPUTE,
        buffer: { type: 'storage' },
      })
      // zreal buffer
      bindGroupLayoutEntries.push({
        binding: 4,
        visibility: GPUShaderStage.COMPUTE,
        buffer: { type: 'storage' },
      })
      // zimag buffer
      bindGroupLayoutEntries.push({
        binding: 5,
        visibility: GPUShaderStage.COMPUTE,
        buffer: { type: 'storage' },
      })

      const bindGroupLayout = device.createBindGroupLayout({
        entries: bindGroupLayoutEntries,
      })

      const pipelineLayout = device.createPipelineLayout({
        bindGroupLayouts: [bindGroupLayout],
      })

      this.pipeline = await device.createComputePipelineAsync({
        label: 'custom fractal pipeline',
        layout: pipelineLayout,
        compute: {
          module: shaderModule,
          entryPoint: 'main',
        },
      })

      this.bindGroupLayout = bindGroupLayout
      return this.pipeline
    } catch (e) {
      ErrorHelpers.logError('Pipeline Creation Failed', e)
      console.error('Iteration function:', iterationFunction)
      console.error('WGSL expression:', wgslIterationExpr)
      console.error('Full shader code:')
      console.error(shaderCode)

      // Reset pipeline key to allow retry with different function
      this.pipelineKey = null
      this.pipeline = null

      // Store error message for user feedback
      this.lastError = ErrorHelpers.format(e)

      // Notify parent about this error
      try {
        const parent = this.ctx?.p ? this.ctx.p : null
        ErrorHelpers.notifyParent(parent, this.lastError)
      } catch (notifyErr) {
        ErrorHelpers.logWarn('Error Notification Failed', notifyErr)
      }
      // Return error to caller instead of throwing
      return null
    }
  }

  generateShader(iterationExpr, doSmooth, _bailout, supersampling) {
    const ssScale = supersampling > 0 ? supersampling : 1
    const ssSamples = ssScale * ssScale

    return `
struct Spec {
  width: u32,
  height: u32,
  max_iter: u32,
  bailout: f32,
  refr: f32,
  refi: f32,
  ddr0: f32,
  ddi0: f32,
  ddr: f32,
  ddi: f32,
  z0x: f32,
  z0y: f32,
  ssScale: u32,
  isJulia: u32,
  juliaCr: f32,
  juliaCi: f32,
};

@group(0) @binding(0) var<uniform> spec: Spec;
@group(0) @binding(1) var<storage, read_write> values: array<i32>;
${doSmooth ? '@group(0) @binding(2) var<storage, read_write> smoothValues: array<u32>;' : ''}
@group(0) @binding(3) var<storage, read_write> signsBuffer: array<u32>;
@group(0) @binding(4) var<storage, read_write> zrealBuffer: array<f32>;
@group(0) @binding(5) var<storage, read_write> zimagBuffer: array<f32>;

${CUSTOM_FUNCTION_WGSL_HELPERS}

struct IterResult {
  iterVal: u32,
  smoothVal: u32,
  signVal: u32,
  escZr: f32,
  escZi: f32,
}

// Returns IterResult(iterValue, smoothValue, signValue, escapeZr, escapeZi)
// signValue: 0=in-set, 1=same sign at escape, 2=different sign at escape
fn iterate(z_init: vec2<f32>, c: vec2<f32>) -> IterResult {
  var z = z_init;
  var iter: i32 = -1;
  var zq: f32 = z.x * z.x + z.y * z.y;

  while (zq <= spec.bailout) {
    iter = iter + 1;
    if (iter == i32(spec.max_iter)) {
      return IterResult(${SHADER_CONSTANTS.IN_SET_INDEX}u, 0u, 0u, 0.0, 0.0);
    }

    // Custom iteration expression
    let n = f32(iter);
    let z_next = ${iterationExpr};

    // Robust NaN/Inf validation (WGSL doesn't have isFinite, use self-equality for NaN check)
    let is_valid = (z_next.x == z_next.x) && (z_next.y == z_next.y) &&
                   abs(z_next.x) < ${SHADER_CONSTANTS.MAX_VALID_VALUE_WGSL} && abs(z_next.y) < ${SHADER_CONSTANTS.MAX_VALID_VALUE_WGSL};

    if (!is_valid) {
      // Treat as diverged
      break;
    }

    z = z_next;
    zq = z.x * z.x + z.y * z.y;
  }

  // Compute sign of z at escape point
  let sameSign = (z.x >= 0.0) == (z.y >= 0.0);
  let signVal = select(2u, 1u, sameSign);

  ${
    doSmooth
      ? `
  // Smooth coloring - use normalized iteration count
  var smoothVal: u32 = 0u;
  if (iter >= 0 && zq > 4.0) {
    let nu = log2(log2(zq)) - 1.0;
    let nuFloor = floor(nu);
    iter = i32(floor(f32(iter) + 1.0 - nu));
    smoothVal = u32(floor(${SHADER_CONSTANTS.SMOOTH_SCALE} - ${SHADER_CONSTANTS.SMOOTH_SCALE} * (nu - nuFloor)));
  }

  return IterResult(u32(iter + ${SHADER_CONSTANTS.ESCAPE_OFFSET}), smoothVal, signVal, z.x, z.y);
  `
      : `
  // No smooth coloring - return iteration count directly
  return IterResult(u32(iter + ${SHADER_CONSTANTS.ESCAPE_OFFSET}), 0u, signVal, z.x, z.y);
  `
  }
}

@compute @workgroup_size(${this.workgroupSizeX}, ${this.workgroupSizeY}, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.y * spec.width + gid.x;
  if (gid.x >= spec.width || gid.y >= spec.height) {
    return;
  }

  ${
    supersampling > 0
      ? `
  // Supersampling (matching Mandelbrot WebGPU float averaging)
  var total_iter: f32 = 0.0;
  var total_smooth: f32 = 0.0;
  var capturedSign: u32 = 0u;
  var capturedZr: f32 = 0.0;
  var capturedZi: f32 = 0.0;
  let ssScale = spec.ssScale;
  let ss_step = 1.0 / f32(ssScale);

  for (var sy: u32 = 0u; sy < ssScale; sy = sy + 1u) {
    for (var sx: u32 = 0u; sx < ssScale; sx = sx + 1u) {
      let offset_x = (f32(sx) + 0.5) * ss_step - 0.5;
      let offset_y = (f32(sy) + 0.5) * ss_step - 0.5;

      let cr = spec.refr + spec.ddr0 + (f32(gid.x) + offset_x) * spec.ddr;
      let ci = spec.refi + spec.ddi0 + (f32(gid.y) + offset_y) * spec.ddi;
	      let pixel = vec2<f32>(cr, ci);
	      let c = select(pixel, vec2<f32>(spec.juliaCr, spec.juliaCi), spec.isJulia != 0u);
	      let z0 = select(vec2<f32>(spec.z0x, spec.z0y), pixel, spec.isJulia != 0u);

      let result = iterate(z0, c);
      total_iter = total_iter + f32(result.iterVal);
      ${doSmooth ? 'total_smooth = total_smooth + f32(result.smoothVal);' : ''}
      // Capture sign and escape z from first sample (sy=0, sx=0)
      if (sy == 0u && sx == 0u) {
        capturedSign = result.signVal;
        capturedZr = result.escZr;
        capturedZi = result.escZi;
      }
    }
  }

  let avg_iter = total_iter / ${ssSamples}.0;
  values[idx] = i32(round(avg_iter));
  ${doSmooth ? `smoothValues[idx] = u32(round(total_smooth / ${ssSamples}.0));` : ''}
  signsBuffer[idx] = capturedSign;
  zrealBuffer[idx] = capturedZr;
  zimagBuffer[idx] = capturedZi;
  `
      : `
  // No supersampling
  let cr = spec.refr + spec.ddr0 + f32(gid.x) * spec.ddr;
  let ci = spec.refi + spec.ddi0 + f32(gid.y) * spec.ddi;
	  let pixel = vec2<f32>(cr, ci);
	  let c = select(pixel, vec2<f32>(spec.juliaCr, spec.juliaCi), spec.isJulia != 0u);
	  let z0 = select(vec2<f32>(spec.z0x, spec.z0y), pixel, spec.isJulia != 0u);

  let result = iterate(z0, c);
  values[idx] = i32(result.iterVal);
  ${doSmooth ? 'smoothValues[idx] = result.smoothVal;' : ''}
  signsBuffer[idx] = result.signVal;
  zrealBuffer[idx] = result.escZr;
  zimagBuffer[idx] = result.escZi;
  `
  }
}
`
  }

  /**
   * Execute the compute shader and retrieve results
   * @param {Object} params - Rendering parameters
   * @returns {Promise<{values: Int32Array, smooth: Uint8ClampedArray|null}>}
   */
  async run(params) {
    const device = await this.devicePromise
    const w = params.w
    const h = params.h

    // Create buffers using helpers
    const specBuffer = BufferHelpers.createBuffer(
      device,
      SHADER_CONSTANTS.SPEC_SIZE,
      GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    )

    const valuesBuffer = BufferHelpers.createStorageBuffer(device, w * h * 4)

    let smoothBuffer = null
    if (params.doSmooth) {
      smoothBuffer = BufferHelpers.createStorageBuffer(device, w * h * 4)
    }

    const signsBuffer = BufferHelpers.createStorageBuffer(device, w * h * 4)
    const zrealBuffer = BufferHelpers.createStorageBuffer(device, w * h * 4)
    const zimagBuffer = BufferHelpers.createStorageBuffer(device, w * h * 4)
    const specData = new ArrayBuffer(SHADER_CONSTANTS.SPEC_SIZE)
    const specView = new DataView(specData)
    let offset = 0
    specView.setUint32(offset, w, true)
    offset += 4
    specView.setUint32(offset, h, true)
    offset += 4
    specView.setUint32(offset, params.max_iter, true)
    offset += 4
    specView.setFloat32(offset, params.bailout, true)
    offset += 4
    specView.setFloat32(offset, params.refr, true)
    offset += 4
    specView.setFloat32(offset, params.refi, true)
    offset += 4
    specView.setFloat32(offset, params.ddr0, true)
    offset += 4
    specView.setFloat32(offset, params.ddi0, true)
    offset += 4
    specView.setFloat32(offset, params.ddr, true)
    offset += 4
    specView.setFloat32(offset, params.ddi, true)
    offset += 4
    specView.setFloat32(offset, params.z0[0], true)
    offset += 4
    specView.setFloat32(offset, params.z0[1], true)
    offset += 4
    specView.setUint32(offset, params.supersampling || 1, true)
    offset += 4
    specView.setUint32(offset, params.isJulia ? 1 : 0, true)
    offset += 4
    specView.setFloat32(offset, params.juliaC?.[0] ?? 0.0, true)
    offset += 4
    specView.setFloat32(offset, params.juliaC?.[1] ?? 0.0, true)

    device.queue.writeBuffer(specBuffer, 0, specData)

    // Create bind group
    const bindGroupEntries = [
      { binding: 0, resource: { buffer: specBuffer } },
      { binding: 1, resource: { buffer: valuesBuffer } },
    ]

    if (params.doSmooth) {
      bindGroupEntries.push({ binding: 2, resource: { buffer: smoothBuffer } })
    }

    bindGroupEntries.push({ binding: 3, resource: { buffer: signsBuffer } })
    bindGroupEntries.push({ binding: 4, resource: { buffer: zrealBuffer } })
    bindGroupEntries.push({ binding: 5, resource: { buffer: zimagBuffer } })

    const bindGroup = device.createBindGroup({
      layout: this.bindGroupLayout,
      entries: bindGroupEntries,
    })

    // Dispatch compute shader
    const commandEncoder = device.createCommandEncoder()
    const passEncoder = commandEncoder.beginComputePass()
    passEncoder.setPipeline(this.pipeline)
    passEncoder.setBindGroup(0, bindGroup)

    const workgroupsX = Math.ceil(w / this.workgroupSizeX)
    const workgroupsY = Math.ceil(h / this.workgroupSizeY)
    passEncoder.dispatchWorkgroups(workgroupsX, workgroupsY, 1)
    passEncoder.end()

    // Read back results
    const valuesReadBuffer = BufferHelpers.createReadBuffer(device, w * h * 4)
    commandEncoder.copyBufferToBuffer(valuesBuffer, 0, valuesReadBuffer, 0, w * h * 4)

    let smoothReadBuffer = null
    if (params.doSmooth) {
      smoothReadBuffer = BufferHelpers.createReadBuffer(device, w * h * 4)
      commandEncoder.copyBufferToBuffer(smoothBuffer, 0, smoothReadBuffer, 0, w * h * 4)
    }

    const signsReadBuffer = BufferHelpers.createReadBuffer(device, w * h * 4)
    commandEncoder.copyBufferToBuffer(signsBuffer, 0, signsReadBuffer, 0, w * h * 4)

    const zrealReadBuffer = BufferHelpers.createReadBuffer(device, w * h * 4)
    commandEncoder.copyBufferToBuffer(zrealBuffer, 0, zrealReadBuffer, 0, w * h * 4)

    const zimagReadBuffer = BufferHelpers.createReadBuffer(device, w * h * 4)
    commandEncoder.copyBufferToBuffer(zimagBuffer, 0, zimagReadBuffer, 0, w * h * 4)

    device.queue.submit([commandEncoder.finish()])

    // Wait for results
    await valuesReadBuffer.mapAsync(GPUMapMode.READ)
    const valuesData = new Int32Array(valuesReadBuffer.getMappedRange())
    const values = new Int32Array(valuesData)
    valuesReadBuffer.unmap()

    let smooth = null
    if (params.doSmooth) {
      await smoothReadBuffer.mapAsync(GPUMapMode.READ)
      const smoothData = new Uint32Array(smoothReadBuffer.getMappedRange())
      smooth = new Uint8ClampedArray(w * h)
      for (let i = 0; i < w * h; i++) {
        smooth[i] = smoothData[i] & 0xff
      }
      smoothReadBuffer.unmap()
    }

    await signsReadBuffer.mapAsync(GPUMapMode.READ)
    const signsU32 = new Uint32Array(signsReadBuffer.getMappedRange().slice())
    signsReadBuffer.unmap()
    const signs = new Int8Array(w * h)
    for (let i = 0; i < w * h; i++) {
      signs[i] = signsU32[i] & 0xff
    }

    await zrealReadBuffer.mapAsync(GPUMapMode.READ)
    const zreal = new Float32Array(zrealReadBuffer.getMappedRange().slice())
    zrealReadBuffer.unmap()

    await zimagReadBuffer.mapAsync(GPUMapMode.READ)
    const zimag = new Float32Array(zimagReadBuffer.getMappedRange().slice())
    zimagReadBuffer.unmap()

    // Cleanup
    BufferHelpers.destroyBuffer(specBuffer)
    BufferHelpers.destroyBuffer(valuesBuffer)
    BufferHelpers.destroyBuffer(valuesReadBuffer)
    BufferHelpers.destroyBuffer(smoothBuffer)
    BufferHelpers.destroyBuffer(smoothReadBuffer)
    BufferHelpers.destroyBuffer(signsBuffer)
    BufferHelpers.destroyBuffer(signsReadBuffer)
    BufferHelpers.destroyBuffer(zrealBuffer)
    BufferHelpers.destroyBuffer(zrealReadBuffer)
    BufferHelpers.destroyBuffer(zimagBuffer)
    BufferHelpers.destroyBuffer(zimagReadBuffer)

    return { values, smooth, signs, zreal, zimag }
  }

  async finish() {
    // No-op for direct rendering
  }
}

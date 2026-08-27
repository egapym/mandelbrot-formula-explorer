/**
 * Based on bertbaron/mandelbrot by Bert Baron
 * This file is part of the Mandelbrot Explorer project.
 * Licensed under GPL-3.0.
 */

import { CUSTOM_FUNCTION_WGSL_HELPERS } from './wgslCompiler.mjs'

export class BuddhabrotWebGPU {
  constructor(options = {}) {
    this.devicePromise = this.initGpu()
    this.width = options.width || 800
    this.height = options.height || 600
    this.samples = options.samples || 100000
    this.maxIter = options.maxIter || 1000
    this.onProgress = options.onProgress || (() => {})
    this.onChunk = options.onChunk || (() => {})
    this.onComplete = options.onComplete || (() => {})
    this.running = false

    // BuddhabrotRunner と同様に、密度チャンネルはメインスレッド側に持つ
    const total = Math.max(0, this.width * this.height)
    this.densityR = new Float32Array(total)
    this.densityG = new Float32Array(total)
    this.densityB = new Float32Array(total)

    this.brightness = options.brightness ?? 1.8
    // GPU カーネル未対応の種別向けに CPU ランナーを内部で使えるようにしておく
    this._cpuRunner = null
    this.gamma = options.gamma ?? 0.8
  }

  // GPU 上だけで乱択サンプリングと軌道加算を行う。
  // 実装は実用寄りの簡略版で、u32 atomic add を使って密度を積み上げる。

  async _startGpu(params, deviceParam) {
    let device = deviceParam
    const fractalType = params.fractalType || params.type || 'mandelbrot'
    const width = this.width
    const height = this.height
    const len = width * height
    // float の加算量を atomic add 用の整数へ変換する量子化スケール
    const QUANT_SCALE = 1024 // 調整可能。大きいほど精度は上がるがオーバーフローしやすい

    // params.palette から band 情報を取り出す
    let bands = []
    if (params.palette?.bands && Array.isArray(params.palette.bands)) {
      bands = params.palette.bands.map((b) => ({
        // 色が 0..255 のときは 0..1 に正規化する
        color: b.color ? b.color.slice(0, 3).map((c) => c / 255) : [0, 0, 0],
        // cum は number のときだけ採用する。0 などの偽値を安易に有効扱いしない
        cum: typeof b.cum === 'number' ? b.cum : null,
      }))
    }
    // band が無ければ既定の 3 band に戻す
    if (bands.length === 0) {
      bands = [
        { color: [1, 0, 0], cum: 0.01 }, // 赤
        { color: [0, 1, 0], cum: 0.1 }, // 緑
        { color: [0, 0, 1], cum: 1.0 }, // 青
      ]
    }
    let bandCount = bands.length
    const MAX_BANDS = 16
    if (bandCount > MAX_BANDS) {
      console.warn(`buddhabrotWebGPU: bandCount=${bandCount} exceeds MAX_BANDS=${MAX_BANDS}, truncating`)
      bandCount = MAX_BANDS
      bands = bands.slice(0, MAX_BANDS)
    }

    // R/G/B 用の u32 atomic buffer を作る
    const bufSize = len * 4 // 各チャンネルあたりの byte 数（u32）
    // このサイズの buffer を束縛できるか確認する。
    // 足りなければ requiredLimits を付けて device の再取得を試みる。
    try {
      const curMax = device.limits?.maxStorageBufferBindingSize || 0
      if (bufSize > curMax) {
        console.warn(
          `buddhabrotWebGPU: required bufSize=${bufSize} > device limit=${curMax}, attempting to request device with higher limit`,
        )
        // adapter を取り直し、必要な上限付きで device を再取得する
        try {
          const adapter = await navigator.gpu.requestAdapter({
            powerPreference: 'high-performance',
          })
          if (adapter) {
            // adapter が示す上限内で必要サイズを要求する
            const requested = bufSize
            const newDevice = await adapter.requestDevice({
              requiredLimits: { maxStorageBufferBindingSize: requested },
            })
            if (newDevice) {
              device = newDevice
            }
          }
        } catch (e) {
          console.warn(
            'buddhabrotWebGPU: failed to request device with larger storage buffer limit',
            e?.message ? e.message : e,
          )
        }
      }
    } catch (e) {
      console.warn('buddhabrotWebGPU: storage buffer size check failed', e?.message ? e.message : e)
    }
    // 可能なら GPU buffer を使い回し、毎回の再確保を避ける
    this._cachedBuffers = this._cachedBuffers || {}
    let rBuf, gBuf, bBuf
    if (
      this._cachedBuffers.bufSize === bufSize &&
      this._cachedBuffers.width === width &&
      this._cachedBuffers.height === height
    ) {
      rBuf = this._cachedBuffers.rBuf
      gBuf = this._cachedBuffers.gBuf
      bBuf = this._cachedBuffers.bBuf
    } else {
      rBuf = device.createBuffer({
        size: bufSize,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
      })
      gBuf = device.createBuffer({
        size: bufSize,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
      })
      bBuf = device.createBuffer({
        size: bufSize,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
      })
      this._cachedBuffers.bufSize = bufSize
      this._cachedBuffers.width = width
      this._cachedBuffers.height = height
      this._cachedBuffers.rBuf = rBuf
      this._cachedBuffers.gBuf = gBuf
      this._cachedBuffers.bBuf = bBuf
    }

    // band 用の uniform buffer。小さな配列なので storage より uniform が向いている
    const bandBufSize = MAX_BANDS * 16 // vec4 境界に合わせて MAX_BANDS 分を確保する
    let bandBuf = this._cachedBuffers?.bandBuf
    if (!bandBuf) {
      bandBuf = device.createBuffer({
        size: bandBufSize,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
      })
      this._cachedBuffers.bandBuf = bandBuf
    }

    // band ごとの整数寄与量を持つ小さな uniform buffer
    const bandContribBufSize = MAX_BANDS * 16 // vec4<u32> 境界に合わせて確保する
    let bandContribBuf = this._cachedBuffers?.bandContribBuf
    if (!bandContribBuf) {
      bandContribBuf = device.createBuffer({
        size: bandContribBufSize,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
      })
      this._cachedBuffers.bandContribBuf = bandContribBuf
    }

    // Uniforms: width, height, maxIter, samplesPerThread などをまとめて保持する
    let uniformBuf = this._cachedBuffers?.uniformBuf
    if (!uniformBuf) {
      uniformBuf = device.createBuffer({
        size: 96,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
      })
      this._cachedBuffers.uniformBuf = uniformBuf
    }

    // 簡易 WGSL カーネルを組み立てる。
    // CPU 版と完全一致ではないが、GPU 側での加算処理を行える。
    let customWGSLIteration = null
    let customIterationSupported =
      fractalType === 'mandelbrot' || fractalType === 'julia' || fractalType === 'julia-custom'
    if (params.iterationFunction) {
      try {
        const parser = await import('./customFunctionParser.mjs')
        const compiler = await import('./wgslCompiler.mjs')
        const jsExpr = parser.getParsedExpression(params.iterationFunction)
        // パース済み式を安全な WGSL 式へ変換する
        let wgslExpr = compiler.jsExprToWGSL_safe(jsExpr)
        // WGSL で使えない二重アンダースコア識別子を置き換える
        if (/\b__\w+/.test(wgslExpr)) {
          console.warn('buddhabrotWebGPU: replacing double-underscore identifiers in custom WGSL expression')
          wgslExpr = wgslExpr.replace(/\b__([A-Za-z0-9_]+)/g, '_$1')
        }
        if (wgslExpr && wgslExpr.length > 0) {
          customWGSLIteration = wgslExpr
          customIterationSupported = true
        }
      } catch (e) {
        console.warn('buddhabrotWebGPU: custom iteration -> WGSL compile failed:', e?.message ? e.message : e)
        customWGSLIteration = null
      }
    }
    // カスタム式が無ければ既定の Mandelbrot 式を使う。
    // これで標準とカスタムの両方を同じカーネル生成経路で扱える。
    if (!customWGSLIteration) {
      // 現在の z と c から次の z を求める式
      customWGSLIteration = `vec2<f32>(z.x * z.x - z.y * z.y + c.x, 2.0 * z.x * z.y + c.y)`
    }

    // customWGSLIteration を使って統一形式の WGSL カーネルを作る
    let wgsl

    // カスタム式が用意できたら、それを使うカーネルを組み立てる
    if (customWGSLIteration) {
      try {
        const customWgsl = `
struct Band { color: vec4<f32>, };
const MAX_BANDS: u32 = 16u;
struct Uniforms {
  width: u32, height: u32, maxIter: u32, samplesPerThread: u32, contrib: u32,
  bandCount: u32, invocations: u32, invocationOffset: u32, seed: u32, mode: u32,
  bandMode: u32,
  centerX: f32, centerY: f32, zoom: f32,
  z0x: f32, z0y: f32,
  escapeRadius: f32,
  isJulia: u32,
  juliaCr: f32,
  juliaCi: f32,
};
@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var<storage, read_write> rAcc: array<atomic<u32>>;
@group(0) @binding(2) var<storage, read_write> gAcc: array<atomic<u32>>;
@group(0) @binding(3) var<storage, read_write> bAcc: array<atomic<u32>>;
@group(0) @binding(4) var<uniform> bands: array<Band, MAX_BANDS>;
@group(0) @binding(5) var<uniform> bandContribs: array<vec4<u32>, MAX_BANDS>;

${CUSTOM_FUNCTION_WGSL_HELPERS}

// 注入したカスタムカーネルだけで完結するよう、
// rand() と coordToIndex() をローカル定義する。
// 既定カーネルと同じ定義を使い、単独コンパイル時の未解決参照を防ぐ。
fn rand(state: ptr<function, u32>) -> f32 {
  var s = (*state);
  s ^= s << 13u;
  s ^= s >> 17u;
  s ^= s << 5u;
  (*state) = s;
  return f32(s) / 4294967295.0;
}

fn coordToIndex(x: f32, y: f32) -> vec2<f32> {
  let scale = u.zoom * f32(u.width);
  // CPU レンダラーと同じ座標変換を使う。横方向の表示幅は 4 / zoom
  let fx = (x - u.centerX) * scale / 4.0 + f32(u.width) / 2.0;
  let fy = (y - u.centerY) * scale / 4.0 + f32(u.height) / 2.0;
  return vec2<f32>(fx, fy);
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let id = gid.x + u.invocationOffset;
  if (id >= u.invocations) { return; }
  var state: u32 = ((id * 747796405u) ^ u.seed) + 2891336453u;
  let spt = u.samplesPerThread;
  // サンプリングで繰り返し使う不変値を先に計算しておく
  let scale_local = u.zoom * f32(u.width);
  let invDenom_local = 4.0 / scale_local;
  let halfW = f32(u.width) / 2.0;
  let halfH = f32(u.height) / 2.0;
    for (var si: u32 = 0u; si < spt; si = si + 1u) {
	    let rpx = rand(&state) * f32(u.width);
	    let rpy = rand(&state) * f32(u.height);
	  let rx = (rpx - halfW) * invDenom_local + u.centerX;
	  let ry = (rpy - halfH) * invDenom_local + u.centerY;
	    let sample = vec2<f32>(rx, ry);
	    let c = select(sample, vec2<f32>(u.juliaCr, u.juliaCi), u.isJulia != 0u);
	    var z = select(vec2<f32>(u.z0x, u.z0y), sample, u.isJulia != 0u);
	    var escaped = false;
	    var iter: u32 = 0u;
    // カスタム式で反復する。式は vec2<f32> を返す必要がある
    loop {
      if (iter >= u.maxIter) { break; }
      // 一度一時変数へ入れてから妥当性を確認する
      let n = f32(iter);
      var _tmp_iter = ${customWGSLIteration};
      // WGSL には isnan / isfinite がないため、自己比較と上限値で判定する
      if (!(_tmp_iter.x == _tmp_iter.x) || !(_tmp_iter.y == _tmp_iter.y) || abs(_tmp_iter.x) > 1e20 || abs(_tmp_iter.y) > 1e20) {
        // 原点へ潰れて真っ黒にならないよう c へ戻す
        _tmp_iter = c;
      }
      z = _tmp_iter;
      let zzq = dot(z, z);
      if (zzq > u.escapeRadius * u.escapeRadius) { escaped = true; break; }
      iter = iter + 1u;
	    }
	    if (!((escaped && u.mode == 0u) || (!escaped && u.mode == 1u))) { continue; }
	  // z0 から軌道をもう一度たどり、密度へ加算する
	  z = select(vec2<f32>(u.z0x, u.z0y), sample, u.isJulia != 0u);
    // bandMode == 1（perTrajectory）なら軌道全体で使う band を先に決める
    var trajBandIdx: u32 = 0u;
    if (u.bandMode == 1u) {
      var denom_t: u32 = 1u;
      if (iter > 1u) { denom_t = iter - 1u; }
      let fracTraj = f32(iter) / f32(max(1u, u.maxIter));
      var bi_t: u32 = 0u;
      while (bi_t < u.bandCount && fracTraj >= bands[bi_t].color.a) { bi_t = bi_t + 1u; }
      trajBandIdx = bi_t;
      if (trajBandIdx >= u.bandCount) { trajBandIdx = u.bandCount - 1u; }
    }
    if (u.isJulia != 0u) {
      let pcoords0 = coordToIndex(sample.x, sample.y);
      let fx0 = pcoords0.x;
      let fy0 = pcoords0.y;
      if (!(fx0 < 0.0 || fy0 < 0.0 || fx0 >= f32(u.width) || fy0 >= f32(u.height))) {
        let px0 = u32(fx0);
        let py0 = u32(fy0);
        let idx0 = py0 * u.width + px0;
        var bandIdx0: u32 = 0u;
        if (u.bandMode == 1u) {
          bandIdx0 = trajBandIdx;
        } else {
          var bi0: u32 = 0u;
          while (bi0 < u.bandCount && 0.0 >= bands[bi0].color.a) { bi0 = bi0 + 1u; }
          bandIdx0 = bi0;
          if (bandIdx0 >= u.bandCount) { bandIdx0 = u.bandCount - 1u; }
        }
        let contribVec0 = bandContribs[bandIdx0];
        atomicAdd(&rAcc[idx0], contribVec0.x);
        atomicAdd(&gAcc[idx0], contribVec0.y);
        atomicAdd(&bAcc[idx0], contribVec0.z);
      }
    }
    for (var oi: u32 = 0u; oi < iter; oi = oi + 1u) {
      // 一時変数へ評価してから妥当性を確認する
      let n = f32(oi);
      var _tmp_iter = ${customWGSLIteration};
      if (!(_tmp_iter.x == _tmp_iter.x) || !(_tmp_iter.y == _tmp_iter.y) || abs(_tmp_iter.x) > 1e20 || abs(_tmp_iter.y) > 1e20) {
        // 原点へ潰れて真っ黒にならないよう c へ戻す
        _tmp_iter = c;
      }
      z = _tmp_iter;
      let zzq2 = dot(z, z);
      if (zzq2 > 1e10) { break; }
      let pcoords = coordToIndex(z.x, z.y);
      let fx_px = pcoords.x;
      let fy_py = pcoords.y;
      if (fx_px < 0.0 || fy_py < 0.0 || fx_px >= f32(u.width) || fy_py >= f32(u.height)) { continue; }
      let px = u32(fx_px);
      let py = u32(fy_py);
      let idx = py * u.width + px;
      // CPU worker と同様に、軌道位置を [0..1] に正規化する
      var denom: u32 = 1u;
      if (iter > 1u) { denom = iter - 1u; }
      var frac: f32 = 0.0;
      if (u.bandMode == 1u) {
        // perTrajectory では frac は点ごとの選択に使わない
        frac = 0.0;
      } else if (u.bandMode == 2u) {
  // perPoint（旧 perIteration）では軌道インデックスを maxIter 比率へ変換する
        frac = f32(oi) / f32(max(1u, u.maxIter));
      } else {
        // 既定の perPoint でも maxIter に対する比率を使う
        frac = f32(oi) / f32(max(1u, u.maxIter));
      }
      // CPU と同じになるよう、frac >= 累積値 の間は次の band へ進める
      var bandIdx: u32 = 0u;
      if (u.bandMode == 1u) {
        bandIdx = trajBandIdx;
      } else {
        var bi: u32 = 0u;
        while (bi < u.bandCount && frac >= bands[bi].color.a) { bi = bi + 1u; }
        bandIdx = bi;
        if (bandIdx >= u.bandCount) { bandIdx = u.bandCount - 1u; }
      }
  let bandColor = bands[bandIdx].color;
  // あらかじめ計算した整数寄与量を bandContribs から読む
  let contribVec = bandContribs[bandIdx];
  let rContrib = contribVec.x;
  let gContrib = contribVec.y;
  let bContrib = contribVec.z;
  atomicAdd(&rAcc[idx], rContrib);
  atomicAdd(&gAcc[idx], gContrib);
  atomicAdd(&bAcc[idx], bContrib);
    }
  }
}
`
        wgsl = customWgsl
      } catch (e) {
        console.warn('buddhabrotWebGPU: failed to assemble custom WGSL kernel', e)
        // 必要な binding だけ定義した安全なフォールバックカーネルを使う。
        // これで `wgsl` が未定義にならず、安全にパイプラインを作成できる。
        wgsl = `struct Band { color: vec4<f32>, };
struct Uniforms { width: u32, height: u32, maxIter: u32, samplesPerThread: u32, contrib: u32, bandCount: u32, invocations: u32, invocationOffset: u32, seed: u32, mode: u32, bandMode: u32, centerX: f32, centerY: f32, zoom: f32, z0x: f32, z0y: f32, escapeRadius: f32 };
@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var<storage, read_write> rAcc: array<atomic<u32>>;
@group(0) @binding(2) var<storage, read_write> gAcc: array<atomic<u32>>;
@group(0) @binding(3) var<storage, read_write> bAcc: array<atomic<u32>>;
@group(0) @binding(4) var<uniform> bands: array<Band>;
@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  return;
}
`
      }
    }

    const module = device.createShaderModule({ code: wgsl })
    // 取得できる環境ではシェーダーのコンパイルメッセージを出力する
    try {
      if (module && typeof module.compilationInfo === 'function') {
        module.compilationInfo().then((info) => {
          for (const msg of info.messages) {
            const prefix = `buddhabrotWebGPU shader ${msg.type}:`
            if (msg.type === 'error') console.error(prefix, msg.message)
            else console.warn(prefix, msg.message)
          }
        })
      }
    } catch (e) {
      console.warn('buddhabrotWebGPU: compilationInfo not available or failed', e)
    }
    const pipeline = device.createComputePipeline({
      layout: 'auto',
      compute: { module, entryPoint: 'main' },
    })

    const bindGroup = device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: uniformBuf } },
        { binding: 1, resource: { buffer: rBuf } },
        { binding: 2, resource: { buffer: gBuf } },
        { binding: 3, resource: { buffer: bBuf } },
        { binding: 4, resource: { buffer: bandBuf } },
        { binding: 5, resource: { buffer: bandContribBuf } },
      ],
    })

    // uniform へ width, height, maxIter, samplesPerThread などを書き込む
    const totalSamples = Math.max(0, Math.floor(params.samples || this.samples))

    // invocation 数を決める。通常はピクセル数ベースだが、
    // サンプル数が少ないときは totalSamples に合わせる。
    let invocations = len
    if (totalSamples < len) {
      invocations = totalSamples
    }
    // CPU 側の寄与量を量子化する
    const cpuContrib = 1.2
    const contribInt = Math.max(1, Math.round(cpuContrib * QUANT_SCALE))

    // 再利用可能な u32 の uniform view を作る。
    // samplesPerThread、seed、invocationOffset はバッチごとに更新する。
    const modeVal = params.mode === 'antibuddha' || params.mode === 'anti' ? 1 : 0
    const u32view = new Uint32Array(11)
    u32view[0] = width
    u32view[1] = height
    u32view[2] = this.maxIter
    // u32view[3] = samplesPerThread（バッチごとに設定）
    u32view[4] = contribInt
    u32view[5] = bandCount
    u32view[6] = invocations
    // u32view[7] = invocationOffset（チャンクごとに設定）
    // u32view[8] = seed（バッチごとに設定）
    u32view[9] = modeVal
    let bandModeVal = 0
    if (params.buddhaBandMode === 'perTrajectory') bandModeVal = 1
    // 後方互換のため 'perIteration' は廃止済みでも 'perPoint' と同じ扱いにする
    else if (params.buddhaBandMode === 'perIteration') bandModeVal = 0
    u32view[10] = bandModeVal
    // まず固定部分の u32 を書き込む
    device.queue.writeBuffer(uniformBuf, 0, u32view.buffer, 0, u32view.byteLength)
    const f32view = new Float32Array([params.center?.x ?? 0.0, params.center?.y ?? 0.0, params.zoom ?? 1.0])
    // centerX, centerY, zoom を byte offset 44 から書き込む
    device.queue.writeBuffer(uniformBuf, 44, f32view.buffer, 0, f32view.byteLength)
    // 必要なら z0 を center/zoom の後ろへ書き込む
    const z0x = params.z0Real !== undefined ? params.z0Real : 0.0
    const z0y = params.z0Imag !== undefined ? params.z0Imag : 0.0
    const z0view = new Float32Array([z0x, z0y])
    device.queue.writeBuffer(uniformBuf, 56, z0view.buffer, 0, z0view.byteLength)
    // escapeRadius を offset 64 に書き込む
    const erView = new Float32Array([params.escapeRadius !== undefined ? params.escapeRadius : 4.0])
    device.queue.writeBuffer(uniformBuf, 64, erView.buffer, 0, erView.byteLength)
    const isJulia = fractalType === 'julia' || fractalType === 'julia-custom'
    const isJuliaView = new Uint32Array([isJulia ? 1 : 0])
    device.queue.writeBuffer(uniformBuf, 68, isJuliaView.buffer, 0, isJuliaView.byteLength)
    const juliaView = new Float32Array([params.juliaRe ?? 0.0, params.juliaIm ?? 0.0])
    device.queue.writeBuffer(uniformBuf, 72, juliaView.buffer, 0, juliaView.byteLength)

    // Mandelbrot 以外でも反復式を WGSL 化できるなら GPU 経路を使う。
    // 変換できない式だけは、上位で CPU フォールバックできるよう例外にする。
    if (!customIterationSupported) {
      throw new Error(`GPU iteration kernel is not available for fractalType='${fractalType}'`)
    }

    // band 情報を書き込む前に、重みと累積値を JS 側で正規化する。
    // 黒 band も CPU 側の挙動に合わせて残し、空白帯の表現を維持する。
    let normalizedBands = bands.map((b) => ({
      r: b.color?.[0] || 0,
      g: b.color?.[1] || 0,
      b: b.color?.[2] || 0,
      weight: typeof b.weight === 'number' ? b.weight : typeof b.ratio === 'number' ? b.ratio : null,
      cum: typeof b.cum === 'number' ? b.cum : null,
    }))

    // 結果が空になった場合は元の band 一覧へ戻す
    if (normalizedBands.length === 0) {
      normalizedBands = bands.map((b) => ({
        r: b.color?.[0] || 0,
        g: b.color?.[1] || 0,
        b: b.color?.[2] || 0,
        weight: typeof b.weight === 'number' ? b.weight : typeof b.ratio === 'number' ? b.ratio : null,
        cum: typeof b.cum === 'number' ? b.cum : null,
      }))
    }

    // 重み配列を決める。weight を優先し、無ければ cum 差分、それも無ければ均等配分にする
    const weights = new Array(normalizedBands.length).fill(0)
    let haveWeights = false
    for (let i = 0; i < normalizedBands.length; ++i) {
      if (typeof normalizedBands[i].weight === 'number') {
        weights[i] = Math.max(0, normalizedBands[i].weight)
        haveWeights = true
      }
    }
    if (!haveWeights) {
      // cum から導けるか試す
      let haveCum = true
      for (let i = 0; i < normalizedBands.length; ++i) {
        if (normalizedBands[i].cum == null) {
          haveCum = false
          break
        }
      }
      if (haveCum) {
        // 差分を重みに変換する
        let prev = 0
        for (let i = 0; i < normalizedBands.length; ++i) {
          const cur = Math.max(0, normalizedBands[i].cum || 0)
          weights[i] = Math.max(0, cur - prev)
          prev = cur
        }
      } else {
        // 均等配分
        for (let i = 0; i < normalizedBands.length; ++i) weights[i] = 1
      }
    }

    // 累積比率へ正規化する
    let totalW = 0
    for (const w of weights) totalW += Math.max(0, w)
    if (totalW <= 0) totalW = normalizedBands.length
    const cums = new Array(weights.length)
    let acc = 0
    for (let i = 0; i < weights.length; ++i) {
      acc += Math.max(0, weights[i])
      cums[i] = acc / totalW
    }

    // 常に MAX_BANDS 分を確保し、未使用分は 0 で埋める
    const bandData = new Float32Array(MAX_BANDS * 4)
    for (let i = 0; i < MAX_BANDS; ++i) {
      if (i < normalizedBands.length) {
        bandData[i * 4 + 0] = Number(normalizedBands[i].r) || 0.0
        bandData[i * 4 + 1] = Number(normalizedBands[i].g) || 0.0
        bandData[i * 4 + 2] = Number(normalizedBands[i].b) || 0.0
        bandData[i * 4 + 3] = Number(cums[i]) || 0.0
      } else {
        bandData[i * 4 + 0] = 0.0
        bandData[i * 4 + 1] = 0.0
        bandData[i * 4 + 2] = 0.0
        bandData[i * 4 + 3] = 0.0
      }
    }
    device.queue.writeBuffer(bandBuf, 0, bandData.buffer, 0, bandData.byteLength)

    // シェーダー内で float→u32 変換しないよう、band ごとの整数寄与量を先に作る
    const bandContribs = new Uint32Array(MAX_BANDS * 4)
    for (let i = 0; i < MAX_BANDS; ++i) {
      if (i < normalizedBands.length) {
        const r = Math.round((bandData[i * 4 + 0] || 0.0) * contribInt)
        const g = Math.round((bandData[i * 4 + 1] || 0.0) * contribInt)
        const b = Math.round((bandData[i * 4 + 2] || 0.0) * contribInt)
        bandContribs[i * 4 + 0] = r
        bandContribs[i * 4 + 1] = g
        bandContribs[i * 4 + 2] = b
        bandContribs[i * 4 + 3] = 0
      } else {
        bandContribs[i * 4 + 0] = 0
        bandContribs[i * 4 + 1] = 0
        bandContribs[i * 4 + 2] = 0
        bandContribs[i * 4 + 3] = 0
      }
    }
    device.queue.writeBuffer(bandContribBuf, 0, bandContribs.buffer, 0, bandContribs.byteLength)

    // デバッグ用に uniform と band を読み戻し、転送内容を確認する
    try {
      const readUniform = device.createBuffer({
        size: 80,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
      })
      const readBands = device.createBuffer({
        size: Math.max(16, bandData.byteLength),
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
      })
      const dbgEnc = device.createCommandEncoder()
      dbgEnc.copyBufferToBuffer(uniformBuf, 0, readUniform, 0, 80)
      dbgEnc.copyBufferToBuffer(bandBuf, 0, readBands, 0, bandData.byteLength)
      device.queue.submit([dbgEnc.finish()])
      await Promise.all([readUniform.mapAsync(GPUMapMode.READ), readBands.mapAsync(GPUMapMode.READ)])

      try {
        readBands.unmap()
      } catch (e) {
        console.warn('buddhabrotWebGPU: failed to read band buf', e?.message ? e.message : e)
      }
    } catch (e) {
      console.warn('buddhabrotWebGPU: debug readback failed', e?.message ? e.message : e)
    }

    // 実行前にバッファを 0 へ戻し、毎回同じ初期状態にする
    const zeroInit = new Uint8Array(bufSize)
    try {
      device.queue.writeBuffer(this._cachedBuffers.rBuf, 0, zeroInit)
      device.queue.writeBuffer(this._cachedBuffers.gBuf, 0, zeroInit)
      device.queue.writeBuffer(this._cachedBuffers.bBuf, 0, zeroInit)
    } catch (_e) {
      // writeBuffer が失敗する環境では一時 staging buffer に戻す
      const zeroStaging = device.createBuffer({
        size: bufSize,
        usage: GPUBufferUsage.COPY_SRC,
        mappedAtCreation: true,
      })
      const zm = zeroStaging.getMappedRange()
      new Uint8Array(zm).set(zeroInit)
      zeroStaging.unmap()
      const zeroEncoder = device.createCommandEncoder()
      zeroEncoder.copyBufferToBuffer(zeroStaging, 0, this._cachedBuffers.rBuf, 0, bufSize)
      zeroEncoder.copyBufferToBuffer(zeroStaging, 0, this._cachedBuffers.gBuf, 0, bufSize)
      zeroEncoder.copyBufferToBuffer(zeroStaging, 0, this._cachedBuffers.bBuf, 0, bufSize)
      device.queue.submit([zeroEncoder.finish()])
    }

    // サンプル数が多い場合に備え、compute workgroup はバッチ単位で流す
    const workgroupsTotal = Math.ceil(invocations / 64)
    // 次元ごとの最大 compute workgroup 数を取得する
    const maxPerDim = device.limits?.maxComputeWorkgroupsPerDimension || 65535
    const maxPerDimSafe = Math.max(1, maxPerDim)

    // 極端に長い GPU ループを避けるための安全上限
    const MAX_SAMPLES_PER_THREAD = 1000000 // per-invocation sample upper bound
    const MAX_SAMPLES_PER_BATCH = 50000000 // total samples per host batch

    let samplesRemaining = totalSamples
    try {
      let batchIndex = 0
      while (samplesRemaining > 0) {
        // このバッチで処理するサンプル数を決める
        const samplesThisBatch = Math.min(samplesRemaining, invocations * MAX_SAMPLES_PER_THREAD, MAX_SAMPLES_PER_BATCH)
        const samplesPerThreadThis = Math.max(1, Math.ceil(samplesThisBatch / invocations))
        console.info(
          `buddhabrotWebGPU: batch[${batchIndex}] will request samplesThisBatch=${samplesThisBatch} (MAX_SAMPLES_PER_BATCH=${MAX_SAMPLES_PER_BATCH})`,
        )

        // このバッチ用の samplesPerThread と seed を設定する
        u32view[3] = samplesPerThreadThis
        // 新しい seed
        let batchSeed = 0
        try {
          if (typeof crypto !== 'undefined' && crypto.getRandomValues)
            batchSeed = crypto.getRandomValues(new Uint32Array(1))[0]
          else batchSeed = Math.floor(Math.random() * 0xffffffff)
        } catch (_e) {
          batchSeed = Math.floor(Math.random() * 0xffffffff)
        }
        u32view[8] = batchSeed
        // 更新分だけ uniform に反映する
        device.queue.writeBuffer(uniformBuf, 0, u32view.buffer, 0, u32view.byteLength)
        console.info(
          `buddhabrotWebGPU: dispatching batch samplesThisBatch=${samplesThisBatch} samplesPerThread=${samplesPerThreadThis} invocations=${invocations}`,
        )

        // workgroup が上限を超える場合は分割して dispatch する
        let remaining = workgroupsTotal
        let invocationOffset = 0
        while (remaining > 0) {
          const thisChunk = Math.min(remaining, maxPerDimSafe)
          const encoder = device.createCommandEncoder()
          const pass = encoder.beginComputePass()
          pass.setPipeline(pipeline)
          // invocationOffset を uniform buffer の index 7 へ書き込む
          const offView = new Uint32Array([invocationOffset])
          device.queue.writeBuffer(uniformBuf, 4 * 7, offView.buffer, 0, 4)
          pass.setBindGroup(0, bindGroup)
          pass.dispatchWorkgroups(thisChunk)
          pass.end()
          device.queue.submit([encoder.finish()])

          remaining -= thisChunk
          invocationOffset += thisChunk * 64
        }

        // 次のバッチへ進む前に完了待ちする。
        // uniform の上書きや読み戻しの競合を防ぎ、進捗通知の区切りにもなる。
        if (device.queue && typeof device.queue.onSubmittedWorkDone === 'function') {
          // 次のバッチへ進む前に完了待ちする。
          // uniform の上書きや読み戻しの競合を防ぎ、進捗通知の区切りにもなる。
          // GPU 完了待ちと abort を競合させ、stop() で中断できるようにする
          const waitRes = await Promise.race([device.queue.onSubmittedWorkDone(), this._abortPromise])
          if (waitRes === 'abort' || !this.running) {
            console.info('buddhabrotWebGPU: aborted during batch wait — reading back partial density')
            this.running = false
            // ここまでにたまった密度だけでも読み戻し、画面が空白にならないようにする
            try {
              const abortEncoder = device.createCommandEncoder()
              const abortCombinedBuf = device.createBuffer({
                size: bufSize * 3,
                usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
              })
              abortEncoder.copyBufferToBuffer(rBuf, 0, abortCombinedBuf, 0, bufSize)
              abortEncoder.copyBufferToBuffer(gBuf, 0, abortCombinedBuf, bufSize, bufSize)
              abortEncoder.copyBufferToBuffer(bBuf, 0, abortCombinedBuf, bufSize * 2, bufSize)
              device.queue.submit([abortEncoder.finish()])
              await abortCombinedBuf.mapAsync(GPUMapMode.READ)
              const abortRange = abortCombinedBuf.getMappedRange()
              const abortU32 = new Uint32Array(abortRange)
              const abortR = abortU32.subarray(0, bufSize / 4)
              const abortG = abortU32.subarray(bufSize / 4, (bufSize * 2) / 4)
              const abortB = abortU32.subarray((bufSize * 2) / 4, (bufSize * 3) / 4)
              const abortLen = this.width * this.height
              for (let i = 0; i < abortLen; i++) {
                this.densityR[i] = (abortR[i] || 0) / QUANT_SCALE
                this.densityG[i] = (abortG[i] || 0) / QUANT_SCALE
                this.densityB[i] = (abortB[i] || 0) / QUANT_SCALE
              }
              abortCombinedBuf.unmap()
            } catch (abortReadErr) {
              console.warn(
                'buddhabrotWebGPU: failed to read back density on abort:',
                abortReadErr?.message ? abortReadErr.message : abortReadErr,
              )
            }
            this.onComplete({
              densityMap: null,
              width: this.width,
              height: this.height,
              aborted: true,
            })
            return
          }
        } else {
          console.info(
            'buddhabrotWebGPU: device.queue.onSubmittedWorkDone not available; relying on buffer mapAsync for sync',
          )
        }

        // このバッチで実際に処理したサンプル数を反映する
        const processed = Math.min(samplesRemaining, invocations * samplesPerThreadThis)
        samplesRemaining -= processed
        this.onProgress({ delta: processed, total: totalSamples })
        console.info(
          `buddhabrotWebGPU: batch[${batchIndex}] completed processed=${processed} samplesRemaining=${samplesRemaining} samplesPerThread=${samplesPerThreadThis}`,
        )
        batchIndex++
      }
    } catch (e) {
      console.error('buddhabrotWebGPU: dispatchWorkgroups threw', e?.message ? e.message : e)
      this.running = false
      this.onComplete({
        densityMap: null,
        width: this.width,
        height: this.height,
        error: 'dispatchWorkgroups failed',
      })
      return
    }

    // 新しい encoder で u32 バッファを読み戻す
    const readEncoder = device.createCommandEncoder()
    // R/G/B を連続配置で受け取る 1 本の読み出し buffer を作る
    const combinedSize = bufSize * 3
    const readCombined = device.createBuffer({
      size: combinedSize,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    })
    readEncoder.copyBufferToBuffer(rBuf, 0, readCombined, 0, bufSize)
    readEncoder.copyBufferToBuffer(gBuf, 0, readCombined, bufSize, bufSize)
    readEncoder.copyBufferToBuffer(bBuf, 0, readCombined, bufSize * 2, bufSize)
    device.queue.submit([readEncoder.finish()])

    // abort 済みでも密度を読めるよう、必ず mapAsync を待つ
    try {
      await readCombined.mapAsync(GPUMapMode.READ)
    } catch (e) {
      console.error('buddhabrotWebGPU: mapAsync for combined read buffer failed', e?.message ? e.message : e)
      readCombined.unmap()
      this.running = false
      this.onComplete({
        densityMap: null,
        width: this.width,
        height: this.height,
        error: 'mapAsync failed',
      })
      return
    }
    // 中断有無にかかわらず、この後で密度バッファは更新する
    // 余計なコピーを避けるため、読み取り範囲へ直接 view を作る
    const combinedRange = readCombined.getMappedRange()
    const combinedU32 = new Uint32Array(combinedRange)
    const ar = combinedU32.subarray(0, bufSize / 4)
    const ag = combinedU32.subarray(bufSize / 4, (bufSize * 2) / 4)
    const ab = combinedU32.subarray((bufSize * 2) / 4, (bufSize * 3) / 4)

    // 量子化した整数を float の密度へ戻す
    const total = width * height

    for (let i = 0; i < total; i++) {
      this.densityR[i] = (ar[i] || 0) / QUANT_SCALE
      this.densityG[i] = (ag[i] || 0) / QUANT_SCALE
      this.densityB[i] = (ab[i] || 0) / QUANT_SCALE
    }

    readCombined.unmap()

    return
  }

  async initGpu() {
    if (!navigator.gpu) {
      console.warn('buddhabrotWebGPU: navigator.gpu not available')
      return null
    }
    try {
      const adapter = await navigator.gpu.requestAdapter({
        powerPreference: 'high-performance',
      })
      if (!adapter) {
        console.warn('buddhabrotWebGPU: no GPU adapter available')
        return null
      }
      const device = await adapter.requestDevice()
      return device
    } catch (e) {
      console.warn('buddhabrotWebGPU: WebGPU init failed:', e?.message ? e.message : e)
      return null
    }
  }

  // start は BuddhabrotRunner.start とほぼ同じ引数を受け取る
  async start(params = {}) {
    if (this.running) return
    this.running = true
    // stop() で待機中バッチを中断できるよう abort Promise を初期化する
    this._abortPromise = new Promise((resolve) => {
      this._abortResolve = resolve
    })
    this.maxIter = params.maxIter ?? this.maxIter
    this.samples = params.samples ?? this.samples
    this.width = params.width || this.width
    this.height = params.height || this.height

    const total = Math.max(0, this.width * this.height)
    this.densityR = new Float32Array(total)
    this.densityG = new Float32Array(total)
    this.densityB = new Float32Array(total)

    // ここではまず最小構成の GPU 経路を用意し、
    // 必要に応じて CPU フォールバックも扱えるようにしている。

    // brightness / gamma を反映し、後段の描画で正しい値を使えるようにする
    if (typeof params.brightness === 'number') this.brightness = params.brightness
    if (typeof params.gamma === 'number') this.gamma = params.gamma

    let device = await this.devicePromise
    if (!device) {
      console.warn('buddhabrotWebGPU.start: no device available, attempting CPU fallback')
      // GPU が使えなくても実際の Buddhabrot を見られるよう CPU に戻す
      try {
        const mod = await import('./buddhabrot.mjs')
        const CPU = mod?.BuddhabrotRunner ? mod.BuddhabrotRunner : null
        if (CPU) {
          const total = Math.max(0, this.width * this.height)
          this.densityR = new Float32Array(total)
          this.densityG = new Float32Array(total)
          this.densityB = new Float32Array(total)
          const cpu = new CPU({
            workerCount: Math.max(1, navigator.hardwareConcurrency || 4),
            width: this.width,
            height: this.height,
            maxIter: this.maxIter,
            samples: params.samples || this.samples,
            onProgress: (d) => this.onProgress(d),
            onChunk: (chunk) => {
              try {
                if (chunk.indices && chunk.indices.length > 0) {
                  const inds = chunk.indices
                  const r = chunk.r
                  const g = chunk.g
                  const b = chunk.b
                  for (let i = 0; i < inds.length; i++) {
                    const srcIdx = inds[i]
                    const dstIdx = chunk.y * this.width + chunk.x + srcIdx
                    if (r) this.densityR[dstIdx] += r[i]
                    if (g) this.densityG[dstIdx] += g[i]
                    if (b) this.densityB[dstIdx] += b[i]
                  }
                } else {
                  const r = chunk.r || []
                  const g = chunk.g || []
                  const b = chunk.b || []
                  const dstW = this.width
                  for (let row = 0; row < chunk.h; row++) {
                    const dstRowBase = (chunk.y + row) * dstW + chunk.x
                    const srcRowBase = row * chunk.w
                    for (let col = 0; col < chunk.w; col++) {
                      const dstIdx = dstRowBase + col
                      const srcIdx = srcRowBase + col
                      if (r?.length) this.densityR[dstIdx] += r[srcIdx]
                      if (g?.length) this.densityG[dstIdx] += g[srcIdx]
                      if (b?.length) this.densityB[dstIdx] += b[srcIdx]
                    }
                  }
                }
              } catch (e) {
                console.warn('buddhabrotWebGPU: error merging CPU chunk:', e)
              }
              this.onChunk?.(chunk)
            },
            onComplete: (result) => {
              this.onProgress({
                delta: result.width * result.height,
                total: result.width * result.height,
              })
              this.onComplete?.({
                densityMap: null,
                width: result.width,
                height: result.height,
              })
            },
          })
          this._cpuRunner = cpu
          this.running = true
          await cpu.start(params)
          this.running = false
          return
        }
      } catch (e) {
        console.warn('buddhabrotWebGPU: CPU fallback failed during no-device path:', e?.message ? e.message : e)
      }
      // フォールバック時は 0 進捗を送ってから終了し、UI の 100% ジャンプを防ぐ
      this.onProgress({ delta: 0, total: this.samples })
      // 空結果を返して UI 側で描画だけは継続できるようにする
      this.running = false
      this.onComplete({
        densityMap: null,
        width: this.width,
        height: this.height,
      })
      return
    }

    // GPU Buddhabrot が要求されたら、反復式を WGSL 化できる限り fractalType に関係なく試す。
    const fractalType = params.fractalType || params.type || 'mandelbrot'

    if (params.gpu) {
      try {
        // 必要な storage buffer をこの device が束縛できるか確認する
        const bufSize = this.width * this.height * 4 // bytes per channel
        const curMax = device.limits?.maxStorageBufferBindingSize || 0
        if (bufSize > curMax) {
          console.warn(
            `buddhabrotWebGPU.start: required bufSize=${bufSize} > device limit=${curMax}, attempting to request device with higher limit`,
          )
          try {
            const adapter = await navigator.gpu.requestAdapter({
              powerPreference: 'high-performance',
            })
            if (adapter) {
              const requested = bufSize
              const newDevice = await adapter.requestDevice({
                requiredLimits: { maxStorageBufferBindingSize: requested },
              })
              if (newDevice) {
                device = newDevice
                // 次回以降も再利用できるよう devicePromise を更新する
                this.devicePromise = Promise.resolve(device)
              }
            }
          } catch (e) {
            console.warn(
              'buddhabrotWebGPU.start: failed to request device with larger storage buffer limit',
              e?.message ? e.message : e,
            )
          }
        }

        // UI がいきなり 100% を示さないよう、開始時に 0 を送る
        this.onProgress({ delta: 0, total: params.samples || this.samples })
        await this._startGpu(params, device)
        // _startGpu 側で進捗は通知されるので、戻ったら完了処理だけ行う
        this.running = false
        this.onComplete({
          densityMap: null,
          width: this.width,
          height: this.height,
        })
        return
      } catch (e) {
        console.warn('buddhabrotWebGPU GPU path failed, falling back to test pattern:', e?.message ? e.message : e)
        // まず CPU フォールバックを試し、実際の Buddhabrot を優先する
        try {
          const mod = await import('./buddhabrot.mjs')
          const CPU = mod?.BuddhabrotRunner ? mod.BuddhabrotRunner : null
          if (CPU) {
            // 内部バッファのサイズを合わせる
            const total = Math.max(0, this.width * this.height)
            this.densityR = new Float32Array(total)
            this.densityG = new Float32Array(total)
            this.densityB = new Float32Array(total)
            const cpu = new CPU({
              workerCount: Math.max(1, navigator.hardwareConcurrency || 4),
              width: this.width,
              height: this.height,
              maxIter: this.maxIter,
              samples: params.samples || this.samples,
              onProgress: (d) => this.onProgress(d),
              onChunk: (chunk) => {
                try {
                  // 受け取ったチャンクを自前の密度バッファへ加算する
                  if (chunk.indices && chunk.indices.length > 0) {
                    const inds = chunk.indices
                    const r = chunk.r
                    const g = chunk.g
                    const b = chunk.b
                    for (let i = 0; i < inds.length; i++) {
                      const srcIdx = inds[i]
                      const dstIdx = chunk.y * this.width + chunk.x + srcIdx
                      if (r) this.densityR[dstIdx] += r[i]
                      if (g) this.densityG[dstIdx] += g[i]
                      if (b) this.densityB[dstIdx] += b[i]
                    }
                  } else {
                    const r = chunk.r || []
                    const g = chunk.g || []
                    const b = chunk.b || []
                    const dstW = this.width
                    for (let row = 0; row < chunk.h; row++) {
                      const dstRowBase = (chunk.y + row) * dstW + chunk.x
                      const srcRowBase = row * chunk.w
                      for (let col = 0; col < chunk.w; col++) {
                        const dstIdx = dstRowBase + col
                        const srcIdx = srcRowBase + col
                        if (r?.length) this.densityR[dstIdx] += r[srcIdx]
                        if (g?.length) this.densityG[dstIdx] += g[srcIdx]
                        if (b?.length) this.densityB[dstIdx] += b[srcIdx]
                      }
                    }
                  }
                } catch (e) {
                  console.warn('buddhabrotWebGPU: error merging CPU chunk:', e)
                }
                this.onChunk?.(chunk)
              },
              onComplete: (result) => {
                this.onProgress({
                  delta: result.width * result.height,
                  total: result.width * result.height,
                })
                this.onComplete?.({
                  densityMap: null,
                  width: result.width,
                  height: result.height,
                })
              },
            })
            this._cpuRunner = cpu
            this.running = true
            await cpu.start(params)
            this.running = false
            return
          }
        } catch (e2) {
          console.warn('buddhabrotWebGPU: CPU fallback also failed:', e2?.message ? e2.message : e2)
        }
        // CPU でも失敗した場合だけ、下の簡易テストパターンへ進む
      }
    }

    // 結果を書き込む GPU バッファを作り、後で CPU 側へ読み戻す
    try {
      const resultBufferSize = this.width * this.height * 4 * 3 // 3 channels float32
      const gpuBuffer = device.createBuffer({
        size: resultBufferSize,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC | GPUBufferUsage.STORAGE,
      })

      // ピクセル座標から簡単な RGB パターンを書く WGSL シェーダー。
      // WebGPU 経路の確認用であり、最終的な Buddhabrot 実装ではない。
      const shaderCode = `
        @group(0) @binding(0) var<storage, read_write> outBuf: array<f32>;
        @compute @workgroup_size(64)
        fn main(@builtin(global_invocation_id) gid : vec3u) {
          let pid = gid.x; // ピクセル index
          if (pid >= ${this.width * this.height}u) { return; }
          let x = f32(pid % ${this.width}u);
          let y = f32(pid / ${this.width}u);
          let fx = x / f32(${this.width});
          let fy = y / f32(${this.height});
          // sin 波と放射グラデーションによる簡単な模様
          let r = 0.5 + 0.5 * sin(6.2831853 * fx + fy * 3.14159);
          let g = 0.5 + 0.5 * sin(6.2831853 * fy + fx * 3.14159);
          let dx = fx - 0.5;
          let dy = fy - 0.5;
          let dist = sqrt(dx*dx + dy*dy);
          let b = 1.0 - clamp(dist*1.5, 0.0, 1.0);
          let base = pid * 3u;
          outBuf[base + 0u] = r;
          outBuf[base + 1u] = g;
          outBuf[base + 2u] = b;
        }
      `

      const module = device.createShaderModule({ code: shaderCode })
      // 取得できる環境ではシェーダーコンパイルメッセージを出す
      try {
        if (module && typeof module.compilationInfo === 'function') {
          module.compilationInfo().then((info) => {
            for (const msg of info.messages) {
              const prefix = `buddhabrotWebGPU shader ${msg.type}:`
              if (msg.type === 'error') console.error(prefix, msg.message)
              else console.warn(prefix, msg.message)
            }
          })
        }
      } catch (e) {
        console.warn('buddhabrotWebGPU: compilationInfo not available or failed', e)
      }
      const pipeline = device.createComputePipeline({
        layout: 'auto',
        compute: { module, entryPoint: 'main' },
      })

      const bindGroupLayout = pipeline.getBindGroupLayout(0)
      const bindGroup = device.createBindGroup({
        layout: bindGroupLayout,
        entries: [{ binding: 0, resource: { buffer: gpuBuffer } }],
      })

      const encoder = device.createCommandEncoder()
      const pass = encoder.beginComputePass()
      pass.setPipeline(pipeline)
      pass.setBindGroup(0, bindGroup)
      const workgroups = Math.ceil((this.width * this.height) / 64)
      pass.dispatchWorkgroups(workgroups)
      pass.end()

      // 読み取り可能な buffer へコピーする
      const readBuffer = device.createBuffer({
        size: resultBufferSize,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
      })
      encoder.copyBufferToBuffer(gpuBuffer, 0, readBuffer, 0, resultBufferSize)
      const commandBuffer = encoder.finish()
      device.queue.submit([commandBuffer])

      await readBuffer.mapAsync(GPUMapMode.READ)
      const arrayBuf = readBuffer.getMappedRange()
      const f32 = new Float32Array(arrayBuf.slice(0))

      // GPU 出力が空でないかをざっくり確認する
      let nonZero = 0
      for (let i = 0; i < f32.length; i++) {
        if (f32[i] !== 0) {
          nonZero++
          // いくつか非ゼロが見えたら十分なので早めに抜ける
          if (nonZero > 8) break
        }
      }
      if (nonZero === 0) {
        // 全部 0 の場合は見かけだけのテスト画像を出さず、
        // エラーとして扱って UI 側に判断を委ねる。
        console.error('buddhabrotWebGPU: GPU output all zeros — aborting render and reporting error')
        this.running = false
        this.onComplete({
          densityMap: null,
          width: this.width,
          height: this.height,
          error: 'GPU output all zeros',
        })
        readBuffer.unmap()
        return
      } else {
        // 値を密度チャンネルへコピーする
        const len = this.width * this.height
        for (let i = 0; i < len; i++) {
          this.densityR[i] = f32[i * 3]
          this.densityG[i] = f32[i * 3 + 1]
          this.densityB[i] = f32[i * 3 + 2]
        }
      }
      readBuffer.unmap()

      // 簡易テストパターン経路でも進捗と完了は通知する
      this.onProgress({ delta: this.samples, total: this.samples })
      this.running = false
      this.onComplete({
        densityMap: null,
        width: this.width,
        height: this.height,
      })
    } catch (e) {
      console.warn('buddhabrotWebGPU error during compute:', e?.message ? e.message : e)
      this.running = false
      this.onComplete({
        densityMap: null,
        width: this.width,
        height: this.height,
        error: e?.message ? e.message : String(e),
      })
    }
  }

  stop() {
    // running=false にし、待機中バッチがすぐ中断できるよう abort を解決する
    this.running = false
    try {
      if (this._abortResolve) {
        this._abortResolve('abort')
        // 二重呼び出しを避ける
        this._abortResolve = null
      }
    } catch (e) {
      console.warn('buddhabrotWebGPU.stop: failed to resolve abort promise', e)
    }
  }
}

// GPU で色変換を行い、3 本の float32 密度バッファを RGBA の ImageBitmap に変換する。
// 主にハイブリッド経路向けで、CPU 側で作った densityR/G/B を GPU で着色する。
export async function colorMapDensity({ rBuf, gBuf, bBuf, width, height, brightness = 1.8, gamma = 0.8 }) {
  if (!navigator.gpu) {
    throw new Error('WebGPU not available')
  }
  const adapter = await navigator.gpu.requestAdapter()
  if (!adapter) throw new Error('No GPU adapter')
  const device = await adapter.requestDevice()

  const len = width * height
  // 入力バッファ: float32 を 3 * len 個並べる
  const inputSize = len * 3 * 4
  const inputBuffer = device.createBuffer({
    size: inputSize,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  })

  // 出力バッファ: RGBA8 を len ピクセル分
  const outputSize = len * 4
  const outputBuffer = device.createBuffer({
    size: outputSize,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
  })

  // 入力を書き込むための staging buffer を作る
  const staging = device.createBuffer({
    size: inputSize,
    usage: GPUBufferUsage.MAP_WRITE | GPUBufferUsage.COPY_SRC,
  })
  await staging.mapAsync(GPUMapMode.WRITE)
  const map = staging.getMappedRange()
  const f32 = new Float32Array(map)
  // r, g, b を交互に並べてコピーする
  for (let i = 0; i < len; i++) {
    f32[i * 3] = rBuf[i] || 0
    f32[i * 3 + 1] = gBuf[i] || 0
    f32[i * 3 + 2] = bBuf[i] || 0
  }
  staging.unmap()

  // compute shader: float 3 要素を読み、u32 に詰めた RGBA8 を書く
  const shader = `
  struct Params { invLogDenom: f32, brightness: f32, gamma: f32, denomScale: f32, width: u32, height: u32 };
    @group(0) @binding(0) var<storage, read> inBuf: array<f32>;
    @group(0) @binding(1) var<storage, read_write> outBuf: array<u32>;
    @group(0) @binding(2) var<uniform> params: Params;

    fn toU32(r: f32, g: f32, b: f32) -> u32 {
      let ri = u32(clamp(r * 255.0, 0.0, 255.0));
      let gi = u32(clamp(g * 255.0, 0.0, 255.0));
      let bi = u32(clamp(b * 255.0, 0.0, 255.0));
      let ai = u32(255);
      return (ai << 24) | (bi << 16) | (gi << 8) | ri;
    }

    @compute @workgroup_size(64)
    fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
      let idx = gid.x;
      let total = params.width * params.height;
      if (idx >= total) { return; }
      let base = idx * 3u;
      let rv = inBuf[base + 0u];
      let gv = inBuf[base + 1u];
      let bv = inBuf[base + 2u];
      let lr = log(1.0 + rv) / log(10.0) * params.invLogDenom;
      let lg = log(1.0 + gv) / log(10.0) * params.invLogDenom;
      let lb = log(1.0 + bv) / log(10.0) * params.invLogDenom;
  let rn = min(1.0, pow((lr * params.brightness) / params.denomScale, params.gamma));
  let gn = min(1.0, pow((lg * params.brightness) / params.denomScale, params.gamma));
  let bn = min(1.0, pow((lb * params.brightness) / params.denomScale, params.gamma));
      outBuf[idx] = toU32(rn, gn, bn);
    }
  `

  const module = device.createShaderModule({ code: shader })
  const pipeline = device.createComputePipeline({
    layout: 'auto',
    compute: { module, entryPoint: 'main' },
  })

  // uniform buffer を作る
  const maxVal = Math.max(
    1,
    Math.max(...rBuf.map((v) => v || 0), ...gBuf.map((v) => v || 0), ...bBuf.map((v) => v || 0)),
  )
  const invLogDenom = 1 / Math.log10(1 + maxVal)
  // denomScale は CPU 側と同じ考え方で、密度が大きいときの明るさを調整する
  const denomScale = 1 + Math.log10(1 + maxVal) / 4
  const uniformData = new Float32Array([invLogDenom, brightness, gamma, denomScale])
  // 16 byte 境界に合わせつつ width / height も入れる
  const uniformBuffer = device.createBuffer({
    size: 16,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  })
  device.queue.writeBuffer(uniformBuffer, 0, uniformData.buffer, 0, 16)
  // float 4 個の後ろへ width / height を u32 で書く
  const wh = new Uint32Array([width, height])
  device.queue.writeBuffer(uniformBuffer, 16, wh.buffer, 0, 8)

  const bindGroup = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: inputBuffer } },
      { binding: 1, resource: { buffer: outputBuffer } },
      { binding: 2, resource: { buffer: uniformBuffer } },
    ],
  })

  // staging から input へコピーし、compute pass を実行する
  const encoder = device.createCommandEncoder()
  encoder.copyBufferToBuffer(staging, 0, inputBuffer, 0, inputSize)
  const pass = encoder.beginComputePass()
  pass.setPipeline(pipeline)
  pass.setBindGroup(0, bindGroup)
  const workgroups = Math.ceil(len / 64)
  pass.dispatchWorkgroups(workgroups)
  pass.end()

  // 出力を読み出し用 buffer へコピーする
  const readBuffer = device.createBuffer({
    size: outputSize,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  })
  encoder.copyBufferToBuffer(outputBuffer, 0, readBuffer, 0, outputSize)
  device.queue.submit([encoder.finish()])

  await readBuffer.mapAsync(GPUMapMode.READ)
  const mapped = readBuffer.getMappedRange()
  // 読み出した byte 配列から ImageData を作る
  const rgba = new Uint8ClampedArray(mapped.slice(0))
  readBuffer.unmap()

  // ImageData から ImageBitmap を作る
  const imageData = new ImageData(rgba, width, height)
  const bitmap = await createImageBitmap(imageData)
  return bitmap
}

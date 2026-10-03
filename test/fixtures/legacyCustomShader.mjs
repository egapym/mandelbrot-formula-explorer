// Frozen shader generator before SS OFF iteration-limit optimization.
// Based on Bert Baron's renderer; keep independent of production changes.
import { CUSTOM_FUNCTION_WGSL_HELPERS } from '../../wgslCompiler.mjs'
import { BAILOUT_MIN, BAILOUT_SMOOTH } from '../../sharedCalculations.mjs'

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

export function legacyCustomShader(iterationExpr, doSmooth, _bailout, supersampling, historyRequirements = { zAt: [], zDelay: [] }) {
    const ssScale = supersampling > 0 ? supersampling : 1
    const ssSamples = ssScale * ssScale
    const historyDeclarations = [
      ...historyRequirements.zAt.map((index) => `  var historyZAt_${index}: vec2<f32> = vec2<f32>(0.0, 0.0);`),
      ...historyRequirements.zDelay
        .filter((index) => index > 0)
        .map((index) => `  var historyZDelayStorage_${index}: array<vec2<f32>, ${index}>;`),
    ].join('\n')
    // private 配列の初期値には依存せず、履歴不足を必ず (0, 0) にする。
    const historyInitialize = historyRequirements.zDelay
      .filter((index) => index > 0)
      .map(
        (index) =>
          `  for (var historyInit_${index}: u32 = 0u; historyInit_${index} < ${index}u; historyInit_${index} = historyInit_${index} + 1u) { historyZDelayStorage_${index}[historyInit_${index}] = vec2<f32>(0.0, 0.0); }`,
      )
      .join('\n')
    const historyBefore = [
      ...historyRequirements.zAt.map((index) => `    if (iter == ${index}) { historyZAt_${index} = z; }`),
      ...historyRequirements.zDelay.map((index) =>
        index === 0
          ? `    let historyZDelay_0 = z;`
          : `    let historyZDelay_${index} = select(vec2<f32>(0.0, 0.0), historyZDelayStorage_${index}[u32(iter) % ${index}u], iter >= ${index});`,
      ),
    ].join('\n')
    const historyAfter = historyRequirements.zDelay
      .filter((index) => index > 0)
      .map((index) => `    historyZDelayStorage_${index}[u32(iter) % ${index}u] = z;`)
      .join('\n')

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
${historyDeclarations}
${historyInitialize}

  while (zq <= spec.bailout) {
    iter = iter + 1;
    if (iter == i32(spec.max_iter)) {
      return IterResult(${SHADER_CONSTANTS.IN_SET_INDEX}u, 0u, 0u, 0.0, 0.0);
    }

    // Custom iteration expression
    let n = f32(iter);
${historyBefore}
    let z_next = ${iterationExpr};
${historyAfter}

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

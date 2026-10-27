import { compileIterationFunction, getParsedExpression } from '../customFunctionParser.mjs'
import { jsExprToWGSL_safe } from '../wgslCompiler.mjs'

const assert = (condition, message) => { if (!condition) throw new Error(message) }
const probes = [[0.3, -0.7, -0.4, 0.6], [-0.8, 0.2, 0.7, -0.3], [0, 0, 1, 2]]
// Independent Cartesian reference for c * exp(i * (real + i * imaginary)).
function reference([zr, zi, cr, ci], real, imaginary, addSquare = false) {
  const scale = Math.exp(-imaginary)
  return [
    scale * (cr * Math.cos(real) - ci * Math.sin(real)) + (addSquare ? zr * zr - zi * zi : 0),
    scale * (cr * Math.sin(real) + ci * Math.cos(real)) + (addSquare ? 2 * zr * zi : 0),
  ]
}
const cases = [
  ['rotate(c, 0)', () => [0, 0]],
  ['rotate(c, pi/2)', () => [Math.PI / 2, 0]],
  ['rotate(c, i)', () => [0, 1]],
  ['rotate(c, -i)', () => [0, -1]],
  ['rotate(c, 0.4-0.7*i)', () => [0.4, -0.7]],
  ['rotate(c, z)', ([zr, zi]) => [zr, zi]],
  ['z*z + rotate(c, i*|Im(z)|)', ([, zi]) => [0, Math.abs(zi)], true],
  ['z*z + rotate(c, i*|Re(z)|)', ([zr]) => [0, Math.abs(zr)], true],
]
function close(actual, expected, tolerance, label) {
  for (let i = 0; i < 2; i++) assert(
    Number.isFinite(actual[i]) && Math.abs(actual[i] - expected[i]) <= tolerance * Math.max(1, Math.abs(expected[i])),
    `${label}: component ${i}, got ${actual[i]}, expected ${expected[i]}`,
  )
}

export function checkComplexRotateCpu() {
  for (const [expr, angle, addSquare] of cases) {
    // Exercise both the optimized compiler and the complexity > 10 fallback.
    for (const suffix of ['', ' + 0 + 0 + 0 + 0 + 0 + 0 + 0 + 0 + 0 + 0 + 0']) {
      const fn = compileIterationFunction(expr + suffix)
      for (const probe of probes) close(fn(...probe, 0), reference(probe, ...angle(probe), addSquare), 1e-12, expr + suffix)
    }
  }
}

export async function checkComplexRotateGpu(device) {
  device.pushErrorScope('validation')
  try {
    for (const [expr, angle, addSquare] of cases) {
      const expression = jsExprToWGSL_safe(getParsedExpression(expr))
      const module = device.createShaderModule({ code: `
        @group(0) @binding(0) var<storage, read_write> output: array<vec2<f32>>;
        @compute @workgroup_size(1) fn main(@builtin(global_invocation_id) id: vec3<u32>) {
          let zs = array<vec2<f32>, 3>(${probes.map(p => `vec2<f32>(${p[0]}, ${p[1]})`).join(',')});
          let cs = array<vec2<f32>, 3>(${probes.map(p => `vec2<f32>(${p[2]}, ${p[3]})`).join(',')});
          let z = zs[id.x]; let c = cs[id.x];
          output[id.x] = ${expression};
        }` })
      const pipeline = await device.createComputePipelineAsync({ layout: 'auto', compute: { module, entryPoint: 'main' } })
      const output = device.createBuffer({ size: 24, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC })
      const readback = device.createBuffer({ size: 24, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST })
      try {
        const encoder = device.createCommandEncoder()
        const pass = encoder.beginComputePass()
        pass.setPipeline(pipeline)
        pass.setBindGroup(0, device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries: [{ binding: 0, resource: { buffer: output } }] }))
        pass.dispatchWorkgroups(probes.length)
        pass.end()
        encoder.copyBufferToBuffer(output, 0, readback, 0, 24)
        device.queue.submit([encoder.finish()])
        await readback.mapAsync(GPUMapMode.READ)
        const values = new Float32Array(readback.getMappedRange())
        probes.forEach((probe, i) => close(values.subarray(2 * i, 2 * i + 2), reference(probe, ...angle(probe), addSquare), 2e-6, `GPU ${expr}`))
        readback.unmap()
      } finally {
        output.destroy()
        readback.destroy()
      }
    }
  } finally {
    const error = await device.popErrorScope()
    assert(!error, error?.message)
  }
}

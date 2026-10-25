/** This file is part of Mandelbrot Formula Explorer. Licensed under GPL-3.0. */
// Convert the exact two-word histograms on the GPU. Read back only the three
// Float32 channels the display uses, rather than six integer channels.
export function createGpuDensityReadback(device, lowBuffers, highBuffers, size, scale) {
  if (size * 3 > (device.limits?.maxStorageBufferBindingSize || 134217728) ||
    Math.ceil(size / 4 / 256) > (device.limits?.maxComputeWorkgroupsPerDimension || 65535)) return null
  const output = device.createBuffer({ size: size * 3,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC })
  try {
    const module = device.createShaderModule({ code: `
${[...lowBuffers, ...highBuffers].map((_, i) =>
  `@group(0) @binding(${i}) var<storage, read> counts${i}: array<u32>;`).join('\n')}
@group(0) @binding(6) var<storage, read_write> density: array<f32>;
// Round the integer only once. Casting each word before adding can double
// round the low word at a Float32 midpoint after the counter carries.
fn wideToFloat(low: u32, high: u32) -> f32 {
  if (high == 0u) { return f32(low); }
  let shift = firstLeadingBit(high) + 9u;
  var top: u32;
  var roundUp: bool;
  if (shift < 32u) {
    top = (high << (32u - shift)) | (low >> shift);
    let rest = low & ((1u << shift) - 1u);
    let half = 1u << (shift - 1u);
    roundUp = rest > half || (rest == half && (top & 1u) != 0u);
  } else if (shift == 32u) {
    top = high;
    roundUp = low > 0x80000000u || (low == 0x80000000u && (top & 1u) != 0u);
  } else {
    let highShift = shift - 32u;
    top = high >> highShift;
    let rest = high & ((1u << highShift) - 1u);
    let half = 1u << (highShift - 1u);
    roundUp = rest > half || (rest == half && (low != 0u || (top & 1u) != 0u));
  }
  if (roundUp) { top = top + 1u; }
  return f32(top) * exp2(f32(shift));
}
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= ${size / 4}u) { return; }
  ${lowBuffers.map((_, ch) => `density[i + ${ch * size / 4}u] =
    wideToFloat(counts${ch}[i], counts${ch + 3}[i]) / ${scale}.0;`).join('\n')}
}
` })
    const pipeline = device.createComputePipeline({ layout: 'auto', compute: { module, entryPoint: 'main' } })
    const group = device.createBindGroup({ layout: pipeline.getBindGroupLayout(0),
      entries: [...lowBuffers, ...highBuffers, output].map((buffer, binding) => ({ binding, resource: { buffer } })),
    })
    return {
      output,
      encode(encoder) {
        const pass = encoder.beginComputePass()
        pass.setPipeline(pipeline)
        pass.setBindGroup(0, group)
        pass.dispatchWorkgroups(Math.ceil(size / 4 / 256))
        pass.end()
      },
      destroy() { output.destroy() },
    }
  } catch (error) { output.destroy(); throw error }
}

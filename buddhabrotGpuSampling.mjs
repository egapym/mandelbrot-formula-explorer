/** This file is part of Mandelbrot Formula Explorer. Licensed under GPL-3.0. */
// Fast/slow queues contain original sample IDs, never new random coordinates.
// Each replay starts from the original state, including n/zAt/zDelay history.
export function createGpuSamplingPipelines(device, code, buffers, capacity, compact, highBuffers = []) {
  let ids, counts
  const destroy = () => { ids?.destroy(); counts?.destroy() }
  try {
    if (compact) {
      ids = device.createBuffer({ label: 'Buddhabrot sample IDs', size: capacity * 8, usage: GPUBufferUsage.STORAGE })
      counts = device.createBuffer({ label: 'Buddhabrot sample counts', size: 8, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST })
    }
    const module = device.createShaderModule({ code })
    const entries = buffers.map((buffer, binding) => ({ binding, resource: { buffer } }))
    if (compact) entries.push({ binding: 6, resource: { buffer: ids } }, { binding: 7, resource: { buffer: counts } })
    entries.push(...highBuffers.map((buffer, i) => ({ binding: 8 + i, resource: { buffer } })))
    const makePass = (entryPoint, constants, bindings = entries) => {
      const pipeline = device.createComputePipeline({ layout: 'auto', compute: { module, entryPoint, constants } })
      return { pipeline, group: device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries: bindings }) }
    }
    const passes = compact ? [
      makePass('classify', {}, [entries[0], entries[6], entries[7]]),
      makePass('main', { queueKind: 0 }), makePass('main', { queueKind: 1 }),
    ] : [makePass('main', {})]
    return { passes, counts, destroy }
  } catch (error) { destroy(); throw error }
}

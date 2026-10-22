import assert from 'node:assert/strict'
import { createGpuSamplingPipelines } from '../buddhabrotGpuSampling.mjs'
globalThis.GPUBufferUsage = { STORAGE: 1, COPY_DST: 2 }
function device(failAt = Infinity) {
  const buffers = [], entries = []
  let pipelines = 0
  return {
    buffers, entries,
    createBuffer(options) {
      const buffer = { ...options, destroyed: 0, destroy() { this.destroyed++ } }
      buffers.push(buffer)
      return buffer
    },
    createShaderModule: () => ({}),
    createComputePipeline({ compute }) {
      if (++pipelines === failAt) throw new Error('GPU pipeline unavailable')
      return { compute, getBindGroupLayout: () => ({}) }
    },
    createBindGroup({ entries: bindings }) { entries.push(bindings); return {} },
  }
}
const uniformAndDensity = Array.from({ length: 6 }, () => ({}))
const gpu = device()
const sampling = createGpuSamplingPipelines(gpu, 'shader', uniformAndDensity, 480000, true)
assert.deepEqual(sampling.passes.map(p => p.pipeline.compute.entryPoint), ['classify', 'main', 'main'])
assert.deepEqual(gpu.entries[0].map(e => e.binding), [0, 6, 7], 'Classification must not write density')
assert.equal(gpu.buffers[0].size, 480000 * 8)
assert.equal(sampling.passes[1].pipeline.compute.constants.queueKind, 0)
assert.equal(sampling.passes[2].pipeline.compute.constants.queueKind, 1)
sampling.destroy()
assert.ok(gpu.buffers.every(b => b.destroyed === 1), 'Completion/Stop must release both queues')
for (const failedPipeline of [1, 2, 3]) {
  const broken = device(failedPipeline)
  assert.throws(() => createGpuSamplingPipelines(broken, 'shader', uniformAndDensity, 480000, true))
  assert.ok(broken.buffers.every(b => b.destroyed === 1), 'Partial setup failure leaked queues')
}
const small = device()
const fallback = createGpuSamplingPipelines(small, 'shader', uniformAndDensity, 16, false)
assert.equal(fallback.passes.length, 1)
assert.equal(small.buffers.length, 0, 'Small/limited GPU jobs must avoid queue allocation')
fallback.destroy()
console.log('PASS: generic GPU queue binding, bounded capacity, cleanup on Stop/completion/setup failure and small-job path')

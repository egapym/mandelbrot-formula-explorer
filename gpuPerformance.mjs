// Opt-in diagnostics. No samples or timestamp buffers are created when disabled.
export function beginGpuProfile(renderer) {
  if (globalThis.fractalGpuPerformance?.enabled !== true) return null
  return { renderer, started: performance.now() }
}

export function endGpuProfile(profile, timings = {}) {
  if (!profile) return
  const diagnostics = globalThis.fractalGpuPerformance
  if (!diagnostics) return
  const samples = (diagnostics.samples ||= [])
  samples.push({ renderer: profile.renderer, totalMs: performance.now() - profile.started, ...timings })
  if (samples.length > 120) samples.splice(0, samples.length - 120)
}

// This records compute-pass time, separately from queueing, copying and mapping.
export function createGpuTimer(device, profile) {
  if (!profile || !device.features.has('timestamp-query')) return null
  const querySet = device.createQuerySet({ type: 'timestamp', count: 2 })
  const resolve = device.createBuffer({ size: 16, usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC })
  const read = device.createBuffer({ size: 16, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ })
  return {
    timestampWrites: { querySet, beginningOfPassWriteIndex: 0, endOfPassWriteIndex: 1 },
    encode(encoder) {
      encoder.resolveQuerySet(querySet, 0, 2, resolve, 0)
      encoder.copyBufferToBuffer(resolve, 0, read, 0, 16)
    },
    async result() {
      await read.mapAsync(GPUMapMode.READ)
      const timestamps = new BigUint64Array(read.getMappedRange())
      const ms = Number(timestamps[1] - timestamps[0]) / 1e6
      read.unmap()
      return ms
    },
    destroy() {
      querySet.destroy()
      resolve.destroy()
      read.destroy()
    },
  }
}

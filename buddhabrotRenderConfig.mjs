export const DEFAULT_RENDER_POINT_BATCH_SIZE = 64
export const MAX_RENDER_POINT_BATCH_SIZE = 1024

export function getRenderPointBatchLimit(samples) {
  const count = Number(samples)
  if (!Number.isFinite(count)) return MAX_RENDER_POINT_BATCH_SIZE
  // The runner distributes this global trajectory budget across all workers.
  return Math.max(1, Math.min(MAX_RENDER_POINT_BATCH_SIZE, Math.floor(Math.max(0, count))))
}

export function normalizeRenderPointBatchSize(value, maximum = MAX_RENDER_POINT_BATCH_SIZE) {
  const size = Number(value)
  const limit = getRenderPointBatchLimit(maximum)
  return Number.isFinite(size)
    ? Math.min(limit, Math.max(1, Math.floor(size)))
    : Math.min(limit, DEFAULT_RENDER_POINT_BATCH_SIZE)
}

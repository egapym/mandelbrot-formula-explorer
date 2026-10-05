export const DEFAULT_RENDER_POINT_BATCH_SIZE = 64
export const MAX_RENDER_POINT_BATCH_SIZE = 1024

export function getRenderPointBatchLimit(samples, workerCount = 1) {
  const count = Number(samples)
  if (!Number.isFinite(count)) return MAX_RENDER_POINT_BATCH_SIZE
  const workers = Math.max(1, Math.floor(Number(workerCount) || 1))
  return Math.max(1, Math.min(MAX_RENDER_POINT_BATCH_SIZE, Math.ceil(Math.max(0, count) / workers)))
}

export function normalizeRenderPointBatchSize(value, maximum = MAX_RENDER_POINT_BATCH_SIZE) {
  const size = Number(value)
  const limit = getRenderPointBatchLimit(maximum)
  return Number.isFinite(size)
    ? Math.min(limit, Math.max(1, Math.floor(size)))
    : Math.min(limit, DEFAULT_RENDER_POINT_BATCH_SIZE)
}

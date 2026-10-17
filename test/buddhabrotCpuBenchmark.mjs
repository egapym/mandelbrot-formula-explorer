// Standalone benchmark; no timing instrumentation is added to the application.
// BUDDHA_REFERENCE_WORKER=/path/to/old-worker.mjs node test/buddhabrotCpuBenchmark.mjs
import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import { Worker } from 'node:worker_threads'
import { BUDDHA_PALETTES } from '../buddhaPalettes.mjs'

const workerUrl = new URL('../buddhabrotWorker.mjs', import.meta.url)
const variants = [['current', workerUrl]]
if (process.env.BUDDHA_REFERENCE_WORKER) variants.unshift(['before', process.env.BUDDHA_REFERENCE_WORKER])
const cases = [
  ['Mandelbrot', { mode: 'buddha' }],
  ['Mandelbrot formula', { mode: 'buddha', iterationFunction: 'z*z+c' }],
  ['Rotated', { mode: 'buddha', iterationFunction: 'z*z+c*i' }],
  ['Translated', { mode: 'buddha', iterationFunction: 'z*z+c+i' }],
  ['Anti Mandelbrot', { mode: 'antibuddha' }],
  ['Custom Anti', { mode: 'antibuddha', iterationFunction: 'c*(z+1/(z^2))', z0Real: 1 }],
  ['History Anti', { mode: 'antibuddha', iterationFunction: 'z*z+c+0.1*zDelay(2)' }],
  ['Julia', { mode: 'buddha', fractalType: 'julia', juliaRe: -0.7, juliaIm: 0.2 }],
]
const selectedCases = process.env.BUDDHA_BENCH_CASES
  ? cases.filter(([label]) => new RegExp(process.env.BUDDHA_BENCH_CASES).test(label))
  : cases
const repetitions = Number(process.env.BUDDHA_BENCH_REPEATS) || 3
const results = []
async function createVariant([name, path]) {
  const source = readFileSync(path, 'utf8').replace(
    /from '(\.\/[^']+)'/g,
    (_, specifier) => `from '${new URL(specifier, workerUrl).href}'`,
  )
  const worker = new Worker(
    `
    const { parentPort } = require('node:worker_threads');
    globalThis.self = {};
    globalThis.postMessage = (message, transfers) => parentPort.postMessage(message, transfers);
    Math.random = () => 0.25;
    import(${JSON.stringify(`data:text/javascript;base64,${Buffer.from(source).toString('base64')}`)})
      .then(() => {
        parentPort.on('message', data => self.onmessage({ data }));
        parentPort.postMessage({ type: 'ready' });
      });
  `,
    { eval: true },
  )
  const waitFor = (type) =>
    new Promise((resolve, reject) => {
      const receive = (message) => {
        if (message.type !== type) return
        worker.off('message', receive)
        worker.off('error', reject)
        resolve(message)
      }
      worker.on('message', receive)
      worker.once('error', reject)
    })
  await waitFor('ready')
  return { name, worker, waitFor }
}

const activeVariants = await Promise.all(variants.map(createVariant))
try {
  for (const [label, options] of selectedCases) {
    const runs = activeVariants.map((variant) => ({ ...variant, times: [], chunks: 0, checksum: 0 }))
    for (const run of runs) {
      run.collect = (message) => {
        if (message.type !== 'chunk') return
        run.chunks++
        for (const channel of ['r', 'g', 'b']) {
          const values = message.chunk[channel]
          for (let i = 0; i < values.length; i++) run.checksum += values[i]
        }
      }
      run.worker.on('message', run.collect)
    }
    // Alternate versions so JIT warmup, temperature and background work do not
    // systematically favor the version measured later. Exclude two warmups.
    for (let trial = 0; trial < repetitions + 2; trial++) {
      for (const run of trial % 2 ? runs.toReversed() : runs) {
        run.chunks = 0
        run.checksum = 0
        const done = run.waitFor('done')
        const start = process.hrtime.bigint()
        run.worker.postMessage({
          cmd: 'start',
          jobId: `${label}:${trial}`,
          width: 640,
          height: 360,
          samples: Number(process.env.BUDDHA_BENCH_SAMPLES) || 10000,
          maxIter: 1000,
          escapeRadius: 4,
          renderDelay: 0,
          paletteStops: BUDDHA_PALETTES[0],
          buddhaBandMode: 'perPoint',
          ...options,
        })
        await done
        if (trial >= 2) run.times.push(Number(process.hrtime.bigint() - start) / 1e6)
      }
    }
    for (const run of runs) {
      run.worker.off('message', run.collect)
      assert(run.checksum > 0, `${label}: missing density`)
      run.times.sort((a, b) => a - b)
      results.push({
        variant: run.name,
        case: label,
        medianMs: Math.round(run.times[Math.floor(run.times.length / 2)] * 10) / 10,
        chunks: run.chunks,
        densitySum: Math.round(run.checksum),
      })
    }
  }
} finally {
  await Promise.all(activeVariants.map(({ worker }) => worker.terminate()))
}
console.table(results)

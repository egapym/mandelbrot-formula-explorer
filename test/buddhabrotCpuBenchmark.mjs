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
  ['Anti Mandelbrot', { mode: 'antibuddha' }],
  ['Custom Anti', { mode: 'antibuddha', iterationFunction: 'c*(z+1/(z^2))', z0Real: 1 }],
  ['History Anti', { mode: 'antibuddha', iterationFunction: 'z*z+c+0.1*zDelay(2)' }],
  ['Julia', { mode: 'buddha', fractalType: 'julia', juliaRe: -0.7, juliaIm: 0.2 }],
]
const repetitions = Number(process.env.BUDDHA_BENCH_REPEATS) || 3
const results = []
for (const [name, path] of variants) {
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
  try {
    for (const [label, options] of cases) {
      const times = []
      let chunks = 0
      let checksum = 0
      const collect = (message) => {
        if (message.type !== 'chunk') return
        chunks++
        for (const channel of ['r', 'g', 'b']) {
          const values = message.chunk[channel]
          for (let i = 0; i < values.length; i++) checksum += values[i]
        }
      }
      worker.on('message', collect)
      for (let trial = 0; trial <= repetitions; trial++) {
        chunks = 0
        checksum = 0
        const done = waitFor('done')
        const start = process.hrtime.bigint()
        worker.postMessage({
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
        if (trial > 0) times.push(Number(process.hrtime.bigint() - start) / 1e6)
      }
      worker.off('message', collect)
      assert(checksum > 0, `${label}: missing density`)
      times.sort((a, b) => a - b)
      results.push({
        variant: name,
        case: label,
        medianMs: Math.round(times[Math.floor(times.length / 2)]),
        chunks,
        densitySum: Math.round(checksum),
      })
    }
  } finally {
    await worker.terminate()
  }
}
console.table(results)

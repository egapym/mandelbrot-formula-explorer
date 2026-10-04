/**
 * @author Bert Baron
 * @modified 2026-01-30
 */

import { functionCache } from './customFunctionParser.mjs'
import * as fxp from './fxp.mjs'
import { MandelbrotCustom } from './mandelbrotCustom.mjs'
import { MandelbrotFloat } from './mandelbrotFloat.mjs'
import { MandelbrotPerturbation } from './mandelbrotPerturbation.mjs'
import { MandelbrotPerturbationExtFloat } from './mandelbrotPerturbationExtFloat.mjs'
import { WorkerContext } from './workerContext.mjs'

const ctx = new WorkerContext()

async function initMandelbrotFloat() {
  // 不要なイベントループ待ちを避けるため、そのまま初期化する
  return new MandelbrotFloat(ctx)
}

async function initMandelbrotPerturbation() {
  return new MandelbrotPerturbation(ctx)
}

async function initMandelbrotPerturbationExtFloat() {
  return new MandelbrotPerturbationExtFloat(ctx)
}

async function initMandelbrotCustom() {
  return new MandelbrotCustom(ctx)
}

const mandelbrotFloat = initMandelbrotFloat()
const mandelbrotPerturbation = initMandelbrotPerturbation()
const mandelbrotPerturbationExtFloat = initMandelbrotPerturbationExtFloat()
const mandelbrotCustom = initMandelbrotCustom()

self.onmessage = handleMessage

let previousFractalType = null
let trapSpec = null

async function handleMessage(msg) {
  if (msg.data.type === 'batch') {
    if (msg.data.trapSpec) trapSpec = msg.data.trapSpec
    const answers = []
    for (const task of msg.data.tasks) {
      task.trapSpec = task.useTrap ? trapSpec : null
      answers.push(await processTask(parseMessage({ data: task })))
    }
    sendResult({ type: 'batch-answer', answers }, answers)
    return
  }
  const message = parseMessage(msg)

  if (message.type === 'task') {
    sendResult(await processTask(message))
  }
}

async function processTask(message) {
  // フラクタル種類が変わったら関数キャッシュを消す
  if (message.fractalType !== previousFractalType) {
    functionCache.clear()
    previousFractalType = message.fractalType
  }

  // カスタム系は専用実装を使う
  // Julia は深いズームを想定しないため常に Float 実装を使う
  const implPromise =
    message.fractalType === 'custom' || message.fractalType === 'julia-custom'
      ? mandelbrotCustom
      : message.fractalType === 'julia'
        ? mandelbrotFloat
        : message.requiredPrecision > 1020
          ? mandelbrotPerturbationExtFloat
          : message.requiredPrecision > 58
            ? mandelbrotPerturbation
            : mandelbrotFloat

  const impl = await implPromise

  ctx.initTask(message.jobToken)
  ctx.resetStats()
  return await impl.process(message)
}

function sendResult(result, answers = [result]) {
  // TypedArray のバッファを転送してコピーを避ける
  try {
    const transfers = []
    for (const answer of answers) {
      // Large trap/bitmap configuration stays in this Worker, rather than
      // being cloned back to the main thread with every tile.
      delete answer.task.trapSpec
      for (const name of ['values', 'smooth', 'signs', 'zreal', 'zimag', 'otData']) {
        if (answer[name]?.buffer) transfers.push(answer[name].buffer)
      }
    }
    if (transfers.length > 0) postMessage(result, transfers)
    else postMessage(result)
  } catch (_e) {
    // 転送に失敗したら通常の postMessage に戻す
    postMessage(result)
  }
}

function parseMessage(msg) {
  if (msg.data.type === 'task') {
    msg.data.frameTopLeft[0] = fxp.fromJSON(msg.data.frameTopLeft[0])
    msg.data.frameTopLeft[1] = fxp.fromJSON(msg.data.frameTopLeft[1])
    msg.data.frameBottomRight[0] = fxp.fromJSON(msg.data.frameBottomRight[0])
    msg.data.frameBottomRight[1] = fxp.fromJSON(msg.data.frameBottomRight[1])
  }
  return msg.data
}

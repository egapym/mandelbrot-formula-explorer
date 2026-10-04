import { compileIterationFunction, ITERATION_FUNCTION_GENERATION_CATALOG } from './customFunctionParser.mjs'

export const DEFAULT_RANDOM_ITERATION_FUNCTION_CONFIG = Object.freeze({
  leaves: ITERATION_FUNCTION_GENERATION_CATALOG.leaves.map(({ id }) => id),
  unaryFunctions: ITERATION_FUNCTION_GENERATION_CATALOG.unaryFunctions.map(({ id }) => id),
  binaryFunctions: ITERATION_FUNCTION_GENERATION_CATALOG.binaryFunctions.map(({ id }) => id),
  operators: Object.fromEntries(ITERATION_FUNCTION_GENERATION_CATALOG.operators.map((operator) => [operator, { min: 0, max: 1 }])),
  maxDepth: 4,
  numberMin: -2,
  numberMax: 2,
  historyMin: 1,
  historyMax: 20,
})

const LEAF_IDS = new Set(ITERATION_FUNCTION_GENERATION_CATALOG.leaves.map(({ id }) => id))
const UNARY_IDS = new Set(ITERATION_FUNCTION_GENERATION_CATALOG.unaryFunctions.map(({ id }) => id))
const BINARY_IDS = new Set(ITERATION_FUNCTION_GENERATION_CATALOG.binaryFunctions.map(({ id }) => id))
const MAX_AUTOMATIC_TREE_DEPTH = 8

function integerInRange(random, min, max) {
  return min + Math.floor(random() * (max - min + 1))
}

function shuffled(values, random) {
  const result = [...values]
  for (let i = result.length - 1; i > 0; i--) {
    const j = Math.floor(random() * (i + 1))
    ;[result[i], result[j]] = [result[j], result[i]]
  }
  return result
}

function normalizeInteger(value, fallback, minimum = 0) {
  const number = Number.parseInt(value, 10)
  return Number.isFinite(number) ? Math.max(minimum, number) : fallback
}

function normalizeNumber(value, fallback) {
  const number = Number(value)
  return Number.isFinite(number) ? number : fallback
}

export function normalizeRandomIterationFunctionConfig(config = {}) {
  const defaults = DEFAULT_RANDOM_ITERATION_FUNCTION_CONFIG
  const uniqueKnown = (values, known, fallback) => {
    const list = Array.isArray(values) ? values.filter((value) => known.has(value)) : fallback
    return [...new Set(list)]
  }
  let numberMin = normalizeNumber(config.numberMin ?? config.coefficientMin, defaults.numberMin)
  let numberMax = normalizeNumber(config.numberMax ?? config.coefficientMax, defaults.numberMax)
  if (numberMin > numberMax) [numberMin, numberMax] = [numberMax, numberMin]
  let historyMin = normalizeInteger(config.historyMin, defaults.historyMin, 0)
  let historyMax = normalizeInteger(config.historyMax, defaults.historyMax, 0)
  if (historyMin > historyMax) [historyMin, historyMax] = [historyMax, historyMin]

  const operators = {}
  for (const operator of ITERATION_FUNCTION_GENERATION_CATALOG.operators) {
    const requested = config.operators?.[operator] || defaults.operators[operator]
    let min = normalizeInteger(requested.min, defaults.operators[operator].min, 0)
    let max = normalizeInteger(requested.max, defaults.operators[operator].max, 0)
    if (min > max) [min, max] = [max, min]
    operators[operator] = { min, max }
  }

  return {
    leaves: uniqueKnown(config.leaves, LEAF_IDS, defaults.leaves),
    unaryFunctions: uniqueKnown(config.unaryFunctions, UNARY_IDS, defaults.unaryFunctions),
    binaryFunctions: uniqueKnown(config.binaryFunctions, BINARY_IDS, defaults.binaryFunctions),
    operators,
    maxDepth: normalizeInteger(config.maxDepth, defaults.maxDepth, 1),
    numberMin,
    numberMax,
    historyMin,
    historyMax,
  }
}

function numberLiteral(random, config) {
  const raw = config.numberMin + random() * (config.numberMax - config.numberMin)
  const rounded = Math.round(raw * 1000) / 1000
  if (Object.is(rounded, -0)) return '0'
  // 負数を / や ^ の右辺に置く場合も、単一の数値として解釈させる。
  // 括弧がないと式パーサーは符号を二項演算子として扱う。
  return rounded < 0 ? `(${rounded})` : String(rounded)
}

function removeNeutralArithmetic(expression, random) {
  const replacement = () => (random() < 0.5 ? 'z' : 'c')
  const one = '1(?:\\.0+)?'
  const zero = '0(?:\\.0+)?'
  let result = expression

  // ジェネレーターは二項式を必ず括弧で直列化するため、ここでは
  // リテラルの恒等要素だけを置換し、部分式には触れない。
  result = result.replace(new RegExp(`\\(\\s*${one}\\s*\\*`, 'g'), () => `(${replacement()} *`)
  result = result.replace(new RegExp(`\\*\\s*${one}\\s*\\)`, 'g'), () => `* ${replacement()})`)
  result = result.replace(new RegExp(`\\/\\s*${one}\\s*\\)`, 'g'), () => `/ ${replacement()})`)
  result = result.replace(new RegExp(`\\^\\s*${one}\\s*\\)`, 'g'), () => `^ ${replacement()})`)
  result = result.replace(new RegExp(`\\(\\s*${zero}\\s*\\+`, 'g'), () => `(${replacement()} +`)
  result = result.replace(new RegExp(`\\+\\s*${zero}\\s*\\)`, 'g'), () => `+ ${replacement()})`)
  result = result.replace(new RegExp(`-\\s*${zero}\\s*\\)`, 'g'), () => `- ${replacement()})`)
  return result
}

function leaf(random, config) {
  const candidates = config.leaves.length ? config.leaves : ['number']
  const id = candidates[integerInRange(random, 0, candidates.length - 1)]
  return id === 'number' ? numberLiteral(random, config) : id
}

function operand(random, config) {
  const historyFunctions = config.unaryFunctions.filter((id) => id === 'zAt' || id === 'zDelay')
  if (historyFunctions.length > 0 && random() < 0.25) {
    const id = historyFunctions[integerInRange(random, 0, historyFunctions.length - 1)]
    return formatUnary(id, '', random, config)
  }
  return leaf(random, config)
}

function formatUnary(id, expression, random, config) {
  if (id === 'abs') return `|${expression}|`
  if (id === 'zAt' || id === 'zDelay') return `${id}(${integerInRange(random, config.historyMin, config.historyMax)})`
  return `${id}(${expression})`
}

function formatBinaryFunction(id, left, right) {
  return `${id}(${left}, ${right})`
}

function pickRegularFunction(random, config) {
  const unaryFunctions = config.unaryFunctions.filter((id) => id !== 'zAt' && id !== 'zDelay')
  const binaryFunctions = config.binaryFunctions
  if (unaryFunctions.length === 0 && binaryFunctions.length === 0) return null

  // カテゴリを先に選び、要素数の多い Unary Functions によって
  // Binary Functions が選ばれにくくならないようにする。
  const useUnary = unaryFunctions.length > 0 && (binaryFunctions.length === 0 || random() < 0.5)
  const candidates = useUnary ? unaryFunctions : binaryFunctions
  return { id: candidates[integerInRange(random, 0, candidates.length - 1)], isUnary: useUnary }
}

function applyRegularFunction(expression, selectedFunction, random, config) {
  if (selectedFunction.isUnary) return formatUnary(selectedFunction.id, expression, random, config)
  return formatBinaryFunction(selectedFunction.id, expression, operand(random, config))
}

function buildBalancedTree(leaves, operators) {
  if (leaves.length === 1) return leaves[0]
  const middle = Math.floor(leaves.length / 2)
  const left = buildBalancedTree(leaves.slice(0, middle), operators.slice(0, middle - 1))
  const right = buildBalancedTree(leaves.slice(middle), operators.slice(middle))
  const operator = operators[middle - 1]
  return `(${left} ${operator} ${right})`
}

function minimumDepthForOperators(operatorCount) {
  return Math.ceil(Math.log2(operatorCount + 1)) + 1
}

/**
 * Generates a syntactically valid expression. The injected random source keeps
 * the generator deterministic in tests while the UI uses Math.random.
 */
export function generateRandomIterationFunction(configInput = {}, random = Math.random) {
  const config = normalizeRandomIterationFunctionConfig(configInput)
  const adjustments = []
  const eligibleOperators = ITERATION_FUNCTION_GENERATION_CATALOG.operators.filter(
    (operator) => config.operators[operator].max > 0,
  )
  const availableBinaryFunctions = config.binaryFunctions
  const regularFunctionAvailable = Boolean(pickRegularFunction(() => 0, config))

  const targetCounts = {}
  for (const operator of ITERATION_FUNCTION_GENERATION_CATALOG.operators) {
    const range = config.operators[operator]
    targetCounts[operator] = integerInRange(random, range.min, range.max)
  }

  let compositionOperator = eligibleOperators[integerInRange(random, 0, Math.max(0, eligibleOperators.length - 1))]
  if (!compositionOperator && availableBinaryFunctions.length === 0) {
    compositionOperator = '+'
    targetCounts['+'] = Math.max(targetCounts['+'], 1)
    adjustments.push('No binary composition was selected, so + was added temporarily.')
  }
  if (compositionOperator) targetCounts[compositionOperator] = Math.max(targetCounts[compositionOperator], 1)

  let requestedOperatorCount = Object.values(targetCounts).reduce((total, count) => total + count, 0)
  const requestedDepth = minimumDepthForOperators(requestedOperatorCount)
  let effectiveDepth = config.maxDepth
  if (requestedDepth > effectiveDepth) {
    effectiveDepth = Math.min(requestedDepth, MAX_AUTOMATIC_TREE_DEPTH)
    adjustments.push(`Maximum depth was raised from ${config.maxDepth} to ${effectiveDepth}.`)
  }

  const maximumOperators = 2 ** (effectiveDepth - 1) - 1
  if (requestedOperatorCount > maximumOperators) {
    let excess = requestedOperatorCount - maximumOperators
    // First discard optional occurrences above each selected lower bound.
    for (const operator of ITERATION_FUNCTION_GENERATION_CATALOG.operators) {
      const removable = Math.max(0, targetCounts[operator] - config.operators[operator].min)
      const removed = Math.min(excess, removable)
      targetCounts[operator] -= removed
      excess -= removed
      if (excess === 0) break
    }
    if (excess > 0) {
      // The requested lower bounds themselves cannot fit even after depth
      // expansion, so relax them only as a final fallback.
      for (const operator of ITERATION_FUNCTION_GENERATION_CATALOG.operators) {
        const protectedCount = operator === compositionOperator ? 1 : 0
        const removable = Math.max(0, targetCounts[operator] - protectedCount)
        const removed = Math.min(excess, removable)
        targetCounts[operator] -= removed
        excess -= removed
        if (excess === 0) break
      }
      adjustments.push('Operator minimum counts were reduced after reaching the automatic depth limit.')
    }
    requestedOperatorCount = Object.values(targetCounts).reduce((total, count) => total + count, 0)
  }

  const operatorPool = []
  for (const operator of ITERATION_FUNCTION_GENERATION_CATALOG.operators) {
    for (let count = 0; count < targetCounts[operator]; count++) operatorPool.push(operator)
  }

  // 関数の引数も単項式に偏らないよう、演算子を1つ引数用に予約する。
  // 残り1個の演算子は z と c を主式で結ぶために残す。
  const functionArgumentOperators = []
  if (regularFunctionAvailable && operatorPool.length >= 2) {
    const index = integerInRange(random, 0, operatorPool.length - 1)
    functionArgumentOperators.push(operatorPool.splice(index, 1)[0])
  }

  // 選択済みの二項関数は、演算子を消費せず z と c を結合できる。
  let expression
  let requiredDepth = 2
  if (operatorPool.length === 0 && availableBinaryFunctions.length > 0) {
    const selectedFunction = pickRegularFunction(random, { ...config, unaryFunctions: [], binaryFunctions: availableBinaryFunctions })
    expression = formatBinaryFunction(selectedFunction.id, 'z', 'c')
  } else {
    requiredDepth = minimumDepthForOperators(operatorPool.length)
    const leaves = ['z', 'c']
    while (leaves.length < operatorPool.length + 1) leaves.push(operand(random, config))

    // zAt/zDelay は葉として扱い、それ以外の選択済み関数は葉・部分式へ
    // 注入する。式全体を最後に囲わないため、関数の出現位置が固定されない。
    if (regularFunctionAvailable) {
      if (effectiveDepth <= requiredDepth) {
        const previousDepth = effectiveDepth
        effectiveDepth = Math.min(MAX_AUTOMATIC_TREE_DEPTH + 1, requiredDepth + 1)
        adjustments.push(`Maximum depth was raised from ${previousDepth} to ${effectiveDepth} to include a function.`)
      }

      const addFunctionToRandomLeaf = () => {
        const leafIndex = integerInRange(random, 0, leaves.length - 1)
        const argumentOperator = functionArgumentOperators.shift()
        const argument = argumentOperator
          ? `(${leaves[leafIndex]} ${argumentOperator} ${operand(random, config)})`
          : leaves[leafIndex]
        leaves[leafIndex] = applyRegularFunction(argument, pickRegularFunction(random, config), random, config)
      }
      addFunctionToRandomLeaf()

      let remainingDepth = effectiveDepth - requiredDepth - 1
      while (remainingDepth > 0 && random() < 0.5) {
        addFunctionToRandomLeaf()
        remainingDepth--
      }
    }
    expression = buildBalancedTree(shuffled(leaves, random), shuffled(operatorPool, random))
  }

  expression = removeNeutralArithmetic(expression, random)
  // compileIterationFunction is the source-of-truth syntax validation.
  compileIterationFunction(expression)
  return { expression, adjustments, config }
}

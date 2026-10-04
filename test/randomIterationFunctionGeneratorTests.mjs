import assert from 'node:assert/strict'
import { compileIterationFunction, ITERATION_FUNCTION_GENERATION_CATALOG } from '../customFunctionParser.mjs'
import { generateRandomIterationFunction } from '../randomIterationFunctionGenerator.mjs'

function sequence(values) {
  let index = 0
  return () => values[index++ % values.length]
}

function operators(min = 0, max = 0) {
  return Object.fromEntries(['+', '-', '*', '/', '^'].map((operator) => [operator, { min, max }]))
}

function countOperator(expression, operator) {
  return [...expression].filter((character) => character === operator).length
}

function testSelectedOperatorsAndRequiredTerms() {
  const selectedOperators = operators(1, 1)
  const result = generateRandomIterationFunction(
    {
      leaves: ['z', 'c'],
      unaryFunctions: [],
      binaryFunctions: [],
      operators: selectedOperators,
      maxDepth: 4,
      numberMin: -2,
      numberMax: 2,
    },
    sequence([0.1, 0.4, 0.8, 0.2, 0.6]),
  )

  assert.match(result.expression, /\bz\b/)
  assert.match(result.expression, /\bc\b/)
  for (const operator of Object.keys(selectedOperators)) {
    assert.equal(countOperator(result.expression, operator), 1, `${operator} occurrence count`)
  }
  assert.doesNotThrow(() => compileIterationFunction(result.expression))
}

function testHistoryRangeAndParserValidation() {
  const result = generateRandomIterationFunction(
    {
      leaves: ['z', 'c'],
      unaryFunctions: ['zAt'],
      binaryFunctions: [],
      operators: { ...operators(), '+': { min: 2, max: 2 } },
      maxDepth: 4,
      historyMin: 3,
      historyMax: 3,
    },
    () => 0,
  )
  assert.match(result.expression, /zAt\(3\)/)
  assert.match(result.expression, /\bz\b/)
  assert.match(result.expression, /\bc\b/)
  assert.doesNotThrow(() => compileIterationFunction(result.expression))
}

function testNumberRangeAndNeutralArithmeticRemoval() {
  const result = generateRandomIterationFunction(
    {
      leaves: ['number'],
      unaryFunctions: [],
      binaryFunctions: [],
      operators: { ...operators(), '+': { min: 2, max: 2 } },
      numberMin: 1,
      numberMax: 1,
    },
    () => 0.9,
  )
  assert.match(result.expression, /\b1\b/)
  assert.doesNotMatch(result.expression, /\(\s*1\s*\*|\*\s*1\s*\)|\/\s*1\s*\)|\^\s*1\s*\)|\+\s*0\s*\)|-\s*0\s*\)/)
  assert.doesNotThrow(() => compileIterationFunction(result.expression))
}

function testNeutralNumberLiteralsAreNeverGeneratedAsIdentities() {
  const cases = [
    { operator: '*', number: 1, neutral: /\(\s*1\s*\*|\*\s*1\s*\)/ },
    { operator: '/', number: 1, neutral: /\/\s*1\s*\)/ },
    { operator: '^', number: 1, neutral: /\^\s*1\s*\)/ },
    { operator: '+', number: 0, neutral: /\(\s*0\s*\+|\+\s*0\s*\)/ },
    { operator: '-', number: 0, neutral: /-\s*0\s*\)/ },
  ]

  for (const { operator, number, neutral } of cases) {
    const result = generateRandomIterationFunction(
      {
        leaves: ['number'],
        unaryFunctions: [],
        binaryFunctions: [],
        operators: { ...operators(), [operator]: { min: 2, max: 2 } },
        numberMin: number,
        numberMax: number,
      },
      () => 0,
    )
    assert.doesNotMatch(result.expression, neutral, `${operator} must not receive its neutral number literal`)
    assert.doesNotThrow(() => compileIterationFunction(result.expression))
  }
}

function testAutomaticRelaxation() {
  const result = generateRandomIterationFunction(
    {
      leaves: ['z', 'c'],
      unaryFunctions: [],
      binaryFunctions: [],
      operators: operators(),
      maxDepth: 1,
    },
    () => 0,
  )
  assert.match(result.expression, /\+/)
  assert.match(result.adjustments.join(' '), /\+ was added temporarily/)
  assert.doesNotThrow(() => compileIterationFunction(result.expression))

  const depthResult = generateRandomIterationFunction(
    {
      leaves: ['z', 'c'],
      unaryFunctions: [],
      binaryFunctions: [],
      operators: { ...operators(), '+': { min: 2, max: 2 } },
      maxDepth: 1,
    },
    () => 0,
  )
  assert.match(depthResult.adjustments.join(' '), /Maximum depth was raised/)
}

function testEveryRegularFunctionCanBeGenerated() {
  const baseConfig = {
    leaves: ['z', 'c'],
    operators: { ...operators(), '+': { min: 1, max: 1 } },
    maxDepth: 2,
  }
  const regularUnaryFunctions = ITERATION_FUNCTION_GENERATION_CATALOG.unaryFunctions.filter(
    ({ id }) => id !== 'zAt' && id !== 'zDelay',
  )

  for (const { id } of regularUnaryFunctions) {
    const result = generateRandomIterationFunction(
      { ...baseConfig, unaryFunctions: [id], binaryFunctions: [] },
      () => 0,
    )
    const marker = id === 'abs' ? /\|/ : new RegExp(`\\b${id}\\(`)
    assert.match(result.expression, marker, `${id} must be included when it is the selected unary function`)
    assert.ok(!result.expression.startsWith(`${id}(`) && !result.expression.startsWith('|'), `${id} must be nested, not wrap the whole formula`)
    assert.doesNotThrow(() => compileIterationFunction(result.expression))
  }

  for (const { id } of ITERATION_FUNCTION_GENERATION_CATALOG.binaryFunctions) {
    const result = generateRandomIterationFunction(
      { ...baseConfig, unaryFunctions: [], binaryFunctions: [id] },
      () => 0,
    )
    assert.match(result.expression, new RegExp(`\\b${id}\\(`), `${id} must be included when it is the selected binary function`)
    assert.ok(!result.expression.startsWith(`${id}(`), `${id} must be nested, not wrap the whole formula`)
    assert.doesNotThrow(() => compileIterationFunction(result.expression))
  }
}

function testFunctionArgumentsUseGeneratedSubexpressions() {
  const compoundConfig = {
    leaves: ['z', 'c'],
    operators: { ...operators(), '+': { min: 2, max: 2 } },
    maxDepth: 3,
  }
  const unary = generateRandomIterationFunction(
    { ...compoundConfig, unaryFunctions: ['sin'], binaryFunctions: [] },
    () => 0,
  )
  assert.match(unary.expression, /sin\(\([^()]+\s\+\s[^()]+\)\)/, 'unary function must receive a generated compound expression')
  assert.ok(!unary.expression.startsWith('sin('))

  const binary = generateRandomIterationFunction(
    { ...compoundConfig, unaryFunctions: [], binaryFunctions: ['mod'] },
    () => 0,
  )
  assert.match(binary.expression, /mod\(\([^()]+\s\+\s[^()]+\),/, 'binary function must receive a generated compound first argument')
  assert.ok(!binary.expression.startsWith('mod('))
}

testSelectedOperatorsAndRequiredTerms()
testHistoryRangeAndParserValidation()
testNumberRangeAndNeutralArithmeticRemoval()
testNeutralNumberLiteralsAreNeverGeneratedAsIdentities()
testAutomaticRelaxation()
testEveryRegularFunctionCanBeGenerated()
testFunctionArgumentsUseGeneratedSubexpressions()
console.log('random iteration function generator tests passed')

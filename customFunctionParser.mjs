/**
 * Based on bertbaron/mandelbrot by Bert Baron
 * This file is part of the Mandelbrot Formula Explorer project.
 * Licensed under GPL-3.0.
 */

// ============================================================================
// 定数
// ============================================================================

const CACHE_CONFIG = {
  MAX_SIZE: 100,
}

const NUMERIC_CONFIG = {
  SAFE_EPS: 1e-12, // 安全な除算用の epsilon
  POOL_SIZE: 2, // Float64Array プール数
}

const NUMBER_LITERAL_PATTERN = /^-?(?:\d+\.\d*|\d*\.\d+|\d+)(?:[eE][+-]?\d+)?$/

const RIEMANN_ZETA_CPU_SOURCE = `
            const complexZeta = (s) => {
              // Euler-Maclaurin analytic continuation with six Bernoulli corrections.
              if (Math.hypot(s[0] - 1, s[1]) < 1e-12) return [1e20, 0];
              const terms = 12;
              const coefficients = [
                1 / 12,
                -1 / 720,
                1 / 30240,
                -1 / 1209600,
                1 / 47900160,
                -691 / 1307674368000,
              ];
              const powPositiveBase = (base, exponent) => {
                const logBase = Math.log(base);
                const magnitude = Math.exp(exponent[0] * logBase);
                const angle = exponent[1] * logBase;
                return [magnitude * Math.cos(angle), magnitude * Math.sin(angle)];
              };

              let result = [0, 0];
              for (let k = 1; k < terms; k++) {
                const term = powPositiveBase(k, [-s[0], -s[1]]);
                result[0] += term[0];
                result[1] += term[1];
              }

              const tail = complexDivide(powPositiveBase(terms, [1 - s[0], -s[1]]), [s[0] - 1, s[1]]);
              const halfTerm = powPositiveBase(terms, [-s[0], -s[1]]);
              result[0] += tail[0] + 0.5 * halfTerm[0];
              result[1] += tail[1] + 0.5 * halfTerm[1];

              let rising = [s[0], s[1]];
              for (let r = 1; r <= coefficients.length; r++) {
                if (r > 1) {
                  rising = complexMultiply(rising, [s[0] + 2 * r - 3, s[1]]);
                  rising = complexMultiply(rising, [s[0] + 2 * r - 2, s[1]]);
                }
                const power = powPositiveBase(terms, [-s[0] - 2 * r + 1, -s[1]]);
                const correction = complexMultiply(rising, power);
                result[0] += coefficients[r - 1] * correction[0];
                result[1] += coefficients[r - 1] * correction[1];
              }
              return result;
            };
`

// 再コンパイルを避けるためのキャッシュ
export const functionCache = new Map()

function splitTopLevelArgs(s) {
  const args = []
  let depth = 0
  let start = 0
  for (let i = 0; i < s.length; i++) {
    const ch = s[i]
    if (ch === '(' || ch === '[') depth++
    else if (ch === ')' || ch === ']') depth--
    else if (ch === ',' && depth === 0) {
      args.push(s.substring(start, i).trim())
      start = i + 1
    }
  }
  if (start < s.length) args.push(s.substring(start).trim())
  return args.filter((arg) => arg.length > 0)
}

function buildScalarFunctionCall(functionName, converted) {
  return `[${functionName}(${converted}), 0]`
}

// 高速判定用のパターン対応表
const patternMap = new Map([
  ['z^2+c', 'mendelbrot'],
  ['z*z+c', 'mendelbrot'],
  ['(|re(z)|+i*|im(z)|)^2+c', 'burning_ship'],
  ['fold(z)^2+c', 'burning_ship'],
  ['fold(z)*fold(z)+c', 'burning_ship'],
  ['sin(z)+c', 'sin'],
  ['cos(z)+c', 'cos'],
  ['exp(z)+c', 'exp'],
  ['conj(z)+c', 'conj'],
])

/**
 * AST 最適化を使ってカスタム反復関数をパースし、コンパイルする
 * @param {string} functionStr - 関数文字列（例: "z*z + c"）
 * @returns {Function} (zReal, zImag, cReal, cImag) を受け取り [newReal, newImag] を返す関数
 */
export function compileIterationFunction(functionStr) {
  // まずキャッシュを確認する
  if (functionCache.has(functionStr)) {
    return functionCache.get(functionStr)
  }

  // 関数文字列を正規化する
  const expr = functionStr.trim()

  // よく使う式は高速経路で処理する
  const normalized = expr.toLowerCase().replace(/\s+/g, '')

  // 既知パターンに合えば専用実装へ振り分ける
  const compiledFunc = matchAndCompilePattern(normalized, expr)

  if (compiledFunc) {
    // 結果をキャッシュする
    if (functionCache.size >= CACHE_CONFIG.MAX_SIZE) {
      // 一番古いものを削除する
      const firstKey = functionCache.keys().next().value
      functionCache.delete(firstKey)
    }
    functionCache.set(functionStr, compiledFunc)
    return compiledFunc
  }

  // 未キャッシュの式は AST 最適化を試す
  try {
    const optimizedFunc = createOptimizedFunction(expr)
    if (optimizedFunc) {
      functionCache.set(functionStr, optimizedFunc)
      return optimizedFunc
    }
  } catch (e) {
    console.warn('AST 最適化に失敗したため標準パーサーへ戻します:', e.message)
  }

  // 標準パーサーへフォールバックする
  try {
    const jsCode = parseExpression(expr)

    // 複雑な式でも扱えるように関数本体を組み立てる
    const functionBody = `
            // 速度優先の簡略複素数演算
            const complexAdd = (a, b) => [a[0] + b[0], a[1] + b[1]];
            const complexSubtract = (a, b) => [a[0] - b[0], a[1] - b[1]];
            const complexMultiply = (a, b) => [a[0] * b[0] - a[1] * b[1], a[0] * b[1] + a[1] * b[0]];
      const complexDivide = (a, b) => {
        // 0 除算や Inf を避けるため分母を下限付きで扱う
        const denomRaw = b[0] * b[0] + b[1] * b[1];
        const denom = Math.max(denomRaw, ${NUMERIC_CONFIG.SAFE_EPS});
        return [(a[0] * b[0] + a[1] * b[1]) / denom, (a[1] * b[0] - a[0] * b[1]) / denom];
      };
            const complexPower = (a, n) => {
        if (typeof n === 'number' && Number.isInteger(n)) {
          if (n === 2) return complexMultiply(a, a);
          if (n === 0) return [1, 0];
          if (n === 1) return a;
          if (n === 3) {
            const a2 = complexMultiply(a, a);
            return complexMultiply(a, a2);
          }
          if (n === 4) {
            const a2 = complexMultiply(a, a);
            return complexMultiply(a2, a2);
          }
          if (n > 0) {
            let result = [1, 0];
            let base = a;
            let exp = n;
            while (exp > 0) {
              if (exp & 1) result = complexMultiply(result, base);
              exp >>= 1;
              if (exp > 0) base = complexMultiply(base, base);
            }
            return result;
          }
          if (n < 0) {
            const pos = complexPower(a, -n);
            const denomRaw = pos[0]*pos[0] + pos[1]*pos[1];
            const denom = Math.max(denomRaw, ${NUMERIC_CONFIG.SAFE_EPS});
            return [pos[0]/denom, -pos[1]/denom];
          }
        }
        // 分数乗で log(0) が起きないよう、底 0 の場合を個別に扱う
        if (a[0] === 0 && a[1] === 0) {
          if (typeof n === 'number') {
            if (n > 0) return [0, 0];
            if (n === 0) return [1, 0];
            // 0 の負べきは発散扱いにする
            return [1e20, 0];
          }
          if (Array.isArray(n) && n[1] === 0) {
            if (n[0] > 0) return [0, 0];
            if (n[0] === 0) return [1, 0];
            return [1e20, 0];
          }
          // 一般的な複素指数のフォールバック
          return [0, 0];
        }
        const complexLog = (x) => {
          // Math.hypot でオーバーフローを防ぎ、r≈1 では log1p を使う
          const r = Math.hypot(x[0], x[1]);
          const theta = r === 0 ? 0 : Math.atan2(x[1], x[0]);
          return [r === 0 ? -Infinity : Math.abs(r - 1) < 0.5 ? Math.log1p(r - 1) : Math.log(r), theta];
        };
        let b = typeof n === 'number' ? [n, 0] : n;
        if (b[1] === 0 && Number.isFinite(b[0]) && Number.isInteger(b[0])) {
          const intN = b[0];
          if (intN === 0) return [1, 0];
          if (intN === 1) return a;
          if (intN === 2) return complexMultiply(a, a);
          if (intN < 0) {
            let result = [1, 0];
            for (let i = 0; i < Math.abs(intN); i++) result = complexMultiply(result, a);
            const denom = result[0] * result[0] + result[1] * result[1];
            return [result[0] / denom, -result[1] / denom];
          }
          let result = [1, 0];
          for (let i = 0; i < intN; i++) result = complexMultiply(result, a);
          return result;
        }
        const loga = complexLog(a);
        const mul = complexMultiply(b, loga);
        return complexExp(mul);
            };
            // Math.hypot を使って大きい値でも安全に絶対値を求める
            const complexAbs = (a) => Math.hypot(a[0], a[1]);
            const complexSqrt = (a) => {
                const r = Math.sqrt(complexAbs(a));
                const theta = r === 0 ? 0 : Math.atan2(a[1], a[0]) / 2;
                return [r * Math.cos(theta), r * Math.sin(theta)];
            };
            const complexLog = (a) => {
                const r = complexAbs(a);
                const theta = r === 0 ? 0 : Math.atan2(a[1], a[0]);
                // r≈1 のときは log1p(r-1) で精度低下を抑える
                return [r === 0 ? -Infinity : Math.abs(r - 1) < 0.5 ? Math.log1p(r - 1) : Math.log(r), theta];
            };
            const complexLog10 = (a) => {
                const logNat = complexLog(a);
                const log10 = Math.log(10);
                return [logNat[0] / log10, logNat[1] / log10];
            };
            const complexConj = (a) => [a[0], -a[1]];
            const complexRe = (a) => a[0];
            const complexIm = (a) => a[1];
            const complexExp = (a) => {
                const expReal = Math.exp(a[0]);
                return [expReal * Math.cos(a[1]), expReal * Math.sin(a[1])];
            };
            const complexSin = (a) => [Math.sin(a[0]) * Math.cosh(a[1]), Math.cos(a[0]) * Math.sinh(a[1])];
            const complexCos = (a) => [Math.cos(a[0]) * Math.cosh(a[1]), -Math.sin(a[0]) * Math.sinh(a[1])];
            const complexTan = (a) => complexDivide(complexSin(a), complexCos(a));
            const complexSinh = (a) => [Math.sinh(a[0]) * Math.cos(a[1]), Math.cosh(a[0]) * Math.sin(a[1])];
            const complexCosh = (a) => [Math.cosh(a[0]) * Math.cos(a[1]), Math.sinh(a[0]) * Math.sin(a[1])];
            const complexTanh = (a) => complexDivide(complexSinh(a), complexCosh(a));
            const complexScalar = (a) => Array.isArray(a) || a instanceof Float64Array ? a[0] : a;
            const complexArg = (a) => Math.atan2(a[1] ?? 0, a[0] ?? a);
            const complexAbs2 = (a) => {
              const x = Array.isArray(a) || a instanceof Float64Array ? a[0] : a;
              const y = Array.isArray(a) || a instanceof Float64Array ? a[1] : 0;
              return x * x + y * y;
            };
            const complexFract = (a) => [a[0] - Math.floor(a[0]), a[1] - Math.floor(a[1])];
            const safeMod = (x, y) => y === 0 ? 0 : x - y * Math.floor(x / y);
            const complexMod = (a, b) => {
              const bx = Array.isArray(b) || b instanceof Float64Array ? b[0] : b;
              const by = Array.isArray(b) || b instanceof Float64Array ? b[1] : bx;
              return [safeMod(a[0], bx), safeMod(a[1], by)];
            };
            const complexRotate = (a, angle) => {
              const theta = complexScalar(angle);
              const cs = Math.cos(theta);
              const sn = Math.sin(theta);
              return [a[0] * cs - a[1] * sn, a[0] * sn + a[1] * cs];
            };
            const complexFold = (a) => [Math.abs(a[0]), Math.abs(a[1])];
            const complexBoxFold = (a, radius) => {
              const r = Math.abs(complexScalar(radius));
              const foldValue = (v) => v > r ? 2 * r - v : v < -r ? -2 * r - v : v;
              return [foldValue(a[0]), foldValue(a[1])];
            };
            ${RIEMANN_ZETA_CPU_SOURCE}
            const z = [zReal, zImag];
            const c = [cReal, cImag];
            const iterIndex = Number.isFinite(n) ? n : 0;

            ${jsCode}
        `

    const compiledFunc = new Function('zReal', 'zImag', 'cReal', 'cImag', 'n', functionBody)

    // 生成した関数が数値ペアを返すか簡単に確認する
    try {
      const testResult = compiledFunc(0, 0, 0, 0)
      if ((!Array.isArray(testResult) && !(testResult instanceof Float64Array)) || testResult.length !== 2) {
        throw new Error('Function must return an array of two numbers [real, imag]')
      }
      if (typeof testResult[0] !== 'number' || typeof testResult[1] !== 'number') {
        throw new Error('Function must return numeric values')
      }
    } catch (testError) {
      console.error('Compiled function test failed:', testError)
      console.error('Generated code:', jsCode)
      console.error('Full function body:', functionBody)
      throw new Error(
        `Invalid iteration function: ${testError.message}\n\nGenerated code may contain syntax errors. Please check your expression.`,
      )
    }

    // 結果をキャッシュする
    functionCache.set(functionStr, compiledFunc)
    return compiledFunc
  } catch (e) {
    console.error('Failed to compile iteration function:', e)
    console.error('Input expression:', functionStr)
    console.error('Normalized expression:', expr)

    // 呼び出し側で扱いやすいメッセージにして投げ直す
    throw new Error(`カスタム関数のパースに失敗しました: ${e.message}\n入力式: ${functionStr}`)
  }
}

/**
 * 内部パーサーが生成した右辺式文字列を返す。
 * これは WGSL コンパイラが GPU 用コードを作るときに使う。
 *
 * @param {string} functionStr
 * @returns {string} 次段の変換で使える式文字列
 */
export function getParsedExpression(functionStr) {
  // compileIterationFunction と同様に入力を正規化する
  const expr = functionStr.trim()
  const normalized = expr.toLowerCase().replace(/\s+/g, '')
  // まずは既知パターンに当てる
  if (patternMap.has(normalized)) {
    // よくある式は簡単な生成ロジックで返す
    switch (patternMap.get(normalized)) {
      case 'mendelbrot':
        return '[zReal*zReal - zImag*zImag + cReal, 2*zReal*zImag + cImag]'
      case 'burning_ship':
        return '[abs(zReal)*abs(zReal) - abs(zImag)*abs(zImag) + cReal, 2*abs(zReal)*abs(zImag) + cImag]'
      case 'sin':
        return '[Math.sin(zReal) * Math.cosh(zImag) + cReal, Math.cos(zReal) * Math.sinh(zImag) + cImag]'
      case 'cos':
        return '[Math.cos(zReal) * Math.cosh(zImag) + cReal, -Math.sin(zReal) * Math.sinh(zImag) + cImag]'
      case 'exp':
        return '[Math.exp(zReal) * Math.cos(zImag) + cReal, Math.exp(zReal) * Math.sin(zImag) + cImag]'
      case 'conj':
        return '[zReal + cReal, -zImag + cImag]'
      default:
        break
    }
  }

  // 一般式の場合は内部 parseExpression を呼び出して右辺式を抜き出す
  try {
    const code = parseExpression(expr)
    // parseExpression は `const result = <expr>;\nreturn result;` 形式の文字列を返す
    const m = code.match(/const\s+result\s*=\s*([\s\S]*?);\s*return\s+result;/)
    if (m?.[1]) {
      return m[1].trim()
    }
    return code
  } catch (e) {
    throw new Error('Failed to parse expression for WGSL: ' + e.message)
  }
}

/**
 * AST ベースの解析で最適化された関数を生成する
 * @param {string} expr - 最適化対象の式
 * @returns {Function|null} - 最適化済み関数、または失敗時は null
 */
function createOptimizedFunction(expr) {
  try {
    // 単純な AST 解析: 演算子と複雑度をカウント
    const complexity = analyzeComplexity(expr)

    // 複雑すぎる式はフォールバック用に null を返す
    if (complexity > 10) {
      return null
    }

    // 適度に複雑な式は最適化を試みる
    const optimizedCode = optimizeExpression(expr)

    if (!optimizedCode) {
      return null
    }

    // 最適化された関数本体を生成
    const functionBody = `
            // 最適化した複素数演算 (オーバーヘッド最小)
            const complexAdd = (a, b) => [a[0] + b[0], a[1] + b[1]];
            const complexSubtract = (a, b) => [a[0] - b[0], a[1] - b[1]];
            const complexMultiply = (a, b) => [a[0] * b[0] - a[1] * b[1], a[0] * b[1] + a[1] * b[0]];
      const SAFE_EPS = 1e-12;
      const complexDivide = (a, b) => {
        // WGSL の挙動に合わせ、分母を下限付きで扱って 0 除算を回避
        const denomRaw = b[0] * b[0] + b[1] * b[1];
        const denom = Math.max(denomRaw, SAFE_EPS);
        return [(a[0] * b[0] + a[1] * b[1]) / denom, (a[1] * b[0] - a[0] * b[1]) / denom];
      };
            const complexPower = (a, n) => {
                if (typeof n === 'number' && Number.isInteger(n) && n >= 0 && n <= 4) {
                    if (n === 0) return [1, 0];
                    if (n === 1) return a;
                    if (n === 2) return complexMultiply(a, a);
                    if (n === 3) {
                        const a2 = complexMultiply(a, a);
                        return complexMultiply(a, a2);
                    }
                    if (n === 4) {
                        const a2 = complexMultiply(a, a);
                        return complexMultiply(a2, a2);
                    }
                }
                // 分数乗 / 複素指数の場合、底が 0 のときは特別扱い
        if (a[0] === 0 && a[1] === 0) {
          if (typeof n === 'number') {
            if (n > 0) return [0, 0];
            if (n === 0) return [1, 0];
            return [1e20, 0];
          }
          if (Array.isArray(n) && n[1] === 0) {
            if (n[0] > 0) return [0, 0];
            if (n[0] === 0) return [1, 0];
            return [1e20, 0];
          }
          return [0, 0];
        }
                // 複雑なケースのフォールバック
                const complexLog = (x) => {
          const r = Math.hypot(x[0], x[1]);
          const theta = r === 0 ? 0 : Math.atan2(x[1], x[0]);
          // r≈1 のとき log1p(r-1) で情報落ちを抑える
          return [r === 0 ? -Infinity : Math.abs(r - 1) < 0.5 ? Math.log1p(r - 1) : Math.log(r), theta];
        };
                const complexExp = (x) => {
                    const expReal = Math.exp(x[0]);
                    return [expReal * Math.cos(x[1]), expReal * Math.sin(x[1])];
                };
                const loga = complexLog(a);
                const mul = complexMultiply(typeof n === 'number' ? [n, 0] : n, loga);
                return complexExp(mul);
            };
            // Math.hypot で大きな値でもオーバーフローせずに絶対値を計算
            const complexAbs = (a) => Math.hypot(a[0], a[1]);
            const complexSqrt = (a) => {
                const r = Math.sqrt(complexAbs(a));
                const theta = r === 0 ? 0 : Math.atan2(a[1], a[0]) / 2;
                return [r * Math.cos(theta), r * Math.sin(theta)];
            };
            const complexLog = (a) => {
                const r = complexAbs(a);
                const theta = r === 0 ? 0 : Math.atan2(a[1], a[0]);
                // r≈1 のとき log1p(r-1) で情報落ちを防ぐ（例: r=1.0000001 の正確な log 値）
                return [r === 0 ? -Infinity : Math.abs(r - 1) < 0.5 ? Math.log1p(r - 1) : Math.log(r), theta];
            };
            const complexLog10 = (a) => {
                const logNat = complexLog(a);
                const log10 = Math.log(10);
                return [logNat[0] / log10, logNat[1] / log10];
            };
            const complexConj = (a) => [a[0], -a[1]];
            const complexRe = (a) => a[0];
            const complexIm = (a) => a[1];
            const complexExp = (a) => {
                const expReal = Math.exp(a[0]);
                return [expReal * Math.cos(a[1]), expReal * Math.sin(a[1])];
            };
            const complexSin = (a) => [Math.sin(a[0]) * Math.cosh(a[1]), Math.cos(a[0]) * Math.sinh(a[1])];
            const complexCos = (a) => [Math.cos(a[0]) * Math.cosh(a[1]), -Math.sin(a[0]) * Math.sinh(a[1])];
            const complexTan = (a) => complexDivide(complexSin(a), complexCos(a));
            const complexSinh = (a) => [Math.sinh(a[0]) * Math.cos(a[1]), Math.cosh(a[0]) * Math.sin(a[1])];
            const complexCosh = (a) => [Math.cosh(a[0]) * Math.cos(a[1]), Math.sinh(a[0]) * Math.sin(a[1])];
            const complexTanh = (a) => complexDivide(complexSinh(a), complexCosh(a));
            const complexScalar = (a) => Array.isArray(a) || a instanceof Float64Array ? a[0] : a;
            const complexArg = (a) => Math.atan2(a[1] ?? 0, a[0] ?? a);
            const complexAbs2 = (a) => {
              const x = Array.isArray(a) || a instanceof Float64Array ? a[0] : a;
              const y = Array.isArray(a) || a instanceof Float64Array ? a[1] : 0;
              return x * x + y * y;
            };
            const complexFract = (a) => [a[0] - Math.floor(a[0]), a[1] - Math.floor(a[1])];
            const safeMod = (x, y) => y === 0 ? 0 : x - y * Math.floor(x / y);
            const complexMod = (a, b) => {
              const bx = Array.isArray(b) || b instanceof Float64Array ? b[0] : b;
              const by = Array.isArray(b) || b instanceof Float64Array ? b[1] : bx;
              return [safeMod(a[0], bx), safeMod(a[1], by)];
            };
            const complexRotate = (a, angle) => {
              const theta = complexScalar(angle);
              const cs = Math.cos(theta);
              const sn = Math.sin(theta);
              return [a[0] * cs - a[1] * sn, a[0] * sn + a[1] * cs];
            };
            const complexFold = (a) => [Math.abs(a[0]), Math.abs(a[1])];
            const complexBoxFold = (a, radius) => {
              const r = Math.abs(complexScalar(radius));
              const foldValue = (v) => v > r ? 2 * r - v : v < -r ? -2 * r - v : v;
              return [foldValue(a[0]), foldValue(a[1])];
            };
            ${RIEMANN_ZETA_CPU_SOURCE}
            const z = [zReal, zImag];
            const c = [cReal, cImag];
            const iterIndex = Number.isFinite(n) ? n : 0;

            ${optimizedCode}
        `

    const compiledFunc = new Function('zReal', 'zImag', 'cReal', 'cImag', 'n', functionBody)

    // 関数をテストする
    const testResult = compiledFunc(0, 0, 0, 0)
    if (
      Array.isArray(testResult) &&
      testResult.length === 2 &&
      typeof testResult[0] === 'number' &&
      typeof testResult[1] === 'number'
    ) {
      return compiledFunc
    }

    return null
  } catch (_e) {
    return null
  }
}

/**
 * 式の複雑度を解析する
 * @param {string} expr - 解析対象の式
 * @returns {number} - 複雑度スコア
 */
function analyzeComplexity(expr) {
  let complexity = 0

  // Count function calls
  const funcMatches = expr.match(
    /\b(sin|cos|tan|sinh|cosh|tanh|exp|log|ln|sqrt|zeta|conj|Re|Im|abs|arg|abs2|fract|mod|rotate|fold|boxFold)\s*\(/g,
  )
  if (funcMatches) complexity += funcMatches.length * 2

  // 演算子をカウント
  const opMatches = expr.match(/[+\-*/]/g)
  if (opMatches) complexity += opMatches.length

  // べき乗演算子をカウント
  const powerMatches = expr.match(/\^/g)
  if (powerMatches) complexity += powerMatches.length * 3

  return complexity
}

/**
 * 代数的簡約を適用して式を最適化する
 * @param {string} expr - 最適化対象の式
 * @returns {string} - 最適化済みの式コード
 */
function optimizeExpression(expr) {
  try {
    // 軽微な最適化を適用
    const optimized = expr

    // 解析して最適化
    const jsCode = parseExpression(optimized)

    return jsCode
  } catch (_e) {
    return null
  }
}

/**
 * 式をパースして JavaScript コードに変換する
 */
function parseExpression(expr) {
  const removedFunctionMatch = expr.match(/\b(?:floor|min|max|clamp)\s*\(/i)
  if (removedFunctionMatch) {
    const functionName = removedFunctionMatch[0].replace(/\s*\($/, '')
    throw new Error(`Unsupported function: ${functionName}`)
  }

  // 数学関数と演算子を置換する
  let code = expr
  // 先に pi, e を処理して分割時の誤判定を避ける
  code = code.replace(/\bpi\b/g, '[Math.PI, 0]')
  code = code.replace(/\be\b/g, '[Math.E, 0]')

  // 単独の i は複素数 [0,1] として扱う
  // これを早めに処理することで exp(i) などが complexExp([0,1]) になる
  code = code.replace(/\bi\b/g, '[0, 1]')
  code = code.replace(/\bn\b/g, '[iterIndex, 0]')

  // power 処理は convertGenericExpression 側で行う
  code = code.replace(/\bc\b/g, '[cReal, cImag]')

  // 新しい関数: sqrt, log, ln を扱う
  // ネストした括弧や引数を正しく処理するためにスキャナベースの置換器を使う
  // これにより正規表現だけでは対応できないケースを避ける

  // 複素関数を扱う（ネストした括弧をサポート）
  const replaceNamedFunction = (str, name, replacer, caseInsensitive = false) => {
    let i = 0
    const lname = caseInsensitive ? name.toLowerCase() : name
    while (i < str.length) {
      const searchIn = caseInsensitive ? str.toLowerCase() : str
      const idx = searchIn.indexOf(lname + '(', i)
      if (idx === -1) break
      // 文字列の一部ではなく完全な識別子か確認する（例: 'complexSin' を誤検出しない）
      const prevChar = idx - 1 >= 0 ? str[idx - 1] : null
      if (prevChar && /[A-Za-z0-9_.]/.test(prevChar)) {
        i = idx + 1
        continue
      }
      // 対応する閉じ括弧を探す
      let depth = 0
      const start = idx + name.length + 1
      let end = -1
      for (let k = start; k < str.length; k++) {
        if (str[k] === '(') depth++
        else if (str[k] === ')') {
          if (depth === 0) {
            end = k
            break
          } else {
            depth--
          }
        }
      }
      if (end === -1) {
        // 対応する括弧がない場合はスキップ
        i = idx + 1
        continue
      }
      const inner = str.substring(start, end).trim()
      const args = splitTopLevelArgs(inner)
      const convertedArgs = args.map((arg) => convertGenericExpression(arg))
      const replacement = replacer(convertedArgs[0] ?? convertGenericExpression(inner), convertedArgs, args)
      str = str.substring(0, idx) + replacement + str.substring(end + 1)
      i = idx + replacement.length
    }
    return str
  }

  // ネーム付き関数の置換を安定するまで繰り返して、
  // ネストした関数呼び出しも完全に変換する。
  // 無限ループを避けるため繰り返し回数に上限を設ける。
  // Re/Im は実数スカラーとして扱う。
  // 例: Re(z) + i*Im(z) が元の z と同じ向きになるようにする。
  const fnReplacers = [
    ['Re', (converted) => buildScalarFunctionCall('complexRe', converted)],
    ['Im', (converted) => buildScalarFunctionCall('complexIm', converted)],
    ['arg', (converted) => buildScalarFunctionCall('complexArg', converted)],
    ['abs2', (converted) => buildScalarFunctionCall('complexAbs2', converted)],
    ['conj', (converted) => `complexConj(${converted})`],
    ['exp', (converted) => `complexExp(${converted})`],
    ['sin', (converted) => `complexSin(${converted})`],
    ['tan', (converted) => `complexTan(${converted})`],
    ['cos', (converted) => `complexCos(${converted})`],
    ['sinh', (converted) => `complexSinh(${converted})`],
    ['cosh', (converted) => `complexCosh(${converted})`],
    ['tanh', (converted) => `complexTanh(${converted})`],
    ['sqrt', (converted) => `complexSqrt(${converted})`],
    ['zeta', (converted) => `complexZeta(${converted})`],
    ['fract', (converted) => `complexFract(${converted})`],
    [
      'mod',
      (_converted, convertedArgs) => `complexMod(${convertedArgs[0] ?? '[0, 0]'}, ${convertedArgs[1] ?? '[1, 0]'})`,
    ],
    [
      'rotate',
      (_converted, convertedArgs) => `complexRotate(${convertedArgs[0] ?? '[0, 0]'}, ${convertedArgs[1] ?? '[0, 0]'})`,
    ],
    [
      'boxFold',
      (_converted, convertedArgs) => `complexBoxFold(${convertedArgs[0] ?? '[0, 0]'}, ${convertedArgs[1] ?? '[1, 0]'})`,
    ],
    ['fold', (converted) => `complexFold(${converted})`],
    ['log', (converted) => `complexLog10(${converted})`],
    ['ln', (converted) => `complexLog(${converted})`],
  ]

  let prevCode
  let iter = 0
  do {
    prevCode = code
    for (const [name, replacer] of fnReplacers) {
      code = replaceNamedFunction(code, name, replacer, true)
    }
    iter++
    if (iter > 20) break
  } while (code !== prevCode)
  // 三角関数 / 指数関数の置換が完了した

  // |expr| の一般的な絶対値は、名前付き関数の置換後に処理する。
  // こうすることで |...| 内の関数が先に複素数ヘルパーへ変換される。
  // 内部式は convertGenericExpression を通し、
  // まず complexMultiply 等に変換してから complexAbs を適用する。
  // これをしないと |c*c| が JS では配列同士の乗算になり NaN になる。
  code = code.replace(/\|([^|]+)\|/g, (_, inner) => `[complexAbs(${convertGenericExpression(inner)}), 0]`)

  // 関数呼び出しの数値引数を複素数へ変換する
  // 補足: 以前の汎用引数 regex は削除した。
  // named-function 置換ですでに変換されるため、
  // 入れ子や既に置換済みの呼び出しを誤変換してしまうからだ。

  // 演算子は汎用式パーサーで扱う
  code = convertGenericExpression(code)

  return `const result = ${code};\nreturn result;`
}

/**
 * 再帰的パースで汎用式を変換するコンバータ
 */
function convertGenericExpression(expr) {
  // 外側の空白を削除
  expr = expr.trim()

  // 式全体を包んでいる外側の括弧を削除
  while (expr.startsWith('(') && expr.endsWith(')')) {
    let depth = 0
    let isOuterParen = true
    for (let i = 0; i < expr.length; i++) {
      if (expr[i] === '(') depth++
      else if (expr[i] === ')') depth--
      // 末尾まで到達する前に depth が 0 になったら、外側の括弧は式全体を包んでいない
      if (depth === 0 && i < expr.length - 1) {
        isOuterParen = false
        break
      }
    }
    if (isOuterParen) {
      expr = expr.substring(1, expr.length - 1).trim()
    } else {
      break
    }
  }

  // まずべき乗演算子を処理する（ネストした関数呼び出しや括弧を含む場合も対応）
  // 以前の正規表現方式は Math.abs(...) のようなネストに対応できなかった
  // トップレベルの '^' を走査し、X^2 を complexPower(X, 2) に置き換える
  // ここでは数値の整数べき乗のみ対応する（例: ^2）
  {
    let s = expr
    let i = 0
    let scanIterations = 0
    // '^' を左から右へ走査し、トップレベルの演算を見つける
    while (i < s.length) {
      if (++scanIterations > 200000) {
        throw new Error(
          'parseExpression power-scan exceeded iteration limit; possible infinite loop. context=' +
            s.substring(Math.max(0, i - 40), Math.min(s.length, i + 40)),
        )
      }
      if (s[i] === '^') {
        // ここまでのネスト深度を確認
        let depth = 0
        // デバッグ用ログ
        // console.log('[parseExpression] found ^ at', i, 'snippet:', s.substring(Math.max(0, i-20), Math.min(s.length, i+20)))
        for (let j = 0; j < i; j++) {
          if (s[j] === '(' || s[j] === '[') depth++
          else if (s[j] === ')' || s[j] === ']') depth--
        }
        if (depth === 0) {
          // '^' の左側から底の式を抜き出す
          let baseEnd = i - 1
          // 末尾の空白を後方からスキップ
          while (baseEnd >= 0 && /\s/.test(s[baseEnd])) baseEnd--
          if (baseEnd < 0) {
            i += 2
            continue
          }
          let baseStart = baseEnd
          if (s[baseEnd] === ')') {
            // 対応する '(' を探す
            let d = 0
            for (let k = baseEnd; k >= 0; k--) {
              if (s[k] === ')') d++
              else if (s[k] === '(') d--
              if (d === 0) {
                baseStart = k
                break
              }
            }
            // 関数名があれば '(' の直前まで含める
            // 例: complexConj(z) -> complexConj(
            let nameStart = baseStart
            while (nameStart - 1 >= 0 && /[A-Za-z0-9_.]/.test(s[nameStart - 1])) {
              nameStart--
            }
            baseStart = nameStart
          } else if (s[baseEnd] === ']') {
            let d = 0
            for (let k = baseEnd; k >= 0; k--) {
              if (s[k] === ']') d++
              else if (s[k] === '[') d--
              if (d === 0) {
                baseStart = k
                break
              }
            }
          } else {
            while (baseStart - 1 >= 0 && /[\w.\]]/.test(s[baseStart - 1])) baseStart--
          }
          const base = s.substring(baseStart, baseEnd + 1).trim()

          // 右側の指数を抜き出す（括弧・関数呼び出し・小数・複雑式に対応）
          let expStart = i + 1
          while (expStart < s.length && /\s/.test(s[expStart])) expStart++
          if (expStart >= s.length) {
            i += 1
            continue
          }
          let expEnd = expStart
          if (s[expStart] === '(') {
            // 対応する ')' を探す
            let d = 0
            for (let k = expStart; k < s.length; k++) {
              if (s[k] === '(') d++
              else if (s[k] === ')') d--
              if (d === 0) {
                expEnd = k + 1
                break
              }
            }
          } else if (s[expStart] === '[') {
            let d = 0
            for (let k = expStart; k < s.length; k++) {
              if (s[k] === '[') d++
              else if (s[k] === ']') d--
              if (d === 0) {
                expEnd = k + 1
                break
              }
            }
          } else {
            // 括弧なしの識別子・数値・関数呼び出し
            let k = expStart
            // 先頭の符号 +/ - を指数トークンに含める
            if (s[k] === '+' || s[k] === '-') {
              k++
            }
            while (k < s.length) {
              const ch = s[k]
              if (ch === '(') {
                // 関数呼び出し引数を含める
                let d = 0
                for (let m = k; m < s.length; m++) {
                  if (s[m] === '(') d++
                  else if (s[m] === ')') d--
                  if (d === 0) {
                    k = m + 1
                    break
                  }
                }
                continue
              }
              // トップレベルの区切り文字で終了する
              // 先頭の符号の後に別の + や - が現れたら外側の演算子とみなす
              if (/\s|\+|-|\*|\/|\)|\]|,/.test(ch)) break
              k++
            }
            expEnd = k
          }
          const exponentRaw = s.substring(expStart, expEnd).trim()
          if (!exponentRaw) {
            i += 1
            continue
          }

          // base^exponent を complexPower(convertedBase, convertedExponent) に置き換える
          const convertedBase = convertGenericExpression(base)
          // 指数が単純な数値リテラルか確認
          let convertedExponent
          if (NUMBER_LITERAL_PATTERN.test(exponentRaw)) {
            // 単純な数値ならそのまま渡して最適化
            convertedExponent = exponentRaw
          } else {
            // 複雑式なら複素数に変換
            convertedExponent = convertGenericExpression(exponentRaw)
          }
          s = s.substring(0, baseStart) + `complexPower(${convertedBase}, ${convertedExponent})` + s.substring(expEnd)
          // 複数のべき乗に対応するため先頭から再走査する
          i = 0
          continue
        }
      }
      i++
    }
    expr = s
  }

  // 括弧や関数の外側にある右端の + または - を探す
  let depth = 0
  let mainOpIndex = -1
  let mainOp = null

  for (let i = expr.length - 1; i >= 0; i--) {
    const char = expr[i]
    if (char === ')' || char === ']') {
      depth++
    } else if (char === '(' || char === '[') {
      depth--
    } else if (depth === 0 && (char === '+' || char === '-')) {
      // これが単項マイナスかどうかを確認
      if (char === '-') {
        if (i === 0) {
          continue // 式の先頭の単項マイナス
        }
        // 直前の非空白文字を調べる
        let j = i - 1
        while (j >= 0 && /\s/.test(expr[j])) {
          j--
        }
        if (j < 0) {
          continue // 式の先頭の単項マイナス
        }
        const prevChar = expr[j]
        if (prevChar === '+' || prevChar === '-' || prevChar === '*' || prevChar === '(' || prevChar === '[') {
          continue // 単項マイナスなのでスキップ
        }
      }
      mainOpIndex = i
      mainOp = char
      break
    }
  }

  if (mainOpIndex !== -1) {
    // + または - を見つけたら分割して変換
    const left = expr.substring(0, mainOpIndex).trim()
    const right = expr.substring(mainOpIndex + 1).trim()
    return convertBinaryOp(convertGenericExpression(left), mainOp, convertGenericExpression(right))
  }

  // 乗除算子を探す
  depth = 0
  for (let i = expr.length - 1; i >= 0; i--) {
    const char = expr[i]
    if (char === ')' || char === ']') {
      depth++
    } else if (char === '(' || char === '[') {
      depth--
    } else if (depth === 0 && (char === '*' || char === '/')) {
      const left = expr.substring(0, i).trim()
      const right = expr.substring(i + 1).trim()
      return convertBinaryOp(convertGenericExpression(left), char, convertGenericExpression(right))
    }
  }

  // 単項マイナスを処理
  if (expr.startsWith('-')) {
    const inner = expr.substring(1).trim()
    return `complexMultiply([-1, 0], ${convertGenericExpression(inner)})`
  }

  // 数値リテラルを複素数 [n, 0] に変換
  if (NUMBER_LITERAL_PATTERN.test(expr)) {
    return `[${expr}, 0]`
  }

  // 基本ケース: z, c, complexSin(z) のような項をそのまま返す
  return expr
}

/**
 * 2 項演算を複素数演算に変換する
 */
function convertBinaryOp(left, op, right) {
  switch (op) {
    case '+':
      return `complexAdd(${left}, ${right})`
    case '-':
      return `complexSubtract(${left}, ${right})`
    case '*':
      return `complexMultiply(${left}, ${right})`
    case '/':
      return `complexDivide(${left}, ${right})`
    default:
      throw new Error(`Unsupported binary operator: ${op}`)
  }
}

/**
 * 効率的なルックアップでパターンを判定し、コンパイル済み関数を返す
 * @param {string} normalized - 正規化済みの式
 * @returns {Function|null} - コンパイル済み関数、または null
 */
function matchAndCompilePattern(normalized) {
  const patternType = patternMap.get(normalized)
  if (!patternType) {
    return null
  }

  switch (patternType) {
    case 'mendelbrot':
      return (() => {
        const _p = [new Float64Array(2), new Float64Array(2)]
        let _pi = 0
        return (zReal, zImag, cReal, cImag) => {
          _pi = (_pi + 1) % _p.length
          const out = _p[_pi]
          out[0] = zReal * zReal - zImag * zImag + cReal
          out[1] = 2 * zReal * zImag + cImag
          return out
        }
      })()

    case 'burning_ship':
      return (() => {
        const _p = [new Float64Array(2), new Float64Array(2)]
        let _pi = 0
        return (zReal, zImag, cReal, cImag) => {
          const ar = Math.abs(zReal)
          const ai = Math.abs(zImag)
          _pi = (_pi + 1) % _p.length
          const out = _p[_pi]
          out[0] = ar * ar - ai * ai + cReal
          out[1] = 2 * ar * ai + cImag
          return out
        }
      })()

    case 'sin':
      return (() => {
        const _p = [new Float64Array(2), new Float64Array(2)]
        let _pi = 0
        return (zReal, zImag, cReal, cImag) => {
          _pi = (_pi + 1) % _p.length
          const out = _p[_pi]
          out[0] = Math.sin(zReal) * Math.cosh(zImag) + cReal
          out[1] = Math.cos(zReal) * Math.sinh(zImag) + cImag
          return out
        }
      })()

    case 'cos':
      return (() => {
        const _p = [new Float64Array(2), new Float64Array(2)]
        let _pi = 0
        return (zReal, zImag, cReal, cImag) => {
          _pi = (_pi + 1) % _p.length
          const out = _p[_pi]
          out[0] = Math.cos(zReal) * Math.cosh(zImag) + cReal
          out[1] = -Math.sin(zReal) * Math.sinh(zImag) + cImag
          return out
        }
      })()

    case 'exp':
      return (() => {
        const _p = [new Float64Array(2), new Float64Array(2)]
        let _pi = 0
        return (zReal, zImag, cReal, cImag) => {
          const expReal = Math.exp(zReal)
          _pi = (_pi + 1) % _p.length
          const out = _p[_pi]
          out[0] = expReal * Math.cos(zImag) + cReal
          out[1] = expReal * Math.sin(zImag) + cImag
          return out
        }
      })()

    case 'conj':
      return (() => {
        const _p = [new Float64Array(2), new Float64Array(2)]
        let _pi = 0
        return (zReal, zImag, cReal, cImag) => {
          _pi = (_pi + 1) % _p.length
          const out = _p[_pi]
          out[0] = zReal + cReal
          out[1] = -zImag + cImag
          return out
        }
      })()

    default:
      return null
  }
}

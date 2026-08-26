/**
 * Bert Baron 氏の bertbaron/mandelbrot をベースにしています。
 * このファイルは Mandelbrot Explorer プロジェクトの一部です。
 * GPL-3.0 ライセンスです。
 */

// ============================================================================
// 定数
// ============================================================================

// 使い回す関数名一覧
export const SAFE_FUNCS = ['exp', 'sin', 'cos', 'log', 'sqrt', 'abs', 'atan2']

/**
 * WGSL 互換のため、数値リテラルに必要なら小数点を付ける
 * @param {string} n - 数値リテラル
 * @returns {string}
 * @private
 */
function safeNumberLiteral(n) {
  // 整数は X.0 にし、すでに小数点があるものはそのまま使う
  if (/^-?\d+$/.test(n)) {
    return n + '.0'
  }
  return n // すでに小数点があるか、式であるとみなす
}

/**
 * 最上位のカンマで引数列を分割する
 * @param {string} s - 入力文字列
 * @returns {string[]}
 * @private
 */
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
  return args.filter((a) => a.length > 0)
}

/**
 * JavaScript 式を WGSL コードへ変換する
 * @param {string} expr - JavaScript 式
 * @returns {string} WGSL コード
 */
export function jsExprToWGSL(expr) {
  // 出力は AST ベースのエミッタに委ねる
  if (!expr || typeof expr !== 'string') {
    return ''
  }
  return jsExprToWGSL_ast(expr)
}

function parseExprFromString(str) {
  const input = str.trim()
  if (input.length === 0) return null

  // 最上位のカンマ区切りは tuple として扱う
  const parts = splitTopLevelArgs(input)
  if (parts.length > 1) return { type: 'tuple', items: parts.map(parseExprFromString) }

  // シンプルな再帰下降パーサー
  let i = 0
  function peek() {
    return input[i]
  }
  function consumeWhitespace() {
    while (i < input.length && /\s/.test(input[i])) i++
  }
  function match(re) {
    const m = input.slice(i).match(re)
    if (m && m.index === 0) {
      i += m[0].length
      return m
    }
    return null
  }

  function parseNumberOrIdentOrCall() {
    consumeWhitespace()
    // かっこ式。中に最上位カンマがあれば tuple として扱う
    if (peek() === '(') {
      // 対応する閉じかっこを探す
      let depth = 0
      let j = i + 1
      for (; j < input.length; j++) {
        if (input[j] === '(') depth++
        else if (input[j] === ')') {
          if (depth === 0) break
          depth--
        }
      }
      if (j >= input.length) {
        // 壊れた式なら通常パースへ戻す
        i++
        const expr = parseAddSub()
        consumeWhitespace()
        if (peek() === ')') i++
        return expr
      }
      const inner = input.slice(i + 1, j)
      const parts = splitTopLevelArgs(inner)
      if (parts.length > 1) {
        i = j + 1
        return { type: 'tuple', items: parts.map(parseExprFromString) }
      }
      // 最上位カンマが無ければ通常の式として扱う
      i = j + 1
      return parseExprFromString(inner)
    }
    // 配列リテラル形式の tuple
    if (peek() === '[') {
      // 対応する閉じ角かっこを探して分割する
      const start = i
      let depth = 0
      i++
      while (i < input.length) {
        if (input[i] === '[') depth++
        else if (input[i] === ']') {
          if (depth === 0) break
          depth--
        }
        i++
      }
      if (i < input.length && input[i] === ']') {
        const inner = input.slice(start + 1, i)
        i++ // ']' をスキップ
        const parts = splitTopLevelArgs(inner)
        if (parts.length > 1) return { type: 'tuple', items: parts.map(parseExprFromString) }
        return parseExprFromString(inner)
      }
    }

    // 数値リテラル
    const num = match(/^[+-]?(?:\d+\.\d*|\d*\.\d+|\d+)(?:[eE][+-]?\d+)?/)
    if (num) return { type: 'number', value: num[0] }

    // 識別子または関数呼び出し（ドット付きも含む）
    const id = match(/^[A-Za-z_][A-Za-z0-9_.]*/)
    if (id) {
      let name = id[0]
      consumeWhitespace()
      // <f32> のような generic を任意で受け取る
      if (peek() === '<') {
        // 対応する '>' を探す
        let j = i
        if (input[j] === '<') {
          let depthg = 0
          for (; j < input.length; j++) {
            if (input[j] === '<') depthg++
            else if (input[j] === '>') {
              depthg--
              if (depthg === 0) break
            }
          }
          if (j < input.length && input[j] === '>') {
            const inner = input.slice(i + 1, j)
            name = name + `<${inner}>`
            i = j + 1
          }
        }
        consumeWhitespace()
      }
      const nameFinal = name
      // ここから本体を処理する
      if (peek() === '(') {
        // 関数呼び出し
        i++ // '(' をスキップ
        // 入れ子のカンマを許すため手動で対応する閉じかっこを探す
        const start = i
        let depth = 0
        for (; i < input.length; i++) {
          if (input[i] === '(') depth++
          else if (input[i] === ')') {
            if (depth === 0) break
            depth--
          }
        }
        const inner = input.slice(start, i)
        if (i < input.length && input[i] === ')') i++
        const args = splitTopLevelArgs(inner).map((p) => parseExprFromString(p))
        let node = { type: 'call', name: nameFinal, args }
        // 後続の .x/.y や [0]/[1] に対応する
        consumeWhitespace()
        while (true) {
          if (peek() === '.') {
            const m = input.slice(i).match(/^\.([xy])/)
            if (m) {
              i += m[0].length
              node = { type: 'prop', target: node, prop: m[1] }
              continue
            }
          } else if (peek() === '[') {
            // 単純な添字アクセス
            const rb = input.indexOf(']', i + 1)
            if (rb !== -1) {
              const idxRaw = input.slice(i + 1, rb).trim()
              const m = idxRaw.match(/^(-?\d+)(?:\.(?:0+)?)?$/)
              if (m) {
                node = {
                  type: 'index',
                  target: node,
                  index: parseInt(m[1], 10),
                }
                i = rb + 1
                continue
              }
            }
          }
          break
        }
        return node
      }
      // 通常の識別子
      return { type: 'ident', name: nameFinal }
    }
    // ここまで来たら 1 文字だけ読み進めて識別子扱いにする
    const ch = peek()
    i++
    return { type: 'ident', name: ch }
  }

  function parseUnary() {
    consumeWhitespace()
    if (peek() === '-') {
      i++
      const operand = parseUnary()
      return { type: 'unary', op: '-', arg: operand }
    }
    return parseNumberOrIdentOrCall()
  }

  function parseMulDiv() {
    let left = parseUnary()
    while (true) {
      consumeWhitespace()
      const op = peek()
      if (op === '*' || op === '/') {
        i++
        const right = parseUnary()
        left = { type: 'binary', op, left, right }
        continue
      }
      break
    }
    return left
  }

  function parseAddSub() {
    let left = parseMulDiv()
    while (true) {
      consumeWhitespace()
      const op = peek()
      if (op === '+' || op === '-') {
        i++
        const right = parseMulDiv()
        left = { type: 'binary', op, left, right }
        continue
      }
      break
    }
    return left
  }

  const node = parseAddSub()
  // 末尾まで消費できなくても、識別子として扱って返す
  return node
}

// 長さや分母のフォールバックに使う共通 epsilon。
// NaN / Inf の回避と、微小値の挙動維持のバランスを取るための値。
export const SAFE_EPS = 1e-12

// AST から WGSL の型付きノードを生成する
function astToWGSL(node, tokensTable) {
  if (!node) return { expr: '', kind: 'scalar' }

  function getVec2Args(expr) {
    const s = typeof expr === 'string' ? expr.trim() : ''
    if (!s.startsWith('vec2<f32>(')) return null

    const start = 'vec2<f32>('.length
    let depth = 0
    for (let i = start; i < s.length; i++) {
      const ch = s[i]
      if (ch === '(' || ch === '[') depth++
      else if (ch === ')' || ch === ']') {
        if (depth === 0) {
          const inner = s.slice(start, i)
          const parts = splitTopLevelArgs(inner)
          return parts.length === 2 ? parts : null
        }
        depth--
      }
    }
    return null
  }

  function comp(n, which) {
    if (!n) return which === 'x' ? '0.0' : '0.0'
    if (n.kind === 'vec2') {
      if (n.x !== undefined && n.y !== undefined) return which === 'x' ? String(n.x).trim() : String(n.y).trim()
      return `${n.expr.trim()}.${which}`
    }
    // スカラー値
    if (which === 'x') return String(n.expr).trim()
    return '0.0'
  }

  // 型付きノードから実数スカラー式を安全に取り出す。
  // 純粋な実数と判断できない場合は null を返す。
  function extractRealScalarFromTN(tn) {
    if (!tn) return null
    if (tn.kind === 'scalar') return String(tn.expr).trim()
    if (tn.kind === 'vec2') {
      const yv = (tn.y || '').toString().trim()
      if (yv === '0.0' || yv === '0') return (tn.x || '0.0').toString().trim()
      // expr が vec2<f32>(A, B) 形式なら確認する
      // 第 2 成分が本当に 0 のときだけ純粋な実数として扱う
      const parts = getVec2Args(tn.expr)
      if (parts) {
        const second = parts[1].trim()
        if (second === '0.0' || second === '0') return parts[0].trim()
      }
    }
    if (typeof tn.expr === 'string') {
      const parts = getVec2Args(tn.expr)
      if (parts) {
        const second = parts[1].trim()
        if (second === '0.0' || second === '0') return parts[0].trim()
      }
    }
    return null
  }
  // 実部が 0 のときだけ、型付きノードから虚数スカラー式を取り出す。
  function extractImagScalarFromTN(tn) {
    if (!tn) return null
    if (tn.kind === 'vec2') {
      const xv = (tn.x || '').toString().trim()
      // 実部が文字どおり 0 なら虚部を返す
      if (xv === '0.0' || xv === '0') return String(tn.y || '0.0').trim()
      // expr が vec2<f32>(A, B) 形式か確認する
      const parts = getVec2Args(tn.expr)
      if (parts) {
        if (parts[0].trim() === '0.0' || parts[0].trim() === '0') return parts[1].trim()
      }
    }
    return null
  }

  // 数字だけで構成された式なら JS の数値として試算する。
  // 見た目は複雑でも実質 0 の式を検出し、不要な項を減らすために使う。
  function tryNumericValue(exprStr) {
    if (!exprStr || typeof exprStr !== 'string') return { ok: false }
    if (!/^[0-9eE.+\-\s()]+$/.test(exprStr))
      // 数字、符号、空白、小数点、かっこ、指数表記だけを許可する
      return { ok: false }
    try {
      // 余分な空白を除く
      const s = exprStr.replace(/\s+/g, '')
      // 識別子を含まない式だけを Function で評価する
      const v = Function(`return (${s});`)()
      if (typeof v === 'number' && Number.isFinite(v)) return { ok: true, value: v }
      return { ok: false }
    } catch (_e) {
      return { ok: false }
    }
  }

  function makeVec2FromChildren(a, b) {
    const ax = a && a.kind === 'vec2' ? comp(a, 'x') : a ? a.expr : '0.0'
    const ay = a && a.kind === 'vec2' ? comp(a, 'y') : a ? a.expr : '0.0'
    const bx = b && b.kind === 'vec2' ? comp(b, 'x') : b ? b.expr : '0.0'
    const by = b && b.kind === 'vec2' ? comp(b, 'y') : b ? b.expr : '0.0'
    const outX = ax
    const outY = b && b.kind === 'vec2' ? comp(b, 'y') : b ? b.expr : '0.0'
    // "0.0 - (X)" のような不安定な表記を整える
    function normalizeComp(s) {
      if (!s || typeof s !== 'string') return s
      s = s.trim()
      // 成分が vec2 リテラルなら内側のスカラーだけを取り出し、
      // vec2 の入れ子を作らないようにする
      const vec2Args = getVec2Args(s)
      if (vec2Args) {
        // ここでは x 側として扱う前提で第 1 成分を返す
        return vec2Args[0].trim()
      }
      // 末尾の ' + 0.0' や先頭の '0.0 + ' を消す
      s = s.replace(/\+\s*0(?:\.0+)?\s*$/, '')
      s = s.replace(/^0(?:\.0+)?\s*\+\s*/, '')
      // '0.0 - (X)' や '0 - (X)' を '-(X)' に直す
      const m = s.match(/^0(?:\.0+)?\s*-\s*\((.*)\)$/)
      if (m) return `-(${m[1].trim()})`
      // '0.0 - X' を '-(X)' に直す
      const m2 = s.match(/^0(?:\.0+)?\s*-\s*(.+)$/)
      if (m2) return `-(${m2[1].trim()})`
      return s
    }
    const nx = normalizeComp(String(outX))
    const ny = normalizeComp(String(outY))
    const expr = `vec2<f32>(${nx}, ${ny})`
    // 後続の comp() でも使えるよう、整形済み成分を保持する
    return { kind: 'vec2', expr, x: nx, y: ny }
  }

  function scalarFromNode(n) {
    return comp(n, 'x')
  }

  function safeModExpr(value, divisor) {
    const hasDivisor = `abs(${divisor}) > ${SAFE_EPS}`
    const d = `select(select(-${SAFE_EPS}, ${SAFE_EPS}, ${divisor} >= 0.0), ${divisor}, ${hasDivisor})`
    const remainder = `(${value}) - (${d}) * floor((${value}) / (${d}))`
    return `select(0.0, ${remainder}, ${hasDivisor})`
  }

  if (node.type === 'number') return { expr: safeNumberLiteral(node.value), kind: 'scalar' }
  if (node.type === 'ident') {
    const n = node.name
    if (n === 'z' || n === 'c') return { expr: n, kind: 'vec2' }
    if (n === 'n') return { expr: 'n', kind: 'scalar' }
    return { expr: n, kind: 'scalar' }
  }
  // AST レベルの CSE が導入した一時ノード
  if (node.type === 'tmp') {
    const name = node.name
    const kind = node.tmpKind === 'vec2' ? 'vec2' : 'scalar'
    if (kind === 'vec2') return { expr: name, kind: 'vec2', x: `${name}.x`, y: `${name}.y` }
    return { expr: name, kind: 'scalar' }
  }
  // tmptoken の処理は廃止し、TMP は通常の識別子として扱う
  if (node.type === 'prop') {
    const t = astToWGSL(node.target, tokensTable)
    return { expr: comp(t, node.prop), kind: 'scalar' }
  }
  if (node.type === 'index') {
    const t = astToWGSL(node.target, tokensTable)
    if (node.index === 0) return { expr: comp(t, 'x'), kind: 'scalar' }
    return { expr: comp(t, 'y'), kind: 'scalar' }
  }

  if (node.type === 'call') {
    const name = node.name
    const argNodes = (node.args || []).map((a) => astToWGSL(a, tokensTable))
    // 'vec2<f32>' や 'vec2' の明示コンストラクタを扱う
    if (/^vec2\s*(<.*>)?$/i.test(name)) {
      if (argNodes.length === 2) return makeVec2FromChildren(argNodes[0], argNodes[1])
      if (argNodes.length === 1) {
        // 引数が 1 つなら、vec2 はそのまま通し、それ以外は包む
        const a = argNodes[0]
        if (a.kind === 'vec2') return a
        return makeVec2FromChildren(a, { expr: '0.0', kind: 'scalar' })
      }
    }
    switch (name) {
      case 'complexAdd': {
        const a = argNodes[0]
        const b = argNodes[1]
        return makeVec2FromChildren(
          { expr: `${comp(a, 'x')} + ${comp(b, 'x')}`, kind: 'scalar' },
          { expr: `${comp(a, 'y')} + ${comp(b, 'y')}`, kind: 'scalar' },
        )
      }
      case 'complexSub':
      case 'complexSubtract': {
        const a = argNodes[0]
        const b = argNodes[1]
        return makeVec2FromChildren(
          { expr: `${comp(a, 'x')} - ${comp(b, 'x')}`, kind: 'scalar' },
          { expr: `${comp(a, 'y')} - ${comp(b, 'y')}`, kind: 'scalar' },
        )
      }
      case 'complexMultiply':
      case 'complexMul': {
        const a = argNodes[0]
        const b = argNodes[1]
        // 片側が 0 や 1 の乗算を簡単な式へ落とす補助関数
        function mulClean(p, q) {
          if (!p || !q) return '0.0'
          const ps = p.trim()
          const qs = q.trim()
          if (ps === '0.0' || qs === '0.0') return '0.0'
          if (ps === '1.0') return `(${qs})`
          if (qs === '1.0') return `(${ps})`
          return `(${ps})*(${qs})`
        }
        const ax = comp(a, 'x')
        const ay = comp(a, 'y')
        const bx = comp(b, 'x')
        const by = comp(b, 'y')
        const real = `${mulClean(ax, bx)} - ${mulClean(ay, by)}`
        const imag = `${mulClean(ax, by)} + ${mulClean(ay, bx)}`
        return makeVec2FromChildren({ expr: real, kind: 'scalar' }, { expr: imag, kind: 'scalar' })
      }
      case 'complexDivide':
      case 'complexDiv': {
        const a = argNodes[0]
        const b = argNodes[1]
        const bx = comp(b, 'x')
        const by = comp(b, 'y')
        // CPU とそろえる特別処理。
        // 分子が vec2(A,0)、分母が vec2(0,B) のときは実数比として扱う。
        try {
          if (
            a &&
            a.kind === 'vec2' &&
            b &&
            b.kind === 'vec2' &&
            (String(a.y || '').trim() === '0.0' || String(a.y || '').trim() === '0') &&
            (String(b.x || '').trim() === '0.0' || String(b.x || '').trim() === '0')
          ) {
            const Aexpr = String(a.x || '0.0').trim()
            const Bexpr = String(b.y || '0.0').trim()
            // [A,0] / [0,B] の複素除算結果は [0, -A / B]。
            // GPU 側では SAFE_EPS で分母を下駄履きし、Inf / NaN を防ぐ。
            const realExpr = `0.0`
            // B の符号を保ちつつ、極小分母を避ける安全形へ変換する
            const imagExpr = `(-(${Aexpr}) * sign((${Bexpr}))) / max(abs((${Bexpr})), ${SAFE_EPS})`
            return makeVec2FromChildren({ expr: realExpr, kind: 'scalar' }, { expr: imagExpr, kind: 'scalar' })
          }
        } catch (_e) {
          // 失敗時は通常の複素除算へ進む
        }
        // 一般的な複素除算
        // 成分式は演算子優先順位の崩れを避けるため必ずかっこで包む
        const denom = `max(((${bx})*(${bx}) + (${by})*(${by})), ${SAFE_EPS})`
        return makeVec2FromChildren(
          {
            expr: `((${comp(a, 'x')})*(${bx}) + (${comp(a, 'y')})*(${by})) / ${denom}`,
            kind: 'scalar',
          },
          {
            expr: `((${comp(a, 'y')})*(${bx}) - (${comp(a, 'x')})*(${by})) / ${denom}`,
            kind: 'scalar',
          },
        )
      }
      case 'complexConj': {
        const a = argNodes[0]
        return makeVec2FromChildren(
          { expr: comp(a, 'x'), kind: 'scalar' },
          { expr: `-(${comp(a, 'y')})`, kind: 'scalar' },
        )
      }
      case 'complexRe': {
        const a = argNodes[0]
        return { expr: comp(a, 'x'), kind: 'scalar' }
      }
      case 'complexIm': {
        const a = argNodes[0]
        return { expr: comp(a, 'y'), kind: 'scalar' }
      }
      case 'complexAbs': {
        const a = argNodes[0]
        const full = a.kind === 'vec2' ? a.expr : `vec2<f32>(${a.expr}, 0.0)`
        return { expr: `length(${full})`, kind: 'scalar' }
      }
      case 'complexArg': {
        const a = argNodes[0]
        return { expr: `atan2(${comp(a, 'y')}, ${comp(a, 'x')})`, kind: 'scalar' }
      }
      case 'complexAbs2': {
        const a = argNodes[0]
        const x = comp(a, 'x')
        const y = comp(a, 'y')
        return { expr: `((${x}) * (${x}) + (${y}) * (${y}))`, kind: 'scalar' }
      }
      case 'complexSinh': {
        const a = argNodes[0]
        return makeVec2FromChildren(
          { expr: `sinh(${comp(a, 'x')}) * cos(${comp(a, 'y')})`, kind: 'scalar' },
          { expr: `cosh(${comp(a, 'x')}) * sin(${comp(a, 'y')})`, kind: 'scalar' },
        )
      }
      case 'complexCosh': {
        const a = argNodes[0]
        return makeVec2FromChildren(
          { expr: `cosh(${comp(a, 'x')}) * cos(${comp(a, 'y')})`, kind: 'scalar' },
          { expr: `sinh(${comp(a, 'x')}) * sin(${comp(a, 'y')})`, kind: 'scalar' },
        )
      }
      case 'complexTanh': {
        const a = argNodes[0]
        const sx = `sinh(${comp(a, 'x')}) * cos(${comp(a, 'y')})`
        const sy = `cosh(${comp(a, 'x')}) * sin(${comp(a, 'y')})`
        const cx = `cosh(${comp(a, 'x')}) * cos(${comp(a, 'y')})`
        const cy = `sinh(${comp(a, 'x')}) * sin(${comp(a, 'y')})`
        const denom = `max((${cx}) * (${cx}) + (${cy}) * (${cy}), ${SAFE_EPS})`
        return makeVec2FromChildren(
          { expr: `((${sx}) * (${cx}) + (${sy}) * (${cy})) / ${denom}`, kind: 'scalar' },
          { expr: `((${sy}) * (${cx}) - (${sx}) * (${cy})) / ${denom}`, kind: 'scalar' },
        )
      }
      case 'complexFloor': {
        const a = argNodes[0]
        return makeVec2FromChildren(
          { expr: `floor(${comp(a, 'x')})`, kind: 'scalar' },
          { expr: `floor(${comp(a, 'y')})`, kind: 'scalar' },
        )
      }
      case 'complexFract': {
        const a = argNodes[0]
        return makeVec2FromChildren(
          { expr: `fract(${comp(a, 'x')})`, kind: 'scalar' },
          { expr: `fract(${comp(a, 'y')})`, kind: 'scalar' },
        )
      }
      case 'complexMod': {
        const a = argNodes[0]
        const b = argNodes[1]
        const bx = scalarFromNode(b)
        const by = b?.kind === 'vec2' ? comp(b, 'y') : bx
        return makeVec2FromChildren(
          { expr: safeModExpr(comp(a, 'x'), bx), kind: 'scalar' },
          { expr: safeModExpr(comp(a, 'y'), by), kind: 'scalar' },
        )
      }
      case 'complexMin': {
        const a = argNodes[0]
        const b = argNodes[1]
        const bx = scalarFromNode(b)
        const by = b?.kind === 'vec2' ? comp(b, 'y') : bx
        return makeVec2FromChildren(
          { expr: `min(${comp(a, 'x')}, ${bx})`, kind: 'scalar' },
          { expr: `min(${comp(a, 'y')}, ${by})`, kind: 'scalar' },
        )
      }
      case 'complexMax': {
        const a = argNodes[0]
        const b = argNodes[1]
        const bx = scalarFromNode(b)
        const by = b?.kind === 'vec2' ? comp(b, 'y') : bx
        return makeVec2FromChildren(
          { expr: `max(${comp(a, 'x')}, ${bx})`, kind: 'scalar' },
          { expr: `max(${comp(a, 'y')}, ${by})`, kind: 'scalar' },
        )
      }
      case 'complexClamp': {
        const a = argNodes[0]
        const lo = argNodes[1]
        const hi = argNodes[2]
        const lox = scalarFromNode(lo)
        const loy = lo?.kind === 'vec2' ? comp(lo, 'y') : lox
        const hix = scalarFromNode(hi)
        const hiy = hi?.kind === 'vec2' ? comp(hi, 'y') : hix
        return makeVec2FromChildren(
          { expr: `clamp(${comp(a, 'x')}, ${lox}, ${hix})`, kind: 'scalar' },
          { expr: `clamp(${comp(a, 'y')}, ${loy}, ${hiy})`, kind: 'scalar' },
        )
      }
      case 'complexRotate': {
        const a = argNodes[0]
        const angle = scalarFromNode(argNodes[1])
        const cs = `cos(${angle})`
        const sn = `sin(${angle})`
        return makeVec2FromChildren(
          { expr: `(${comp(a, 'x')}) * (${cs}) - (${comp(a, 'y')}) * (${sn})`, kind: 'scalar' },
          { expr: `(${comp(a, 'x')}) * (${sn}) + (${comp(a, 'y')}) * (${cs})`, kind: 'scalar' },
        )
      }
      case 'complexFold': {
        const a = argNodes[0]
        return makeVec2FromChildren(
          { expr: `abs(${comp(a, 'x')})`, kind: 'scalar' },
          { expr: `abs(${comp(a, 'y')})`, kind: 'scalar' },
        )
      }
      case 'complexBoxFold': {
        const a = argNodes[0]
        const radius = `abs(${scalarFromNode(argNodes[1])})`
        const fold = (value) =>
          `select(select(${value}, -2.0 * (${radius}) - (${value}), (${value}) < -(${radius})), 2.0 * (${radius}) - (${value}), (${value}) > (${radius}))`
        return makeVec2FromChildren(
          { expr: fold(comp(a, 'x')), kind: 'scalar' },
          { expr: fold(comp(a, 'y')), kind: 'scalar' },
        )
      }
      case 'complexCos': {
        const aNode = node.args?.[0]
        // 引数が純粋な実数なら scalar cos を優先する。
        // 純虚数では CPU に合わせて複素数展開を使う。
        if (aNode) {
          const aTN = astToWGSL(aNode, tokensTable)
          const realScalar = extractRealScalarFromTN(aTN)
          if (realScalar) return { expr: `cos(${realScalar})`, kind: 'scalar' }
        }
        // scalarExpr が使えるときの z 倍スケール展開を作る補助関数
        function buildScaledCos(sx, sy) {
          // cosh/sinh を用いることで exp() の二重計算をなくし、Float32 で |y|>88.7 のオーバーフロー脆弱性を低減
          const ch = `cosh(${sy})`
          const sh = `sinh(${sy})`
          return makeVec2FromChildren(
            { expr: `cos(${sx})*${ch}`, kind: 'scalar' },
            { expr: `-sin(${sx})*${sh}`, kind: 'scalar' },
          )
        }
        // 1. 二項演算の z * scalar 形式
        if (aNode && aNode.type === 'binary' && aNode.op === '*') {
          const left = aNode.left,
            right = aNode.right
          let zPart = null,
            scalarPart = null
          if (left && left.type === 'ident' && left.name === 'z') {
            zPart = left
            scalarPart = right
          } else if (right && right.type === 'ident' && right.name === 'z') {
            zPart = right
            scalarPart = left
          }
          if (zPart && scalarPart) {
            const scalarTN = astToWGSL(scalarPart, tokensTable)
            const zTN = astToWGSL(zPart, tokensTable)
            const scalarExpr = extractRealScalarFromTN(scalarTN)
            if (scalarExpr) {
              const sx = `(${comp(zTN, 'x')})*(${scalarExpr})`
              const sy = `(${comp(zTN, 'y')})*(${scalarExpr})`
              return buildScaledCos(sx, sy)
            }
          }
        }
        // 2. complexMultiply(z, scalar) / complexMul 形式
        if (aNode && aNode.type === 'call' && /complexMul/i.test(aNode.name)) {
          const ia = aNode.args || []
          let zPart = null,
            scalarPart = null
          if (ia[0] && ia[0].type === 'ident' && ia[0].name === 'z') {
            zPart = ia[0]
            scalarPart = ia[1]
          } else if (ia[1] && ia[1].type === 'ident' && ia[1].name === 'z') {
            zPart = ia[1]
            scalarPart = ia[0]
          }
          if (zPart && scalarPart) {
            const scalarTN = astToWGSL(scalarPart, tokensTable)
            const zTN = astToWGSL(zPart, tokensTable)
            const scalarExpr = extractRealScalarFromTN(scalarTN)
            if (scalarExpr) {
              const sx = `(${comp(zTN, 'x')})*(${scalarExpr})`
              const sy = `(${comp(zTN, 'y')})*(${scalarExpr})`
              return buildScaledCos(sx, sy)
            }
          }
        }
        // それ以外は通常の複素 cos 展開を使う
        const a = argNodes[0]
        return makeVec2FromChildren(
          {
            // cosh(虚部) で直接計算（exp2回より精度良ま、コード簡潔）
            expr: `cos(${comp(a, 'x')}) * cosh(${comp(a, 'y')})`,
            kind: 'scalar',
          },
          {
            expr: `-sin(${comp(a, 'x')}) * sinh(${comp(a, 'y')})`,
            kind: 'scalar',
          },
        )
      }
      case 'complexSin': {
        const aNode = node.args?.[0]
        // complexSin も同様に、純実数なら scalar sin を優先する
        if (aNode) {
          const aTN = astToWGSL(aNode, tokensTable)
          const realScalar = extractRealScalarFromTN(aTN)
          if (realScalar) return { expr: `sin(${realScalar})`, kind: 'scalar' }
        }
        if (aNode && aNode.type === 'binary' && aNode.op === '*') {
          const left = aNode.left,
            right = aNode.right
          let zPart = null,
            scalarPart = null
          if (left && left.type === 'ident' && left.name === 'z') {
            zPart = left
            scalarPart = right
          } else if (right && right.type === 'ident' && right.name === 'z') {
            zPart = right
            scalarPart = left
          }
          if (zPart && scalarPart) {
            const scalarTN = astToWGSL(scalarPart, tokensTable)
            const zTN = astToWGSL(zPart, tokensTable)
            const scalarExpr = extractRealScalarFromTN(scalarTN)
            if (scalarExpr) {
              const sx = `(${comp(zTN, 'x')})*(${scalarExpr})`
              const sy = `(${comp(zTN, 'y')})*(${scalarExpr})`
              // cosh/sinh で直接計算（exp2回よりコード簡潔、Float32 オーバーフロー脆弱性を低減）
              const ch = `cosh(${sy})`
              const sh = `sinh(${sy})`
              return makeVec2FromChildren(
                { expr: `cos(${sx})*${ch}`, kind: 'scalar' },
                { expr: `-sin(${sx})*${sh}`, kind: 'scalar' },
              )
            }
            // scalarExpr が complex（純粋な実スカラーでない）場合の sin(z*s) 展開
            // sz_y = Im(z) * Re(s) として cosh/sinh を使う（Re(s)=null なので近似計算は難しく、
            // ここでは z*複素数スカラーをイテレーションの虚部として素直に扱う）
            const sy_approx = comp(zTN, 'y')
            return makeVec2FromChildren(
              {
                expr: `sin(${comp(zTN, 'x')}) * cosh(${sy_approx})`,
                kind: 'scalar',
              },
              {
                expr: `cos(${comp(zTN, 'x')}) * sinh(${sy_approx})`,
                kind: 'scalar',
              },
            )
          }
        }
        const a = argNodes[0]
        return makeVec2FromChildren(
          {
            // cosh(虚部) で直接計算（exp2回より精度良ま、コード簡潔）
            expr: `sin(${comp(a, 'x')}) * cosh(${comp(a, 'y')})`,
            kind: 'scalar',
          },
          {
            expr: `cos(${comp(a, 'x')}) * sinh(${comp(a, 'y')})`,
            kind: 'scalar',
          },
        )
      }
      case 'complexLog': {
        const a = argNodes[0]
        const full = a.kind === 'vec2' ? a.expr : `vec2<f32>(${a.expr}, 0.0)`
        // 純実数または純虚数なら、より単純な log / atan 形を使う
        const realScalar = extractRealScalarFromTN(a)
        const imagScalar = extractImagScalarFromTN(a)
        if (realScalar) {
          // log(real) は虚部 0 の形で表せる
          return makeVec2FromChildren(
            { expr: `log(max(${realScalar}, ${SAFE_EPS}))`, kind: 'scalar' },
            { expr: `0.0`, kind: 'scalar' },
          )
        }
        if (imagScalar) {
          // log(i*imag) = log(|imag|) + i * pi/2 * sign(imag)
          return makeVec2FromChildren(
            {
              expr: `log(max(abs(${imagScalar}), ${SAFE_EPS}))`,
              kind: 'scalar',
            },
            { expr: `sign(${imagScalar}) * 1.57079632679`, kind: 'scalar' },
          )
        }
        return makeVec2FromChildren(
          // 0.5*log(dot(z,z)) = log(|z|) だが sqrt の丸め誤差が log に伝播しない分、length() より若干精度良妄
          {
            expr: `0.5 * log(max(dot(${full}, ${full}), ${SAFE_EPS * SAFE_EPS}))`,
            kind: 'scalar',
          },
          { expr: `atan2(${comp(a, 'y')}, ${comp(a, 'x')})`, kind: 'scalar' },
        )
      }
      case 'complexLog10': {
        const a = argNodes[0]
        const full = a.kind === 'vec2' ? a.expr : `vec2<f32>(${a.expr}, 0.0)`
        const log10 = '2.302585092994046' // ln(10)
        return makeVec2FromChildren(
          {
            // 0.5*log(dot(z,z)) で sqrt の層絍誤差を防ぐ
            expr: `0.5 * log(max(dot(${full}, ${full}), ${SAFE_EPS * SAFE_EPS}))/${log10}`,
            kind: 'scalar',
          },
          {
            expr: `atan2(${comp(a, 'y')}, ${comp(a, 'x')}) / ${log10}`,
            kind: 'scalar',
          },
        )
      }
      case 'complexExp': {
        const a = argNodes[0]
        return makeVec2FromChildren(
          { expr: `exp(${comp(a, 'x')})*cos(${comp(a, 'y')})`, kind: 'scalar' },
          { expr: `exp(${comp(a, 'x')})*sin(${comp(a, 'y')})`, kind: 'scalar' },
        )
      }
      case 'complexTan': {
        const a = argNodes[0]
        // cosh/sinh で直接計算（exp の 4 回計算を 2 回に削減、Float32 オーバーフロー脆弱性を低減）
        // sin の各成分
        const sx = `sin(${comp(a, 'x')}) * cosh(${comp(a, 'y')})`
        const sy = `cos(${comp(a, 'x')}) * sinh(${comp(a, 'y')})`
        // cos の各成分
        const cx = `cos(${comp(a, 'x')}) * cosh(${comp(a, 'y')})`
        const cy = `-sin(${comp(a, 'x')}) * sinh(${comp(a, 'y')})`
        const denom = `max((${cx}*${cx} + ${cy}*${cy}), ${SAFE_EPS})`
        return makeVec2FromChildren(
          { expr: `((${sx}*${cx} + ${sy}*${cy}) / ${denom})`, kind: 'scalar' },
          { expr: `((${sy}*${cx} - ${sx}*${cy}) / ${denom})`, kind: 'scalar' },
        )
      }
      case 'complexSqrt': {
        const a = argNodes[0]
        const full = a.kind === 'vec2' ? a.expr : `vec2<f32>(${a.expr}, 0.0)`
        const r = `length(${full})`
        // 純実数または純虚数なら、より単純な sqrt 形を使う
        const realScalar = extractRealScalarFromTN(a)
        const imagScalar = extractImagScalarFromTN(a)
        if (realScalar) {
          // sqrt(real) は負数のとき虚部側へ回す
          return makeVec2FromChildren(
            { expr: `sqrt(max(${realScalar}, 0.0))`, kind: 'scalar' },
            // 負の実数では主値の虚部は正になる
            { expr: `sqrt(max(-${realScalar}, 0.0))`, kind: 'scalar' },
          )
        }
        if (imagScalar) {
          // sqrt(i*imag) の既知形を使う
          return makeVec2FromChildren(
            { expr: `sqrt(abs(${imagScalar})/2.0)`, kind: 'scalar' },
            {
              expr: `sign(${imagScalar}) * sqrt(abs(${imagScalar})/2.0)`,
              kind: 'scalar',
            },
          )
        }
        const realPart = `sqrt(max((${r} + ${comp(a, 'x')}) / 2.0, 0.0))`
        // 虚部: r ≈ a.x の場合（z が正の実軸付近）は (r - a.x) で桁落ちが生じる
        // a.x >= 0 のとき: abs(y)/sqrt(2*(r+x)) が等価かつ数値的に安定（r+x は桁落ちしない）
        // a.x < 0  のとき: r - a.x = r + |x| なので元の式のまま安定
        // WGSL の select(false_val, true_val, condition) で分岐
        const imagPart =
          `sign(${comp(a, 'y')}) * select(` +
          `sqrt(max((${r} - ${comp(a, 'x')}) / 2.0, 0.0)), ` +
          `abs(${comp(a, 'y')}) / sqrt(max(2.0 * (${r} + ${comp(a, 'x')}), ${SAFE_EPS})), ` +
          `(${comp(a, 'x')}) >= 0.0)`
        return makeVec2FromChildren({ expr: realPart, kind: 'scalar' }, { expr: imagPart, kind: 'scalar' })
      }
      case 'complexPower': {
        const base = argNodes[0]
        const expNode = argNodes[1]
        const fullBase = base.kind === 'vec2' ? base.expr : `vec2<f32>(${base.expr}, 0.0)`
        // 底 0 は特別扱いし、log(0) 由来の NaN を防ぐ
        // CPU と同様に 0^0=1、0^(正)=0、0^(負)=Infinity 扱いにする
        const baseLen = `length(${fullBase})`
        const expX = comp(expNode, 'x')
        const expY = comp(expNode, 'y')
        const isZeroBase = `(${baseLen} < ${SAFE_EPS})`
        const isZeroExp = `(abs(${expX}) < ${SAFE_EPS} && abs(${expY}) < ${SAFE_EPS})`
        // 虚部がほぼ 0 なら純実数指数として扱う
        const isPureRealExp = `(abs(${expY}) < ${SAFE_EPS})`
        const isNegPureRealExp = `(${isPureRealExp} && (${expX} < 0.0))`
        // 底が 0 の場合の戻り値を場合分けする
        const zeroResult = `select(
          select(vec2<f32>(0.0, 0.0), vec2<f32>(1e20, 0.0), ${isNegPureRealExp}),
          vec2<f32>(1.0, 0.0),
          ${isZeroExp}
        )`
        const safeLen = `max(length(${fullBase}), ${SAFE_EPS})`
        // 角度 A と大きさ mag の式を組み立てる。
        // 0 との乗算は後段で壊れやすいため、最初から省いた形を作る。
        const baseAtan = `atan2(${comp(base, 'y')}, ${comp(base, 'x')})`
        const baseLog = `log(${safeLen})`
        const wrap = (s) => `(${s})`
        const A_parts = []
        if (!tryNumericValue(expX).ok || tryNumericValue(expX).value !== 0)
          A_parts.push(`${wrap(expX)} * ${wrap(baseAtan)}`)
        if (!tryNumericValue(expY).ok || tryNumericValue(expY).value !== 0)
          A_parts.push(`${wrap(expY)} * ${wrap(baseLog)}`)
        const A = A_parts.length > 0 ? A_parts.join(' + ') : '0.0'
        const mag_parts = []
        if (!tryNumericValue(expX).ok || tryNumericValue(expX).value !== 0)
          mag_parts.push(`${wrap(expX)} * ${wrap(baseLog)}`)
        if (!tryNumericValue(expY).ok || tryNumericValue(expY).value !== 0)
          mag_parts.push(`- ${wrap(expY)} * ${wrap(baseAtan)}`)
        const mag = mag_parts.length > 0 ? `exp(${mag_parts.join(' + ')})` : `exp(0.0)`
        const normalResult = makeVec2FromChildren(
          { expr: `${mag} * cos(${A})`, kind: 'scalar' },
          { expr: `${mag} * sin(${A})`, kind: 'scalar' },
        )
        // 底が 0 なら zeroResult、それ以外は通常結果を返す
        return {
          expr: `select(${normalResult.expr}, ${zeroResult}, ${isZeroBase})`,
          kind: 'vec2',
        }
      }
      case 'pow': {
        const base = argNodes[0]
        const expN = argNodes[1]
        const fullBase = base.kind === 'vec2' ? base.expr : `vec2<f32>(${base.expr}, 0.0)`
        const safeLen = `max(length(${fullBase}), ${SAFE_EPS})`
        const expX = comp(expN, 'x')
        const expY = comp(expN, 'y')
        const baseAtan = `atan2(${comp(base, 'y')}, ${comp(base, 'x')})`
        const baseLog = `log(${safeLen})`
        // complexPower と同様に底 0 を特別扱いする
        const isZeroBase = `(length(${fullBase}) < ${SAFE_EPS})`
        const isZeroExp = `(abs(${expX}) < ${SAFE_EPS} && abs(${expY}) < ${SAFE_EPS})`
        const isPureRealExp = `(abs(${expY}) < ${SAFE_EPS})`
        const isNegPureRealExp = `(${isPureRealExp} && (${expX} < 0.0))`
        const A_parts = []
        if (expX !== '0.0' && expX !== '0') A_parts.push(`${expX} * ${baseAtan}`)
        if (expY !== '0.0' && expY !== '0') A_parts.push(`${expY} * ${baseLog}`)
        const A = A_parts.length > 0 ? A_parts.join(' + ') : '0.0'
        const mag_parts = []
        if (expX !== '0.0' && expX !== '0') mag_parts.push(`${expX} * ${baseLog}`)
        if (expY !== '0.0' && expY !== '0') mag_parts.push(`- ${expY} * ${baseAtan}`)
        const mag = mag_parts.length > 0 ? `exp(${mag_parts.join(' ')})` : `exp(0.0)`
        const normalResult = makeVec2FromChildren(
          { expr: `${mag} * cos(${A})`, kind: 'scalar' },
          { expr: `${mag} * sin(${A})`, kind: 'scalar' },
        )
        const zeroResult = `select(
          select(vec2<f32>(0.0, 0.0), vec2<f32>(1e20, 0.0), ${isNegPureRealExp}),
          vec2<f32>(1.0, 0.0),
          ${isZeroExp}
        )`
        return {
          expr: `select(${normalResult.expr}, ${zeroResult}, ${isZeroBase})`,
          kind: 'vec2',
        }
      }
      default: {
        const scalarFuncs = new Set(['log', 'sqrt', 'sin', 'cos', 'exp', 'abs'])
        if (scalarFuncs.has(name) && argNodes.length === 1) {
          const a = argNodes[0]
          // vec2 でも純実数なら scalar 形式を優先する。
          // 純虚数は CPU に合わせて複素展開を使う。
          if (a.kind === 'vec2') {
            const realScalar = extractRealScalarFromTN(a)
            if (realScalar) {
              return { expr: `${name}(${realScalar})`, kind: 'scalar' }
            }
            // 複素引数向けに主要スカラー関数を展開する
            switch (name) {
              case 'sin':
                return makeVec2FromChildren(
                  {
                    // cosh/sinh で exp 2回計算を削減、Float32 オーバーフロー脆弱性を低減
                    expr: `sin(${comp(a, 'x')}) * cosh(${comp(a, 'y')})`,
                    kind: 'scalar',
                  },
                  {
                    expr: `cos(${comp(a, 'x')}) * sinh(${comp(a, 'y')})`,
                    kind: 'scalar',
                  },
                )
              case 'cos':
                return makeVec2FromChildren(
                  {
                    expr: `cos(${comp(a, 'x')}) * cosh(${comp(a, 'y')})`,
                    kind: 'scalar',
                  },
                  {
                    expr: `-sin(${comp(a, 'x')}) * sinh(${comp(a, 'y')})`,
                    kind: 'scalar',
                  },
                )
              case 'exp':
                return makeVec2FromChildren(
                  {
                    expr: `exp(${comp(a, 'x')})*cos(${comp(a, 'y')})`,
                    kind: 'scalar',
                  },
                  {
                    expr: `exp(${comp(a, 'x')})*sin(${comp(a, 'y')})`,
                    kind: 'scalar',
                  },
                )
              case 'log': {
                const full = a.kind === 'vec2' ? a.expr : `vec2<f32>(${a.expr}, 0.0)`
                return makeVec2FromChildren(
                  {
                    expr: `log(max(length(${full}), ${SAFE_EPS}))`,
                    kind: 'scalar',
                  },
                  {
                    expr: `atan2(${comp(a, 'y')}, ${comp(a, 'x')})`,
                    kind: 'scalar',
                  },
                )
              }
              case 'sqrt': {
                const full = a.kind === 'vec2' ? a.expr : `vec2<f32>(${a.expr}, 0.0)`
                const r = `length(${full})`
                const realPart = `sqrt(max((${r} + ${comp(a, 'x')}) / 2.0, 0.0))`
                const imagPart = `sign(${comp(a, 'y')}) * sqrt(max((${r} - ${comp(a, 'x')}) / 2.0, 0.0))`
                return makeVec2FromChildren({ expr: realPart, kind: 'scalar' }, { expr: imagPart, kind: 'scalar' })
              }
              default:
                return { expr: `${name}(length(${a.expr}))`, kind: 'scalar' }
            }
          }
          return { expr: `${name}(${a.expr})`, kind: 'scalar' }
        }
        return {
          expr: `${name}(${argNodes.map((n) => n.expr).join(', ')})`,
          kind: 'scalar',
        }
      }
    }
  }

  if (node.type === 'binary') {
    const L = astToWGSL(node.left, tokensTable)
    const R = astToWGSL(node.right, tokensTable)
    const op = node.op
    // どちらかが vec2 なら、演算子ごとの複素数ルールで扱う
    if (L.kind === 'vec2' || R.kind === 'vec2') {
      const lx = comp(L, 'x')
      const ly = comp(L, 'y')
      const rx = comp(R, 'x')
      const ry = comp(R, 'y')
      // scalar と vec2 の乗除算では両成分に同じ係数を掛ける
      const Lvec = L.kind === 'vec2'
      const Rvec = R.kind === 'vec2'
      if (!Lvec && Rvec) {
        // 左が scalar、右が vec2
        if (op === '*' || op === '/') {
          // scalar を両成分へ複製する
          return makeVec2FromChildren(
            { expr: `(${L.expr}) ${op} (${rx})`, kind: 'scalar' },
            { expr: `(${L.expr}) ${op} (${ry})`, kind: 'scalar' },
          )
        }
        // 加減算では scalar は実部側だけへ作用させる
        return makeVec2FromChildren(
          { expr: `(${L.expr}) ${op} (${rx})`, kind: 'scalar' },
          { expr: `0.0 ${op} (${ry})`, kind: 'scalar' },
        )
      }
      if (Lvec && !Rvec) {
        if (op === '*' || op === '/') {
          return makeVec2FromChildren(
            { expr: `(${lx}) ${op} (${R.expr})`, kind: 'scalar' },
            { expr: `(${ly}) ${op} (${R.expr})`, kind: 'scalar' },
          )
        }
        return makeVec2FromChildren(
          { expr: `(${lx}) ${op} (${R.expr})`, kind: 'scalar' },
          { expr: `(${ly}) ${op} (0.0)`, kind: 'scalar' },
        )
      }
      // 両方 vec2 のとき、加減算は成分ごと、乗除算は複素数演算にする
      if (op === '*') {
        // 複素数乗算: (a+ib)*(c+id)
        const real = `(${lx})*(${rx}) - (${ly})*(${ry})`
        const imag = `(${lx})*(${ry}) + (${ly})*(${rx})`
        return makeVec2FromChildren({ expr: real, kind: 'scalar' }, { expr: imag, kind: 'scalar' })
      }
      if (op === '/') {
        // 複素数除算: (a+ib)/(c+id)
        // 2 乗対象は必ずかっこで包む
        const denom = `max(((${rx})*(${rx}) + (${ry})*(${ry})), ${SAFE_EPS})`
        const real = `((${lx})*(${rx}) + (${ly})*(${ry})) / ${denom}`
        const imag = `((${ly})*(${rx}) - (${lx})*(${ry})) / ${denom}`
        return makeVec2FromChildren({ expr: real, kind: 'scalar' }, { expr: imag, kind: 'scalar' })
      }
      // 加減算は成分ごとに処理する
      return makeVec2FromChildren(
        { expr: `(${lx}) ${op} (${rx})`, kind: 'scalar' },
        { expr: `(${ly}) ${op} (${ry})`, kind: 'scalar' },
      )
    }
    // 両方ともスカラー
    return { expr: `(${L.expr}) ${op} (${R.expr})`, kind: 'scalar' }
  }

  if (node.type === 'unary') {
    const a = astToWGSL(node.arg, tokensTable)
    if (a.kind === 'vec2') {
      return makeVec2FromChildren(
        { expr: `-(${comp(a, 'x')})`, kind: 'scalar' },
        { expr: `-(${comp(a, 'y')})`, kind: 'scalar' },
      )
    }
    return { expr: `-(${a.expr})`, kind: 'scalar' }
  }

  if (node.type === 'tuple') {
    // [0, [cReal, cImag]] のような入れ子 tuple を特別扱いし、
    // 文字列置換で拾えなかった形を直接 vec2 として組み立てる。
    if (node.items && node.items.length === 2) {
      const a = node.items[0]
      const b = node.items[1]
      // [0, [cReal, cImag]] -> vec2(0.0, cImag)
      if (a.type === 'number' && (a.value === '0' || a.value === '0.0') && b.type === 'tuple' && b.items.length === 2) {
        const bx = b.items[0]
        const by = b.items[1]
        // cImag / cReal が見えていれば優先して使う
        if (by.type === 'ident' && /cImag|c\.y/.test(by.name || '')) {
          return makeVec2FromChildren(
            { expr: '0.0', kind: 'scalar' },
            {
              expr: by.type === 'ident' ? (by.name === 'cImag' ? 'c.y' : by.name) : '0.0',
              kind: 'scalar',
            },
          )
        }
        // それ以外は内側を WGSL 化して y 成分を使う
        const inner = astToWGSL(b, tokensTable)
        return makeVec2FromChildren({ expr: '0.0', kind: 'scalar' }, inner)
      }
      // [[cReal, cImag][1], 0] -> vec2(cImag, 0.0)
      if (b.type === 'number' && (b.value === '0' || b.value === '0.0') && a.type === 'tuple' && a.items.length === 2) {
        const ax = a.items[0]
        const ay = a.items[1]
        if (ax.type === 'ident' && ay.type === 'ident' && /cReal|cImag/.test(ax.name || '' + ay.name)) {
          // vec2(cReal, 0) を作り、後で c.x へ正規化させる
          const left = astToWGSL(a, tokensTable)
          return makeVec2FromChildren(left, { expr: '0.0', kind: 'scalar' })
        }
      }
    }
    const items = node.items.map((it) => astToWGSL(it, tokensTable))
    if (items.length === 2) return makeVec2FromChildren(items[0], items[1])
    return makeVec2FromChildren(
      { expr: items.map((it) => it.expr).join(', '), kind: 'scalar' },
      { expr: '0.0', kind: 'scalar' },
    )
  }
  return { expr: '', kind: 'scalar' }
}

export function jsExprToWGSL_ast(expr) {
  if (!expr || typeof expr !== 'string') return ''
  // 軽めの事前正規化を行う
  let s = expr.trim()
  // Math.* 呼び出しを通常名へ直し、sinh / cosh も扱う
  function replaceMathFns(input) {
    let out = ''
    let i = 0
    while (i < input.length) {
      const m = input.slice(i).match(/^Math\.(sin|cos|exp|log|sqrt|abs|atan2|pow|tan|sinh|cosh)\s*\(/)
      if (!m) {
        out += input[i]
        i++
        continue
      }
      const name = m[1]
      // この呼び出しに対応する閉じかっこを探す
      const start = i + m[0].indexOf('(') + 1
      let depth = 0
      let j = start
      let found = false
      for (; j < input.length; j++) {
        const ch = input[j]
        if (ch === '(') depth++
        else if (ch === ')') {
          if (depth === 0) {
            found = true
            break
          }
          depth--
        }
      }
      if (!found) {
        // 壊れた式なら残りをそのまま出す
        out += input.slice(i)
        break
      }
      const inner = input.slice(start, j)
      if (name === 'sinh') {
        // WGSL \u306b\u306f\u7d44\u307f\u8fbc\u307f sinh/cosh \u304c\u3042\u308b\u306e\u3067\u305d\u306e\u307e\u307e\u4f7f\u7528\uff08exp 2\u56de\u8a08\u7b97\u306e\u5c55\u958b\u306f\u4e0d\u8981\uff09
        out += `sinh(${inner})`
      } else if (name === 'cosh') {
        out += `cosh(${inner})`
      } else {
        // 単純な改名だけでよい
        out += `${name}(${inner})`
      }
      i = j + 1
    }
    return out
  }
  s = replaceMathFns(s)
  // 入力に残った Math 定数を数値へ置き換える
  s = s.replace(/\bMath\.E\b/g, '2.718281828459045')
  s = s.replace(/\bMath\.PI\b/g, '3.141592653589793')
  // 単独の e も必要なら数値定数へ置き換える
  s = s.replace(/\be\b/g, '2.718281828459045')
  // '(' や ',' の直後の単項 + を消す
  s = s.replace(/([(,])\s*\+\s*/g, '$1')
  s = s.replace(/\bz\s*\[\s*0\s*\]/g, 'z.x').replace(/\bz\s*\[\s*1\s*\]/g, 'z.y')
  s = s.replace(/\bc\s*\[\s*0\s*\]/g, 'c.x').replace(/\bc\s*\[\s*1\s*\]/g, 'c.y')
  s = s.replace(/\bRe\s*\(\s*z\s*\)/g, 'z.x').replace(/\bIm\s*\(\s*z\s*\)/g, 'z.y')
  // parser 特有の配列インデックス形を先に処理する。
  // cReal -> c.x の変換前に済ませておく必要がある。
  s = s.replace(/\[\[\s*cReal\s*,\s*cImag\s*\]\s*\[\s*0\s*\]\s*,\s*0\s*\]/g, 'vec2<f32>(c.x, 0.0)')
  s = s.replace(/\[\[\s*cReal\s*,\s*cImag\s*\]\s*\[\s*1\s*\]\s*,\s*0\s*\]/g, 'vec2<f32>(c.y, 0.0)')
  // 先頭が 0、後ろが複素数ペアの形を処理する
  s = s.replace(/\[\s*0\s*,\s*\[\s*cReal\s*,\s*cImag\s*\]\s*\]/g, 'vec2<f32>(0.0, c.y)')
  // [0, [cReal, cImag][1]] のような形を処理する
  s = s.replace(/\[\s*0\s*,\s*\[\s*cReal\s*,\s*cImag\s*\]\s*\[\s*1(?:\.0)?\s*\]\s*\]/g, 'vec2<f32>(0.0, c.y)')
  // すでに c.x / c.y へ変換済みの形も処理する
  s = s.replace(/\[\s*0\s*,\s*\[\s*c\.x\s*,\s*c\.y\s*\]\s*\[\s*1(?:\.0)?\s*\]\s*\]/g, 'vec2<f32>(0.0, c.y)')
  s = s.replace(/\[\s*0\s*,\s*\[\s*cReal\s*,\s*cImag\s*\]\s*\]\s*\[\s*1\s*\]/g, 'vec2<f32>(0.0, c.y)')
  s = s.replace(/\[\s*cReal\s*,\s*cImag\s*\]\s*\[\s*1(?:\.0)?\s*\]/g, 'cImag')
  s = s.replace(/\[\s*cReal\s*,\s*cImag\s*\]\s*\[\s*0(?:\.0)?\s*\]/g, 'cReal')
  // 部分的に評価済みの [0, cImag] や [cReal, 0] も扱う
  s = s.replace(/\[\s*0\s*,\s*cImag\s*\]/g, 'vec2<f32>(0.0, c.y)')
  s = s.replace(/\[\s*cReal\s*,\s*0\s*\]/g, 'vec2<f32>(c.x, 0.0)')
  // [c.x, c.y][1] のような変換済みの形も処理する
  s = s.replace(/\[\s*c\.x\s*,\s*c\.y\s*\]\s*\[\s*1(?:\.0)?\s*\]/g, 'c.y')
  s = s.replace(/\[\s*c\.x\s*,\s*c\.y\s*\]\s*\[\s*0(?:\.0)?\s*\]/g, 'c.x')
  // parser 固有名を WGSL の成分参照へそろえる
  s = s.replace(/\bcReal\b/g, 'c.x').replace(/\bcImag\b/g, 'c.y')
  s = s.replace(/\bzReal\b/g, 'z.x').replace(/\bzImag\b/g, 'z.y')
  s = s.replace(/\biterIndex\b/g, 'n')

  // 添字アクセスでない配列リテラル [a,b] を (a, b) へ変換する。
  // parseExprFromString はこれを tuple として扱い、後で vec2 化する。
  function replaceArrayLiteralsToTuples(input) {
    let out = ''
    let i = 0
    while (i < input.length) {
      if (input[i] === '[') {
        const prev = i > 0 ? input[i - 1] : ''
        const isIndexing = /[A-Za-z0-9_.)\]]/.test(prev)
        // 対応する閉じ角かっこを探す
        let depth = 0
        let j = i + 1
        let found = false
        for (; j < input.length; j++) {
          if (input[j] === '[') depth++
          else if (input[j] === ']') {
            if (depth === 0) {
              found = true
              break
            }
            depth--
          }
        }
        if (!found) {
          out += input.slice(i)
          break
        }
        if (isIndexing) {
          // 添字アクセスはそのまま残す
          out += input.slice(i, j + 1)
          i = j + 1
          continue
        }
        const inner = input.slice(i + 1, j)
        const parts = splitTopLevelArgs(inner)
        if (parts.length === 2) {
          out += `(${parts[0]}, ${parts[1]})`
          i = j + 1
          continue
        }
        // 2 要素 tuple でなければ元のまま残す
        out += input.slice(i, j + 1)
        i = j + 1
        continue
      }
      out += input[i]
      i++
    }
    return out
  }
  s = replaceArrayLiteralsToTuples(s)

  // tokensTable は後方互換のため残しておく。
  // 今は中身を埋めないが、後段コードが常に存在を前提にできるよう空配列を渡す。
  const tokensTable = []
  // tuple の添字アクセスをその場で展開する
  function inlineTupleIndexing(input) {
    let out = ''
    let i = 0
    while (i < input.length) {
      const p = input.indexOf('(', i)
      if (p === -1) {
        out += input.slice(i)
        break
      }
      out += input.slice(i, p)
      // 対応する閉じかっこを探す
      let depth = 0
      let j = p + 1
      let found = false
      for (; j < input.length; j++) {
        const ch = input[j]
        if (ch === '(') depth++
        else if (ch === ')') {
          if (depth === 0) {
            found = true
            break
          }
          depth--
        }
      }
      if (!found) {
        out += input.slice(p)
        break
      }
      const inner = input.slice(p + 1, j)
      // 後ろに [0] / [1] が続くか確認する
      let k = j + 1
      while (k < input.length && /\s/.test(input[k])) k++
      if (k < input.length && input[k] === '[') {
        // 閉じ角かっこを探す
        let rb = k + 1
        let bdepth = 0
        for (; rb < input.length; rb++) {
          if (input[rb] === '[') bdepth++
          else if (input[rb] === ']') {
            if (bdepth === 0) break
            bdepth--
          }
        }
        if (rb < input.length && input[rb] === ']') {
          const idxRaw = input.slice(k + 1, rb).trim()
          const m = idxRaw.match(/^(-?\d+)(?:\.(?:0+)?)?$/)
          if (m) {
            const idx = parseInt(m[1], 10)
            const parts = splitTopLevelArgs(inner)
            if (parts.length === 2 && (idx === 0 || idx === 1)) {
              out += parts[idx].trim()
              i = rb + 1
              continue
            }
          }
        }
      }
      // tuple 添字でなければ元のかっこを残す
      out += '(' + inner + ')'
      i = j + 1
    }
    return out
  }
  s = inlineTupleIndexing(s)

  const ast = parseExprFromString(s)
  // 控えめな AST 簡約を行い、CSE 前の不要な算術ノイズを減らす
  function simplifyAst(node) {
    if (!node || typeof node !== 'object') return node
    switch (node.type) {
      case 'number':
      case 'ident':
        return node
      case 'unary':
        node.arg = simplifyAst(node.arg)
        return node
      case 'binary': {
        node.left = simplifyAst(node.left)
        node.right = simplifyAst(node.right)
        // 単純な数値演算はここで畳み込む
        if (node.left && node.left.type === 'number' && node.right && node.right.type === 'number') {
          const a = parseFloat(node.left.value)
          const b = parseFloat(node.right.value)
          switch (node.op) {
            case '+':
              return { type: 'number', value: String(a + b) }
            case '-':
              return { type: 'number', value: String(a - b) }
            case '*':
              return { type: 'number', value: String(a * b) }
            case '/':
              return { type: 'number', value: String(a / b) }
          }
        }
        // x + 0 => x, 0 + x => x
        if (node.op === '+') {
          if (node.right && node.right.type === 'number' && /^0(?:\.0*)?$/.test(node.right.value)) return node.left
          if (node.left && node.left.type === 'number' && /^0(?:\.0*)?$/.test(node.left.value)) return node.right
        }
        // x - 0 => x
        if (node.op === '-') {
          if (node.right && node.right.type === 'number' && /^0(?:\.0*)?$/.test(node.right.value)) return node.left
        }
        // x * 1 => x, 1 * x => x, x * 0 => 0, 0 * x => 0
        if (node.op === '*') {
          if (node.right && node.right.type === 'number') {
            if (/^1(?:\.0*)?$/.test(node.right.value)) return node.left
            if (/^0(?:\.0*)?$/.test(node.right.value)) return { type: 'number', value: '0' }
          }
          if (node.left && node.left.type === 'number') {
            if (/^1(?:\.0*)?$/.test(node.left.value)) return node.right
            if (/^0(?:\.0*)?$/.test(node.left.value)) return { type: 'number', value: '0' }
          }
        }
        // x / 1 => x
        if (node.op === '/') {
          if (node.right && node.right.type === 'number' && /^1(?:\.0*)?$/.test(node.right.value)) return node.left
        }
        return node
      }
      case 'call':
        node.args = (node.args || []).map(simplifyAst)
        return node
      case 'tuple':
        node.items = (node.items || []).map(simplifyAst)
        // tuple はそのまま保つ
        return node
      case 'prop':
        node.target = simplifyAst(node.target)
        return node
      case 'index':
        node.target = simplifyAst(node.target)
        return node
      default:
        return node
    }
  }
  // CSE 前に 1 回だけ簡約をかける
  const simplifiedAst = simplifyAst(ast)
  // AST を正規化し、識別子や関数名のゆれをそろえる
  function normalizeAst(node) {
    if (!node || typeof node !== 'object') return node
    switch (node.type) {
      case 'number':
        return node
      case 'ident': {
        // parser 固有の仮名を標準的な dot 形式へ直す
        if (node.name === 'cReal') return { type: 'ident', name: 'c.x' }
        if (node.name === 'cImag') return { type: 'ident', name: 'c.y' }
        if (node.name === 'zReal') return { type: 'ident', name: 'z.x' }
        if (node.name === 'zImag') return { type: 'ident', name: 'z.y' }
        return node
      }
      case 'call': {
        // Math.* 呼び出しを通常名へ直す
        if (typeof node.name === 'string' && node.name.startsWith('Math.')) {
          node.name = node.name.slice(5)
        }
        node.args = (node.args || []).map(normalizeAst)
        return node
      }
      case 'prop':
        node.target = normalizeAst(node.target)
        return node
      case 'index':
        node.target = normalizeAst(node.target)
        return node
      case 'tuple':
        node.items = (node.items || []).map(normalizeAst)
        return node
      case 'binary':
        node.left = normalizeAst(node.left)
        node.right = normalizeAst(node.right)
        return node
      case 'unary':
        node.arg = normalizeAst(node.arg)
        return node
      default:
        return node
    }
  }
  normalizeAst(ast)
  // AST レベルの CSE 用に、重複部分木を探して tmp ノードへ置き換える準備をする
  // 構造キーはメモ化して安定して生成する
  function structuralKey(node, memo = new Map()) {
    if (!node) return { key: '', size: 0 }
    if (memo.has(node)) return memo.get(node)
    let res
    switch (node.type) {
      case 'number':
        res = { key: `n:${node.value}`, size: 1 }
        break
      case 'ident':
        res = { key: `id:${node.name}`, size: 1 }
        break
      case 'prop': {
        const t = structuralKey(node.target, memo)
        res = { key: `prop:${t.key}.${node.prop}`, size: t.size + 1 }
        break
      }
      case 'index': {
        const t = structuralKey(node.target, memo)
        res = { key: `idx:${t.key}[${node.index}]`, size: t.size + 1 }
        break
      }
      case 'call': {
        const parts = []
        let total = 1
        for (const a of node.args || []) {
          const r = structuralKey(a, memo)
          parts.push(r.key)
          total += r.size
        }
        res = { key: `call:${node.name}(${parts.join(',')})`, size: total }
        break
      }
      case 'tuple': {
        const parts = []
        let total = 1
        for (const it of node.items || []) {
          const r = structuralKey(it, memo)
          parts.push(r.key)
          total += r.size
        }
        res = { key: `tuple:(${parts.join(',')})`, size: total }
        break
      }
      default:
        res = { key: JSON.stringify(node), size: 1 }
    }
    memo.set(node, res)
    return res
  }

  // 以前の AST レベル CSE は `_tmp0` のような一時識別子を残し、
  // 複雑な式で未解決参照を生むことがあった。
  // 安全性を優先し、ここでは CSE 置換を使わず簡約 AST から直接出力する。
  const reducedAst = simplifiedAst
  const cseBindings = []
  const wgslObj = astToWGSL(reducedAst, tokensTable)
  // 最終出力は明示的な vec2<f32>(x, y) 形式で組み立てる
  let wgsl = ''
  if (wgslObj && typeof wgslObj === 'object' && wgslObj.kind === 'vec2') {
    // x / y が明示されていればそれを使う。
    // すでに vec2 を返す式なら二重ラップを避けてそのまま返す。
    if (wgslObj.x !== undefined && wgslObj.y !== undefined) {
      wgsl = `vec2<f32>(${wgslObj.x}, ${wgslObj.y})`
    } else {
      const expr = (wgslObj.expr || '').toString().trim()
      if (/^vec2<f32>\s*\(/.test(expr) || /^select\s*\(/.test(expr)) {
        wgsl = expr
      } else {
        const x = expr || '0.0'
        wgsl = `vec2<f32>(${x}, 0.0)`
      }
    }
  } else if (wgslObj && typeof wgslObj === 'object' && wgslObj.kind === 'scalar') {
    wgsl = `vec2<f32>(${wgslObj.expr}, 0.0)`
  } else if (typeof wgslObj === 'string') {
    // 文字列だけなら可能な範囲で vec2 へ包む
    wgsl = ensureVec2Wrapper(wgslObj)
  } else {
    wgsl = ''
  }
  if (cseBindings && cseBindings.length > 0) {
    // CSE binding があれば tmp を式へ展開する
    // `let` 宣言を式中に出すと無効になるため、tmp 名を実際の式へ置換する
    for (const b of cseBindings) {
      const esc = b.name.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')
      // tmp 名を単語境界で置換する
      wgsl = wgsl.replace(new RegExp(`\\b${esc}\\b`, 'g'), `(${b.expr})`)
    }
  }
  // 安全策として、複雑な式では後段の強い文字列置換を避ける。
  // 少し冗長でも、壊れた WGSL を出さないことを優先する。
  const vec2Count = (wgsl.match(/vec2<f32>/g) || []).length
  if (wgsl.includes(String(SAFE_EPS)) || wgsl.includes('1e-') || wgsl.includes('max(') || vec2Count > 3) {
    // 指数表記、max()、多重 vec2、SAFE_EPS を含む式では強い再書換えを避ける
    return ensureVec2Wrapper(wgsl)
  }
  // 生成後の後処理として、明らかな vec2 の入れ子だけを保守的に畳む
  let out = wgsl
  let safety = 50
  while (safety-- > 0) {
    const before = out
    // 左側の引数が入れ子 vec2 の場合
    out = out.replace(/vec2<f32>\(\s*vec2<f32>\(([^,]+),([^)]+)\)\s*,\s*([^)]+)\s*\)/g, (m, a, b, c) => {
      return `vec2<f32>(${a.trim()}, ${c.trim()})`
    })
    // 右側の引数が入れ子 vec2 の場合
    out = out.replace(/vec2<f32>\(\s*([^,]+),\s*vec2<f32>\(([^,]+),([^)]+)\)\s*\)/g, (m, a, b, c) => {
      return `vec2<f32>(${a.trim()}, ${c.trim()})`
    })
    if (out === before) break
  }

  // 小さな vec2 からの単純な .x / .y 参照を畳む
  // vec2<f32>(EXPR, 0.0).x -> EXPR
  out = out.replace(/vec2<f32>\(\s*([^,]+?),\s*0\.0\s*\)\.x/g, (m, a) => a.trim())
  // vec2<f32>(0.0, EXPR).y -> EXPR
  out = out.replace(/vec2<f32>\(\s*0\.0\s*,\s*([^)]+?)\s*\)\.y/g, (m, b) => b.trim())
  // vec2<f32>(EXPR, 0.0).y と vec2<f32>(0.0, EXPR).x は 0.0 にできる
  out = out.replace(/vec2<f32>\(\s*([^,]+?),\s*0\.0\s*\)\.y/g, '0.0')
  out = out.replace(/vec2<f32>\(\s*0\.0\s*,\s*([^)]+?)\s*\)\.x/g, '0.0')
  // vec2<f32>(c.x,0.0).y < 0.0 のような比較を単純化する
  out = out.replace(/vec2<f32>\(\s*([^,]+?),\s*([^)]+?)\s*\)\.y\s*<\s*0\.0/g, (m, a, b) => {
    // 第 2 引数が単純な変数なら、その比較へ置き換える
    const second = b.trim()
    if (/^[A-Za-z0-9_.]+$/.test(second) && second !== '0.0') return `${second} < 0.0`
    // それ以外は変更しない
    return m
  })
  // 常に偽になる select は定数へ畳む
  out = out.replace(/select\(\s*1\.0\s*,\s*-1\.0\s*,\s*0\.0\s*<\s*0\.0\s*\)/g, '1.0')

  // 第 2 成分が 0 の length は abs に置き換えられる
  out = out.replace(/length\(\s*vec2<f32>\(\s*([^,]+?)\s*,\s*0\.0\s*\)\s*\)/g, (m, x) => {
    return `abs(${x.trim()})`
  })
  // 第 1 成分が 0 の length も abs に置き換えられる
  out = out.replace(/length\(\s*vec2<f32>\(\s*0\.0\s*,\s*([^)]+?)\s*\)\s*\)/g, (m, y) => {
    return `abs(${y.trim()})`
  })

  // 念のため、同様の単純参照ももう一度畳む
  out = out.replace(/vec2<f32>\(\s*([^,]+?)\s*,\s*0\.0\s*\)\.x/g, (m, a) => a.trim())
  out = out.replace(/vec2<f32>\(\s*0\.0\s*,\s*([^)]+?)\s*\)\.y/g, (m, b) => b.trim())

  // 残った配列・tuple 添字も可能な範囲で畳む
  try {
    out = out.replace(/\[\s*([^,\]]+?)\s*,\s*([^\]]+?)\s*\]\s*\[\s*0(?:\.0)?\s*\]/g, (m, a, b) => a.trim())
    out = out.replace(/\[\s*([^,\]]+?)\s*,\s*([^\]]+?)\s*\]\s*\[\s*1(?:\.0)?\s*\]/g, (m, a, b) => b.trim())
    out = out.replace(/\(\s*([^,)]+?)\s*,\s*([^)]+?)\s*\)\s*\[\s*0(?:\.0)?\s*\]/g, (m, a, b) => a.trim())
    out = out.replace(/\(\s*([^,)]+?)\s*,\s*([^)]+?)\s*\)\s*\[\s*1(?:\.0)?\s*\]/g, (m, a, b) => b.trim())
  } catch (e) {}

  // かっこ対応の flatten を追加で行う。
  // 入れ子 vec2 の内部引数に複雑な式があっても安全に処理する。
  function flattenVec2Constructs(code) {
    let res = ''
    let i = 0
    while (i < code.length) {
      const pos = code.indexOf('vec2<f32>(', i)
      if (pos === -1) {
        res += code.slice(i)
        break
      }
      res += code.slice(i, pos)
      // この vec2 に対応する閉じかっこを探す
      let depth = 0
      let j = pos + 'vec2<f32>('.length
      let found = false
      for (; j < code.length; j++) {
        const ch = code[j]
        if (ch === '(') depth++
        else if (ch === ')') {
          if (depth === 0) {
            found = true
            break
          }
          depth--
        }
      }
      if (!found) {
        res += code.slice(pos)
        break
      }
      const inner = code.slice(pos + 'vec2<f32>('.length, j)
      const parts = splitTopLevelArgs(inner)
      if (parts.length === 2) {
        let left = parts[0].trim()
        let right = parts[1].trim()
        // 左が vec2<f32>(...) なら内側の引数を取り出す
        if (left.startsWith('vec2<f32>(')) {
          const li = left.indexOf('(')
          const linner = left.slice(li + 1, left.lastIndexOf(')'))
          const lparts = splitTopLevelArgs(linner)
          if (lparts.length === 2) {
            left = lparts[0].trim()
          }
        }
        // 右が vec2<f32>(...) なら内側の引数を取り出す
        if (right.startsWith('vec2<f32>(')) {
          const ri = right.indexOf('(')
          const rinner = right.slice(ri + 1, right.lastIndexOf(')'))
          const rparts = splitTopLevelArgs(rinner)
          if (rparts.length === 2) {
            right = rparts[1].trim()
          }
        }
        res += `vec2<f32>(${left}, ${right})`
        i = j + 1
        continue
      }
      // 単純な 2 要素でなければ元のまま残す
      res += code.slice(pos, j + 1)
      i = j + 1
    }
    return res
  }

  // 変化がなくなるまで flatten を繰り返す
  let loop = 20
  while (loop-- > 0) {
    const prevOut = out
    out = flattenVec2Constructs(out)
    if (out === prevOut) break
  }

  // 後段の文字列簡約として、0 を含む積や +0.0 を安全に畳む
  function collapseZeroProducts(code) {
    let s = code
    let iter = 0
    while (iter++ < 10) {
      const before = s
      // ... * 0.0 を見つけたら 0.0 に置き換える
      s = s.replace(/([A-Za-z0-9_.)(]+)\s*\*\s*0\.0/g, '0.0')
      s = s.replace(/0\.0\s*\*\s*([A-Za-z0-9_.)(]+)/g, '0.0')
      // 不要な +0.0 / -0.0 を消す
      s = s.replace(/\+\s*0\.0\b/g, '')
      s = s.replace(/-\s*0\.0\b/g, '')
      if (s === before) break
    }
    return s
  }

  out = collapseZeroProducts(out)

  // 変換途中で引数が落ちた場合に備え、空呼び出しは 0.0 を補う
  for (const f of SAFE_FUNCS) {
    // f() や f(   ) を検出する
    out = out.replace(new RegExp(`\\b${f}\\s*\\(\\s*\\)`, 'g'), `${f}(0.0)`)
  }

  // f( + X) や f(0.0 + X) のような崩れた形を直す
  for (const f of SAFE_FUNCS) {
    // f( + X) -> f(X)
    out = out.replace(new RegExp(`\\b${f}\\s*\\(\\s*\\+\\s*`, 'g'), `${f}(`)
    // f(0.0 + X) -> f(X)
    out = out.replace(new RegExp(`\\b${f}\\s*\\(\\s*0\\.0\\s*\\+\\s*`, 'g'), `${f}(`)
  }

  // 誤って生じた / 0.0 は SAFE_EPS を使う安全形へ直す
  try {
    out = out.replace(/\/\)\s*\/\s*0\.0/g, ') / max(0.0, ' + String(SAFE_EPS) + ')')
    out = out.replace(/\b\/\s*0\.0/g, '/ max(0.0, ' + String(SAFE_EPS) + ')')
  } catch (e) {}

  // 符号とかっこの保守的な正規化を行う
  function normalizeSigns(code) {
    // 複雑な式は触らずそのまま返す
    if (code.includes(String(SAFE_EPS)) || /1e-|1E-/.test(code) || code.includes('max(')) return code

    let s = code

    // + -X や + +X を整理する
    s = s.replace(/\+\s*\+\s*/g, '+')
    s = s.replace(/\+\s*-\s*/g, '-')

    // 二重否定を簡単なケースだけ畳む
    let iter = 0
    while (iter++ < 10) {
      const prev = s
      // -( -something ) -> something
      s = s.replace(/-\(\s*-\s*([^)]+?)\s*\)/g, (m, inner) => {
        // vec2 引数らしいものは安全のため触らない
        if (/,/.test(inner) || /vec2\s*</.test(inner)) return m
        return `(${inner.trim()})`
      })
      if (s === prev) break
    }

    // --X も単純な場合だけ整理する
    s = s.replace(/--(?=[A-Za-z0-9_([])/g, '+')

    // 末尾側の + 0.0 を消す
    s = s.replace(/\+\s*0\.0\b/g, '')

    return s
  }

  try {
    out = normalizeSigns(out)
  } catch (_e) {
    // 正規化に失敗しても既存の出力は保持する
  }

  // 正規化処理の結果、隣り合うサブ式の間から「+」が誤って削除される
  // 場合に対する安全対策です。このバグは虚数単位を含む式（例:
  // `i*cos(Im(z))`）を変換した際に観察され、生成された WGSL が
  // `sin(z.x) (sin(0.0)...` のような GPU では構文エラーになる形に
  // なっていました。
  // 原因は complexAdd と正規化の相互作用で、一方の項がゼロに簡約
  // されると「+」が取り除かれてしまうことにあります。
  // 上流の変換を追いかける代わりに、演算子が抜けている場所には
  // 明示的に「+」を挿入し直すことで修正します。
  //
  // この処理は意図的に保守的で、閉じる識別子や括弧の直後に
  // 開く括弧または識別子が空白を挟んで続くというよくあるパターン
  // のみに適用します。正当な WGSL ではこのような記法を使わない
  // ため、「+」を挿入しても安全であり、期待した加算を復元します。
  out = out.replace(/([0-9A-Za-z_)\]])\s+\(/g, '$1 + (')
  out = out.replace(/([0-9A-Za-z_)\]])\s+(?=[A-Za-z_])/g, '$1 + ')

  return ensureVec2Wrapper(out)
}

// 文字列レベルの CSE は不安定だったため廃止した。
// 呼び出し側は右辺式だけを期待するので、let tmp の生成も避ける。

// 呼び出し側向けのラッパー関数
export function jsExprToWGSL_withCSE(expr) {
  const base = jsExprToWGSL_ast(expr)
  // AST 側で十分に処理しているため、文字列レベルの CSE は行わない
  return base
}

// 生成結果が top-level の裸 tuple になった場合は vec2<f32> で包む。
function ensureVec2Wrapper(s) {
  if (!s || typeof s !== 'string') return s
  const trimmed = s.trim()
  if (/^vec2<f32>\s*\(/.test(trimmed)) return s
  // top-level の 2 要素ペアなら vec2<f32>(a, b) として包む
  // 1 組の外側かっこだけで包まれている場合は判定のために外す
  let probe = trimmed
  if (!/^vec2<f32>\s*\(/.test(probe) && probe[0] === '(' && probe[probe.length - 1] === ')') {
    // 外側のかっこが全体を包んでいるか確認する
    let d = 0
    let encloses = false
    for (let i = 0; i < probe.length; i++) {
      const ch = probe[i]
      if (ch === '(') d++
      else if (ch === ')') {
        d--
        if (d === 0 && i === probe.length - 1) encloses = true
      }
    }
    if (encloses) {
      // 外側 1 組を外す
      probe = probe.slice(1, -1).trim()
    }
  }

  let depth = 0
  let splitIndex = -1
  for (let i = 0; i < probe.length; i++) {
    const ch = probe[i]
    if (ch === '(' || ch === '[') depth++
    else if (ch === ')' || ch === ']') depth = Math.max(0, depth - 1)
    else if (ch === ',' && depth === 0) {
      splitIndex = i
      break
    }
  }
  if (splitIndex !== -1) {
    const left = probe.slice(0, splitIndex).trim()
    const right = probe.slice(splitIndex + 1).trim()
    // 左右どちらも空でないことを確認する
    if (left.length > 0 && right.length > 0) {
      return `vec2<f32>(${left}, ${right})`
    }
  }
  return s
}

// 呼び出し側が使う安全版ラッパー
// 式中の変数名を検証する補助関数
function validateVariables(expr) {
  // 式中の識別子を集める
  const identifierPattern = /\b[A-Za-z_][A-Za-z0-9_]*/g
  const identifiers = new Set()
  let match
  while (true) {
    match = identifierPattern.exec(expr)
    if (match === null) break
    identifiers.add(match[0])
  }

  // 許可する識別子
  const validNames = new Set([
    // 変数
    'z',
    'c',
    'n',
    // プロパティ
    'x',
    'y',
    'r',
    // 数学関数
    'sin',
    'cos',
    'tan',
    'exp',
    'log',
    'sqrt',
    'abs',
    'atan2',
    'pow',
    'sinh',
    'cosh',
    'tanh',
    'asin',
    'acos',
    'atan',
    'min',
    'max',
    'clamp',
    'floor',
    'fract',
    'ceil',
    'round',
    'sign',
    'length',
    'dot',
    'cross',
    'normalize',
    'distance',
    // 複素数関数
    'complexAdd',
    'complexSub',
    'complexMul',
    'complexDiv',
    'complexPower',
    'complexExp',
    'complexLog',
    'complexLog10',
    'complexSin',
    'complexCos',
    'complexTan',
    'complexSinh',
    'complexCosh',
    'complexTanh',
    'complexArg',
    'complexAbs2',
    'complexFloor',
    'complexFract',
    'complexMod',
    'complexMin',
    'complexMax',
    'complexClamp',
    'complexRotate',
    'complexFold',
    'complexBoxFold',
    'complexRe',
    'complexIm',
    'complexAbs',
    'complexConj',
    'complexSqrt',
    // 補助関数
    'vec2',
    'f32',
    'select',
    // 定数
    'PI',
    'E',
  ])

  const invalidVars = []
  for (const id of identifiers) {
    if (!validNames.has(id)) {
      invalidVars.push(id)
    }
  }

  if (invalidVars.length > 0) {
    throw new Error(
      `Invalid variable(s) in expression: "${invalidVars.join('", "')}".\n` +
        `Only "z", "c", and "n" are valid variables. Did you mean to use a function or property?\n` +
        `Example: z^2 + c (not zz + c)`,
    )
  }
}

export function jsExprToWGSL_safe(expr) {
  // parser 由来の名前を WGSL の成分参照へそろえる
  let s = jsExprToWGSL(expr)
  if (s && typeof s === 'string') {
    // 置換前に不正な変数名を検出する
    validateVariables(s)

    s = s.replace(/\bcReal\b/g, 'c.x').replace(/\bcImag\b/g, 'c.y')
    s = s.replace(/\bzReal\b/g, 'z.x').replace(/\bzImag\b/g, 'z.y')
  }
  return ensureVec2Wrapper(s)
}

// 一時的なデバッグ用ヘルパー
export function __debug_parse_expr(s) {
  return parseExprFromString(s)
}

export function __debug_ast_to_wgsl(node) {
  const tn = astToWGSL(node, [])
  if (tn && typeof tn.expr === 'string') {
    tn.expr = tn.expr.replace(/\bcReal\b/g, 'c.x').replace(/\bcImag\b/g, 'c.y')
    tn.expr = tn.expr.replace(/\bzReal\b/g, 'z.x').replace(/\bzImag\b/g, 'z.y')
    // x / y 側に残った cReal / cImag も正規化する
    if (tn.x && typeof tn.x === 'string') tn.x = tn.x.replace(/\bcReal\b/g, 'c.x').replace(/\bcImag\b/g, 'c.y')
    if (tn.y && typeof tn.y === 'string') tn.y = tn.y.replace(/\bcReal\b/g, 'c.x').replace(/\bcImag\b/g, 'c.y')
  }
  return tn
}

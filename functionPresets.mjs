/**
 * Based on bertbaron/mandelbrot by Bert Baron
 * This file is part of the Mandelbrot Explorer project.
 * Licensed under GPL-3.0.
 */

/**
 * 定義済みのフラクタル反復式一覧。
 *
 * 各プリセットには次の項目を含められる。
 * - expr: 反復式
 * - label: 表示名
 * - fractalType: 'mandelbrot' または 'custom'
 * - z0Real, z0Imag: 初期 z 値
 * - coordX, coordY: 表示座標
 */
export const functionPresets = [
  {
    expr: 'z*z + c',
    label: 'Mandelbrot',
    fractalType: 'mandelbrot',
    z0Real: 0,
    z0Imag: 0,
    coordX: -0.5,
    coordY: 0,
  },
  {
    expr: '(|Re(z)| + i*|Im(z)|)^2 + c',
    label: 'Burning Ship',
    z0Real: 0,
    z0Imag: 0,
    coordX: -0.5,
    coordY: -0.5,
  },
  {
    expr: 'conj(z)^2 + c',
    label: 'Tricorn',
  },
  { expr: 'sin(z) + c', z0Real: 0, z0Imag: 0 },
  { expr: 'cos(z) + c', z0Real: 0, z0Imag: 0 },
  { expr: 'tan(z) + c', z0Real: 0, z0Imag: 0 },
  { expr: 'exp(z) + c', z0Real: 0, z0Imag: 0 },
  { expr: 'zeta(z) + c', label: 'Riemann zeta', z0Real: 0, z0Imag: 0, coordX: 0, coordY: 0 },
  { expr: 'sinh(0.4*z) + c', label: 'sinh stretch', z0Real: 0, z0Imag: 0 },
  { expr: 'cosh(0.25*z) + c', label: 'cosh arch', z0Real: 0, z0Imag: 0 },
  { expr: 'tanh(z*z) + c', label: 'tanh square', z0Real: 0, z0Imag: 0 },
  { expr: 'z*z + 0.1*arg(z+c) + c', label: 'argument drift', z0Real: 0, z0Imag: 0 },
  { expr: 'z*z + c/(1+abs2(z))', label: 'abs2 damping', z0Real: 0, z0Imag: 0 },
  { expr: 'rotate(z*z, pi/2) + c', label: 'z^2 rotated 90°', z0Real: 0, z0Imag: 0 },
  { expr: 'fold(z)^2 + c', label: 'fold mirror', z0Real: 0, z0Imag: 0 },
  { expr: 'boxFold(z, 1.2)^2 + c', label: 'boxFold mirror', z0Real: 0, z0Imag: 0 },
  { expr: 'fract(z*z) + c', label: 'fract tiling', z0Real: 0, z0Imag: 0 },
  { expr: 'mod(z*z, 1) + c', label: 'mod lattice', z0Real: 0, z0Imag: 0 },
  { expr: 'z*z + c + 0.01*sin(n*z)', label: 'iteration ripple', z0Real: 0, z0Imag: 0 },

  {
    expr: '(Re(z) + i*|Im(z)|)^2 + c',
    z0Real: 0,
    z0Imag: 0,
    coordX: -0.5,
    coordY: 0,
  },
  { expr: '(|Re(z)| + i*|Im(z)|) * conj(c) + c', z0Real: 0, z0Imag: 0 },
  { expr: 'conj(z) * sin(z) + c', z0Real: 0, z0Imag: 0 },
  { expr: 'sin(z) * (1 + |z|) + c', z0Real: 0, z0Imag: 0 },
  {
    expr: 'sqrt(cos(z * i) ^ pi) + c * ln(c ^ e) / (c / e)',
    z0Real: 0,
    z0Imag: 0,
  },
  {
    expr: 'sin(|Re(z)| + i*Im(c)) * conj(z) + exp(i*|z|) * c',
    z0Real: 0,
    z0Imag: 0,
  },

  { expr: 'z^(z/c) + c', z0Real: 0, z0Imag: 0 },
  { expr: 'z/(c+1) + c', z0Real: 0, z0Imag: 0, coordX: -1, coordY: 0 },
  {
    expr: '(1.000333 * conj(z)^3) + (0.000333 * conj(z)^2 * i) - (0.932635 * conj(z)) - (0.361745 * conj(z) * i) + c',
    z0Real: 0,
    z0Imag: 0,
  },
  { expr: 'c*(z+1/z)', z0Real: 1, z0Imag: 0 },
  { expr: 'z*z / c*c', z0Real: 1, z0Imag: 0 },
  { expr: 'z*z + 0.5/c', z0Real: 0, z0Imag: 0, coordX: 0.5, coordY: 0 },
  { expr: 'z*z + z/c', z0Real: -1, z0Imag: 0 },
  { expr: '((z + (c^2)-1) / c^2)^2', z0Real: 0, z0Imag: 0 },
  { expr: 'exp((z^2-1.00001*z)/c^3)', z0Real: 0, z0Imag: 0 },
  { expr: 'c/z^(c/z)', z0Real: 0, z0Imag: 0 },
  {
    expr: 'conj(z) * (|Re(c)| - i*|Im(c)|) + sin(Re(z) ^ 2) + ((z^c)/(z+c))^(0.1) + 0.05/(c*0.14)',
    z0Real: 0,
    z0Imag: 0,
  },
  {
    expr: '(1/z)*(1/z) + z/c',
    z0Real: 1,
    z0Imag: 0,
  },
  { expr: 'z/i*Re(|c|) + c' },
  { expr: '((1/z)/c)*(z+1/z)', z0Real: 1, z0Imag: 0 },
  { expr: 'conj(c/z)^2 + z', z0Real: 1, z0Imag: 0 },
]

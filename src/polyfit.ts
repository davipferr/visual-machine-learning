/**
 * Least-squares polynomial fitting: the simplest model whose capacity is a
 * single dial (the degree). Used to show underfitting and overfitting without
 * any training noise in the way.
 */

/**
 * Legendre polynomials P₀…P_degree at x. On [−1, 1] they stay between −1 and 1,
 * unlike raw powers x¹⁵, which keeps the least-squares system well conditioned.
 */
export function legendre(x: number, degree: number): number[] {
  const p = [1, x]
  for (let n = 1; n < degree; n++) p.push(((2 * n + 1) * x * p[n] - n * p[n - 1]) / (n + 1))
  return p.slice(0, degree + 1)
}

export interface PolynomialFit {
  degree: number
  coefficients: number[]
  predict: (x: number) => number
}

/**
 * Minimize Σ(p(xᵢ) − yᵢ)² + ridge·Σcₖ² over the coefficients, by solving the
 * normal equations (AᵀA + ridge·I)·c = Aᵀy. A tiny ridge keeps the system
 * solvable when there are fewer points than coefficients.
 */
export function fitPolynomial(xs: readonly number[], ys: readonly number[], degree: number, ridge = 1e-9): PolynomialFit {
  const k = degree + 1
  const ata = Array.from({ length: k }, () => new Array<number>(k).fill(0))
  const aty = new Array<number>(k).fill(0)
  xs.forEach((x, i) => {
    const row = legendre(x, degree)
    for (let a = 0; a < k; a++) {
      aty[a] += row[a] * ys[i]
      for (let b = 0; b < k; b++) ata[a][b] += row[a] * row[b]
    }
  })
  for (let a = 0; a < k; a++) ata[a][a] += ridge
  const coefficients = solve(ata, aty)
  return {
    degree,
    coefficients,
    predict: (x) => {
      const basis = legendre(x, degree)
      let sum = 0
      for (let a = 0; a < k; a++) sum += coefficients[a] * basis[a]
      return sum
    },
  }
}

/** Solve A·x = b by Gaussian elimination with partial pivoting. A and b are copied. */
export function solve(matrix: readonly (readonly number[])[], rhs: readonly number[]): number[] {
  const n = rhs.length
  const a = matrix.map((row) => [...row])
  const b = [...rhs]
  for (let col = 0; col < n; col++) {
    let pivot = col
    for (let r = col + 1; r < n; r++) if (Math.abs(a[r][col]) > Math.abs(a[pivot][col])) pivot = r
    ;[a[col], a[pivot]] = [a[pivot], a[col]]
    ;[b[col], b[pivot]] = [b[pivot], b[col]]
    const diag = a[col][col] || 1e-300
    for (let r = col + 1; r < n; r++) {
      const factor = a[r][col] / diag
      if (factor === 0) continue
      for (let c = col; c < n; c++) a[r][c] -= factor * a[col][c]
      b[r] -= factor * b[col]
    }
  }
  const x = new Array<number>(n).fill(0)
  for (let r = n - 1; r >= 0; r--) {
    let sum = b[r]
    for (let c = r + 1; c < n; c++) sum -= a[r][c] * x[c]
    x[r] = sum / (a[r][r] || 1e-300)
  }
  return x
}

/** Mean squared error of a fit on some points. */
export function meanSquaredError(predict: (x: number) => number, xs: readonly number[], ys: readonly number[]) {
  let sum = 0
  xs.forEach((x, i) => (sum += (predict(x) - ys[i]) ** 2))
  return sum / Math.max(1, xs.length)
}

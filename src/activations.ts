export type Activation = 'relu' | 'tanh' | 'sigmoid' | 'linear'

export interface ActivationFn {
  /** a = f(z) */
  f: (z: number) => number
  /**
   * da/dz. It gets both z and a = f(z) because the cheap formula differs:
   * sigmoid' = a·(1 − a) and tanh' = 1 − a² reuse the output, ReLU' looks at z.
   */
  df: (z: number, a: number) => number
}

export const activations: Record<Activation, ActivationFn> = {
  relu: { f: (z) => (z > 0 ? z : 0), df: (z) => (z > 0 ? 1 : 0) },
  tanh: { f: Math.tanh, df: (_, a) => 1 - a * a },
  sigmoid: { f: (z) => 1 / (1 + Math.exp(-z)), df: (_, a) => a * (1 - a) },
  linear: { f: (z) => z, df: () => 1 },
}

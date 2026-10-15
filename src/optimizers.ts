export type OptimizerName = 'sgd' | 'momentum' | 'adam'

export interface OptimizerConfig {
  name: OptimizerName
  /** Step size η. */
  learningRate: number
  /** Momentum's β: how much of the previous velocity survives each step. Default 0.9. */
  momentum?: number
  /** Adam's decay rates for the running mean and the running square. Defaults 0.9 and 0.999. */
  beta1?: number
  beta2?: number
  epsilon?: number
}

export interface Optimizer {
  readonly config: Required<OptimizerConfig>
  /** Update every parameter array in place from the matching gradient array. */
  step(params: Float64Array[], grads: Float64Array[]): void
  /** Forget velocities and running averages. */
  reset(): void
}

export const optimizerLabels: Record<OptimizerName, string> = {
  sgd: 'Gradient descent',
  momentum: 'Momentum',
  adam: 'Adam',
}

export function resolveOptimizer(config: OptimizerConfig): Required<OptimizerConfig> {
  return { momentum: 0.9, beta1: 0.9, beta2: 0.999, epsilon: 1e-8, ...config }
}

/**
 * The three update rules, for a parameter θ with gradient g:
 *
 *  - Gradient descent: θ ← θ − η·g
 *  - Momentum:         v ← β·v + g;  θ ← θ − η·v
 *    The velocity remembers past gradients: it builds speed along a consistent
 *    slope and cancels out back-and-forth zig-zags across a narrow valley.
 *  - Adam:             m ← β₁·m + (1 − β₁)·g;  s ← β₂·s + (1 − β₂)·g²
 *                      θ ← θ − η·m̂ / (√ŝ + ε)   (m̂, ŝ: bias-corrected)
 *    Each parameter gets its own step size: dividing by √s makes steep and flat
 *    directions move at similar speeds.
 *
 * `config` is read on every step, so the learning rate can be changed mid-training.
 */
export function createOptimizer(config: OptimizerConfig): Optimizer {
  const resolved = resolveOptimizer(config)
  let first: Float64Array[] = []
  let second: Float64Array[] = []
  let t = 0

  const state = (store: Float64Array[], index: number, size: number) => {
    if (!store[index] || store[index].length !== size) store[index] = new Float64Array(size)
    return store[index]
  }

  return {
    config: resolved,
    reset() {
      first = []
      second = []
      t = 0
    },
    step(params, grads) {
      const { name, learningRate: lr, momentum, beta1, beta2, epsilon } = resolved
      t++
      for (let p = 0; p < params.length; p++) {
        const theta = params[p]
        const g = grads[p]
        if (name === 'sgd') {
          for (let i = 0; i < theta.length; i++) theta[i] -= lr * g[i]
        } else if (name === 'momentum') {
          const v = state(first, p, theta.length)
          for (let i = 0; i < theta.length; i++) {
            v[i] = momentum * v[i] + g[i]
            theta[i] -= lr * v[i]
          }
        } else {
          const m = state(first, p, theta.length)
          const s = state(second, p, theta.length)
          const c1 = 1 - beta1 ** t
          const c2 = 1 - beta2 ** t
          for (let i = 0; i < theta.length; i++) {
            m[i] = beta1 * m[i] + (1 - beta1) * g[i]
            s[i] = beta2 * s[i] + (1 - beta2) * g[i] * g[i]
            theta[i] -= (lr * (m[i] / c1)) / (Math.sqrt(s[i] / c2) + epsilon)
          }
        }
      }
    },
  }
}

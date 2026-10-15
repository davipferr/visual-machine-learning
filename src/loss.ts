import type { Activation } from './activations'

export type Loss = 'mse' | 'bce'

/** Probabilities are kept this far from 0 and 1 so log() stays finite. */
const EPS = 1e-12

/**
 * Loss for one sample.
 *  - `mse`: ½·Σ(a − t)². The ½ makes the gradient just (a − t).
 *  - `bce`: binary cross-entropy −Σ[t·ln a + (1 − t)·ln(1 − a)], for probabilities.
 */
export function lossValue(loss: Loss, output: ArrayLike<number>, target: ArrayLike<number>): number {
  let sum = 0
  for (let i = 0; i < output.length; i++) {
    const a = output[i]
    const t = target[i]
    if (loss === 'mse') {
      sum += 0.5 * (a - t) * (a - t)
    } else {
      const p = Math.min(1 - EPS, Math.max(EPS, a))
      sum -= t * Math.log(p) + (1 - t) * Math.log(1 - p)
    }
  }
  return sum
}

/**
 * ∂L/∂z for the output layer: the "error signal" backpropagation starts from.
 *
 * In general that's ∂L/∂a · f'(z). Cross-entropy paired with a sigmoid output
 * is special: the two derivatives cancel to exactly (a − t). Using the
 * shortcut avoids dividing by a·(1 − a), which is 0 when the sigmoid saturates.
 */
export function outputDelta(
  loss: Loss,
  activation: Activation,
  z: number,
  a: number,
  t: number,
  derivative: (z: number, a: number) => number,
): number {
  if (loss === 'bce' && activation === 'sigmoid') return a - t
  if (loss === 'mse') return (a - t) * derivative(z, a)
  const p = Math.min(1 - EPS, Math.max(EPS, a))
  return ((p - t) / (p * (1 - p))) * derivative(z, a)
}

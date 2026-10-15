import { lossValue, type Loss } from './loss'
import { flatGradients, type Gradients, type Network, type Sample } from './network'
import { createOptimizer, type Optimizer, type OptimizerConfig } from './optimizers'
import { createRng, shuffle, type Rng } from './random'

export interface TrainerConfig {
  loss: Loss
  optimizer: OptimizerConfig
  /** Samples per gradient step. 1 is pure stochastic GD; the whole set is full-batch GD. */
  batchSize: number
  /** L2 penalty λ: adds λ/2·Σw² to the loss, which pulls every weight towards 0. Default 0. */
  l2?: number
  /** Seed for the shuffling order. */
  seed?: number
}

export interface Evaluation {
  /** Mean loss per sample (without the L2 penalty). */
  loss: number
  /** Fraction classified correctly, thresholding the first output at 0.5. Only for `bce`. */
  accuracy: number
}

/**
 * Mini-batch training: shuffle, cut into batches, and for each batch average
 * the gradients and let the optimizer take one step.
 */
export class Trainer {
  readonly network: Network
  config: TrainerConfig
  optimizer: Optimizer
  /** Completed passes over the training set. */
  epoch = 0
  /** Optimizer steps taken. */
  steps = 0
  /** RMS of ∂L/∂W per layer from the latest step: how strongly each layer is being pushed. */
  layerGradients: number[] = []
  private rng: Rng

  constructor(network: Network, config: TrainerConfig) {
    this.network = network
    this.config = config
    this.optimizer = createOptimizer(config.optimizer)
    this.rng = createRng(config.seed ?? 1)
  }

  /** Swap the update rule. Velocities and running averages start over. */
  setOptimizer(config: OptimizerConfig) {
    this.config = { ...this.config, optimizer: config }
    this.optimizer = createOptimizer(config)
  }

  /** One gradient step on one batch. Returns the batch's mean loss before the step. */
  step(batch: readonly Sample[]): number {
    const { network } = this
    const lambda = this.config.l2 ?? 0
    const grads = network.gradients(batch, this.config.loss)
    if (lambda > 0) {
      network.layers.forEach((layer, l) => {
        const g = grads.weights[l]
        for (let i = 0; i < g.length; i++) g[i] += lambda * layer.weights[i]
      })
    }
    this.layerGradients = rmsPerLayer(grads)
    let loss = 0
    for (const sample of batch) loss += network.loss(sample.input, sample.target, this.config.loss)
    this.optimizer.step(network.parameters(), flatGradients(grads))
    this.steps++
    return loss / Math.max(1, batch.length)
  }

  /** One pass over the training set in shuffled mini-batches. Returns the mean batch loss. */
  trainEpoch(train: readonly Sample[]): number {
    const order = shuffle([...train], this.rng)
    const size = Math.max(1, Math.min(this.config.batchSize, order.length))
    let total = 0
    let batches = 0
    for (let start = 0; start < order.length; start += size) {
      total += this.step(order.slice(start, start + size))
      batches++
    }
    this.epoch++
    return total / Math.max(1, batches)
  }
}

export function evaluate(network: Network, samples: readonly Sample[], loss: Loss): Evaluation {
  let total = 0
  let correct = 0
  for (const sample of samples) {
    const output = network.predict(sample.input)
    total += lossValue(loss, output, sample.target)
    if (output[0] >= 0.5 === sample.target[0] >= 0.5) correct++
  }
  const n = Math.max(1, samples.length)
  return { loss: total / n, accuracy: loss === 'bce' ? correct / n : Number.NaN }
}

/** Sum of squared weights: the quantity the L2 penalty keeps small. */
export function weightNorm(network: Network): number {
  let sum = 0
  for (const layer of network.layers) for (const w of layer.weights) sum += w * w
  return sum
}

function rmsPerLayer(grads: Gradients): number[] {
  return grads.weights.map((g) => {
    let sum = 0
    for (const v of g) sum += v * v
    return Math.sqrt(sum / Math.max(1, g.length))
  })
}

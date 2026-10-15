import { activations, type Activation } from './activations'
import { lossValue, outputDelta, type Loss } from './loss'
import { createRng, gaussian, type Rng } from './random'

export interface NetworkConfig {
  /** Neurons per layer, input first: [2, 4, 4, 1] is 2 inputs, two hidden layers of 4, 1 output. */
  sizes: number[]
  /** Activation of every hidden layer. Default `tanh`. */
  hidden?: Activation
  /** Activation of the output layer. Default `sigmoid`. */
  output?: Activation
  /** Seed for the initial weights. Same seed, same network. */
  seed?: number
}

/**
 * One fully connected layer: z = W·x + b, a = f(z).
 * Weights are stored row-major: `weights[j * inSize + i]` connects input i to neuron j.
 */
export interface Layer {
  inSize: number
  outSize: number
  activation: Activation
  weights: Float64Array
  biases: Float64Array
}

/** Everything the forward pass computed, which is exactly what backprop needs. */
export interface Trace {
  /** Pre-activations per layer. */
  z: Float64Array[]
  /** Activations: `a[0]` is the input, `a[l + 1]` is the output of layer l. */
  a: Float64Array[]
}

/** Gradients shaped like the parameters. */
export interface Gradients {
  weights: Float64Array[]
  biases: Float64Array[]
}

export class Network {
  readonly layers: Layer[]

  constructor(config: NetworkConfig) {
    const { sizes, hidden = 'tanh', output = 'sigmoid', seed = 1 } = config
    if (sizes.length < 2) throw new Error('A network needs at least an input and an output layer')
    const rng = createRng(seed)
    this.layers = []
    for (let l = 0; l < sizes.length - 1; l++) {
      const inSize = sizes[l]
      const outSize = sizes[l + 1]
      const activation = l === sizes.length - 2 ? output : hidden
      this.layers.push({
        inSize,
        outSize,
        activation,
        weights: initWeights(inSize, outSize, activation, rng),
        biases: new Float64Array(outSize),
      })
    }
  }

  get sizes(): number[] {
    return [this.layers[0].inSize, ...this.layers.map((layer) => layer.outSize)]
  }

  get parameterCount(): number {
    let count = 0
    for (const layer of this.layers) count += layer.weights.length + layer.biases.length
    return count
  }

  /** Parameter arrays in a fixed order (W₀, b₀, W₁, b₁, …), matching `flatGradients`. */
  parameters(): Float64Array[] {
    return this.layers.flatMap((layer) => [layer.weights, layer.biases])
  }

  /** Run the input through every layer, keeping intermediate values. */
  trace(input: ArrayLike<number>): Trace {
    const z: Float64Array[] = []
    const a: Float64Array[] = [Float64Array.from(input as ArrayLike<number>)]
    for (const layer of this.layers) {
      const x = a[a.length - 1]
      const zl = new Float64Array(layer.outSize)
      const al = new Float64Array(layer.outSize)
      const f = activations[layer.activation].f
      for (let j = 0; j < layer.outSize; j++) {
        let sum = layer.biases[j]
        const row = j * layer.inSize
        for (let i = 0; i < layer.inSize; i++) sum += layer.weights[row + i] * x[i]
        zl[j] = sum
        al[j] = f(sum)
      }
      z.push(zl)
      a.push(al)
    }
    return { z, a }
  }

  /** Forward pass: the output only. */
  predict(input: ArrayLike<number>): Float64Array {
    const { a } = this.trace(input)
    return a[a.length - 1]
  }

  loss(input: ArrayLike<number>, target: ArrayLike<number>, loss: Loss): number {
    return lossValue(loss, this.predict(input), target)
  }

  /**
   * Backpropagation for one sample. Returns ∂L/∂W and ∂L/∂b for every layer.
   *
   * δ (delta) is ∂L/∂z, the loss's sensitivity to a neuron's pre-activation.
   *  - Output layer: δ comes straight from the loss (see `outputDelta`).
   *  - Hidden layer: δⱼ = f'(zⱼ) · Σₖ Wₖⱼ·δₖ (the chain rule, one layer back).
   * Then ∂L/∂Wⱼᵢ = δⱼ·aᵢ (the input that weight multiplied) and ∂L/∂bⱼ = δⱼ.
   */
  backward(trace: Trace, target: ArrayLike<number>, loss: Loss): Gradients {
    const L = this.layers.length
    const weights: Float64Array[] = new Array(L)
    const biases: Float64Array[] = new Array(L)

    const last = this.layers[L - 1]
    const lastFn = activations[last.activation]
    let delta = new Float64Array(last.outSize)
    for (let j = 0; j < last.outSize; j++) {
      delta[j] = outputDelta(loss, last.activation, trace.z[L - 1][j], trace.a[L][j], target[j], lastFn.df)
    }

    for (let l = L - 1; l >= 0; l--) {
      const layer = this.layers[l]
      const input = trace.a[l]
      const gw = new Float64Array(layer.weights.length)
      for (let j = 0; j < layer.outSize; j++) {
        const row = j * layer.inSize
        for (let i = 0; i < layer.inSize; i++) gw[row + i] = delta[j] * input[i]
      }
      weights[l] = gw
      biases[l] = delta

      if (l > 0) {
        const below = this.layers[l - 1]
        const df = activations[below.activation].df
        const next = new Float64Array(layer.inSize)
        for (let i = 0; i < layer.inSize; i++) {
          let sum = 0
          for (let j = 0; j < layer.outSize; j++) sum += layer.weights[j * layer.inSize + i] * delta[j]
          next[i] = sum * df(trace.z[l - 1][i], trace.a[l][i])
        }
        delta = next
      }
    }
    return { weights, biases }
  }

  /** Average gradient over a batch of samples. */
  gradients(batch: readonly Sample[], loss: Loss): Gradients {
    const total = zeroGradients(this)
    for (const sample of batch) {
      const g = this.backward(this.trace(sample.input), sample.target, loss)
      for (let l = 0; l < this.layers.length; l++) {
        addInto(total.weights[l], g.weights[l])
        addInto(total.biases[l], g.biases[l])
      }
    }
    const scale = 1 / Math.max(1, batch.length)
    for (let l = 0; l < this.layers.length; l++) {
      scaleInPlace(total.weights[l], scale)
      scaleInPlace(total.biases[l], scale)
    }
    return total
  }

  /** Deep copy, e.g. to remember the weights at the best validation epoch. */
  clone(): Network {
    const copy = Object.create(Network.prototype) as Network
    ;(copy as { layers: Layer[] }).layers = this.layers.map((layer) => ({
      ...layer,
      weights: layer.weights.slice(),
      biases: layer.biases.slice(),
    }))
    return copy
  }

  /** Copy weights from a network of the same shape. */
  copyFrom(other: Network) {
    this.layers.forEach((layer, l) => {
      layer.weights.set(other.layers[l].weights)
      layer.biases.set(other.layers[l].biases)
    })
  }
}

export interface Sample {
  input: number[]
  target: number[]
}

/** Gradients in the same order as `Network.parameters()`. */
export function flatGradients(g: Gradients): Float64Array[] {
  return g.weights.flatMap((w, l) => [w, g.biases[l]])
}

export function zeroGradients(network: Network): Gradients {
  return {
    weights: network.layers.map((layer) => new Float64Array(layer.weights.length)),
    biases: network.layers.map((layer) => new Float64Array(layer.biases.length)),
  }
}

/**
 * Random starting weights, scaled by fan-in. Too big and tanh/sigmoid saturate
 * from the first step; too small and the signal fades layer by layer.
 *  - He (√(2/n)) for ReLU, which zeroes half its inputs.
 *  - Xavier/Glorot (√(1/n)) for the rest.
 */
function initWeights(inSize: number, outSize: number, activation: Activation, rng: Rng) {
  const scale = Math.sqrt((activation === 'relu' ? 2 : 1) / inSize)
  const weights = new Float64Array(inSize * outSize)
  for (let i = 0; i < weights.length; i++) weights[i] = gaussian(rng) * scale
  return weights
}

function addInto(target: Float64Array, source: Float64Array) {
  for (let i = 0; i < target.length; i++) target[i] += source[i]
}

function scaleInPlace(target: Float64Array, scale: number) {
  for (let i = 0; i < target.length; i++) target[i] *= scale
}

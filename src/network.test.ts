import { describe, expect, it } from 'vitest'
import type { Activation } from './activations'
import { makeClassification, makeRegression, split } from './data'
import type { Loss } from './loss'
import { Network, type Sample } from './network'
import { Trainer, evaluate } from './trainer'

/** Numerical ∂L/∂θ by central differences, for comparing against backprop. */
function numericGradient(network: Network, sample: Sample, loss: Loss, params: Float64Array, i: number) {
  const h = 1e-6
  const original = params[i]
  params[i] = original + h
  const up = network.loss(sample.input, sample.target, loss)
  params[i] = original - h
  const down = network.loss(sample.input, sample.target, loss)
  params[i] = original
  return (up - down) / (2 * h)
}

describe('backpropagation', () => {
  const cases: [Activation, Activation, Loss][] = [
    ['tanh', 'sigmoid', 'bce'],
    ['sigmoid', 'sigmoid', 'mse'],
    ['relu', 'linear', 'mse'],
    ['tanh', 'tanh', 'mse'],
  ]
  for (const [hidden, output, loss] of cases) {
    it(`matches finite differences (${hidden} → ${output}, ${loss})`, () => {
      const network = new Network({ sizes: [3, 5, 4, 2], hidden, output, seed: 7 })
      const sample: Sample = { input: [0.3, -0.7, 0.5], target: [1, 0] }
      const grads = network.backward(network.trace(sample.input), sample.target, loss)
      network.layers.forEach((layer, l) => {
        for (let i = 0; i < layer.weights.length; i++) {
          expect(grads.weights[l][i]).toBeCloseTo(numericGradient(network, sample, loss, layer.weights, i), 6)
        }
        for (let i = 0; i < layer.biases.length; i++) {
          expect(grads.biases[l][i]).toBeCloseTo(numericGradient(network, sample, loss, layer.biases, i), 6)
        }
      })
    })
  }
})

describe('network', () => {
  it('is deterministic for a seed', () => {
    const a = new Network({ sizes: [2, 4, 1], seed: 3 })
    const b = new Network({ sizes: [2, 4, 1], seed: 3 })
    const c = new Network({ sizes: [2, 4, 1], seed: 4 })
    expect(a.predict([0.2, 0.4])[0]).toBe(b.predict([0.2, 0.4])[0])
    expect(a.predict([0.2, 0.4])[0]).not.toBe(c.predict([0.2, 0.4])[0])
  })

  it('counts parameters', () => {
    // (2·4 + 4) + (4·1 + 1)
    expect(new Network({ sizes: [2, 4, 1] }).parameterCount).toBe(17)
  })

  it('clones independently', () => {
    const net = new Network({ sizes: [2, 3, 1] })
    const copy = net.clone()
    net.layers[0].weights[0] += 1
    expect(copy.layers[0].weights[0]).not.toBe(net.layers[0].weights[0])
    copy.copyFrom(net)
    expect(copy.predict([0.1, 0.2])[0]).toBe(net.predict([0.1, 0.2])[0])
  })
})

describe('training', () => {
  it('learns XOR, which no straight line can separate', () => {
    const data = makeClassification('xor', 200, 0, 2)
    const network = new Network({ sizes: [2, 8, 1], seed: 1 })
    const trainer = new Trainer(network, {
      loss: 'bce',
      optimizer: { name: 'adam', learningRate: 0.03 },
      batchSize: 10,
    })
    for (let i = 0; i < 150; i++) trainer.trainEpoch(data)
    expect(evaluate(network, data, 'bce').accuracy).toBeGreaterThan(0.97)
  })

  it('a network with no hidden layer cannot learn XOR', () => {
    const data = makeClassification('xor', 200, 0, 2)
    const network = new Network({ sizes: [2, 1], seed: 1 })
    const trainer = new Trainer(network, {
      loss: 'bce',
      optimizer: { name: 'adam', learningRate: 0.03 },
      batchSize: 10,
    })
    for (let i = 0; i < 150; i++) trainer.trainEpoch(data)
    expect(evaluate(network, data, 'bce').accuracy).toBeLessThan(0.75)
  })

  it('a big network on a few noisy points overfits: train loss falls far below test loss', () => {
    const { train, test } = split(makeRegression(120, 0.25, 5), 0.12, 5)
    const network = new Network({ sizes: [1, 40, 40, 1], hidden: 'tanh', output: 'linear', seed: 2 })
    const trainer = new Trainer(network, {
      loss: 'mse',
      optimizer: { name: 'adam', learningRate: 0.01 },
      batchSize: train.length,
    })
    for (let i = 0; i < 3000; i++) trainer.trainEpoch(train)
    const trainLoss = evaluate(network, train, 'mse').loss
    const testLoss = evaluate(network, test, 'mse').loss
    expect(trainLoss).toBeLessThan(0.005)
    expect(testLoss).toBeGreaterThan(trainLoss * 5)
  })

  it('L2 regularization keeps the weights smaller', () => {
    const data = makeRegression(20, 0.2, 1)
    const run = (l2: number) => {
      const network = new Network({ sizes: [1, 20, 1], output: 'linear', seed: 1 })
      const trainer = new Trainer(network, {
        loss: 'mse',
        optimizer: { name: 'adam', learningRate: 0.01 },
        batchSize: 20,
        l2,
      })
      for (let i = 0; i < 800; i++) trainer.trainEpoch(data)
      return network.layers.reduce((s, layer) => s + layer.weights.reduce((a, w) => a + w * w, 0), 0)
    }
    expect(run(0.01)).toBeLessThan(run(0) * 0.7)
  })
})

describe('data', () => {
  it('splits without losing or duplicating samples', () => {
    const data = makeClassification('circle', 100, 0.1)
    const { train, test } = split(data, 0.7)
    expect(train.length).toBe(70)
    expect(test.length).toBe(30)
    expect(new Set([...train, ...test]).size).toBe(100)
  })

  it('makes balanced classes', () => {
    for (const name of ['circle', 'xor', 'gaussians', 'moons', 'spiral'] as const) {
      const ones = makeClassification(name, 100, 0).filter((s) => s.target[0] === 1).length
      expect(ones).toBe(50)
    }
  })
})

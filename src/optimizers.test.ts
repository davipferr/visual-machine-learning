import { describe, expect, it } from 'vitest'
import { createOptimizer } from './optimizers'
import { fitPolynomial, meanSquaredError, solve } from './polyfit'
import { createRng } from './random'
import { descend, surfaces } from './surfaces'

const distance = (p: [number, number], q: [number, number]) => Math.hypot(p[0] - q[0], p[1] - q[1])

describe('gradient descent on surfaces', () => {
  it('every surface gradient matches finite differences', () => {
    const h = 1e-6
    for (const s of Object.values(surfaces)) {
      for (const [x, y] of [[0.3, -0.8], [1.7, 0.4], [-2.1, 1.3]]) {
        const [gx, gy] = s.grad(x, y)
        expect(gx).toBeCloseTo((s.f(x + h, y) - s.f(x - h, y)) / (2 * h), 4)
        expect(gy).toBeCloseTo((s.f(x, y + h) - s.f(x, y - h)) / (2 * h), 4)
      }
    }
  })

  it('reaches the bottom of the bowl', () => {
    const s = surfaces.bowl
    const { path } = descend(s, { name: 'sgd', learningRate: s.learningRate }, s.start, 200)
    expect(distance(path.at(-1)!, [0, 0])).toBeLessThan(1e-6)
  })

  it('diverges when the learning rate exceeds 2 / curvature', () => {
    // Curvature of the bowl is 1, so η = 2 is the edge of stability.
    const s = surfaces.bowl
    expect(descend(s, { name: 'sgd', learningRate: 1.9 }, s.start, 500).diverged).toBe(false)
    expect(descend(s, { name: 'sgd', learningRate: 2.1 }, s.start, 500).diverged).toBe(true)
  })

  it('momentum crosses the narrow valley much faster than plain gradient descent', () => {
    const s = surfaces.ravine
    const steps = 100
    const sgd = descend(s, { name: 'sgd', learningRate: s.learningRate }, s.start, steps).path.at(-1)!
    const mom = descend(s, { name: 'momentum', learningRate: s.learningRate }, s.start, steps).path.at(-1)!
    expect(distance(mom, [0, 0])).toBeLessThan(distance(sgd, [0, 0]) / 5)
  })

  it('Adam finds the Rosenbrock minimum', () => {
    const s = surfaces.rosenbrock
    const { path } = descend(s, { name: 'adam', learningRate: 0.05 }, s.start, 3000)
    expect(distance(path.at(-1)!, [1, 1])).toBeLessThan(0.05)
  })

  it('on a bumpy surface, the start decides which valley you end in', () => {
    const s = surfaces.bumpy
    const end = (start: [number, number]) =>
      descend(s, { name: 'sgd', learningRate: s.learningRate }, start, 500).path.at(-1)!
    const a = end([1, -1])
    const b = end([-5.2, 3.6])
    expect(distance(a, [1.35, -1.35])).toBeLessThan(0.1)
    expect(s.f(b[0], b[1])).toBeGreaterThan(s.f(a[0], a[1]) + 0.5) // stuck higher up
  })

  it('reads the learning rate live', () => {
    const opt = createOptimizer({ name: 'sgd', learningRate: 1 })
    const theta = Float64Array.of(10)
    opt.step([theta], [Float64Array.of(1)])
    opt.config.learningRate = 5
    opt.step([theta], [Float64Array.of(1)])
    expect(theta[0]).toBe(4)
  })
})

describe('polynomial fitting', () => {
  it('solves linear systems', () => {
    expect(solve([[0, 2], [3, 1]], [4, 5]).map((v) => +v.toFixed(10))).toEqual([1, 2])
  })

  it('recovers an exact cubic', () => {
    const xs = Array.from({ length: 20 }, (_, i) => -1 + i / 10)
    const ys = xs.map((x) => 2 * x ** 3 - x + 0.5)
    const fit = fitPolynomial(xs, ys, 3)
    expect(fit.predict(0.37)).toBeCloseTo(2 * 0.37 ** 3 - 0.37 + 0.5, 8)
  })

  it('a high enough degree passes through every training point', () => {
    const rng = createRng(1)
    const xs = Array.from({ length: 8 }, () => rng() * 2 - 1)
    const ys = xs.map(() => rng())
    expect(meanSquaredError(fitPolynomial(xs, ys, 7).predict, xs, ys)).toBeLessThan(1e-6)
  })
})

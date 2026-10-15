import { createOptimizer, type OptimizerConfig } from './optimizers'

export type SurfaceName = 'bowl' | 'ravine' | 'rosenbrock' | 'bumpy' | 'himmelblau'

/**
 * A loss with only two parameters, so the whole landscape fits on screen.
 * Real networks have thousands of parameters, but gradient descent does the
 * same thing in every direction at once.
 */
export interface Surface {
  name: SurfaceName
  label: string
  f: (x: number, y: number) => number
  grad: (x: number, y: number) => [number, number]
  x: [number, number]
  y: [number, number]
  /** A learning rate that works for plain gradient descent here. */
  learningRate: number
  start: [number, number]
  /** Where the lowest points are (global minima). */
  minima: [number, number][]
}

export const surfaces: Record<SurfaceName, Surface> = {
  bowl: {
    name: 'bowl',
    label: 'Round bowl',
    f: (x, y) => 0.5 * (x * x + y * y),
    grad: (x, y) => [x, y],
    x: [-4, 4],
    y: [-2.5, 2.5],
    learningRate: 0.1,
    start: [-3.4, 2],
    minima: [[0, 0]],
  },
  ravine: {
    // Steep across (curvature 12), shallow along (curvature 0.1). Plain
    // gradient descent bounces between the walls while crawling along the floor.
    name: 'ravine',
    label: 'Narrow valley',
    f: (x, y) => 0.05 * x * x + 6 * y * y,
    grad: (x, y) => [0.1 * x, 12 * y],
    x: [-4, 4],
    y: [-2.5, 2.5],
    learningRate: 0.15,
    start: [-3.6, 1.6],
    minima: [[0, 0]],
  },
  rosenbrock: {
    // A curved, nearly flat banana-shaped valley: easy to reach, slow to follow.
    name: 'rosenbrock',
    label: 'Banana valley',
    f: (x, y) => (1 - x) ** 2 + 10 * (y - x * x) ** 2,
    grad: (x, y) => [-2 * (1 - x) - 40 * x * (y - x * x), 20 * (y - x * x)],
    x: [-2.4, 2.4],
    y: [-1.2, 3],
    learningRate: 0.01,
    start: [-1.8, 2.6],
    minima: [[1, 1]],
  },
  bumpy: {
    // A gentle bowl with ripples: several valleys, and the one you end up in
    // depends on where you start.
    name: 'bumpy',
    label: 'Bumpy (local minima)',
    f: (x, y) => 0.06 * (x * x + y * y) + Math.sin(x) * Math.sin(y),
    grad: (x, y) => [0.12 * x + Math.cos(x) * Math.sin(y), 0.12 * y + Math.sin(x) * Math.cos(y)],
    x: [-7, 7],
    y: [-4.4, 4.4],
    learningRate: 0.3,
    start: [-5.2, 3.6],
    minima: [
      [1.35, -1.35],
      [-1.35, 1.35],
    ],
  },
  himmelblau: {
    // Four equally deep minima.
    name: 'himmelblau',
    label: 'Four minima',
    f: (x, y) => (x * x + y - 11) ** 2 + (x + y * y - 7) ** 2,
    grad: (x, y) => {
      const a = x * x + y - 11
      const b = x + y * y - 7
      return [4 * x * a + 2 * b, 2 * a + 4 * y * b]
    },
    x: [-6, 6],
    y: [-5, 5],
    learningRate: 0.01,
    start: [-0.3, -0.6],
    minima: [
      [3, 2],
      [-2.805118, 3.131312],
      [-3.77931, -3.283186],
      [3.584428, -1.848126],
    ],
  },
}

export interface DescentResult {
  /** Visited points, start included. */
  path: [number, number][]
  /** The loss shot off to infinity (or NaN): the learning rate is too big. */
  diverged: boolean
}

/** Run an optimizer on a 2D surface for `steps` steps. */
export function descend(
  surface: Surface,
  optimizer: OptimizerConfig,
  start: [number, number],
  steps: number,
): DescentResult {
  const opt = createOptimizer(optimizer)
  const theta = Float64Array.from(start)
  const path: [number, number][] = [[theta[0], theta[1]]]
  const grad = new Float64Array(2)
  for (let i = 0; i < steps; i++) {
    const [gx, gy] = surface.grad(theta[0], theta[1])
    grad[0] = gx
    grad[1] = gy
    opt.step([theta], [grad])
    const value = surface.f(theta[0], theta[1])
    if (!Number.isFinite(value) || Math.abs(value) > 1e8) {
      return { path, diverged: true }
    }
    path.push([theta[0], theta[1]])
  }
  return { path, diverged: false }
}

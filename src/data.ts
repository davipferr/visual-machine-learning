import type { Sample } from './network'
import { createRng, gaussian, shuffle, type Rng } from './random'

export type DatasetName = 'circle' | 'xor' | 'gaussians' | 'moons' | 'spiral'

export const datasetLabels: Record<DatasetName, string> = {
  circle: 'Circle',
  xor: 'XOR',
  gaussians: 'Two blobs',
  moons: 'Moons',
  spiral: 'Spiral',
}

/**
 * Two-class points in roughly [−1, 1]², labelled 0 or 1.
 * `noise` is the standard deviation of Gaussian jitter added to every coordinate.
 */
export function makeClassification(name: DatasetName, count: number, noise: number, seed = 1): Sample[] {
  const rng = createRng(seed)
  const samples: Sample[] = []
  for (let i = 0; i < count; i++) {
    const label = i % 2
    const [x, y] = point(name, label, i, count, rng)
    samples.push({
      input: [x + gaussian(rng) * noise, y + gaussian(rng) * noise],
      target: [label],
    })
  }
  return shuffle(samples, rng)
}

function point(name: DatasetName, label: number, i: number, count: number, rng: Rng): [number, number] {
  switch (name) {
    case 'circle': {
      const r = label === 1 ? rng() * 0.45 : 0.65 + rng() * 0.3
      const angle = rng() * Math.PI * 2
      return [r * Math.cos(angle), r * Math.sin(angle)]
    }
    case 'xor': {
      // Keep a small gap around the axes so the four quadrants are separable.
      const pad = 0.06
      const sx = rng() < 0.5 ? -1 : 1
      const sy = label === 1 ? sx : -sx
      return [sx * (pad + rng() * (0.95 - pad)), sy * (pad + rng() * (0.95 - pad))]
    }
    case 'gaussians': {
      const c = label === 1 ? 0.45 : -0.45
      return [c + gaussian(rng) * 0.2, c + gaussian(rng) * 0.2]
    }
    case 'moons': {
      const t = rng() * Math.PI
      return label === 1
        ? [Math.cos(t) * 0.6 - 0.3, Math.sin(t) * 0.6 - 0.15]
        : [0.3 - Math.cos(t) * 0.6, 0.15 - Math.sin(t) * 0.6]
    }
    case 'spiral': {
      const k = Math.floor(i / 2) / Math.ceil(count / 2)
      const r = 0.08 + k * 0.85
      const angle = k * Math.PI * 3.5 + label * Math.PI
      return [r * Math.cos(angle), r * Math.sin(angle)]
    }
  }
}

/** The curve the regression data is sampled from: y = 0.8·sin(π·x). */
export const regressionTruth = (x: number) => 0.8 * Math.sin(Math.PI * x)

/** Points on `regressionTruth` with x in [−1, 1] and Gaussian noise on y. */
export function makeRegression(count: number, noise: number, seed = 1): Sample[] {
  const rng = createRng(seed)
  const samples: Sample[] = []
  for (let i = 0; i < count; i++) {
    const x = rng() * 2 - 1
    samples.push({ input: [x], target: [regressionTruth(x) + gaussian(rng) * noise] })
  }
  return samples
}

/**
 * Shuffle and cut into a training set (what the model learns from) and a test
 * set (held back, used only to measure how well it generalizes).
 */
export function split(samples: readonly Sample[], trainFraction: number, seed = 1) {
  const shuffled = shuffle([...samples], createRng(seed))
  const cut = Math.max(1, Math.min(shuffled.length - 1, Math.round(shuffled.length * trainFraction)))
  return { train: shuffled.slice(0, cut), test: shuffled.slice(cut) }
}

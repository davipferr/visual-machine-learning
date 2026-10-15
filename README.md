# Mini Neural Net

A small, educational neural network library in TypeScript, with an interactive playground
that trains networks live in the browser. There are no ML libraries: the forward pass,
backpropagation and the optimizers are written by hand, in a few hundred lines.

```ts
import { Network, Trainer, evaluate, makeClassification, split } from 'mini-neural-net'

const { train, test } = split(makeClassification('spiral', 400, 0.05), 0.7)
const network = new Network({ sizes: [2, 10, 10, 1], hidden: 'tanh', output: 'sigmoid' })
const trainer = new Trainer(network, {
  loss: 'bce',
  optimizer: { name: 'adam', learningRate: 0.01 },
  batchSize: 10,
  l2: 0.001,
})

for (let epoch = 0; epoch < 300; epoch++) trainer.trainEpoch(train)
evaluate(network, test, 'bce') // { loss, accuracy }
```

## Scripts

| Command | What it does |
| --- | --- |
| `npm run dev` | The playground (see below) |
| `npm test` | Vitest: backprop vs finite differences, learning XOR, overfitting, optimizers, polynomial fits |
| `npm run build` | Library build (ESM + CJS + `.d.ts`) into `dist/` |
| `npm run build:playground` | Static playground build into `dist-playground/` |
| `npm run typecheck` | `tsc --noEmit` over library and playground |

## Playground

`npm run dev` opens an interactive page about how neural networks learn. The sidebar sets up one
experiment (dataset, noise, train/test split, layers, activation, optimizer, learning rate, batch
size, L2) with presets such as *Overfit*, *LR too high* and *Vanishing*. Sections 1–4 all watch
that one live training run ([`playground/session.ts`](playground/session.ts)):

1. **What the network believes**: its predicted probability over the whole input plane, the
   decision boundary, training points and held-out test points. Hover to query it.
2. **Inside the network**: every neuron drawn as a map of where it fires, and every weight as a
   line whose thickness and colour show its size and sign.
3. **Training loss vs test loss**: the two curves, the best test epoch, a shaded overfitting zone
   once they split apart, and a button that rewinds to the best epoch (early stopping).
4. **Backpropagation**: the gradient size reaching each layer, where deep sigmoid stacks show
   vanishing gradients.

The rest have their own controls:

5. **Gradient descent on a landscape**: gradient descent, momentum and Adam racing on two-parameter
   losses (round bowl, narrow valley, Rosenbrock, local minima, four equal minima). Drag the start.
6. **The learning rate**: too small, good, zig-zagging and your own choice on the same bowl, and why
   the limit is `η < 2 / curvature`.
7. **One neuron, one step at a time**: linear regression as a one-neuron network. Shows the loss
   landscape over (w, b), and the gradient and update with the real numbers filled in.
8. **Overfitting, up close**: a network with over 1,000 parameters fitting 12 noisy points until it threads
   through every one. Try more data, a smaller network, L2 or early stopping.
9. **Model capacity**: polynomial degree against training and test error (the U-shaped curve), with
   fits to other samples drawn faintly to show variance.

## How it works

### 1. The network: [`src/network.ts`](src/network.ts)

Fully connected layers: `z = W·x + b`, `a = f(z)`. Weights are `Float64Array`s, row-major.
Initial weights are Gaussian, scaled by fan-in (He for ReLU, Xavier for the rest). Without
that scaling, tanh and sigmoid saturate from the first step or the signal fades layer by layer.
`trace()` keeps every `z` and `a` from the forward pass, because backprop needs them.

### 2. Backpropagation

`δ = ∂L/∂z` is computed for the output layer first, then passed back one layer at a time:

- output: `δ = (a − t)·f′(z)` for MSE. For **cross-entropy with a sigmoid** the derivatives cancel
  to exactly `δ = a − t` ([`src/loss.ts`](src/loss.ts)), which also avoids dividing by zero when the
  sigmoid saturates;
- hidden: `δⱼ = f′(zⱼ) · Σₖ Wₖⱼ·δₖ`;
- gradients: `∂L/∂Wⱼᵢ = δⱼ·aᵢ` and `∂L/∂bⱼ = δⱼ`.

The tests compare every gradient against central finite differences for four
activation/loss combinations.

### 3. Optimizers: [`src/optimizers.ts`](src/optimizers.ts)

All three work on arrays of parameter vectors, so the same code trains a network and walks a 2D
landscape ([`src/surfaces.ts`](src/surfaces.ts)):

- **Gradient descent**: `θ ← θ − η·g`
- **Momentum**: `v ← β·v + g; θ ← θ − η·v`. It speeds up along consistent slopes and averages
  away zig-zags.
- **Adam**: running mean and running square of the gradient, bias-corrected, so each parameter
  gets its own step size.

The learning rate is read on every step, so it can change mid-training.

### 4. Training: [`src/trainer.ts`](src/trainer.ts)

Shuffled mini-batches, averaged gradients, optional L2 penalty (`λ/2·Σw²`, which adds `λ·w` to each
weight gradient). The trainer also records the RMS gradient per layer for the backprop chart.
`evaluate()` reports mean loss and accuracy.

### 5. Data and capacity

- [`src/data.ts`](src/data.ts): seeded 2D datasets (circle, XOR, two blobs, moons, spiral),
  a noisy sine for regression, and a train/test `split`.
- [`src/polyfit.ts`](src/polyfit.ts): least-squares polynomials in the Legendre basis (raw powers
  up to x¹⁴ make the normal equations badly conditioned), solved by Gaussian elimination.
- [`src/random.ts`](src/random.ts): Mulberry32. Everything takes a seed, so every run can be replayed.

## API

| Export | |
| --- | --- |
| `Network` | `new Network({ sizes, hidden, output, seed })`, `predict`, `trace`, `backward`, `gradients`, `loss`, `parameters`, `clone`, `copyFrom` |
| `Trainer` | `step(batch)`, `trainEpoch(train)`, `setOptimizer`, `epoch`, `layerGradients` |
| `evaluate`, `weightNorm` | Mean loss / accuracy on a set, sum of squared weights |
| `createOptimizer` | `{ name: 'sgd' \| 'momentum' \| 'adam', learningRate, momentum, beta1, beta2 }` |
| `surfaces`, `descend` | 2D loss landscapes and an optimizer path on one |
| Data | `makeClassification`, `makeRegression`, `regressionTruth`, `split` |
| Math | `activations`, `lossValue`, `outputDelta`, `fitPolynomial`, `legendre`, `solve`, `createRng`, `gaussian`, `shuffle` |

## Known limitations

These are deliberate, to keep the project small:

- Dense layers only: no convolutions, no dropout, no batch normalization.
- One sample at a time through the network (no matrix batching), which is plenty for a few hundred
  points and keeps backprop readable.
- Losses: MSE and binary cross-entropy. No softmax / multi-class.
- Early stopping in the playground uses the test set for simplicity. Real projects use a separate
  validation set for that, so the test set stays untouched.

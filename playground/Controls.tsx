import { datasetLabels, optimizerLabels, type Activation, type DatasetName, type OptimizerName } from 'mini-neural-net'
import type { ReactNode } from 'react'
import { short, useRerender } from './canvas'
import type { TrainingSession } from './session'

export interface Params {
  dataset: DatasetName
  noise: number
  trainShare: number
  points: number
  layers: number
  neurons: number
  activation: Activation
  optimizer: OptimizerName
  learningRate: number
  /** 0 means the whole training set (full-batch gradient descent). */
  batchSize: number
  l2: number
  speed: number
}

export const defaultParams: Params = {
  dataset: 'circle',
  noise: 0.08,
  trainShare: 0.6,
  points: 300,
  layers: 2,
  neurons: 6,
  activation: 'tanh',
  optimizer: 'adam',
  learningRate: 0.01,
  batchSize: 10,
  l2: 0,
  speed: 1,
}

const presets: Record<string, { hint: string; params: Partial<Params> }> = {
  Healthy: { hint: 'A sensible setup that learns the circle quickly.', params: defaultParams },
  Spiral: {
    hint: 'A hard shape: it needs depth and patience.',
    params: { dataset: 'spiral', noise: 0.03, trainShare: 0.7, points: 400, layers: 3, neurons: 10, activation: 'tanh', optimizer: 'adam', learningRate: 0.01, batchSize: 10, l2: 0 },
  },
  Overfit: {
    hint: 'Few noisy points, a big network, no brakes. Watch the test loss turn upward.',
    params: { dataset: 'gaussians', noise: 0.3, trainShare: 0.12, points: 300, layers: 3, neurons: 14, activation: 'tanh', optimizer: 'adam', learningRate: 0.03, batchSize: 0, l2: 0, speed: 4 },
  },
  'Overfit + L2': {
    hint: 'The same setup with weight decay: smoother boundary, smaller gap.',
    params: { dataset: 'gaussians', noise: 0.3, trainShare: 0.12, points: 300, layers: 3, neurons: 14, activation: 'tanh', optimizer: 'adam', learningRate: 0.03, batchSize: 0, l2: 0.01, speed: 4 },
  },
  'LR too high': {
    hint: 'Plain gradient descent with steps far too big: the loss jumps around or explodes.',
    params: { dataset: 'circle', optimizer: 'sgd', learningRate: 3, batchSize: 1, layers: 2, neurons: 6, activation: 'relu', l2: 0 },
  },
  'LR too low': {
    hint: 'Tiny steps: it is going the right way, just very slowly.',
    params: { dataset: 'circle', optimizer: 'sgd', learningRate: 0.002, batchSize: 10, layers: 2, neurons: 6, activation: 'tanh', l2: 0 },
  },
  Vanishing: {
    hint: 'Six sigmoid layers: the gradient shrinks at every layer on the way back, so the first layers barely move.',
    params: { dataset: 'xor', noise: 0.05, layers: 6, neurons: 6, activation: 'sigmoid', optimizer: 'sgd', learningRate: 0.3, batchSize: 10, l2: 0 },
  },
  'No hidden layer': {
    hint: 'Just a weighted sum and a sigmoid: it can only draw a straight line.',
    params: { dataset: 'xor', noise: 0.05, layers: 0, optimizer: 'adam', learningRate: 0.03, l2: 0 },
  },
}

const batchOptions = [1, 4, 10, 32, 0]
const l2Options = [0, 0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03]

interface SliderProps {
  label: ReactNode
  value: number
  min: number
  max: number
  step: number
  display?: string
  onChange: (value: number) => void
}

export function Slider({ label, value, min, max, step, display, onChange }: SliderProps) {
  return (
    <label className="slider">
      <span className="slider-label">
        <span>{label}</span>
        <output>{display ?? value}</output>
      </span>
      <input type="range" min={min} max={max} step={step} value={value} onChange={(e) => onChange(Number(e.target.value))} />
    </label>
  )
}

/** A slider that moves through powers of ten: 0.001 → 0.01 → 0.1 take equal distance. */
export function LogSlider({
  label,
  value,
  min,
  max,
  onChange,
}: {
  label: ReactNode
  value: number
  min: number
  max: number
  onChange: (value: number) => void
}) {
  return (
    <Slider
      label={label}
      value={Math.log10(value)}
      min={Math.log10(min)}
      max={Math.log10(max)}
      step={0.01}
      display={short(value)}
      onChange={(exponent) => onChange(Number((10 ** exponent).toPrecision(2)))}
    />
  )
}

export function Controls({
  params,
  session,
  onChange,
  onReset,
}: {
  params: Params
  session: TrainingSession
  onChange: (params: Params) => void
  onReset: () => void
}) {
  const set = (patch: Partial<Params>) => onChange({ ...params, ...patch })
  useRerender(session)

  const latest = session.latest
  const gap = latest.testLoss - latest.trainLoss
  const shape = [2, ...Array(params.layers).fill(params.neurons), 1].join(' → ')
  const activePreset = Object.entries(presets).find(([, p]) =>
    Object.entries(p.params).every(([k, v]) => params[k as keyof Params] === v),
  )

  return (
    <aside className="card controls">
      <div>
        <h2>The experiment</h2>
        <p className="muted small">One network, trained live. Sections 1–4 all watch it.</p>
      </div>

      <div className="presets">
        {Object.entries(presets).map(([name, preset]) => (
          <button
            key={name}
            className={`chip${activePreset?.[0] === name ? ' active' : ''}`}
            title={preset.hint}
            onClick={() => onChange({ ...defaultParams, ...preset.params })}
          >
            {name}
          </button>
        ))}
      </div>
      {activePreset && <p className="hint">{activePreset[1].hint}</p>}

      <div className="row">
        <button className="primary play" onClick={() => session.toggle()} disabled={session.diverged}>
          {session.running ? '❚❚ Pause' : '▶ Train'}
        </button>
        <button onClick={() => session.stepEpoch()} disabled={session.running || session.diverged}>
          Step
        </button>
        <button onClick={onReset}>Reset</button>
      </div>

      <dl className="stats">
        <div>
          <dt>Epoch</dt>
          <dd>{latest.epoch}</dd>
        </div>
        <div>
          <dt>Test accuracy</dt>
          <dd>{Number.isNaN(latest.testAccuracy) ? '–' : `${(latest.testAccuracy * 100).toFixed(1)}%`}</dd>
        </div>
        <div>
          <dt>Train loss</dt>
          <dd className="train-text">{short(latest.trainLoss)}</dd>
        </div>
        <div>
          <dt>Test loss</dt>
          <dd className="test-text">{short(latest.testLoss)}</dd>
        </div>
        <div className="wide">
          <dt>Generalization gap (test − train)</dt>
          <dd className={gap > 0.1 ? 'warn' : ''}>
            {session.diverged ? 'diverged: the loss blew up' : short(gap)}
          </dd>
        </div>
      </dl>

      <h3>Data</h3>
      <label className="select">
        Dataset
        <select value={params.dataset} onChange={(e) => set({ dataset: e.target.value as DatasetName })}>
          {Object.entries(datasetLabels).map(([value, label]) => (
            <option key={value} value={value}>
              {label}
            </option>
          ))}
        </select>
      </label>
      <Slider label="Noise" value={params.noise} min={0} max={0.5} step={0.01} onChange={(noise) => set({ noise })} />
      <Slider label="Points" value={params.points} min={40} max={600} step={20} onChange={(points) => set({ points })} />
      <Slider
        label="Used for training"
        value={params.trainShare}
        min={0.05}
        max={0.9}
        step={0.01}
        display={`${Math.round(params.trainShare * 100)}%`}
        onChange={(trainShare) => set({ trainShare })}
      />

      <h3>Network</h3>
      <Slider label="Hidden layers" value={params.layers} min={0} max={6} step={1} onChange={(layers) => set({ layers })} />
      <Slider label="Neurons per layer" value={params.neurons} min={1} max={16} step={1} onChange={(neurons) => set({ neurons })} />
      <label className="select">
        Activation
        <select value={params.activation} onChange={(e) => set({ activation: e.target.value as Activation })}>
          <option value="tanh">tanh</option>
          <option value="relu">ReLU</option>
          <option value="sigmoid">sigmoid</option>
          <option value="linear">linear (none)</option>
        </select>
      </label>
      <p className="hint mono">
        {shape} · {session.network.parameterCount} parameters
      </p>

      <h3>Training</h3>
      <label className="select">
        Optimizer
        <select value={params.optimizer} onChange={(e) => set({ optimizer: e.target.value as OptimizerName })}>
          {Object.entries(optimizerLabels).map(([value, label]) => (
            <option key={value} value={value}>
              {label}
            </option>
          ))}
        </select>
      </label>
      <LogSlider
        label={<>Learning rate <var>η</var></>}
        value={params.learningRate}
        min={0.0001}
        max={5}
        onChange={(learningRate) => set({ learningRate })}
      />
      <Slider
        label="Batch size"
        value={batchOptions.indexOf(params.batchSize)}
        min={0}
        max={batchOptions.length - 1}
        step={1}
        display={params.batchSize === 0 ? `all (${session.setup.train.length})` : `${params.batchSize}`}
        onChange={(i) => set({ batchSize: batchOptions[i] })}
      />
      <Slider
        label={<>L2 regularization <var>λ</var></>}
        value={Math.max(0, l2Options.indexOf(params.l2))}
        min={0}
        max={l2Options.length - 1}
        step={1}
        display={params.l2 === 0 ? 'off' : `${params.l2}`}
        onChange={(i) => set({ l2: l2Options[i] })}
      />
      <Slider
        label="Speed"
        value={params.speed}
        min={1}
        max={20}
        step={1}
        display={`${params.speed} epoch${params.speed > 1 ? 's' : ''} / frame`}
        onChange={(speed) => set({ speed })}
      />
    </aside>
  )
}

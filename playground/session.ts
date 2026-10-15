import {
  Network,
  Trainer,
  evaluate,
  type Activation,
  type Loss,
  type OptimizerName,
  type Sample,
} from 'mini-neural-net'
import { onFrame } from './loop'

const SMOOTHING = 0.1

export interface Hyper {
  learningRate: number
  optimizer: OptimizerName
  batchSize: number
  l2: number
  /** Epochs per animation frame. */
  speed: number
}

export interface EpochPoint {
  epoch: number
  trainLoss: number
  testLoss: number
  trainAccuracy: number
  testAccuracy: number
  /**
   * Exponential moving averages of the two losses. Mini-batch training is noisy
   * from epoch to epoch; the trend is what tells you whether it is overfitting.
   */
  trainSmooth: number
  testSmooth: number
  /** RMS weight gradient per layer, input side first. */
  gradients: number[]
}

export interface SessionSetup {
  train: Sample[]
  test: Sample[]
  sizes: number[]
  hidden: Activation
  output: Activation
  loss: Loss
  seed: number
}

/**
 * A network being trained live, one or more epochs per animation frame.
 * Several sections watch the same session (like the shared spring in the
 * physics playground): the boundary, the diagram, the loss curves.
 */
export class TrainingSession {
  network!: Network
  trainer!: Trainer
  setup!: SessionSetup
  hyper: Hyper
  history: EpochPoint[] = []
  /** Epoch with the lowest (smoothed) test loss so far, and a copy of the weights from then. */
  best: { epoch: number; testLoss: number; network: Network } | null = null
  diverged = false
  maxEpochs = 5000

  private stopLoop: (() => void) | null = null
  private listeners = new Set<() => void>()

  constructor(setup: SessionSetup, hyper: Hyper) {
    this.hyper = hyper
    this.load(setup)
  }

  get running() {
    return this.stopLoop !== null
  }

  get epoch() {
    return this.trainer.epoch
  }

  get latest(): EpochPoint {
    return this.history[this.history.length - 1]
  }

  /** New data or a new architecture: start over from fresh random weights. */
  load(setup: SessionSetup) {
    this.setup = setup
    this.network = new Network({ sizes: setup.sizes, hidden: setup.hidden, output: setup.output, seed: setup.seed })
    this.trainer = new Trainer(this.network, {
      loss: setup.loss,
      optimizer: { name: this.hyper.optimizer, learningRate: this.hyper.learningRate },
      batchSize: this.hyper.batchSize,
      l2: this.hyper.l2,
      seed: setup.seed,
    })
    this.history = []
    this.best = null
    this.diverged = false
    this.record()
    this.emit()
  }

  /** Learning rate, batch size and L2 change on the fly, mid-training. */
  setHyper(hyper: Hyper) {
    const previous = this.hyper
    this.hyper = hyper
    if (hyper.optimizer !== previous.optimizer) {
      this.trainer.setOptimizer({ name: hyper.optimizer, learningRate: hyper.learningRate })
    }
    this.trainer.optimizer.config.learningRate = hyper.learningRate
    this.trainer.config = { ...this.trainer.config, batchSize: hyper.batchSize, l2: hyper.l2 }
  }

  play() {
    if (this.stopLoop || this.diverged) return
    if (this.epoch >= this.maxEpochs) return
    this.stopLoop = onFrame(() => {
      for (let i = 0; i < this.hyper.speed && this.running; i++) this.stepEpoch(false)
      this.emit()
    })
    this.emit()
  }

  pause() {
    this.stopLoop?.()
    this.stopLoop = null
    this.emit()
  }

  toggle() {
    if (this.running) this.pause()
    else this.play()
  }

  stepEpoch(notify = true) {
    if (this.diverged) return
    this.trainer.trainEpoch(this.setup.train)
    this.record()
    if (this.diverged || this.epoch >= this.maxEpochs) this.pause()
    if (notify) this.emit()
  }

  /** Put the weights from the best test epoch back: early stopping, after the fact. */
  restoreBest() {
    if (!this.best) return
    this.pause()
    this.network.copyFrom(this.best.network)
    this.history = this.history.filter((p) => p.epoch <= this.best!.epoch)
    this.trainer.epoch = this.best.epoch
    this.trainer.optimizer.reset()
    this.emit()
  }

  subscribe(listener: () => void) {
    this.listeners.add(listener)
    return () => {
      this.listeners.delete(listener)
    }
  }

  dispose() {
    this.pause()
  }

  private record() {
    const { loss, train, test } = this.setup
    const tr = evaluate(this.network, train, loss)
    const te = evaluate(this.network, test, loss)
    if (!Number.isFinite(tr.loss) || tr.loss > 1e6) this.diverged = true
    const previous = this.history[this.history.length - 1]
    const ema = (prev: number | undefined, value: number) =>
      prev === undefined || !Number.isFinite(prev) ? value : prev * (1 - SMOOTHING) + value * SMOOTHING
    const point: EpochPoint = {
      epoch: this.epoch,
      trainLoss: tr.loss,
      testLoss: te.loss,
      trainSmooth: ema(previous?.trainSmooth, tr.loss),
      testSmooth: ema(previous?.testSmooth, te.loss),
      trainAccuracy: tr.accuracy,
      testAccuracy: te.accuracy,
      gradients: this.trainer.layerGradients.slice(),
    }
    this.history.push(point)
    if (test.length > 0 && Number.isFinite(point.testSmooth) && (!this.best || point.testSmooth < this.best.testLoss)) {
      this.best = { epoch: this.epoch, testLoss: point.testSmooth, network: this.network.clone() }
    }
  }

  private emit() {
    for (const listener of this.listeners) listener()
  }
}

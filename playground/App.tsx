import { makeClassification, split } from 'mini-neural-net'
import { useEffect, useMemo, useState } from 'react'
import { Controls, defaultParams, type Params } from './Controls'
import { TrainingSession, type Hyper, type SessionSetup } from './session'
import { Capacity } from './sections/Capacity'
import { DecisionBoundary } from './sections/DecisionBoundary'
import { LayerGradients } from './sections/LayerGradients'
import { LearningRate } from './sections/LearningRate'
import { LineFit } from './sections/LineFit'
import { LossCurves } from './sections/LossCurves'
import { LossSurface } from './sections/LossSurface'
import { NetworkDiagram } from './sections/NetworkDiagram'
import { Overfitting } from './sections/Overfitting'

function makeSetup(params: Params, seed: number): SessionSetup {
  const data = makeClassification(params.dataset, params.points, params.noise, 11)
  const { train, test } = split(data, params.trainShare, 11)
  return {
    train,
    test,
    sizes: [2, ...Array<number>(params.layers).fill(params.neurons), 1],
    hidden: params.activation,
    output: 'sigmoid',
    loss: 'bce',
    seed,
  }
}

function makeHyper(params: Params, trainSize: number): Hyper {
  return {
    learningRate: params.learningRate,
    optimizer: params.optimizer,
    batchSize: params.batchSize === 0 ? trainSize : params.batchSize,
    l2: params.l2,
    speed: params.speed,
  }
}

export function App() {
  const [params, setParams] = useState<Params>(defaultParams)
  const [seed, setSeed] = useState(1)

  const { dataset, noise, points, trainShare, layers, neurons, activation } = params
  const setup = useMemo(
    () => makeSetup(params, seed),
    // Only things that change the data or the architecture start a new run.
    [dataset, noise, points, trainShare, layers, neurons, activation, seed],
  )
  const hyper = makeHyper(params, setup.train.length)

  // One shared experiment watched by sections 1–4.
  const session = useMemo(() => new TrainingSession(setup, hyper), [])
  useEffect(() => {
    if (session.setup !== setup) session.load(setup)
  }, [session, setup])
  useEffect(() => {
    session.setHyper(hyper)
  }, [session, hyper.learningRate, hyper.optimizer, hyper.batchSize, hyper.l2, hyper.speed])
  useEffect(() => () => session.dispose(), [session])

  return (
    <div className="page">
      <header className="page-header">
        <h1>Neural networks, visually</h1>
        <p>
          Every model on this page is trained live in your browser by a small hand-written library: a forward pass,
          backpropagation and gradient descent, with no ML framework underneath. Press <b>Train</b> and watch it learn,
          then push it until it <b>overfits</b>.
        </p>
      </header>

      <div className="layout">
        <Controls params={params} session={session} onChange={setParams} onReset={() => setSeed((s) => s + 1)} />
        <main className="sections">
          <DecisionBoundary session={session} />
          <NetworkDiagram session={session} />
          <LossCurves session={session} />
          <LayerGradients session={session} />
          <LossSurface />
          <LearningRate />
          <LineFit />
          <Overfitting />
          <Capacity />
        </main>
      </div>
    </div>
  )
}

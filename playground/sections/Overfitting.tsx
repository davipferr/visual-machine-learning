import { makeRegression, regressionTruth } from 'mini-neural-net'
import { useEffect, useMemo, useState } from 'react'
import { clip, grid, plotArea, polyline, short, useCanvas, useRerender, type Frame } from '../canvas'
import { Slider } from '../Controls'
import { TrainingSession, type SessionSetup } from '../session'
import { SectionHead } from './SectionHead'

const TEST_POINTS = 200
const PATIENCE = 300
const l2Options = [0, 0.0001, 0.0003, 0.001, 0.003, 0.01]
const log = (v: number) => Math.log10(Math.max(1e-6, v))

interface Options {
  points: number
  noise: number
  neurons: number
  seed: number
}

function makeSetup({ points, noise, neurons, seed }: Options): SessionSetup {
  const data = makeRegression(points + TEST_POINTS, noise, seed)
  return {
    train: data.slice(0, points),
    test: data.slice(points),
    sizes: [1, neurons, neurons, 1],
    hidden: 'tanh',
    output: 'linear',
    loss: 'mse',
    seed,
  }
}

export function Overfitting() {
  const [options, setOptions] = useState<Options>({ points: 12, noise: 0.25, neurons: 32, seed: 3 })
  const [l2, setL2] = useState(0)
  const [earlyStop, setEarlyStop] = useState(false)
  const setup = useMemo(() => makeSetup(options), [options])
  const session = useMemo(
    () =>
      new TrainingSession(setup, { learningRate: 0.005, optimizer: 'adam', batchSize: setup.train.length, l2: 0, speed: 25 }),
    [],
  )
  session.maxEpochs = 8000
  useEffect(() => {
    if (session.setup !== setup) session.load(setup)
    session.setHyper({ ...session.hyper, batchSize: setup.train.length, l2 })
  }, [session, setup, l2])
  useEffect(() => () => session.dispose(), [session])

  // Early stopping: once the test loss hasn't improved for a while, go back to its best.
  useEffect(() => {
    if (!earlyStop) return
    return session.subscribe(() => {
      const { best, latest } = session
      if (session.running && best && latest.epoch - best.epoch > PATIENCE) session.restoreBest()
    })
  }, [session, earlyStop])

  useRerender(session)

  const curve = useCanvas((frame: Frame) => {
    const { ctx, color } = frame
    const xRange: [number, number] = [-1.05, 1.05]
    const yRange: [number, number] = [-1.6, 1.6]
    const plot = plotArea(frame, xRange, yRange, { left: 36, right: 10, top: 10, bottom: 24 })
    grid(frame, plot, xRange, yRange)
    ctx.save()
    clip(ctx, plot)

    for (const s of session.setup.test) {
      ctx.strokeStyle = color('--test')
      ctx.globalAlpha = 0.5
      ctx.lineWidth = 1.2
      ctx.beginPath()
      ctx.arc(plot.x(s.input[0]), plot.y(s.target[0]), 2.5, 0, Math.PI * 2)
      ctx.stroke()
    }
    ctx.globalAlpha = 1

    const truth: [number, number][] = []
    const model: [number, number][] = []
    for (let x = -1.05; x <= 1.05; x += 0.005) {
      truth.push([plot.x(x), plot.y(regressionTruth(x))])
      model.push([plot.x(x), plot.y(session.network.predict([x])[0])])
    }
    ctx.setLineDash([6, 5])
    ctx.strokeStyle = color('--muted')
    ctx.lineWidth = 1.5
    polyline(ctx, truth)
    ctx.setLineDash([])
    ctx.strokeStyle = color('--accent')
    ctx.lineWidth = 2.5
    polyline(ctx, model)

    for (const s of session.setup.train) {
      ctx.fillStyle = color('--train')
      ctx.beginPath()
      ctx.arc(plot.x(s.input[0]), plot.y(s.target[0]), 5, 0, Math.PI * 2)
      ctx.fill()
      ctx.strokeStyle = color('--surface')
      ctx.lineWidth = 1.5
      ctx.stroke()
    }
    ctx.restore()
  })

  const losses = useCanvas((frame: Frame) => {
    const { ctx, color } = frame
    const { history, best } = session
    const xRange: [number, number] = [0, Math.max(100, session.latest.epoch)]
    const yRange: [number, number] = [-4, 0.5]
    const plot = plotArea(frame, xRange, yRange, { left: 44, right: 10, top: 10, bottom: 24 })
    grid(frame, plot, xRange, yRange, { x: (v) => `${Math.round(v)}`, y: (v) => short(10 ** v) })
    ctx.save()
    clip(ctx, plot)
    // Noise sets a floor: even the true curve has this much test error.
    const floor = (options.noise * options.noise) / 2
    ctx.setLineDash([3, 4])
    ctx.strokeStyle = color('--muted')
    ctx.lineWidth = 1
    polyline(ctx, [[plot.left, plot.y(log(floor))], [plot.left + plot.width, plot.y(log(floor))]])
    ctx.setLineDash([])
    ctx.lineWidth = 2
    ctx.strokeStyle = color('--train')
    polyline(ctx, history.map((p) => [plot.x(p.epoch), plot.y(log(p.trainLoss))] as [number, number]))
    ctx.strokeStyle = color('--test')
    polyline(ctx, history.map((p) => [plot.x(p.epoch), plot.y(log(p.testLoss))] as [number, number]))
    if (best) {
      ctx.fillStyle = color('--test')
      ctx.beginPath()
      ctx.arc(plot.x(best.epoch), plot.y(log(best.testLoss)), 4.5, 0, Math.PI * 2)
      ctx.fill()
    }
    ctx.restore()
    ctx.fillStyle = color('--muted')
    ctx.textAlign = 'left'
    ctx.fillText('noise floor', plot.left + 4, plot.y(log(floor)) - 4)
  })

  const set = (patch: Partial<Options>) => setOptions({ ...options, ...patch })
  const { latest, best } = session
  const memorised = latest.trainLoss < ((options.noise * options.noise) / 2) * 0.3
  return (
    <section className="card">
      <SectionHead step={8} title="Overfitting, up close">
        A network with {session.network.parameterCount} parameters learns a curve from only{' '}
        {options.points} noisy points (green). Dashed is the truth it should find; rings are fresh test points from the same
        curve. Let it train long enough and it bends itself through every green point, noise and all. That's overfitting:
        the training loss goes on falling while the test loss climbs.
      </SectionHead>
      <div className="two-up">
        <canvas ref={curve.ref} className="canvas" aria-label="Noisy points, the true curve, and the network's fit" />
        <canvas ref={losses.ref} className="canvas" aria-label="Training and test loss over epochs" />
      </div>
      <div className="legend">
        <span><i className="dot" style={{ background: 'var(--train)' }} /> training points</span>
        <span><i className="dot ring" style={{ borderColor: 'var(--test)' }} /> test points</span>
        <span><i className="swatch" style={{ background: 'var(--accent)' }} /> network</span>
        <span><i className="swatch dashed" /> true curve</span>
        <span><i className="swatch" style={{ background: 'var(--train)' }} /> train loss</span>
        <span><i className="swatch" style={{ background: 'var(--test)' }} /> test loss</span>
      </div>
      <div className="controls-grid">
        <Slider label="Training points" value={options.points} min={4} max={80} step={1} onChange={(points) => set({ points })} />
        <Slider label="Noise" value={options.noise} min={0} max={0.5} step={0.01} onChange={(noise) => set({ noise })} />
        <Slider label="Neurons per layer (2 layers)" value={options.neurons} min={2} max={64} step={1} onChange={(neurons) => set({ neurons })} />
        <Slider
          label={<>L2 regularization <var>λ</var></>}
          value={l2Options.indexOf(l2)}
          min={0}
          max={l2Options.length - 1}
          step={1}
          display={l2 === 0 ? 'off' : `${l2}`}
          onChange={(i) => setL2(l2Options[i])}
        />
      </div>
      <p className="callout">
        Epoch {latest.epoch}: train loss <b className="train-text">{short(latest.trainLoss)}</b>, test loss{' '}
        <b className="test-text">{short(latest.testLoss)}</b>
        {best && ` (best ${short(best.testLoss)} at epoch ${best.epoch})`}.{' '}
        {memorised
          ? 'The training loss is now well below the noise floor. The only way to do that is to memorise the noise, which is exactly what fails on new points.'
          : latest.epoch === 0
            ? 'Press Train.'
            : 'Three cures to try: more training points, a smaller network, or L2 regularization, which penalises large weights and so favours smooth curves. Or stop early, before the network gets to the noise.'}
      </p>
      <div className="row">
        <button className="primary" onClick={() => session.toggle()}>
          {session.running ? '❚❚ Pause' : '▶ Train'}
        </button>
        <button onClick={() => set({ seed: options.seed + 1 })}>New sample</button>
        <button onClick={() => session.load(setup)}>Reset weights</button>
        <label className="checkbox">
          <input type="checkbox" checked={earlyStop} onChange={(e) => setEarlyStop(e.target.checked)} /> Early stopping
          (patience {PATIENCE} epochs)
        </label>
      </div>
    </section>
  )
}

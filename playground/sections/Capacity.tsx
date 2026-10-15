import { fitPolynomial, makeRegression, meanSquaredError, regressionTruth } from 'mini-neural-net'
import { useMemo, useState } from 'react'
import { clip, grid, plotArea, polyline, short, useCanvas, type Frame } from '../canvas'
import { Slider } from '../Controls'
import { SectionHead } from './SectionHead'

const MAX_DEGREE = 14
const TRAIN = 15
const NOISE = 0.2
const OTHER_SAMPLES = 8

const sample = (seed: number) => {
  const data = makeRegression(TRAIN, NOISE, seed)
  return { xs: data.map((s) => s.input[0]), ys: data.map((s) => s.target[0]) }
}

/**
 * Polynomials make capacity a single dial. Degree 1 is a line, degree 14 can
 * pass through all 15 training points. Least squares finds the best fit
 * directly, so there is no training noise in the way.
 */
export function Capacity() {
  const [degree, setDegree] = useState(3)
  const [seed, setSeed] = useState(1)
  const [showOthers, setShowOthers] = useState(true)

  const test = useMemo(() => makeRegression(400, NOISE, 999), [])
  const testXs = useMemo(() => test.map((s) => s.input[0]), [test])
  const testYs = useMemo(() => test.map((s) => s.target[0]), [test])
  const train = useMemo(() => sample(seed), [seed])
  const others = useMemo(() => Array.from({ length: OTHER_SAMPLES }, (_, i) => sample(seed * 100 + i + 1)), [seed])

  // Error for every degree, averaged over several training samples so the curve is smooth.
  const errors = useMemo(
    () =>
      Array.from({ length: MAX_DEGREE + 1 }, (_, d) => {
        let trainErr = 0
        let testErr = 0
        for (const s of [train, ...others]) {
          const fit = fitPolynomial(s.xs, s.ys, d)
          trainErr += meanSquaredError(fit.predict, s.xs, s.ys)
          testErr += meanSquaredError(fit.predict, testXs, testYs)
        }
        const n = others.length + 1
        return { degree: d, train: trainErr / n, test: testErr / n }
      }),
    [train, others, testXs, testYs],
  )
  const bestDegree = errors.reduce((a, e) => (e.test < a.test ? e : a)).degree
  const fit = useMemo(() => fitPolynomial(train.xs, train.ys, degree), [train, degree])

  const fitCanvas = useCanvas((frame: Frame) => {
    const { ctx, color } = frame
    const xRange: [number, number] = [-1, 1]
    const yRange: [number, number] = [-1.6, 1.6]
    const plot = plotArea(frame, xRange, yRange, { left: 36, right: 10, top: 10, bottom: 24 })
    grid(frame, plot, xRange, yRange)
    ctx.save()
    clip(ctx, plot)
    const trace = (predict: (x: number) => number): [number, number][] => {
      const points: [number, number][] = []
      for (let x = -1; x <= 1.0001; x += 0.005) points.push([plot.x(x), plot.y(predict(x))])
      return points
    }
    if (showOthers) {
      ctx.strokeStyle = color('--accent')
      ctx.globalAlpha = 0.22
      ctx.lineWidth = 1.5
      for (const s of others) polyline(ctx, trace(fitPolynomial(s.xs, s.ys, degree).predict))
      ctx.globalAlpha = 1
    }
    ctx.setLineDash([6, 5])
    ctx.strokeStyle = color('--muted')
    ctx.lineWidth = 1.5
    polyline(ctx, trace(regressionTruth))
    ctx.setLineDash([])
    ctx.strokeStyle = color('--accent')
    ctx.lineWidth = 2.5
    polyline(ctx, trace(fit.predict))
    for (let i = 0; i < train.xs.length; i++) {
      ctx.fillStyle = color('--train')
      ctx.beginPath()
      ctx.arc(plot.x(train.xs[i]), plot.y(train.ys[i]), 5, 0, Math.PI * 2)
      ctx.fill()
      ctx.strokeStyle = color('--surface')
      ctx.lineWidth = 1.5
      ctx.stroke()
    }
    ctx.restore()
  })

  const curveCanvas = useCanvas((frame: Frame) => {
    const { ctx, color } = frame
    const log = (v: number) => Math.log10(Math.max(1e-8, v))
    const xRange: [number, number] = [0, MAX_DEGREE]
    const yRange: [number, number] = [-4, 1]
    const plot = plotArea(frame, xRange, yRange, { left: 44, right: 10, top: 10, bottom: 24 })
    grid(frame, plot, xRange, yRange, { x: (v) => `${v}`, y: (v) => short(10 ** v) })

    // Underfitting on the left, overfitting on the right.
    ctx.fillStyle = color('--muted')
    ctx.textAlign = 'left'
    ctx.fillText('← underfit', plot.left + 6, plot.top + 14)
    ctx.textAlign = 'right'
    ctx.fillText('overfit →', plot.left + plot.width - 6, plot.top + 14)

    ctx.save()
    clip(ctx, plot)
    ctx.strokeStyle = color('--grid')
    ctx.lineWidth = 18
    polyline(ctx, [[plot.x(degree), plot.top], [plot.x(degree), plot.top + plot.height]])
    for (const [key, token] of [['train', '--train'], ['test', '--test']] as const) {
      ctx.strokeStyle = color(token)
      ctx.fillStyle = color(token)
      ctx.lineWidth = 2
      polyline(ctx, errors.map((e) => [plot.x(e.degree), plot.y(log(e[key]))] as [number, number]))
      for (const e of errors) {
        ctx.beginPath()
        ctx.arc(plot.x(e.degree), plot.y(log(e[key])), e.degree === degree ? 5 : 2.5, 0, Math.PI * 2)
        ctx.fill()
      }
    }
    ctx.restore()
  })

  const current = errors[degree]
  return (
    <section className="card">
      <SectionHead step={9} title="Model capacity: the U-shaped curve">
        How flexible should a model be? Here capacity is a single dial: the degree of a polynomial fitted to {TRAIN} noisy
        points. Too low and it can't follow the curve (<b>underfitting</b>, high bias). Too high and it chases the noise
        (<b>overfitting</b>, high variance). The faint lines are fits to {OTHER_SAMPLES} other samples of the same curve: watch how much
        they disagree at high degrees.
      </SectionHead>
      <div className="two-up">
        <canvas ref={fitCanvas.ref} className="canvas" aria-label="Polynomial fit to noisy points" />
        <canvas ref={curveCanvas.ref} className="canvas" aria-label="Training and test error by polynomial degree" />
      </div>
      <div className="legend">
        <span><i className="swatch" style={{ background: 'var(--accent)' }} /> fit (faint: other samples)</span>
        <span><i className="swatch dashed" /> true curve</span>
        <span><i className="swatch" style={{ background: 'var(--train)' }} /> training error</span>
        <span><i className="swatch" style={{ background: 'var(--test)' }} /> test error</span>
        <span className="muted">mean squared error, log scale</span>
      </div>
      <Slider label="Polynomial degree" value={degree} min={0} max={MAX_DEGREE} step={1} onChange={setDegree} />
      <p className="callout">
        Degree {degree}: {degree + 1} coefficients. Training error <b className="train-text">{short(current.train)}</b>, test
        error <b className="test-text">{short(current.test)}</b>.{' '}
        {degree < bestDegree - 1
          ? 'Too stiff: it misses the shape, so it is wrong on training and test points alike.'
          : degree > bestDegree + 2
            ? 'Training error keeps dropping, but test error has shot up. Each new sample gives a wildly different curve: the fit describes the noise in this sample, not the truth.'
            : `Near the sweet spot: degree ${bestDegree} has the lowest test error.`}{' '}
        A neural network's size, training time and regularization all turn this same dial.
      </p>
      <div className="row">
        <button onClick={() => setSeed(seed + 1)}>New sample</button>
        <label className="checkbox">
          <input type="checkbox" checked={showOthers} onChange={(e) => setShowOthers(e.target.checked)} /> show fits on other
          samples
        </label>
      </div>
    </section>
  )
}

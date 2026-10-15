import { clip, grid, plotArea, polyline, short, useCanvas, useRerender, type Frame } from '../canvas'
import type { TrainingSession } from '../session'
import { SectionHead } from './SectionHead'

const log = (v: number) => Math.log10(Math.max(1e-6, v))

export function LossCurves({ session }: { session: TrainingSession }) {
  const draw = (frame: Frame) => {
    const { ctx, color } = frame
    const { history, best } = session
    const lastEpoch = history[history.length - 1].epoch
    const xRange: [number, number] = [0, Math.max(20, lastEpoch)]

    let lo = Infinity
    let hi = -Infinity
    for (const p of history) {
      for (const v of [p.trainLoss, p.testLoss]) {
        if (!Number.isFinite(v)) continue
        lo = Math.min(lo, log(v))
        hi = Math.max(hi, log(v))
      }
    }
    if (!Number.isFinite(lo)) [lo, hi] = [-2, 0]
    if (hi - lo < 0.5) [lo, hi] = [lo - 0.25, hi + 0.25]
    const yRange: [number, number] = [lo - 0.05, Math.min(hi + 0.05, 3)]
    const plot = plotArea(frame, xRange, yRange, { left: 52, right: 14, top: 12, bottom: 26 })
    grid(frame, plot, xRange, yRange, { x: (v) => `${Math.round(v)}`, y: (v) => short(10 ** v) })

    ctx.save()
    clip(ctx, plot)

    // Shade where the test loss has been climbing since its best point.
    if (best && lastEpoch - best.epoch > 10 && session.latest.testSmooth > best.testLoss * 1.05) {
      ctx.fillStyle = color('--overfit-zone')
      ctx.fillRect(plot.x(best.epoch), plot.top, plot.x(lastEpoch) - plot.x(best.epoch), plot.height)
      ctx.fillStyle = color('--test')
      ctx.textAlign = 'left'
      ctx.fillText('overfitting', plot.x(best.epoch) + 6, plot.top + 14)
    }

    // Raw values faint, the smoothed trend on top.
    for (const [raw, smooth, token] of [
      ['trainLoss', 'trainSmooth', '--train'],
      ['testLoss', 'testSmooth', '--test'],
    ] as const) {
      ctx.strokeStyle = color(token)
      ctx.lineWidth = 1
      ctx.globalAlpha = 0.3
      polyline(ctx, history.map((p) => [plot.x(p.epoch), plot.y(log(p[raw]))] as [number, number]))
      ctx.globalAlpha = 1
      ctx.lineWidth = 2.2
      polyline(ctx, history.map((p) => [plot.x(p.epoch), plot.y(log(p[smooth]))] as [number, number]))
    }

    if (best) {
      const x = plot.x(best.epoch)
      ctx.strokeStyle = color('--muted')
      ctx.lineWidth = 1
      ctx.setLineDash([4, 4])
      polyline(ctx, [[x, plot.top], [x, plot.top + plot.height]])
      ctx.setLineDash([])
      ctx.fillStyle = color('--test')
      ctx.beginPath()
      ctx.arc(x, plot.y(log(best.testLoss)), 5, 0, Math.PI * 2)
      ctx.fill()
    }
    ctx.restore()
  }

  const { ref } = useCanvas(draw)
  useRerender(session)

  const { best, latest, history } = session
  const sinceBest = best ? latest.epoch - best.epoch : 0
  let message: string
  if (session.diverged) {
    message = 'The loss exploded to infinity. The learning rate is too big: each step overshoots by more than the last. Lower it and press Reset.'
  } else if (history.length < 2) {
    message = 'Press Train. Both losses start near ln 2 ≈ 0.69: a fresh network guesses 50/50 for everything.'
  } else if (best && sinceBest > 30 && latest.testSmooth > best.testLoss * 1.1) {
    message = `Overfitting: the training loss keeps falling, but the test loss bottomed out at epoch ${best.epoch} and has risen ${Math.round((latest.testSmooth / best.testLoss - 1) * 100)}% since. The network is memorising the training points, noise included, instead of the shape behind them.`
  } else if (latest.trainSmooth > 0.65 && latest.epoch > 15) {
    message = 'Not learning: the loss is no better than guessing 50/50 (≈ 0.69), or worse. If it jumps around, the learning rate is too big: each step overshoots, and with ReLU a big step can push neurons into the flat zero region where they never recover ("dead" neurons). If it is flat, the steps are too small or the gradient has vanished.'
  } else if (latest.trainSmooth > 0.4 && latest.epoch > 100) {
    message = 'Underfitting: even the training loss is stuck high. The model is too simple for this shape, or the learning rate is too small to get anywhere.'
  } else {
    message = 'Still learning: both curves are falling together, so what the network learns from the training set carries over to points it has never seen.'
  }

  return (
    <section className="card">
      <SectionHead step={3} title="Training loss vs test loss">
        The <b>loss</b> measures how wrong the network is (cross-entropy here: −ln of the probability it gave the right
        answer). Training minimises the loss on the training set only. The test set is the honest check. When the two
        curves split apart, the network has started to <b>overfit</b>.
      </SectionHead>
      <canvas ref={ref} className="canvas" aria-label="Training and test loss over epochs, log scale" />
      <div className="legend">
        <span><i className="swatch" style={{ background: 'var(--train)' }} /> training loss</span>
        <span><i className="swatch" style={{ background: 'var(--test)' }} /> test loss</span>
        <span><i className="dot" style={{ background: 'var(--test)' }} /> best test loss {best ? `(epoch ${best.epoch}, ${short(best.testLoss)})` : ''}</span>
        <span className="muted">by epoch · log scale · faint = raw, bold = smoothed</span>
      </div>
      <p className="callout">{message}</p>
      <div className="row">
        <button onClick={() => session.restoreBest()} disabled={!best || sinceBest === 0}>
          Rewind to epoch {best?.epoch ?? 0} (early stopping)
        </button>
        <span className="hint">
          Early stopping keeps the weights from the epoch where the test loss was lowest. In practice a separate
          validation set plays this role, so the test set stays untouched.
        </span>
      </div>
    </section>
  )
}

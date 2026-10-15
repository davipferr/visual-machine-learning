import { clip, grid, plotArea, polyline, short, useCanvas, useRerender, type Frame } from '../canvas'
import type { TrainingSession } from '../session'
import { SectionHead } from './SectionHead'

const palette = ['--layer-1', '--layer-2', '--layer-3', '--layer-4', '--layer-5', '--layer-6', '--layer-7']
const log = (v: number) => Math.log10(Math.max(1e-12, v))

export function LayerGradients({ session }: { session: TrainingSession }) {
  const draw = (frame: Frame) => {
    const { ctx, width, color } = frame
    const history = session.history.filter((p) => p.gradients.length > 0)
    const layers = session.network.layers.length
    const barSpace = 30 + layers * 18
    const lastEpoch = session.latest.epoch
    const xRange: [number, number] = [0, Math.max(20, lastEpoch)]

    let lo = Infinity
    let hi = -Infinity
    for (const p of history) for (const g of p.gradients) if (g > 0) (lo = Math.min(lo, log(g))), (hi = Math.max(hi, log(g)))
    if (!Number.isFinite(lo)) [lo, hi] = [-4, 0]
    const yRange: [number, number] = [Math.floor(lo), Math.max(Math.ceil(hi), Math.floor(lo) + 1)]

    const plot = plotArea({ ...frame, width: width - barSpace }, xRange, yRange, { left: 52, right: 12, top: 12, bottom: 26 })
    grid(frame, plot, xRange, yRange, { x: (v) => `${Math.round(v)}`, y: (v) => short(10 ** v) })

    ctx.save()
    clip(ctx, plot)
    ctx.lineWidth = 1.6
    for (let l = 0; l < layers; l++) {
      ctx.strokeStyle = color(palette[l % palette.length])
      // Gradients are noisy step to step; a short running average reads better.
      let smooth = 0
      const points: [number, number][] = []
      history.forEach((p, i) => {
        const v = log(p.gradients[l] ?? 1e-12)
        smooth = i === 0 ? v : smooth * 0.8 + v * 0.2
        points.push([plot.x(p.epoch), plot.y(smooth)])
      })
      polyline(ctx, points)
    }
    ctx.restore()

    // Current values as bars, one per layer.
    const latest = history[history.length - 1]
    const barW = 14
    const x0 = plot.left + plot.width + 24
    ctx.textAlign = 'center'
    for (let l = 0; l < layers; l++) {
      const x = x0 + l * 18
      const v = latest ? log(latest.gradients[l]) : yRange[0]
      const y = Math.max(plot.top, Math.min(plot.y(yRange[0]), plot.y(v)))
      ctx.fillStyle = color(palette[l % palette.length])
      ctx.fillRect(x, y, barW, plot.y(yRange[0]) - y)
      ctx.fillStyle = color('--muted')
      ctx.fillText(`${l + 1}`, x + barW / 2, plot.top + plot.height + 16)
    }
  }

  const { ref } = useCanvas(draw)
  useRerender(session)

  const { network, latest } = session
  const g = latest.gradients
  const ratio = g.length > 1 ? g[0] / g[g.length - 1] : Number.NaN
  const activation = network.layers[0].activation

  return (
    <section className="card">
      <SectionHead step={4} title="Backpropagation: how much each layer is pushed">
        To learn, every weight needs ∂<var>L</var>/∂<var>w</var>: how the loss would change if that weight moved. Backpropagation
        gets them all in one backward sweep using the chain rule: the error at the output is passed back layer by layer,
        multiplied by each layer's weights and by the slope of its activation <var>f′</var>(<var>z</var>). This chart shows
        the typical size of the gradient in each layer of weights (1 = closest to the input).
      </SectionHead>
      <canvas ref={ref} className="canvas canvas-short" aria-label="Gradient size per layer over epochs" />
      <div className="legend">
        {network.layers.map((_, l) => (
          <span key={l}>
            <i className="swatch" style={{ background: `var(${palette[l % palette.length]})` }} />
            {l === network.layers.length - 1 ? 'output layer' : `layer ${l + 1}`}
          </span>
        ))}
        <span className="muted">RMS of ∂L/∂W, log scale</span>
      </div>
      <p className="callout">
        {g.length < 2
          ? 'Train the network (with at least one hidden layer) to see gradients flow.'
          : ratio < 0.05
            ? `Vanishing gradients: the first layer gets a push ${short(1 / ratio)}× weaker than the output layer. Each ${activation} layer multiplies the signal by f′(z)${activation === 'sigmoid' ? ' ≤ 0.25' : ''} on the way back, so deep stacks starve their early layers. ReLU (f′ = 1 when active) and good initialisation help.`
            : ratio > 20
              ? `Exploding gradients: the first layer is pushed ${short(ratio)}× harder than the output layer. Lower the learning rate.`
              : `Healthy: gradients reach the first layer at a similar size (first ÷ last layer = ${short(ratio)}). Try the "Vanishing" preset to see them die out.`}
      </p>
    </section>
  )
}

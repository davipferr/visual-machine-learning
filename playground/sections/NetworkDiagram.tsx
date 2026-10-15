import { diverging, heatmap, parseColor, useCanvas, useRerender, type Frame } from '../canvas'
import { DOMAIN, probabilityScale } from '../classes'
import type { TrainingSession } from '../session'
import { SectionHead } from './SectionHead'

const CELLS = 18

export function NetworkDiagram({ session }: { session: TrainingSession }) {
  const draw = (frame: Frame) => {
    const { ctx, width, height, color } = frame
    const { network } = session
    const sizes = network.sizes
    const columns = sizes.length

    // Run the whole grid through the network once and keep every neuron's activation.
    const [d0, d1] = DOMAIN
    const span = d1 - d0
    const activations: Float64Array[][] = [] // [cell][layer][neuron]
    for (let r = 0; r < CELLS; r++) {
      for (let c = 0; c < CELLS; c++) {
        activations.push(network.trace([d0 + ((c + 0.5) / CELLS) * span, d1 - ((r + 0.5) / CELLS) * span]).a)
      }
    }

    const tallest = Math.max(...sizes)
    const top = 34
    const bottom = 14
    const box = Math.max(10, Math.min(46, (height - top - bottom) / tallest - 8))
    const colGap = (width - 40 - box) / Math.max(1, columns - 1)
    const pos = (l: number, j: number): [number, number] => {
      const n = sizes[l]
      const gap = Math.min(box + 18, (height - top - bottom - box) / Math.max(1, n - 1))
      const total = (n - 1) * gap
      return [20 + l * colGap, top + (height - top - bottom - box) / 2 - total / 2 + j * gap]
    }

    // Edges: thickness is |weight|, colour is its sign.
    let maxW = 1e-9
    for (const layer of network.layers) for (const w of layer.weights) maxW = Math.max(maxW, Math.abs(w))
    network.layers.forEach((layer, l) => {
      for (let j = 0; j < layer.outSize; j++) {
        for (let i = 0; i < layer.inSize; i++) {
          const w = layer.weights[j * layer.inSize + i]
          const [x1, y1] = pos(l, i)
          const [x2, y2] = pos(l + 1, j)
          ctx.strokeStyle = color(w >= 0 ? '--class1' : '--class0')
          ctx.globalAlpha = 0.25 + 0.6 * Math.min(1, Math.abs(w) / maxW)
          ctx.lineWidth = 0.4 + 4 * Math.min(1, Math.abs(w) / maxW)
          ctx.beginPath()
          ctx.moveTo(x1 + box, y1 + box / 2)
          ctx.bezierCurveTo(x1 + box + colGap * 0.4, y1 + box / 2, x2 - colGap * 0.4, y2 + box / 2, x2, y2 + box / 2)
          ctx.stroke()
        }
      }
    })
    ctx.globalAlpha = 1

    // Neurons: each one drawn as what it responds to across the input plane.
    const neutral = parseColor(color('--surface'))
    const signed = diverging(neutral, parseColor(color('--class0')), parseColor(color('--class1')), 0.9)
    const probability = probabilityScale(frame, 0.9)
    for (let l = 0; l < columns; l++) {
      const activation = l === 0 ? 'linear' : network.layers[l - 1].activation
      for (let j = 0; j < sizes[l]; j++) {
        let peak = 1e-9
        for (const a of activations) peak = Math.max(peak, Math.abs(a[l][j]))
        const [x, y] = pos(l, j)
        heatmap(ctx, { left: x, top: y, width: box, height: box }, CELLS, CELLS, (c, r) => {
          const a = activations[r * CELLS + c][l][j]
          if (l === columns - 1) return probability(a)
          if (activation === 'sigmoid') return signed(2 * a - 1)
          if (activation === 'tanh') return signed(a)
          return signed(a / peak) // ReLU and linear: unbounded, so normalise per neuron
        })
        ctx.strokeStyle = color('--border')
        ctx.lineWidth = 1
        ctx.strokeRect(x + 0.5, y + 0.5, box - 1, box - 1)
      }
      // Column caption
      const [cx] = pos(l, 0)
      ctx.fillStyle = color('--muted')
      ctx.textAlign = 'center'
      const caption =
        l === 0 ? 'input' : l === columns - 1 ? 'output · sigmoid' : `hidden ${l} · ${network.layers[l - 1].activation}`
      ctx.fillText(caption, Math.min(width - 50, Math.max(40, cx + box / 2)), 16)
    }
    ctx.textAlign = 'left'
    if (sizes[0] === 2) {
      const [x0, y0] = pos(0, 0)
      const [, y1] = pos(0, 1)
      ctx.fillStyle = color('--text')
      ctx.fillText('x₁', x0 + box + 4, y0 + 10)
      ctx.fillText('x₂', x0 + box + 4, y1 + 10)
    }
  }

  const { ref } = useCanvas(draw)
  useRerender(session)

  const { network } = session
  return (
    <section className="card">
      <SectionHead step={2} title="Inside the network">
        Each square is one neuron, drawn as a map of how strongly it fires across the input plane. The first hidden layer
        can only draw straight-edged ramps (a weighted sum of <var>x</var>₁ and <var>x</var>₂, squashed). Every later
        layer mixes the maps before it, which is how curves and islands appear. Lines are weights: thickness is size,
        blue is positive, orange negative.
      </SectionHead>
      <canvas
        ref={ref}
        className="canvas canvas-network"
        aria-label="Diagram of the network with each neuron's activation map"
      />
      <p className="callout">
        Each neuron computes <code>a = f(w·x + b)</code>. With {network.layers.length - 1} hidden layer
        {network.layers.length === 2 ? '' : 's'} this network has <b>{network.parameterCount}</b> numbers to tune, and
        training nudges every one of them each step.
        {network.layers.length === 1 && ' With no hidden layer it is just a weighted sum: the boundary must be a straight line.'}
        {network.layers.length > 1 && network.layers[0].activation === 'linear' &&
          ' Linear activations collapse: stacking linear layers is still one linear function.'}
      </p>
    </section>
  )
}

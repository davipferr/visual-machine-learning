import { Network, createRng, gaussian, type Sample } from 'mini-neural-net'
import { useEffect, useMemo, useRef, useState } from 'react'
import { fmt, grid, heatmap, mix, parseColor, plotArea, polyline, useCanvas, type Frame } from '../canvas'
import { LogSlider } from '../Controls'
import { onFrame } from '../loop'
import { SectionHead } from './SectionHead'

const W_RANGE: [number, number] = [-2, 3]
const B_RANGE: [number, number] = [-2, 2.5]
const START: [number, number] = [-1.5, -1.6]

function makeData(): Sample[] {
  const rng = createRng(4)
  return Array.from({ length: 14 }, () => {
    const x = rng() * 2 - 1
    return { input: [x], target: [0.9 * x + 0.5 + gaussian(rng) * 0.18] }
  })
}

/** Mean ½(w·x + b − y)²: the same loss the network reports, written out for speed. */
function lossAt(data: Sample[], w: number, b: number) {
  let sum = 0
  for (const s of data) sum += 0.5 * (w * s.input[0] + b - s.target[0]) ** 2
  return sum / data.length
}

/**
 * The smallest possible network: one neuron, no activation. ŷ = w·x + b.
 * Fitting it with gradient descent is linear regression.
 */
export function LineFit() {
  const data = useMemo(makeData, [])
  const net = useMemo(() => new Network({ sizes: [1, 1], output: 'linear' }), [])
  const [path, setPath] = useState<[number, number][]>([START])
  const [eta, setEta] = useState(0.5)
  const [running, setRunning] = useState(false)
  const [w, b] = path[path.length - 1]

  // Gradients come from the library's backprop, on the one-neuron network.
  net.layers[0].weights[0] = w
  net.layers[0].biases[0] = b
  const grads = net.gradients(data, 'mse')
  const gw = grads.weights[0][0]
  const gb = grads.biases[0][0]
  const loss = lossAt(data, w, b)

  const stepOnce = () =>
    setPath((p) => {
      const [pw, pb] = p[p.length - 1]
      net.layers[0].weights[0] = pw
      net.layers[0].biases[0] = pb
      const g = net.gradients(data, 'mse')
      const next: [number, number] = [pw - eta * g.weights[0][0], pb - eta * g.biases[0][0]]
      if (!Number.isFinite(next[0]) || Math.abs(next[0]) > 1e3) return p
      return [...p, next].slice(-400)
    })

  useEffect(() => {
    if (!running) return
    let acc = 0
    return onFrame((dt) => {
      acc += dt
      if (acc > 0.12) {
        acc = 0
        stepOnce()
      }
    })
  }, [running, eta])

  const fit = useCanvas((frame: Frame) => {
    const { ctx, color } = frame
    const plot = plotArea(frame, [-1.1, 1.1], [-1.2, 2], { left: 36, right: 10, top: 10, bottom: 24 })
    grid(frame, plot, [-1.1, 1.1], [-1.2, 2])
    // Residuals: the errors the loss squares and averages.
    ctx.strokeStyle = color('--force')
    ctx.lineWidth = 1
    for (const s of data) {
      polyline(ctx, [
        [plot.x(s.input[0]), plot.y(s.target[0])],
        [plot.x(s.input[0]), plot.y(w * s.input[0] + b)],
      ])
    }
    ctx.strokeStyle = color('--accent')
    ctx.lineWidth = 2.5
    polyline(ctx, [
      [plot.x(-1.1), plot.y(w * -1.1 + b)],
      [plot.x(1.1), plot.y(w * 1.1 + b)],
    ])
    for (const s of data) {
      ctx.fillStyle = color('--text')
      ctx.beginPath()
      ctx.arc(plot.x(s.input[0]), plot.y(s.target[0]), 4, 0, Math.PI * 2)
      ctx.fill()
    }
    ctx.fillStyle = color('--muted')
    ctx.textAlign = 'left'
    ctx.fillText('x →', plot.left + plot.width - 24, plot.y(0) - 6)
  })

  const land = useCanvas((frame: Frame) => {
    const { ctx, color } = frame
    const plot = plotArea(frame, W_RANGE, B_RANGE, { left: 36, right: 10, top: 10, bottom: 24 })
    const low = parseColor(color('--land-low'))
    const high = parseColor(color('--land-high'))
    const line = parseColor(color('--land-line'))
    const cols = 200
    const rows = 170
    const ceiling = lossAt(data, W_RANGE[0], B_RANGE[0])
    heatmap(ctx, plot, cols, rows, (c, r) => {
      const pw = W_RANGE[0] + ((c + 0.5) / cols) * (W_RANGE[1] - W_RANGE[0])
      const pb = B_RANGE[1] - ((r + 0.5) / rows) * (B_RANGE[1] - B_RANGE[0])
      const t = Math.min(1, (lossAt(data, pw, pb) / ceiling) ** 0.4)
      const base = mix(low, high, t)
      return (t * 12) % 1 < 0.1 ? mix(base, line, 0.3) : base
    })
    grid(frame, plot, W_RANGE, B_RANGE)
    ctx.fillStyle = color('--muted')
    ctx.textAlign = 'right'
    ctx.fillText('weight w →', plot.left + plot.width - 4, plot.top + plot.height - 6)
    ctx.textAlign = 'left'
    ctx.fillText('↑ bias b', plot.left + 6, plot.top + 14)

    ctx.strokeStyle = color('--live')
    ctx.fillStyle = color('--live')
    ctx.lineWidth = 2
    polyline(ctx, path.map(([pw, pb]) => [plot.x(pw), plot.y(pb)] as [number, number]))
    for (const [pw, pb] of path) {
      ctx.beginPath()
      ctx.arc(plot.x(pw), plot.y(pb), 2, 0, Math.PI * 2)
      ctx.fill()
    }
    // The negative gradient: the direction of the next step.
    const scale = Math.min(1, 1.2 / Math.hypot(gw, gb)) * eta
    ctx.strokeStyle = color('--force')
    polyline(ctx, [
      [plot.x(w), plot.y(b)],
      [plot.x(w - gw * scale), plot.y(b - gb * scale)],
    ])
    ctx.beginPath()
    ctx.arc(plot.x(w), plot.y(b), 6, 0, Math.PI * 2)
    ctx.fillStyle = color('--live')
    ctx.fill()
  })

  const dragging = useRef(false)
  const setFromPointer = (e: React.PointerEvent<HTMLCanvasElement>) => {
    const rect = e.currentTarget.getBoundingClientRect()
    const plot = plotArea({ width: rect.width, height: rect.height } as Frame, W_RANGE, B_RANGE, {
      left: 36,
      right: 10,
      top: 10,
      bottom: 24,
    })
    const pw = W_RANGE[0] + ((e.clientX - rect.left - plot.left) / plot.width) * (W_RANGE[1] - W_RANGE[0])
    const pb = B_RANGE[1] - ((e.clientY - rect.top - plot.top) / plot.height) * (B_RANGE[1] - B_RANGE[0])
    setPath([[pw, pb]])
  }

  const nw = w - eta * gw
  const nb = b - eta * gb
  return (
    <section className="card">
      <SectionHead step={7} title="One neuron, one step at a time">
        The smallest network there is: one neuron with no activation, <var>ŷ</var> = <var>w</var>·<var>x</var> + <var>b</var>.
        Its loss over the two parameters (<var>w</var>, <var>b</var>) is a bowl like the ones above. Every number below comes
        from the library's backpropagation. Step through it by hand, or <b>drag on the right map</b> to start somewhere else.
      </SectionHead>
      <div className="two-up">
        <canvas ref={fit.ref} className="canvas" aria-label="Data points and the fitted line" />
        <canvas
          ref={land.ref}
          className="canvas draggable"
          aria-label="Loss over weight and bias, with the gradient descent path"
          onPointerDown={(e) => {
            dragging.current = true
            e.currentTarget.setPointerCapture(e.pointerId)
            setFromPointer(e)
          }}
          onPointerMove={(e) => dragging.current && setFromPointer(e)}
          onPointerUp={() => (dragging.current = false)}
        />
      </div>
      <div className="equation">
        <div>
          <span className="muted">loss</span> <var>L</var> = mean ½(<var>ŷ</var> − <var>y</var>)² = <b>{fmt(loss, 4)}</b>
        </div>
        <div>
          ∂<var>L</var>/∂<var>w</var> = mean((<var>ŷ</var> − <var>y</var>)·<var>x</var>) = <b className="eq-term damping">{fmt(gw, 3)}</b>
          <span className="spacer" />∂<var>L</var>/∂<var>b</var> = mean(<var>ŷ</var> − <var>y</var>) ={' '}
          <b className="eq-term damping">{fmt(gb, 3)}</b>
        </div>
        <div className="eq-update">
          <var>w</var> ← {fmt(w, 3)} − {fmt(eta, 2)} × {fmt(gw, 3)} = <b className="net">{fmt(nw, 3)}</b>
          <span className="spacer" />
          <var>b</var> ← {fmt(b, 3)} − {fmt(eta, 2)} × {fmt(gb, 3)} = <b className="net">{fmt(nb, 3)}</b>
        </div>
      </div>
      <div className="legend">
        <span><i className="swatch" style={{ background: 'var(--force)' }} /> errors (left) and −gradient, the next step (right)</span>
        <span><i className="swatch" style={{ background: 'var(--live)' }} /> path taken: {path.length - 1} steps</span>
      </div>
      <LogSlider label={<>Learning rate <var>η</var></>} value={eta} min={0.02} max={4} onChange={setEta} />
      <div className="row">
        <button className="primary" onClick={stepOnce} disabled={running}>
          Take one step
        </button>
        <button onClick={() => setRunning(!running)}>{running ? 'Pause' : 'Run'}</button>
        <button onClick={() => setPath([START])}>Reset</button>
      </div>
    </section>
  )
}

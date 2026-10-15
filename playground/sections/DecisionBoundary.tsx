import { useRef, useState } from 'react'
import { heatmap, short, useCanvas, useRerender, type Frame } from '../canvas'
import { DOMAIN, drawPoints, probabilityScale, squarePlot } from '../classes'
import type { TrainingSession } from '../session'
import { SectionHead } from './SectionHead'

const CELLS = 80

export function DecisionBoundary({ session }: { session: TrainingSession }) {
  const [showTest, setShowTest] = useState(true)
  const [showMistakes, setShowMistakes] = useState(false)
  const hover = useRef<[number, number] | null>(null)
  const readout = useRef<HTMLSpanElement>(null)

  const draw = (frame: Frame) => {
    const { ctx, color } = frame
    const { network, setup } = session
    const plot = squarePlot(frame)
    const scale = probabilityScale(frame)
    const [d0, d1] = DOMAIN
    const span = d1 - d0

    // Ask the network about every point of a grid: that is its whole belief.
    const n = CELLS + 1
    const p = new Float64Array(n * n)
    for (let r = 0; r < n; r++) {
      for (let c = 0; c < n; c++) p[r * n + c] = network.predict([d0 + (c / CELLS) * span, d1 - (r / CELLS) * span])[0]
    }
    heatmap(ctx, plot, n, n, (c, r) => scale(p[r * n + c]))

    // The 0.5 contour, where the network changes its mind: dots between grid points that disagree.
    ctx.fillStyle = color('--text')
    const cell = plot.width / CELLS
    for (let r = 0; r < CELLS; r++) {
      for (let c = 0; c < CELLS; c++) {
        const i = r * n + c
        const v = p[i] >= 0.5
        if (v !== p[i + 1] >= 0.5 || v !== p[i + n] >= 0.5) {
          ctx.fillRect(plot.left + (c + 0.5) * cell - 0.75, plot.top + (r + 0.5) * cell - 0.75, 1.5, 1.5)
        }
      }
    }

    ctx.strokeStyle = color('--border')
    ctx.lineWidth = 1
    ctx.strokeRect(plot.left, plot.top, plot.width, plot.height)

    drawPoints(frame, plot, setup.train, 'train', showMistakes ? network : undefined)
    if (showTest) drawPoints(frame, plot, setup.test, 'test', showMistakes ? network : undefined)

    if (hover.current && readout.current) {
      const [hx, hy] = hover.current
      const p = network.predict([hx, hy])[0]
      readout.current.textContent = `At (${hx.toFixed(2)}, ${hy.toFixed(2)}) the network says P(blue) = ${short(p)}`
    }
  }

  const { ref, redraw } = useCanvas(draw)
  useRerender(session)

  const onMove = (e: React.PointerEvent<HTMLCanvasElement>) => {
    const canvas = e.currentTarget
    const rect = canvas.getBoundingClientRect()
    const side = Math.min(rect.width, rect.height) - 24
    const left = (rect.width - side) / 2
    const top = (rect.height - side) / 2
    const [d0, d1] = DOMAIN
    const x = d0 + ((e.clientX - rect.left - left) / side) * (d1 - d0)
    const y = d0 + (1 - (e.clientY - rect.top - top) / side) * (d1 - d0)
    hover.current = x >= d0 && x <= d1 && y >= d0 && y <= d1 ? [x, y] : null
    if (!hover.current && readout.current) readout.current.textContent = 'Hover the plane to ask the network.'
    redraw()
  }

  const latest = session.latest
  return (
    <section className="card">
      <SectionHead step={1} title="What the network believes">
        The network takes a point (<var>x</var>₁, <var>x</var>₂) and outputs the probability that it is blue. Asking it
        about every pixel paints its beliefs: the dotted line is the <b>decision boundary</b>, where it switches from
        orange to blue. Solid dots are the training set it learns from; rings are test points it never sees.
      </SectionHead>
      <canvas
        ref={ref}
        className="canvas canvas-boundary"
        onPointerMove={onMove}
        onPointerLeave={() => {
          hover.current = null
          if (readout.current) readout.current.textContent = 'Hover the plane to ask the network.'
        }}
        aria-label="Decision boundary of the network over the input plane"
      />
      <div className="legend">
        <span><i className="dot" style={{ background: 'var(--class0)' }} /> class 0</span>
        <span><i className="dot" style={{ background: 'var(--class1)' }} /> class 1</span>
        <span><i className="dot ring" /> test point (held out)</span>
        <label className="checkbox">
          <input type="checkbox" checked={showTest} onChange={(e) => setShowTest(e.target.checked)} /> show test points
        </label>
        <label className="checkbox">
          <input type="checkbox" checked={showMistakes} onChange={(e) => setShowMistakes(e.target.checked)} /> cross out
          mistakes
        </label>
      </div>
      <p className="callout">
        <span ref={readout}>Hover the plane to ask the network.</span>
        <br />
        <span className="muted small">
          Train accuracy {Number.isNaN(latest.trainAccuracy) ? '–' : `${(latest.trainAccuracy * 100).toFixed(1)}%`} ·
          test accuracy {Number.isNaN(latest.testAccuracy) ? '–' : `${(latest.testAccuracy * 100).toFixed(1)}%`} on{' '}
          {session.setup.train.length} / {session.setup.test.length} points
        </span>
      </p>
    </section>
  )
}

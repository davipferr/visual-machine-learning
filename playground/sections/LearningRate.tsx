import { useEffect, useMemo, useRef, useState } from 'react'
import { clip, grid, plotArea, polyline, short, useCanvas, type Frame } from '../canvas'
import { LogSlider } from '../Controls'
import { onFrame } from '../loop'
import { SectionHead } from './SectionHead'

/** L(w) = ½·k·w², so ∇L = k·w and one step multiplies w by (1 − η·k). */
const K = 1
const START = 2.5
const STEPS = 20
const loss = (w: number) => 0.5 * K * w * w

export function LearningRate() {
  const [custom, setCustom] = useState(2.1)
  const cases = useMemo(
    () => [
      { eta: 0.08, label: 'Too small', token: '--under' },
      { eta: 0.7, label: 'Good', token: '--critical' },
      { eta: 1.75, label: 'Too big: zig-zags', token: '--over' },
      { eta: custom, label: 'Your choice', token: '--live' },
    ],
    [custom],
  )
  const paths = useMemo(
    () =>
      cases.map((c) => {
        const w = [START]
        for (let i = 0; i < STEPS; i++) w.push(w[i] - c.eta * K * w[i])
        return w
      }),
    [cases],
  )

  const step = useRef(0)
  const { ref, redraw } = useCanvas((frame: Frame) => {
    const { ctx, width, color } = frame
    const k = Math.floor(step.current)
    const panelH = 150
    const gap = 12
    const narrow = width < 560
    const perRow = narrow ? 2 : 4
    const panelW = (width - gap * (perRow - 1)) / perRow
    const rows = Math.ceil(cases.length / perRow)

    cases.forEach((c, i) => {
      const left = (i % perRow) * (panelW + gap)
      const top = Math.floor(i / perRow) * (panelH + gap)
      const plot = plotArea({ ...frame, width: panelW, height: panelH }, [-3.2, 3.2], [-0.3, 5.5], {
        left: 6,
        right: 6,
        top: 22,
        bottom: 6,
      })
      const px = (w: number) => left + plot.x(w)
      const py = (l: number) => top + plot.y(l)
      ctx.strokeStyle = color('--border')
      ctx.lineWidth = 1
      ctx.strokeRect(left + 0.5, top + 0.5, panelW - 1, panelH - 1)
      ctx.fillStyle = color(c.token)
      ctx.textAlign = 'left'
      ctx.font = '600 12px ui-sans-serif, system-ui, sans-serif'
      ctx.fillText(`${c.label}  η = ${short(c.eta)}`, left + 8, top + 15)
      ctx.font = '11px ui-sans-serif, system-ui, sans-serif'

      ctx.save()
      ctx.beginPath()
      ctx.rect(left, top + 20, panelW, panelH - 20)
      ctx.clip()
      ctx.strokeStyle = color('--target')
      ctx.lineWidth = 1.5
      const curve: [number, number][] = []
      for (let w = -3.2; w <= 3.2; w += 0.05) curve.push([px(w), py(loss(w))])
      polyline(ctx, curve)

      const path = paths[i].slice(0, k + 1)
      ctx.strokeStyle = color(c.token)
      ctx.globalAlpha = 0.6
      ctx.lineWidth = 1.5
      polyline(ctx, path.map((w) => [px(w), py(loss(w))] as [number, number]))
      ctx.globalAlpha = 1
      const w = path[path.length - 1]
      ctx.fillStyle = color(c.token)
      ctx.beginPath()
      ctx.arc(px(Math.max(-3.4, Math.min(3.4, w))), py(Math.min(6, loss(w))), 6, 0, Math.PI * 2)
      ctx.fill()
      ctx.restore()
    })

    // Loss over steps, all four together.
    const top = rows * (panelH + gap) + 4
    const xRange: [number, number] = [0, STEPS]
    const yRange: [number, number] = [-6, 2]
    const plot = plotArea(frame, xRange, yRange, { left: 52, right: 12, top, bottom: 26 })
    grid(frame, plot, xRange, yRange, { x: (v) => `${v}`, y: (v) => short(10 ** v) })
    ctx.save()
    clip(ctx, plot)
    cases.forEach((c, i) => {
      ctx.strokeStyle = color(c.token)
      ctx.lineWidth = 2
      polyline(
        ctx,
        paths[i].slice(0, k + 1).map((w, s) => [plot.x(s), plot.y(Math.log10(Math.max(1e-9, loss(w))))] as [number, number]),
      )
    })
    ctx.restore()
    ctx.fillStyle = color('--muted')
    ctx.textAlign = 'left'
    ctx.fillText('loss (log) by step', plot.left + 6, plot.top + 12)
  })

  useEffect(() => {
    step.current = 0
    let hold = 0
    return onFrame((dt) => {
      if (step.current >= STEPS) {
        hold += dt
        if (hold > 1.5) {
          hold = 0
          step.current = 0
        }
      } else {
        step.current = Math.min(STEPS, step.current + dt * 3)
      }
      redraw()
    })
  }, [paths, redraw])

  const factor = 1 - custom * K
  return (
    <section className="card">
      <SectionHead step={6} title="The learning rate">
        The learning rate <var>η</var> is the size of each step. On this simple bowl, <var>L</var>(<var>w</var>) = ½<var>w</var>²,
        the slope is <var>w</var> itself, so every step multiplies <var>w</var> by (1 − <var>η</var>). Too small and progress is
        slow. A bit too big and it overshoots back and forth. Past <var>η</var> = 2 every overshoot is bigger than the last,
        and training blows up.
      </SectionHead>
      <canvas ref={ref} className="canvas canvas-lr" aria-label="Four learning rates on the same bowl" />
      <LogSlider label={<>Your learning rate <var>η</var></>} value={custom} min={0.01} max={3} onChange={setCustom} />
      <p className="callout">
        With <var>η</var> = {short(custom)} each step multiplies the distance to the bottom by{' '}
        <b>{short(factor)}</b>.{' '}
        {Math.abs(factor) < 1e-9
          ? 'That is exactly 0: one step lands on the minimum. That only works because this bowl is perfectly round and we know its curvature.'
          : Math.abs(factor) >= 1
            ? '|factor| ≥ 1, so the distance never shrinks: training diverges. In a real network the loss shoots up to NaN.'
            : factor < 0
              ? 'It is negative, so it jumps to the other side every step, but it still shrinks, so it converges.'
              : factor > 0.8
                ? 'Close to 1: safe, but painfully slow.'
                : 'A healthy shrink each step.'}{' '}
        A steeper loss (curvature <var>k</var>) moves this limit to <var>η</var> &lt; 2/<var>k</var>. In a network the steepest
        direction sets the limit for all of them, which is why one learning rate rarely suits every weight.
      </p>
    </section>
  )
}

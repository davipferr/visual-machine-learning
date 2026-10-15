import { descend, optimizerLabels, surfaces, type OptimizerName, type Surface, type SurfaceName } from 'mini-neural-net'
import { useEffect, useMemo, useRef, useState } from 'react'
import { heatmap, mix, parseColor, plotArea, polyline, short, useCanvas, type Frame } from '../canvas'
import { LogSlider, Slider } from '../Controls'
import { onFrame } from '../loop'
import { SectionHead } from './SectionHead'

const STEPS = 500
const optimizerTokens: Record<OptimizerName, string> = { sgd: '--sgd', momentum: '--momentum', adam: '--adam' }

/** Lowest value on the visible part of the surface, so the colours can use log(f − min). */
function floorOf(s: Surface) {
  let lo = Infinity
  for (const [x, y] of s.minima) lo = Math.min(lo, s.f(x, y))
  for (let i = 0; i <= 60; i++) {
    for (let j = 0; j <= 60; j++) {
      lo = Math.min(lo, s.f(s.x[0] + ((s.x[1] - s.x[0]) * i) / 60, s.y[0] + ((s.y[1] - s.y[0]) * j) / 60))
    }
  }
  return lo
}

export function LossSurface() {
  const [name, setName] = useState<SurfaceName>('ravine')
  const surface = surfaces[name]
  const [learningRate, setLearningRate] = useState(surface.learningRate)
  const [start, setStart] = useState<[number, number]>(surface.start)
  const [enabled, setEnabled] = useState<Record<OptimizerName, boolean>>({ sgd: true, momentum: true, adam: true })
  const [speed, setSpeed] = useState(30)
  const progress = useRef(0)
  const stopRef = useRef<(() => void) | null>(null)
  const [, setTick] = useState(0)

  const floor = useMemo(() => floorOf(surface), [surface])
  const runs = useMemo(
    () =>
      (Object.keys(enabled) as OptimizerName[])
        .filter((o) => enabled[o])
        .map((o) => ({ name: o, ...descend(surface, { name: o, learningRate }, start, STEPS) })),
    [surface, learningRate, start, enabled],
  )

  const pickSurface = (next: SurfaceName) => {
    setName(next)
    setLearningRate(surfaces[next].learningRate)
    setStart(surfaces[next].start)
  }

  const plotFor = (frame: { width: number; height: number }) =>
    plotArea(frame as Frame, surface.x, surface.y, { left: 8, right: 8, top: 8, bottom: 8 })

  const draw = (frame: Frame) => {
    const { ctx, color } = frame
    const plot = plotFor(frame)
    const low = parseColor(color('--land-low'))
    const high = parseColor(color('--land-high'))
    const line = parseColor(color('--land-line'))
    // Colour by (f − min)^0.3: a power scale keeps both the floor and the walls readable.
    const ceiling = Math.max(
      surface.f(surface.x[0], surface.y[0]),
      surface.f(surface.x[0], surface.y[1]),
      surface.f(surface.x[1], surface.y[0]),
      surface.f(surface.x[1], surface.y[1]),
    )
    const level = (x: number, y: number) => Math.max(0, Math.min(1, ((surface.f(x, y) - floor) / (ceiling - floor)) ** 0.3))
    const cols = 260
    const rows = Math.round((cols * plot.height) / plot.width)
    heatmap(ctx, plot, cols, rows, (c, r) => {
      const x = surface.x[0] + ((c + 0.5) / cols) * (surface.x[1] - surface.x[0])
      const y = surface.y[1] - ((r + 0.5) / rows) * (surface.y[1] - surface.y[0])
      const t = level(x, y)
      const base = mix(low, high, t)
      // Thin bands at equal steps read as contour lines.
      return (t * 10) % 1 < 0.08 ? mix(base, line, 0.3) : base
    })

    for (const [mx, my] of surface.minima) {
      ctx.strokeStyle = color('--text')
      ctx.lineWidth = 1.5
      const x = plot.x(mx)
      const y = plot.y(my)
      ctx.beginPath()
      ctx.moveTo(x - 5, y - 5)
      ctx.lineTo(x + 5, y + 5)
      ctx.moveTo(x + 5, y - 5)
      ctx.lineTo(x - 5, y + 5)
      ctx.stroke()
    }

    const shown = Math.floor(progress.current)
    for (const run of runs) {
      const visible = run.path.slice(0, shown + 1)
      ctx.strokeStyle = color(optimizerTokens[run.name])
      ctx.lineWidth = 2
      polyline(ctx, visible.map(([x, y]) => [plot.x(x), plot.y(y)] as [number, number]))
      ctx.fillStyle = ctx.strokeStyle
      for (const [x, y] of visible) {
        ctx.beginPath()
        ctx.arc(plot.x(x), plot.y(y), 1.8, 0, Math.PI * 2)
        ctx.fill()
      }
      const [hx, hy] = visible[visible.length - 1]
      ctx.beginPath()
      ctx.arc(plot.x(hx), plot.y(hy), 6, 0, Math.PI * 2)
      ctx.fill()
      ctx.strokeStyle = color('--surface')
      ctx.lineWidth = 1.5
      ctx.stroke()
    }

    ctx.fillStyle = color('--text')
    ctx.beginPath()
    ctx.arc(plot.x(start[0]), plot.y(start[1]), 4, 0, Math.PI * 2)
    ctx.fill()
  }

  const { ref } = useCanvas(draw)

  const play = () => {
    stopRef.current?.()
    progress.current = 0
    stopRef.current = onFrame((dt) => {
      progress.current = Math.min(STEPS, progress.current + dt * speedRef.current)
      setTick((t) => t + 1) // re-render: redraws the canvas and the table
      if (progress.current >= STEPS) {
        stopRef.current?.()
        stopRef.current = null
      }
    })
  }
  const speedRef = useRef(speed)
  speedRef.current = speed

  // Restart the race whenever the setup changes.
  useEffect(() => {
    play()
    return () => stopRef.current?.()
  }, [runs])

  const dragging = useRef(false)
  const setFromPointer = (e: React.PointerEvent<HTMLCanvasElement>) => {
    const rect = e.currentTarget.getBoundingClientRect()
    const plot = plotFor({ width: rect.width, height: rect.height })
    const x = surface.x[0] + ((e.clientX - rect.left - plot.left) / plot.width) * (surface.x[1] - surface.x[0])
    const y = surface.y[1] - ((e.clientY - rect.top - plot.top) / plot.height) * (surface.y[1] - surface.y[0])
    setStart([
      Math.max(surface.x[0], Math.min(surface.x[1], x)),
      Math.max(surface.y[0], Math.min(surface.y[1], y)),
    ])
  }

  const shown = Math.floor(progress.current)
  return (
    <section className="card">
      <SectionHead step={5} title="Gradient descent: walking downhill">
        Training is a search for the lowest point of the loss. Here the "network" has just two weights, so the loss is a
        landscape you can see (darker is lower, × marks the bottom). Gradient descent can't see the map. At each step
        it feels the slope under its feet, ∇<var>L</var>, and steps the other way: <var>θ</var> ← <var>θ</var> −{' '}
        <var>η</var>·∇<var>L</var>. <b>Click or drag on the map</b> to choose where it starts.
      </SectionHead>
      <div className="presets">
        {(Object.keys(surfaces) as SurfaceName[]).map((s) => (
          <button key={s} className={`chip${s === name ? ' active' : ''}`} onClick={() => pickSurface(s)}>
            {surfaces[s].label}
          </button>
        ))}
      </div>
      <canvas
        ref={ref}
        className="canvas canvas-tall draggable"
        aria-label="Loss landscape with optimizer paths"
        onPointerDown={(e) => {
          dragging.current = true
          e.currentTarget.setPointerCapture(e.pointerId)
          setFromPointer(e)
        }}
        onPointerMove={(e) => dragging.current && setFromPointer(e)}
        onPointerUp={() => (dragging.current = false)}
      />
      <div className="surface-controls">
        <LogSlider
          label={<>Learning rate <var>η</var></>}
          value={learningRate}
          min={surface.learningRate / 30}
          max={surface.learningRate * 30}
          onChange={setLearningRate}
        />
        <Slider label="Animation" value={speed} min={5} max={200} step={5} display={`${speed} steps/s`} onChange={setSpeed} />
      </div>
      <table className="errors">
        <thead>
          <tr>
            <th>Optimizer</th>
            <th>Rule</th>
            <th>Loss after {shown} steps</th>
          </tr>
        </thead>
        <tbody>
          {(Object.keys(optimizerLabels) as OptimizerName[]).map((o) => {
            const run = runs.find((r) => r.name === o)
            const p = run?.path[Math.min(shown, run.path.length - 1)]
            const diverged = run && run.diverged && shown >= run.path.length - 1
            return (
              <tr key={o}>
                <td>
                  <label className="checkbox">
                    <input
                      type="checkbox"
                      checked={enabled[o]}
                      onChange={(e) => setEnabled({ ...enabled, [o]: e.target.checked })}
                    />
                    <i className="swatch" style={{ background: `var(${optimizerTokens[o]})` }} />
                    {optimizerLabels[o]}
                  </label>
                </td>
                <td className="mono small">
                  {o === 'sgd' && 'θ ← θ − η·g'}
                  {o === 'momentum' && 'v ← 0.9·v + g;  θ ← θ − η·v'}
                  {o === 'adam' && 'θ ← θ − η·m̂ / √ŝ'}
                </td>
                <td>{!run ? '–' : diverged ? 'diverged ↗ ∞' : p ? short(surface.f(p[0], p[1]) - floor) : '–'}</td>
              </tr>
            )
          })}
        </tbody>
      </table>
      <p className="callout">
        {name === 'ravine' &&
          'In a narrow valley the slope points mostly across it, not along it. Plain gradient descent zig-zags between the walls and crawls along the floor. Momentum overshoots across the valley too (it is a heavy ball), but it builds up speed along the floor and gets there first; Adam rescales each direction so both move at a similar pace.'}
        {name === 'bowl' &&
          'A round bowl is the easy case: the slope points straight at the bottom. Raise η and watch the steps overshoot; above η = 2 they get bigger every time and fly off.'}
        {name === 'rosenbrock' &&
          'Reaching the curved valley is fast; following it to the minimum at (1, 1) is slow, because along the floor the slope is almost zero.'}
        {name === 'bumpy' &&
          'Gradient descent only ever goes downhill from where it is, so it settles in whichever valley it starts above. That valley may not be the deepest. Drag the start around. Momentum can sometimes roll through a small bump.'}
        {name === 'himmelblau' &&
          'Four valleys, all equally deep. Which one you end up in depends only on the start and the optimizer. Real networks have this too: many different sets of weights work equally well.'}
      </p>
      <div className="row">
        <button className="primary" onClick={play}>
          Replay
        </button>
        <button onClick={() => setStart(surface.start)}>Reset start</button>
      </div>
    </section>
  )
}

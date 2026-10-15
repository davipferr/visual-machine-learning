import type { Network, Sample } from 'mini-neural-net'
import { diverging, parseColor, type Frame, type Plot } from './canvas'

/** The input plane every 2D classification view shows. */
export const DOMAIN: [number, number] = [-1.5, 1.5]

/** Colour for a predicted probability: orange near 0, blue near 1, background at 0.5. */
export function probabilityScale(frame: Frame, strength = 0.75) {
  const scale = diverging(
    parseColor(frame.color('--surface')),
    parseColor(frame.color('--class0')),
    parseColor(frame.color('--class1')),
    strength,
  )
  return (p: number) => scale(2 * p - 1)
}

/** Training points filled, test points as rings, mistakes crossed out. */
export function drawPoints(
  frame: Frame,
  plot: Plot,
  samples: readonly Sample[],
  kind: 'train' | 'test',
  network?: Network,
  radius = 4,
) {
  const { ctx, color } = frame
  const outline = color('--surface')
  for (const s of samples) {
    const x = plot.x(s.input[0])
    const y = plot.y(s.input[1])
    const fill = color(s.target[0] === 1 ? '--class1' : '--class0')
    ctx.beginPath()
    ctx.arc(x, y, radius, 0, Math.PI * 2)
    if (kind === 'train') {
      ctx.fillStyle = fill
      ctx.fill()
      ctx.lineWidth = 1.2
      ctx.strokeStyle = outline
      ctx.stroke()
    } else {
      ctx.lineWidth = 2
      ctx.strokeStyle = fill
      ctx.stroke()
    }
    if (network && network.predict(s.input)[0] >= 0.5 !== s.target[0] >= 0.5) {
      const r = radius + 2.5
      ctx.strokeStyle = color('--text')
      ctx.lineWidth = 1.5
      ctx.beginPath()
      ctx.moveTo(x - r, y - r)
      ctx.lineTo(x + r, y + r)
      ctx.moveTo(x + r, y - r)
      ctx.lineTo(x - r, y + r)
      ctx.stroke()
    }
  }
}

/** A centred square plot area inside a frame. */
export function squarePlot(frame: Frame, pad = 12) {
  const side = Math.min(frame.width, frame.height) - pad * 2
  const left = (frame.width - side) / 2
  const top = (frame.height - side) / 2
  const [d0, d1] = DOMAIN
  return {
    left,
    top,
    width: side,
    height: side,
    x: (v: number) => left + ((v - d0) / (d1 - d0)) * side,
    y: (v: number) => top + (1 - (v - d0) / (d1 - d0)) * side,
    invX: (px: number) => d0 + ((px - left) / side) * (d1 - d0),
    invY: (py: number) => d0 + (1 - (py - top) / side) * (d1 - d0),
  }
}

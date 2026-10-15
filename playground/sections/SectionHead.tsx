import type { ReactNode } from 'react'

export function SectionHead({ step, title, children }: { step: number; title: ReactNode; children: ReactNode }) {
  return (
    <div className="section-head">
      <span className="step">{step}</span>
      <div>
        <h2>{title}</h2>
        <p>{children}</p>
      </div>
    </div>
  )
}

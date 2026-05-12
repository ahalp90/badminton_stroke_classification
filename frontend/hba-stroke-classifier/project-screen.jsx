import { useTheme, Card, SectionHeader } from './shared';
import {
  sections,
  overview,
  dataset,
  method,
  training,
  results,
  improvements,
  team,
} from './project-content';

function PlaceholderBadge({ show }) {
  if (!show) return null;
  return (
    <span
      title="Placeholder content — replace before submission"
      style={{
        marginLeft: 10,
        padding: '2px 8px',
        borderRadius: 4,
        fontSize: 10,
        fontWeight: 700,
        letterSpacing: '0.06em',
        textTransform: 'uppercase',
        background: '#B8860B',
        color: '#fff',
        fontFamily: "'JetBrains Mono', monospace",
        verticalAlign: 'middle',
      }}
    >
      Placeholder
    </span>
  );
}

function Figure({ src, alt, caption }) {
  const { t } = useTheme();
  return (
    <figure style={{ margin: 0, display: 'flex', flexDirection: 'column', gap: 8 }}>
      <div style={{ background: t.surface2, border: `1px solid ${t.border}`, borderRadius: 8, padding: 8 }}>
        <img src={src} alt={alt} loading="lazy" style={{ width: '100%', height: 'auto', display: 'block', borderRadius: 4 }} />
      </div>
      {caption && (
        <figcaption style={{ fontSize: 12, color: t.muted, lineHeight: 1.5 }}>{caption}</figcaption>
      )}
    </figure>
  );
}

function FigureGrid({ figures }) {
  return (
    <div
      style={{
        display: 'grid',
        gridTemplateColumns: 'repeat(auto-fit, minmax(260px, 1fr))',
        gap: 18,
        marginTop: 14,
      }}
    >
      {figures.map((f, i) => <Figure key={i} {...f} />)}
    </div>
  );
}

function MetricsTable({ headers, rows }) {
  const { t } = useTheme();
  return (
    <div style={{ overflowX: 'auto', marginTop: 14, border: `1px solid ${t.border}`, borderRadius: 8 }}>
      <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
        <thead>
          <tr>
            {headers.map(h => (
              <th
                key={h}
                style={{
                  padding: '10px 14px',
                  textAlign: 'left',
                  background: t.surface2,
                  color: t.text,
                  fontWeight: 600,
                  borderBottom: `1px solid ${t.border}`,
                  fontSize: 12,
                  textTransform: 'uppercase',
                  letterSpacing: '0.05em',
                }}
              >
                {h}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((row, i) => (
            <tr key={i}>
              {row.map((cell, j) => (
                <td
                  key={j}
                  style={{
                    padding: '10px 14px',
                    color: t.text,
                    borderBottom: i === rows.length - 1 ? 'none' : `1px solid ${t.border}`,
                    fontFamily: j === 0 ? "'Space Grotesk', sans-serif" : "'JetBrains Mono', monospace",
                  }}
                >
                  {cell}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function Prose({ body }) {
  const { t } = useTheme();
  return body.map((p, i) => (
    <p key={i} style={{ margin: '0 0 10px', color: t.text, lineHeight: 1.65, fontSize: 14 }}>{p}</p>
  ));
}

function SectionHeading({ tag, title, placeholder }) {
  const { t } = useTheme();
  return (
    <h2
      id={tag}
      style={{
        fontSize: 20,
        fontWeight: 700,
        color: t.text,
        margin: '0 0 12px',
        scrollMarginTop: 72,
      }}
    >
      {title}
      <PlaceholderBadge show={placeholder} />
    </h2>
  );
}

const allSections = [overview, dataset, method, training, results, improvements, team];

export function ProjectScreen() {
  const { t } = useTheme();
  const placeholderCount = allSections.filter(s => s.placeholder).length;
  const hasPlaceholders = placeholderCount > 0;

  return (
    <div>
      {hasPlaceholders && (
        <div
          role="status"
          style={{
            background: 'repeating-linear-gradient(45deg,#5a4400,#5a4400 12px,#735500 12px,#735500 24px)',
            color: '#FFE69C',
            borderBottom: `2px solid ${t.warning}`,
            padding: '12px 24px',
            fontSize: 13,
            lineHeight: 1.5,
            fontFamily: "'Space Grotesk', sans-serif",
          }}
        >
          <strong style={{ color: '#FFF3CD' }}>⚠ Placeholder content.</strong>{' '}
          {placeholderCount} of {allSections.length} sections still contain stub
          text and must be replaced before this page is submission-ready. Edit{' '}
          <code style={{ background: 'rgba(0,0,0,0.3)', padding: '1px 6px', borderRadius: 3, fontFamily: "'JetBrains Mono', monospace" }}>
            hba-stroke-classifier/project-content.js
          </code>{' '}
          and set each section&apos;s <code style={{ background: 'rgba(0,0,0,0.3)', padding: '1px 6px', borderRadius: 3, fontFamily: "'JetBrains Mono', monospace" }}>placeholder</code> flag to <code style={{ background: 'rgba(0,0,0,0.3)', padding: '1px 6px', borderRadius: 3, fontFamily: "'JetBrains Mono', monospace" }}>false</code> once its content is final.
        </div>
      )}

      <div
        style={{
          maxWidth: 1100,
          margin: '0 auto',
          padding: 32,
          display: 'grid',
          gridTemplateColumns: '180px 1fr',
          gap: 40,
        }}
      >
        <nav
          aria-label="Page sections"
          style={{
            position: 'sticky',
            top: 72,
            alignSelf: 'start',
            display: 'flex',
            flexDirection: 'column',
            gap: 4,
            borderLeft: `1px solid ${t.border}`,
            paddingLeft: 14,
          }}
        >
          {sections.map(s => (
            <a
              key={s.id}
              href={`#${s.id}`}
              style={{
                color: t.muted,
                textDecoration: 'none',
                fontSize: 13,
                padding: '4px 0',
                fontFamily: "'Space Grotesk', sans-serif",
              }}
            >
              {s.label}
            </a>
          ))}
        </nav>

        <div style={{ minWidth: 0, display: 'flex', flexDirection: 'column', gap: 40 }}>
          <SectionHeader title={overview.title} subtitle="ML project overview" />

          <section>
            <SectionHeading tag="overview" title="Overview" placeholder={overview.placeholder} />
            <Prose body={overview.body} />
          </section>

          <section>
            <SectionHeading tag="dataset" title={dataset.title} placeholder={dataset.placeholder} />
            <Prose body={dataset.body} />
            <FigureGrid figures={dataset.figures} />
          </section>

          <section>
            <SectionHeading tag="method" title={method.title} placeholder={method.placeholder} />
            <Prose body={method.body} />
          </section>

          <section>
            <SectionHeading tag="training" title={training.title} placeholder={training.placeholder} />
            <Prose body={training.body} />
            <FigureGrid figures={training.figures} />
          </section>

          <section>
            <SectionHeading tag="results" title={results.title} placeholder={results.placeholder} />
            <Prose body={results.body} />
            <MetricsTable headers={results.table.headers} rows={results.table.rows} />
          </section>

          <section>
            <SectionHeading tag="improvements" title={improvements.title} placeholder={improvements.placeholder} />
            <Prose body={improvements.body} />
            <ul style={{ listStyle: 'none', padding: 0, margin: '8px 0 0', display: 'flex', flexDirection: 'column', gap: 14 }}>
              {improvements.items.map((item, i) => (
                <li key={i} style={{ borderLeft: `3px solid ${t.blue}`, padding: '4px 0 4px 14px' }}>
                  <div style={{ fontSize: 14, fontWeight: 600, color: t.text, marginBottom: 4 }}>{item.title}</div>
                  <div style={{ fontSize: 13, color: t.muted, lineHeight: 1.55 }}>{item.body}</div>
                </li>
              ))}
            </ul>
          </section>

          <section>
            <SectionHeading tag="team" title={team.title} placeholder={team.placeholder} />
            <Prose body={team.body} />
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(180px, 1fr))', gap: 12, marginTop: 14 }}>
              {team.members.map((m, i) => (
                <Card key={i} style={{ padding: '12px 14px' }}>
                  <div style={{ color: t.text, fontWeight: 600, marginBottom: 2 }}>{m.name}</div>
                  <div style={{ color: t.muted, fontSize: 12 }}>{m.role}</div>
                </Card>
              ))}
            </div>
          </section>
        </div>
      </div>
    </div>
  );
}

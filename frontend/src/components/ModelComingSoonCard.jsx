import { useTheme, Card } from '../shared';

/* Pending-model card: shown for a registered model whose trained run / metrics
   aren't in the repo yet (e.g. BRIC). Deliberately shows NO metrics — no fake
   numbers. Replaced by ModelEvaluationPanel automatically once the registry
   reports status "available". */
export function ModelComingSoonCard({ model }) {
  const { t } = useTheme();
  return (
    <Card style={{ padding: 28 }}>
      <div style={{
        fontSize: 11, fontWeight: 600, color: t.blue,
        textTransform: 'uppercase', letterSpacing: '0.08em', marginBottom: 8,
      }}>
        Results coming soon
      </div>
      <div style={{ fontSize: 13, color: t.muted, lineHeight: 1.65 }}>
        {model.display_name} is trained and its held-out evaluation is in
        progress. Metrics and per-clip results will appear here automatically
        once the run lands in the registry.
      </div>
    </Card>
  );
}

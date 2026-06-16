# BST-X model + CDB-F1 loss

Compact view of the forward pass and how the loss reweights classes between epochs. Full reference: [`bst_x_model_canonical.md`](bst_x_model_canonical.md).

## Forward pass

```mermaid
%%{init: {'theme':'neutral','themeVariables':{
  'primaryColor':'#a78bfa',
  'primaryTextColor':'#000000',
  'primaryBorderColor':'#7c3aed',
  'lineColor':'#6b7280',
  'background':'#ffffff',
  'fontFamily':'sans-serif'
}}}%%
flowchart LR

  inputs["3 input streams per clip<br/>pose JnB · court xy · shuttle xy"]
  tcn["TCN sequencing<br/>pose modulated by court pos,<br/>then 1D stacked dilated convs"]
  trans["Transformer stack<br/>Temporal (2L, CLS per stream) →<br/>Cross (1L, player ← shuttle) →<br/>Interactional (1L, cross-player)"]
  priors["Warm-start priors<br/>CG + AP<br/>factor 1 → 0 by epoch 15"]
  head["MLP head<br/>concat streams → n_classes logits"]
  loss(["CDB-F1 loss"])
  x3d["X3D-S wrist crop<br/>K400 pretrained<br/>planned ~July 2026"]

  inputs --> tcn --> trans --> priors --> head --> loss
  x3d -. fusion point TBD .-> head

  classDef input fill:#22d3ee,stroke:#0e7490,color:#000000
  classDef proc fill:#a78bfa,stroke:#7c3aed,color:#000000
  classDef trans fill:#fbcfe8,stroke:#be185d,color:#000000
  classDef opt fill:#bbf7d0,stroke:#15803d,color:#000000
  classDef out fill:#facc15,stroke:#a16207,color:#000000
  classDef future fill:#d1d5db,stroke:#6b7280,color:#000000

  class inputs input
  class tcn,head proc
  class trans trans
  class priors opt
  class loss out
  class x3d future
```

## CDB-F1 loss

```mermaid
%%{init: {'theme':'neutral','themeVariables':{
  'primaryColor':'#a78bfa',
  'primaryTextColor':'#000000',
  'primaryBorderColor':'#7c3aed',
  'lineColor':'#888a85',
  'secondaryColor':'#22d3ee',
  'tertiaryColor':'#f4f4f5',
  'tertiaryBorderColor':'#d1d5db',
  'tertiaryTextColor':'#000000',
  'edgeLabelBackground':'#ffffff',
  'background':'#ffffff',
  'fontFamily':'sans-serif'
},'themeCSS':'.edgeLabel,.edgeLabel *,.edgeLabel span,.edgeLabel p{color:#000000!important;fill:#000000!important}'}%%
flowchart LR

  context["F1 for performance<br/>not freq;<br/>wrist_smash performs worse<br/>than less common classes"]
  why["Focal loss: static class α.<br/>CDB-F1: per-epoch<br>α = 1 − F1_class<br/>(EMA smoothed)"]

  batch["α[c] scales each example's loss<br/>* per-sample scaling (standard FL)<br/>(α = 1 during warm-up, epochs 1-5)"]
  acc["Track per-class<br/>TP / FP / FN<br/>(this epoch's running tally)"]
  endep["End of epoch:<br/>compute per-class F1<br/>refresh α for next epoch"]

  context -.-> why -.-> batch
  batch --> acc --> endep
  endep -. updated α feeds next epoch .-> batch

  classDef why fill:#facc15,stroke:#a16207,color:#000000
  classDef batch fill:#22d3ee,stroke:#0e7490,color:#000000
  classDef epoch fill:#a78bfa,stroke:#7c3aed,color:#000000

  class context,why why
  class batch,acc batch
  class endep epoch
```

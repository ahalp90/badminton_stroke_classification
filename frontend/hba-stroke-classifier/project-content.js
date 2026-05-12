// Content source of truth for the Project showcase screen.
//
// PLACEHOLDER WORKFLOW:
//   Every section starts with `placeholder: true` while the text is a stub.
//   Once the ML team replaces a section with real content, set its
//   `placeholder` flag to `false`. While ANY section is still flagged, a
//   loud banner renders at the top of the screen so placeholder text can't
//   accidentally ship.
//
// To add a new figure:
//   1. Drop the PNG into hba-stroke-classifier/assets/project/
//   2. Import it below
//   3. Reference it in the relevant section's `figures` array

import badmintonDB from './assets/project/badmintonDB.png';
import tenniSet from './assets/project/tenniSet.png';
import shuttleFixed from './assets/project/shuttleset_25_fixed_width.png';
import shuttleFixedPartial from './assets/project/shuttleset_25_fixed_width_partial_train.png';
import shuttleOurStrategy from './assets/project/shuttleset_25_our_strategy.png';
import lossCurves from './assets/project/loss_curves.png';
import trainingSpeed from './assets/project/training_speed.png';

export const sections = [
  { id: 'overview', label: 'Overview' },
  { id: 'dataset', label: 'Dataset' },
  { id: 'method', label: 'Method' },
  { id: 'training', label: 'Training' },
  { id: 'results', label: 'Results' },
  { id: 'improvements', label: 'Improvements' },
  { id: 'team', label: 'Team' },
];

export const overview = {
  title: 'Badminton Stroke Classification',
  placeholder: true,
  body: [
    'This project classifies individual badminton strokes from broadcast video using a fine-grained action recognition model. The goal is to give coaches and players an automated tool to break match footage down into per-stroke data.',
    'This page summarises the dataset, model, training setup, and the experiments that drove our final results.',
  ],
};

export const dataset = {
  title: 'Dataset',
  placeholder: true,
  body: [
    'We work with the ShuttleSet broadcast badminton dataset, supplemented with our own clip-splitting strategy. The figures below compare the source datasets and show how our splitting strategy differs from the fixed-width baselines.',
  ],
  figures: [
    { src: badmintonDB, alt: 'BadmintonDB sample distribution', caption: 'BadmintonDB class distribution.' },
    { src: tenniSet, alt: 'TenniSet sample distribution', caption: 'TenniSet class distribution for reference.' },
    { src: shuttleFixed, alt: 'ShuttleSet fixed-width split', caption: 'ShuttleSet 25 — fixed-width clip splitting.' },
    { src: shuttleFixedPartial, alt: 'ShuttleSet fixed-width partial training split', caption: 'ShuttleSet 25 — fixed-width with partial training set.' },
    { src: shuttleOurStrategy, alt: 'ShuttleSet our splitting strategy', caption: 'ShuttleSet 25 — our splitting strategy.' },
  ],
};

export const method = {
  title: 'Method',
  placeholder: true,
  body: [
    'Our classifier is the Badminton Stroke Transformer (BST): a dilated TCN over per-frame MMPose keypoints and TrackNetV3 shuttle trajectory, followed by a temporal / cross / interactional transformer stack that jointly attends to both players and the shuttlecock.',
    'Replace this prose with the ML team\'s authoritative architecture description.',
  ],
};

export const training = {
  title: 'Training',
  placeholder: true,
  body: [
    'We trained BST on a single GPU using class-balanced sampling. Loss curves show convergence behaviour across runs; the training-speed plot summarises throughput on our setup.',
  ],
  figures: [
    { src: lossCurves, alt: 'Training and validation loss curves', caption: 'Loss curves across training runs.' },
    { src: trainingSpeed, alt: 'Training throughput', caption: 'Training speed (samples / second).' },
  ],
};

export const results = {
  title: 'Results',
  placeholder: true,
  body: [
    'Headline metrics from our best run, compared with the baselines we measured. Values transcribed from result_table.xlsx — update the rows below as new experiments land.',
  ],
  table: {
    headers: ['Model', 'Accuracy', 'F1 (macro)', 'Notes'],
    rows: [
      ['Baseline (fixed-width)', 'TBD', 'TBD', 'ShuttleSet 25 baseline'],
      ['BST (ours)', 'TBD', 'TBD', 'Our splitting strategy'],
    ],
  },
};

export const improvements = {
  title: 'Improvements',
  placeholder: true,
  body: [
    'A summary of what we tried and what moved the needle. Edit and reorder freely as the project evolves.',
  ],
  items: [
    { title: 'New clip splitting strategy', body: 'Replaced fixed-width windows with a stroke-aware splitting strategy, giving cleaner per-stroke training samples.' },
    { title: 'Class-balanced sampling', body: 'Re-weighted underrepresented stroke classes during training to lift macro F1 on rarer strokes.' },
    { title: 'Pose-feature pipeline refinements', body: 'Stabilised per-frame pose extraction and trajectory features fed into the transformer.' },
  ],
};

export const team = {
  title: 'Team',
  placeholder: true,
  body: ['The ML team behind the model and experiments.'],
  members: [
    { name: 'TBD', role: 'ML lead' },
    { name: 'TBD', role: 'Data / training' },
    { name: 'TBD', role: 'Evaluation' },
  ],
};

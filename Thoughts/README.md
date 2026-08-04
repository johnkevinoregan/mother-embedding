# Thoughts

Short notes that are **not experiments**: identities, relations between things the project already
has, and arguments about what a result means. Each should be readable on its own.

The distinction from a phase directory is that nothing here introduces a measurement to justify a
design decision. If a note turns into an experiment, it moves out and becomes a phase.

| note | what it says |
|:--|:--|
| [`FourierGaborBankRelations.md`](FourierGaborBankRelations.md) | The orientation block is a polar-binned local power spectrum — provably the same numbers a plain FFT of the cell would give. So it is *not* where this front end differs from the Fourier baseline it was built to beat; the pointwise nonlinear layer above it is. Also: what the higher orientation harmonics measure — they are junction-*angle* detectors (`Z₄` for 90°, `Z₆` for 60°, mutually exclusive), and they were tested on a dataset with no 60° junctions. And why the scale axis has no counterpart to the orientation axis' size/shape split — which is what the thickness/fuzziness degeneracy is. |

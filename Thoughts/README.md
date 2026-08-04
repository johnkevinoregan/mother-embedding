# Thoughts

Short notes that are **not experiments**: identities, relations between things the project already
has, and arguments about what a result means. Each should be readable on its own.

The distinction from a phase directory is that nothing here introduces a measurement to justify a
design decision. If a note turns into an experiment, it moves out and becomes a phase.

| note | what it says |
|:--|:--|
| [`FourierGaborBankRelations.md`](FourierGaborBankRelations.md) | The orientation block is a polar-binned local power spectrum — provably the same numbers a plain FFT of the cell would give. Which means it is *not* where this front end differs from the Fourier baseline it was built to beat; the pointwise nonlinear layer above it is. |

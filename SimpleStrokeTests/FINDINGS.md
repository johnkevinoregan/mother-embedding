# Can we make the front end better by giving it more to work with?

*A plain-language summary. The detailed tables are in `SWEEP.md` (a first pass, partly wrong),
`SWEEP_FULLN.md` (the proper version) and `XSCALE.md` (a follow-up). Read this first.*

---

## What the front end is, and what we were asking

The **front end** is a fixed recipe for measuring an image. It looks at every pixel through a
bank of filters tuned to different **orientations** (which way an edge runs) and different
**scales** (how coarse or fine the detail is), and boils the result down to **31 numbers** per
image. Nothing about it is learned — the recipe was designed, and the same recipe is applied to
every image.

To test whether those 31 numbers are any good, we generate pictures of a single stroke on a grey
background and ask a small network to read eight things off them:

| | in plain terms |
|:--|:--|
| **curvedness** | how bent is the stroke |
| **brokenness** | is there a gap in it, and how big |
| **closedness** | is it a closed loop or an open arc |
| **vangle** | if it has a kink, how sharp is the kink |
| **arms** | is it a plain stroke, a T-junction, or a crossing |
| **thickness** | how wide is the stroke |
| **fuzziness** | how blurry are its edges |
| **polarity** | is it lighter or darker than the background |

Scores are **R²**: **1.0** means the property is read off perfectly, **0.0** means no better than
always guessing the average, and **negative** means worse than guessing.

**The question in this round:** the front end has several dials that had never been turned. Would
turning them up make it better?

---

## The four dials we turned

**1. More scales.** Use five filter sizes instead of three, so the range from coarse to fine is
sampled more finely.

**2. Higher harmonics.** At each point the front end summarises *which directions have edges* —
like a compass rose of edge strength. It used to keep only a coarse summary of that rose. We let
it keep a more detailed one. Costs five extra numbers.

**3. More orientations.** Use 16/24/32 filter directions instead of 8/12/16 — a finer compass —
while keeping each filter's tuning just as broad as before.

**4. More probe distances.** To decide whether something is a T-junction or a crossing, the front
end reaches out a fixed distance from each point and asks "is there a stroke over there?". It only
ever reached out *one* distance per scale. We let it reach three.

---

## What happened

**More scales made things worse.** This was the one we most expected to help, and it hurt — every
time. Rejected.

**Higher harmonics helped, cheaply.** Better readings of how *bent* a stroke is, consistently, for
five extra numbers. Sensible in hindsight: curvature is about how the compass rose *spreads out*,
and the coarse summary couldn't describe that shape.

**More orientations did nothing useful — until the test got harder** (see below). On ordinary
tests it bought nothing at all, for twice the computation.

**More probe distances helped the most, and broadly** — better at spotting gaps, kinks and
junctions. This was also the cheapest to justify: we'd checked earlier that two of the three
existing probes were reading something other than what the theory said they should.

---

## The harder test, which changed the answers

Reading properties off images drawn from the *same* pool you trained on is easy. The demanding
test is **training on one range and testing on another** — for example, train only on thin
strokes, then test on thick ones. That asks whether the measurements really capture the property,
or merely memorised the range they saw.

**Almost everything looked different under that test**, and mostly in our favour:

- Gains that seemed large on the easy test **shrank by two to five times**.
- The extra orientations, worthless on the easy test, became one of the **best** dials.
- Combining higher harmonics with more probe distances was slightly *worse* than probe distances
  alone on the easy test, but **clearly better** on the hard one.

**The practical lesson: judge on the hard test.** Judged on the easy test alone we would have
picked the wrong configuration twice over.

---

## The one thing that stayed broken

There is a specific weakness we already knew about. **The front end confuses thickness with
blurriness.** Train it on sharp-edged strokes and show it blurry ones, and its thickness readings
collapse to far worse than guessing (R² ≈ −2). Train it on thin strokes and show it thick ones,
and its blurriness readings collapse the same way.

The reason is intuitive: a **thick sharp** stroke and a **thin blurry** one produce very similar
patterns of filter response. Both put a lot of energy into the coarse filters. Telling them apart
needs something the front end didn't measure.

**So we tried to measure it.** A thick sharp stroke has energy in the coarse filters *and* the
fine ones, because its edges are crisp. A thin blurry stroke has energy only in the coarse ones.
So we added a number recording **what fraction of the energy sits in the finer filters** — two
extra numbers in total.

**On its own, it barely helped.** But **combined with the other improvements it moved the problem
by about 0.16, in both directions, and that replicated almost exactly** (+0.161 one way, +0.160
the other). That's the first thing in the whole project to shift this.

Why only in combination? Most likely because the fine-versus-coarse fraction means different
things at different stroke widths — so it is only interpretable once the readout also knows the
width, which the extra probe distances supply. That's a plausible story, not something we've
proved.

**It is reduced, not fixed.** The scores are still deeply negative (−1.90 and −2.29). A readout
trained on one range remains badly wrong on the other.

---

## Where this leaves the front end

| if you care most about | use | numbers |
|:--|:--|--:|
| shape under changing conditions | harmonics + probe distances | 54 |
| the thickness/blur weakness, and ordinary accuracy | the above **plus** cross-scale | 58 |

**No single setting wins everything** — the second buys the thickness/blur improvement at a small
cost to the shape readings. Which to prefer depends on the eventual application, not on the front
end.

Both are better than the 31-number original on essentially every measure.

---

## Two mistakes worth recording

**We overstated an early result.** A first, cheaper pass (on a fifth of the data) suggested two
dials had solved the thickness/blur problem outright. Repeated properly on the full data, that
gain **vanished entirely**. Small-scale trials systematically exaggerate the benefit of adding
capacity, because extra capacity helps most when data is scarce. They are useful for deciding
*what to test properly*, never for the size of an effect.

**We declared an idea dead too early.** The cross-scale measurement was tested on its own, failed,
and was written off — and the very next test showed it working when combined with the others. An
ingredient that does nothing alone can still matter in company, so testing it alone is not a fair
test of the idea.

---

## What we would do next

The remaining weakness needs a different kind of answer than "measure more of the same". The plan
is to look at a large pretrained network that *doesn't* have this weakness, find which of its
internal measurements distinguish thickness from blurriness when the range shifts, and see what
those measurements actually respond to — rather than guessing at another formula.

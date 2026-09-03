# Phase 13 — Curriculum EMNIST: what the hand-designed front end learns, and how it compares with CNNs

## Purpose of the experiment

This subproject asks a fairly specific question: **does the proposed hand-designed visual front end provide a useful, general representation of character shape without itself having to be learned?** The comparison is with modern convolutional networks, especially ConvNeXt.

The dataset is **EMNIST Balanced**, which contains handwritten digits and upper- and lower-case letters grouped into 47 classes. There are 112,800 training images and 18,800 test images. The original images are 28 × 28 pixels; for the proposed front end they are bilinearly enlarged to 112 × 112.

The central experiment is deliberately unusual. Instead of training repeatedly on the whole training set, the 112,800 training examples are randomly divided into four non-overlapping subsets of 28,200 images. Training proceeds for 60 epochs:

- epochs 1–15: subset 1
- epochs 16–30: subset 2
- epochs 31–45: subset 3
- epochs 46–60: subset 4

The optimizer is **not reset** when the subset changes.

This provides a way of separating two things that ordinary train/test accuracy confounds: learning the general structure of the task, and remembering individual training examples. Because all four subsets are random samples from the same EMNIST distribution, there is no genuine change of task at epochs 15, 30, or 45. In principle, a learner that has extracted only general character structure should hardly notice the switch. A sudden change in performance therefore reveals dependence on the particular images that were being trained on.

A control condition trains for the same 60 epochs but stays on subset 1 throughout. This makes it possible to distinguish the benefit of receiving fresh examples from effects caused simply by further optimization.

## What is the proposed front end?

The front end is **not a neural network and is not trained on EMNIST**. It is a fixed, hand-designed image transform motivated by elementary properties of contour and stroke geometry. An input image is converted into a vector of **381 numbers**. Only a conventional learned classifier placed after this vector is trained.

The first stage is a bank of oriented, approximately Gabor-like filters operating at several spatial scales and orientations. In the Phase 13 configuration there are four scales, corresponding approximately to wavelengths of 56, 29.9, 16, and 8 pixels in the 112 × 112 enlarged image, with progressively finer angular sampling at the finer scales. The use of quadrature energy means that these channels encode the presence and orientation of local strokes while largely discarding the sign of the contrast.

The subsequent fixed operations try to describe properties of how those oriented responses are arranged rather than simply preserving local filter activations. They include:

- pooled oriented energy, describing which orientations and scales are present;
- a low-pass or DC component, preserving mean luminance information that oriented energy deliberately discards;
- an `A1` conjunction measure sensitive to relationships between orientations, roughly capturing junction/corner-like structure;
- an `A2` or end-stopping measure, designed to respond to endings and interruptions along a locally dominant contour;
- ray-based spatial relations between oriented responses;
- a **spatial maximum** for selected conjunction responses, added because very local events such as a small gap may be almost invisible to spatial averaging.

Most features are pooled over a 3 × 3 spatial grid. Thus the representation retains coarse information about *where* a feature occurs, but discards precise pixel position. The spatial-max features provide an additional translation-tolerant signal for whether a strong local event occurs anywhere in the character.

Conceptually, this front end therefore imposes a strong prior: **images are usefully described in terms of multiscale oriented strokes and relations among those strokes**. A CNN starts much closer to raw pixels and has to discover useful filters and combinations through training. Here those early computations are specified in advance.

One technical caveat matters for EMNIST. The finest wavelength, 8 pixels after 4× upsampling, lies at the Nyquist limit of the original 28 × 28 image, so that channel partly reflects the bilinear interpolation rather than genuinely new image structure.

## What was compared?

Four main systems were tested.

| System | Visual representation | What is learned on EMNIST? |
| --- | --- | --- |
| Proposed front end | 381 fixed features | classifier/readout only |
| ConvNeXt-base pretrained on ImageNet | 1024 frozen features | classifier/readout only |
| ConvNeXt-tiny from scratch | 27.9 million parameters | entire CNN |
| ConvNeXt-base from scratch | 87.6 million parameters | entire CNN |

The first two conditions are particularly clean: the representations are frozen and the same type of downstream readout is learned. They therefore compare the *quality of the representations* rather than the ability of one model to alter its representation during EMNIST training.

The from-scratch ConvNeXts answer a different question: if a large CNN is allowed to learn its visual representation specifically for EMNIST, how much better can it eventually become?

## Main result: the fixed front end is surprisingly competitive

At epoch 15, after seeing only the first 28,200 training examples, the proposed representation reaches **85.41% test accuracy**, slightly higher than the frozen ImageNet ConvNeXt-base at **84.70%**, and also slightly higher than the ConvNeXt-tiny trained from scratch at that point (**83.10%**). The enormous ConvNeXt-base trained from scratch is similar at **85.73%**.

By epoch 60, the end-to-end trained CNNs have moved ahead:

| Representation | Test accuracy at epoch 60 | Best observed |
| --- | ---: | ---: |
| ConvNeXt-base from scratch | 88.06% | 88.94% |
| ConvNeXt-tiny from scratch | 87.01% | 88.19% |
| Proposed front end, frozen | 85.97% | 87.18% |
| ImageNet ConvNeXt-base, frozen | 85.70% | 86.78% |

So the defensible conclusion is **not** that the hand-designed front end is better than a trainable CNN. Rather:

**A 381-dimensional, completely untrained representation gets within roughly two percentage points of large CNNs trained end-to-end on EMNIST, and performs about as well as a 1024-dimensional representation learned by ConvNeXt-base from roughly 1.28 million ImageNet photographs.**

The small numerical advantage over frozen ImageNet ConvNeXt is below half a percentage point and comes from single-seed comparisons, so the repository correctly treats the two frozen representations as effectively equivalent rather than claiming a reliable victory.

The striking difference is therefore one of **prior structure and data efficiency**. The proposed front end brings useful assumptions about oriented strokes to the task before seeing any labelled character. A large CNN can ultimately discover a better task-specific representation, but it needs data and optimization to do so.

## What does the subset-switch experiment reveal?

All systems memorize the particular images they are currently seeing. Near the ends of the four training blocks, accuracy on the currently trained subset lies about **10–14 percentage points above test accuracy**. The end-to-end ConvNeXts can drive accuracy on an individual 28,200-image subset to essentially 100%.

Yet this perfecting of the training set contributes relatively little to generalization. In the no-switch ConvNeXt control, training accuracy eventually reaches 100% while held-out accuracy improves only modestly.

A particularly interesting qualitative difference appears when a fresh subset is introduced. The **frozen** front ends — both the proposed representation and frozen ImageNet ConvNeXt — cross the switches with little immediate change in test accuracy. In contrast, both end-to-end ConvNeXts show an abrupt improvement when new examples arrive. This indicates that fresh data immediately changes a trainable visual representation, whereas with a fixed representation new examples can only improve the downstream classifier gradually.

The experiment also shows that forgetting previously trained examples is remarkably similar for the two frozen representations. Once training leaves subset 1, most of its special training-set advantage disappears within the next 15 epochs. This suggests that, when the representation is fixed, this form of forgetting is primarily a property of the **readout and optimizer**, not of whether the fixed representation came from hand-designed filters or an ImageNet CNN.

## A stronger test: entirely unseen character classes

The subproject also asks whether the CNN advantage is merely superior memorization. Ten of the 47 character classes are completely withheld while a ConvNeXt-tiny is trained on the other 37. Representations are then evaluated on few-shot classification of those **never-before-seen classes**.

Here the trained ConvNeXt is clearly superior. At 1-shot classification it scores about **74.5%**, compared with **57.9%** for the proposed front end and **60.2%** for frozen ImageNet ConvNeXt. At 20 shots the scores are approximately **90.2%, 85.8%, and 85.6%**, respectively. Raw pixels are substantially worse.

This is important because memorizing individual training characters cannot directly explain success on classes never encountered during training. The end-to-end CNN has learned transferable regularities of handwritten character construction. Its advantage is therefore genuine representation learning, not simply memorization.

At the same time, the proposed front end and frozen ImageNet ConvNeXt remain remarkably similar in this test, despite the former having only 381 features and **no learned visual parameters at all**.

## Overall interpretation

Phase 13 supports three main conclusions.

First, a relatively compact representation built explicitly from multiscale orientation, contour conjunctions, end-stopping, and coarse spatial pooling contains a great deal of the information needed for handwritten-character recognition. On EMNIST it performs approximately as well as frozen features from a very large CNN pretrained on ImageNet.

Second, explicitly supplying this kind of stroke-based visual structure produces high performance with unusually little task-specific learning. This is the principal advantage of the front end relative to a conventional CNN: **it begins with useful geometric structure instead of having to discover that structure statistically from examples**.

Third, the experiment also identifies the limit of that advantage. A sufficiently large CNN trained end-to-end on the relevant domain eventually learns a better representation, gaining roughly two points in ordinary EMNIST classification and much more in one- and few-shot transfer to previously unseen character classes. Thus the results argue for the proposed front end as a strong, economical **inductive bias**, not as a replacement for learned hierarchical representations in all circumstances.

In short, Phase 13 shows that much of what a CNN needs for this line-drawing problem can be supplied by a small fixed representation based on oriented stroke geometry. Learning remains valuable: once enough relevant data are available, a trainable CNN discovers additional reusable structure that the hand-designed front end does not capture.

## Repository sources

- [P13_CurriculumEMNIST README](https://github.com/johnkevinoregan/mother-embedding/blob/main/P13_CurriculumEMNIST/README.md)
- [P13_CurriculumEMNIST RESULTS](https://github.com/johnkevinoregan/mother-embedding/blob/main/P13_CurriculumEMNIST/RESULTS.md)
- [Extract_Ours.jl — exact Phase 13 front-end configuration](https://github.com/johnkevinoregan/mother-embedding/blob/main/P13_CurriculumEMNIST/Extract_Ours.jl)
- [Frontend.module.jl — implementation and description of feature blocks](https://github.com/johnkevinoregan/mother-embedding/blob/main/P9_P12_SimpleStrokeTests/Frontend.module.jl)

# tracing mathematical error detection in language models 

Yes. These three papers can be combined into a **much cleaner two-day project** than the theorem-prover idea.

The key observation is that they are useful for three different pieces of the paper:

* **Palumbo et al.** gives you a standard for saying an interpretation is actually mechanistic rather than just correlated.
* **Typed Chain-of-Thought** gives you the mathematical object you care about: **valid vs invalid intermediate reasoning**.
* **Compositional Interpretability** gives you a way to ask whether the discovered mechanism is **simple/compressible as well as faithful**. ([arXiv][1])

I would use them to build the following paper.

# Best idea: Where does a math model detect a reasoning error?

### Possible title

**Tracing Mathematical Error Detection in Language Models**

or, more workshop-oriented,

**Can Model Internals Detect Invalid Mathematical Reasoning Before the Output Does?**

The question:

> When a mathematical reasoning trace contains an incorrect step, does a math LLM internally represent that the reasoning has become invalid—and where does this representation emerge?

This is substantially easier than trying to reverse-engineer the whole mathematical reasoning process.

And importantly:

**you do not need to generate any dataset.**

Use **ProcessBench**.

ProcessBench already contains 3,400 mathematical reasoning examples, mostly competition/Olympiad-level, with reasoning steps and labels identifying the erroneous step. ([ACL Anthology][2])

So your data looks roughly like:

$$
\text{problem}
$$

$$
s_1,\;s_2,\;s_3,\;s_4,\ldots
$$

with a label saying something like:

$$
s_3 = \text{first erroneous step}.
$$

That is almost ideal for mechanistic interpretability.

---

# What you actually run

Use something small like:

**Qwen2.5-Math-1.5B-Instruct**

or another 1–2B math/reasoning model that fits comfortably on Colab.

You don't need to train it.

For every reasoning step \(s_k\), give the model:

$$
[\text{problem},s_1,\ldots,s_k]
$$

and extract hidden states:

$$
h_k^{(1)},h_k^{(2)},\ldots,h_k^{(L)}.
$$

Now you already have the ProcessBench label:

$$
y_k =
\begin{cases}
0 & \text{valid reasoning so far}\\
1 & \text{error has occurred}
\end{cases}
$$

Then ask:

> At which layers does \(h_k^{(l)}\) reveal that the mathematical reasoning has become invalid?

---

# Experiment 1: layer-wise error probes

This is the easiest experiment.

For each transformer layer \(l\), train a tiny logistic regression:

$$
P(\text{error}\mid h^{(l)})
=
\sigma(w_l^\top h^{(l)}+b).
$$

No neural-network training.

Just sklearn logistic regression.

Plot:

$$
\text{AUROC}
$$

against:

$$
\text{layer}.
$$

Your figure might hypothetically look like:

```text
Error decodability

1.0 |                         █ █
    |                    █ █████
0.8 |                ███████████
    |             ██████████████
0.6 |        ███████████████████
    | ██████████████████████████
0.5 +-----------------------------
      1  3  5  7  9 ...      28
                Layer
```

And perhaps you discover:

> Invalid reasoning is weakly encoded in early layers but becomes sharply linearly separable around layers 16–20.

That's already an interpretable finding.

But probing alone isn't enough for a strong mech-interp submission.

Which is where your **first paper becomes useful**.

---

# Experiment 2: turn the probe into a causal mechanism

Suppose the probe finds an error direction

$$
v_{\text{error}}
=
\frac{w}{\|w\|}.
$$

Now perform a simple intervention.

Take a valid mathematical reasoning state:

$$
h.
$$

Inject the error direction:

$$
h'=h+\alpha v_{\text{error}}.
$$

Or take an erroneous state and remove it:

$$
h'=h-\alpha v_{\text{error}}.
$$

Then continue the forward pass.

Ask the model something simple such as:

> Is the last reasoning step correct?

Measure the logit difference

$$
\Delta =
\operatorname{logit}(\text{incorrect})
-
\operatorname{logit}(\text{correct}).
$$

Now test whether:

$$
h+\alpha v_{\text{error}}
$$

causes

$$
\Delta\uparrow.
$$

That's much more interesting.

Instead of:

> "We can decode errors."

you can say:

> **A low-dimensional internal direction not only predicts mathematical invalidity but causally modulates the model's own error judgment.**

That is mech interp.

---

# This connects beautifully to paper #1

Palumbo et al. argue that mechanistic interpretations should approximately preserve the model's computation **compositionally**, rather than merely matching input-output behavior. Their validation framework includes notions related to replacing internal computations with their interpreted counterparts. ([arXiv][3])

You don't need to implement their entire formal framework.

Use the principle:

### Predictiveness

Does the proposed "error representation" correspond to error status?

$$
h \rightarrow \text{error label}.
$$

### Replaceability / causal validation

Can manipulating that representation predictably alter downstream behavior?

$$
h
\rightarrow
h+\alpha v
\rightarrow
\text{model verdict}.
$$

That lets you write:

> Inspired by axiomatic validation of mechanistic interpretations, we evaluate both representational correspondence and causal replaceability.

That's a reasonable extension rather than pretending your linear probe is itself a complete circuit.

---

# Where paper #2 becomes important

The Typed CoT paper makes a useful conceptual distinction:

A reasoning chain isn't good merely because it obtains the correct final answer.

Its intermediate operations have to constitute valid transformations.

It explicitly frames reasoning as something closer to a typed proof/program and shows examples where the **final numerical answer can be correct despite an invalid intermediate operation**. ([arXiv][4])

That creates an excellent motivation for your paper.

## Answer correctness ≠ reasoning validity

You can divide ProcessBench examples into categories such as:

### A

valid reasoning
correct final answer

### B

invalid reasoning
wrong final answer

and, where available,

### C

invalid intermediate reasoning
but correct final answer.

Then ask:

> Does the internal error mechanism detect local mathematical invalidity, or merely predict whether the final answer will be wrong?

That is a **much stronger research question**.

---

# This could become your nicest result

Train two probes.

### Probe A

Predict:

$$
\text{reasoning step invalid?}
$$

### Probe B

Predict:

$$
\text{final answer wrong?}
$$

Then compare layer-wise representations.

Perhaps:

```text
             Early      Middle       Late

Step error     .           ███         ███
Answer wrong   .            ██         █████
```

You could find that:

> local mathematical-invalidity information appears earlier than final-answer correctness.

Or alternatively:

> the representations are nearly identical, suggesting the model does not distinguish local validity from outcome prediction.

**Both are interesting.**

---

# Paper #3 gives you another easy experiment

Gauderis et al. argue that interpretation quality involves a tradeoff between:

$$
\text{faithfulness}
$$

and

$$
\text{complexity}.
$$

Their broader point is that a useful interpretation shouldn't merely reproduce model behavior—it should do so using a simpler/compressible representation. ([arXiv][5])

So don't stop at:

> "Layer 18 encodes error."

Ask:

> **How many dimensions are needed to represent mathematical invalidity?**

This is extremely easy.

---

# Experiment 3: dimensionality of mathematical error detection

Use PCA or low-rank projections.

Take your hidden states:

$$
h\in\mathbb R^d.
$$

Restrict the error probe to:

* 1 dimension
* 2 dimensions
* 4 dimensions
* 8
* 16
* 32
* full residual stream.

Plot:

$$
\text{error-detection AUROC}
$$

against:

$$
\text{number of dimensions}.
$$

Maybe:

| dimensions | AUROC |
| ---------: | ----: |
|          1 |   .77 |
|          2 |   .83 |
|          4 |   .88 |
|          8 |   .90 |
|         16 |   .91 |
|        128 |  .912 |
|       full |  .914 |

That would give you the result:

> **Most of the mathematical-error signal occupies an extremely low-dimensional subspace.**

Now you've connected directly to compositional/compressive interpretability.

---

# Even better: test one direction across mathematics domains

ProcessBench includes several datasets/categories.

So train

$$
v_{\text{error}}
$$

on one subset.

For example:

$$
\text{GSM8K}
$$

and test it on harder mathematical reasoning datasets.

ProcessBench itself contains examples spanning multiple benchmarks and primarily competition/Olympiad-level mathematics. ([ACL Anthology][2])

The important question becomes:

> Is mathematical invalidity represented by a **general error direction**, or are there domain-specific error mechanisms?

Imagine:

| Train direction | GSM8K | MATH | Olympiad |
| --------------- | ----: | ---: | -------: |
| GSM8K           |   .91 |  .78 |      .70 |
| MATH            |   .76 |  .89 |      .82 |
| Combined        |   .88 |  .87 |      .84 |

That would be a genuinely useful discovery about mathematical reasoning representations.

---

# Now you have a nice "Interpretability for Discovery" angle

The discovery isn't:

> "We discovered theorem X."

It is:

> **We discovered an internal representation associated with mathematical validity.**

More importantly, you can tie it to AI-assisted mathematical research.

For interpretability to help mathematicians discover things, we need to distinguish:

$$
\text{model believes this looks promising}
$$

from

$$
\text{model internally recognizes this argument as mathematically valid}.
$$

Your paper asks whether those internal signals exist and whether they are causally meaningful.

That's directly relevant to using AI for conjecture/proof discovery.

---

# There is a particularly good variation

## **When does the model know that the proof has gone wrong?**

This is even more interesting.

Suppose ProcessBench says:

$$
s_4
$$

is the first incorrect step.

For each prefix:

$$
s_1
$$

$$
s_1,s_2
$$

$$
s_1,s_2,s_3
$$

$$
s_1,s_2,s_3,s_4
$$

$$
s_1,\ldots,s_5
$$

extract the error probe score:

$$
E_k=w^\top h_k.
$$

Plot:

```text
error score

 1.0 |                   █████
     |                ████████
 0.5 |              ██████████
     | _________██████████████
 0.0 +--------------------------
       s1 s2 s3  s4 s5 s6
                  ↑
             first error
```

Then measure:

$$
\Delta t =
t_{\text{internal detection}}
-
t_{\text{annotated error}}.
$$

You may find:

### Case 1

The representation flips **at exactly the erroneous step**.

Strong mathematical monitoring.

### Case 2

It only changes several steps later.

Delayed recognition.

### Case 3

It changes before the labeled error.

Possibly the model recognizes a flawed trajectory before the first explicit incorrect statement.

Case 3 would be especially interesting.

---

# And you don't need to generate anything

You wanted zero data-generation hassle.

So I'd use one of these existing datasets.

## Best: ProcessBench

About **3,400 examples**, explicit first-error annotation. ([GitHub][6])

Ideal for:

> Where does reasoning become invalid?

---

## Alternative: PRM800K

This is much larger:

> roughly **800,000 human step-level correctness labels** over model-generated solutions to MATH problems. ([GitHub][7])

Great if you need more examples.

But I would **not use 800k examples**.

Sample maybe:

$$
N=2{,}000-5{,}000
$$

steps.

That's more than enough.

---

## Alternative: Math-Shepherd

Already formatted as:

```text
problem
completions = [step1, step2, ...]
labels = [true, true, false, ...]
```

and distributed directly as a Hugging Face dataset. ([Hugging Face][8])

Easiest coding.

---

# I would use ProcessBench

Because it gives you the cleanest scientific object:

$$
\boxed{\text{location of first mathematical error}}
$$

rather than just generic positive/negative reward labels.

---

# What NOT to do with those three papers

There are some traps here.

### Don't extend paper #1 by doing another SAT model

They already provide a 2-SAT model and reverse-engineer an algorithm where early layers parse clauses and later layers enumerate Boolean assignments. They even release the trained model and datasets. ([GitHub][9])

You could replicate it, but:

> 2-SAT → 3-SAT

would probably turn into a very difficult mech-interp project.

Not two-day material.

---

### Don't implement full Typed CoT

Their implementation is considerably heavier than it first appears.

The authors report a pipeline involving typed graphs, segmentation, rule labeling, decoding constraints, GPT-5 labeling and other machinery; their experiments used about 200 GSM8K examples and an A100 runtime. ([arXiv][4])

That's the opposite of what you need.

Use the paper as **conceptual motivation**, not infrastructure.

---

### Definitely don't implement category theory from paper #3

It's primarily a theoretical framework, not a ready-made math-discovery benchmark. ([arXiv][10])

Borrow the idea:

$$
\text{interpretation quality}
=
\text{faithfulness}
+
\text{simplicity}
$$

and operationalize it cheaply.

Don't try to build category-theoretic software by Tuesday.

---

# There is one novelty issue to be aware of

Mechanistic interpretation of mathematical correctness is becoming active.

Recent work already shows that correctness can be decoded from hidden representations on GSM8K, and there are 2026 studies using SAEs and activation patching to study CoT reasoning. ([GitHub][11])

There is also recent work showing that process reward models can rely heavily on formatting artifacts instead of actual mathematical content. ([ACL Anthology][12])

So I would **not claim**:

> "We show that correctness is encoded in LLM representations."

That's no longer novel enough.

Your narrower claim should be:

> **We study the temporal emergence and causal representation of the first locally invalid mathematical reasoning step.**

That's different from final-answer correctness.

---

# Your three exact RQs

I would freeze the project around these.

## RQ1 — Emergence

> At which layers and reasoning steps does an LLM internally distinguish valid from invalid mathematical reasoning?

Experiment:

$$
\text{layerwise linear probes}.
$$

---

## RQ2 — Causality

> Is the discovered validity representation causally involved in the model's own judgment of mathematical correctness?

Experiment:

$$
h'=h\pm\alpha v_{\text{error}}.
$$

Measure correct/incorrect logits.

---

## RQ3 — Compressibility

> Can mathematical invalidity be represented by a compact, transferable subspace?

Experiment:

$$
1,2,4,8,16,\ldots
$$

dimensions + cross-dataset transfer.

That maps almost perfectly onto your three papers:

| Paper           | What you borrow                                                                                |
| --------------- | ---------------------------------------------------------------------------------------------- |
| Palumbo et al.  | interpretations require **causal/compositional validation**                                    |
| Typed CoT       | mathematical reasoning should be evaluated through **intermediate validity**, not answer alone |
| Gauderis et al. | interpretations should balance **faithfulness and complexity**                                 |

---

# Your 3 figures

This is enough for the five-page workshop paper.

### Figure 1 — Where errors emerge

Heatmap:

$$
x=\text{reasoning step},
\qquad
y=\text{model layer},
$$

color:

$$
P(\text{error}).
$$

Overlay the annotated first error.

This could be visually excellent.

---

### Figure 2 — causal intervention

$$
\alpha
$$

on x-axis.

$$
\log P(\text{incorrect})-
\log P(\text{correct})
$$

on y-axis.

Compare:

* error direction;
* random direction.

---

### Figure 3 — complexity vs faithfulness

x-axis:

$$
\text{subspace dimension}
$$

y-axis:

$$
\text{AUROC/error detection}.
$$

Maybe add transfer performance.

---

# The actual paper story

Something like:

> As AI systems increasingly participate in mathematical research, reliable discovery requires more than producing correct final answers: models must distinguish locally valid from invalid reasoning. We study this question mechanistically using human-annotated mathematical reasoning traces from ProcessBench. We find that [result], identify a low-dimensional representation predictive of the first invalid reasoning step, and test its causal role through activation interventions. Inspired by axiomatic and compositional approaches to mechanistic interpretability, we evaluate both causal faithfulness and representational complexity.

That's **far stronger** than “I ran probes on GSM8K.”

---

## My recommendation

Given your deadline and Colab constraint, I would now commit to:

**ProcessBench + Qwen2.5-Math-1.5B + residual-stream probing + one activation-steering experiment.**

Skip SAEs.

Skip attention-head circuits.

Skip Lean.

Skip theorem generation.

Skip dataset creation.

Skip typed-CoT implementation.

If you manage those two experiments cleanly, then add the low-rank/compression analysis as the third result.

This is probably the best intersection I've seen so far between **your deadline, mathematics, existing data, real mechanistic interpretability, and this workshop's discovery/validation framing**. ([arXiv][1])

[1]: https://arxiv.org/abs/2407.13594 "[2407.13594] Validating Mechanistic Interpretations: An Axiomatic Approach"
[2]: https://aclanthology.org/2025.acl-long.50/?utm_source=chatgpt.com "ProcessBench: Identifying Process Errors in Mathematical Reasoning - ACL Anthology"
[3]: https://arxiv.org/html/2407.13594v2 "Validating Mechanistic Interpretations: An Axiomatic Approach"
[4]: https://arxiv.org/html/2510.01069v1 "Typed Chain-of-Thought: A Curry-Howard Framework for Verifying LLM Reasoning"
[5]: https://arxiv.org/abs/2605.08934 "[2605.08934] From Mechanistic to Compositional Interpretability"
[6]: https://github.com/QwenLM/ProcessBench?utm_source=chatgpt.com "GitHub - QwenLM/ProcessBench: Official repository for ACL 2025 paper \"ProcessBench: Identifying Process Errors in Mathematical Reasoning\" · GitHub"
[7]: https://github.com/openai/prm800k/blob/main/README.md?plain=1&utm_source=chatgpt.com "prm800k/README.md at main · openai/prm800k · GitHub"
[8]: https://huggingface.co/datasets/trl-lib/math_shepherd?utm_source=chatgpt.com "trl-lib/math_shepherd · Datasets at Hugging Face"
[9]: https://github.com/nilspalumbo/axiomatic-validation?utm_source=chatgpt.com "GitHub - nilspalumbo/axiomatic-validation: Implementation for \"Validating Mechanistic Interpretations: An Axiomatic Approach\" · GitHub"
[10]: https://arxiv.org/html/2605.08934v2 "From Mechanistic to Compositional Interpretability"
[11]: https://github.com/EladMoshe98/cot-interpretability?utm_source=chatgpt.com "GitHub - EladMoshe98/cot-interpretability: Mechanistic interpretability of chain-of-thought reasoning in LLMs — 12 experiments on Gemma-2-2B-IT and Gemma-3-27B (GSM8K). MSc Research Capstone, IE Madrid 2026. · GitHub"
[12]: https://aclanthology.org/2026.eacl-short.31/?utm_source=chatgpt.com "Out of Distribution, Out of Luck: Process Rewards Misguide Reasoning Models - ACL Anthology"

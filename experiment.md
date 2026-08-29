Yes — there is a very plausible workshop paper here, but with the time/compute constraint I would **not** try to invent a new protein model or train a new SAE/transcoder. I would build a tight mechanistic-interpretability study around **why a protein language model makes zero-shot mutation-effect predictions**, using existing pretrained interpretability machinery and ProteinGym as experimental validation.

The workshop is almost tailor-made for this framing. Its stated goal is not generic explainability; it specifically asks whether interpretability can turn learned representations into **novel, testable knowledge**, including methods for unfamiliar modalities, empirical discovery case studies, and well-executed negative/failure results. ([Interpretability for Discovery][1]) The submission is only **5 main-text pages**, double-blind and non-archival, and reproducibility/code availability will explicitly be considered. ([Interpretability for Discovery][2])

## My first-choice paper

### **What Does ESM-2 Use to Score Mutations? Causal Circuits for Zero-Shot Variant Effects**

The central question would be:

> **Can the zero-shot mutation-effect predictions of a protein language model be reduced to a small causal circuit of interpretable protein features, and do those features identify experimentally constrained functional residues?**

That question cleanly connects all four papers you gave me.

### Why the four papers point to this gap

| Prior work                     | What it gives you                                                                                                                                                                             | What is still missing                                                                                                                                                                               |
| ------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Meier et al., NeurIPS 2021** | Shows that masked-language-model probabilities can predict DMS mutation effects zero-shot. ([NeurIPS Proceedings][3])                                                                         | It tells us **that** the model predicts functional effects, not **what internal computation causes those predictions**.                                                                             |
| **ProteinGym**                 | >250 standardized DMS assays, millions of mutants, structures and model scores; an excellent source of external experimental ground truth. ([PubMed][4])                                      | Primarily a benchmark; it does not mechanistically explain the PLMs.                                                                                                                                |
| **ProtoMech**                  | Cross-layer transcoders recover sparse pLM circuits; <1% of latents can retain substantial task performance, and the authors release ESM2-8M/35M models and a Colab workflow. ([alphaXiv][5]) | Their DMS/function circuits are anchored mainly to **supervised function-prediction probes**, rather than directly explaining the native masked-marginal zero-shot computation. ([ResearchGate][6]) |
| **Nainani et al.**             | Gives you causal activation/feature patching for pLMs and finds an early-motif → later-domain mechanism for contact prediction. ([BioRxiv][7])                                                | It studies **contact prediction in two case-study proteins**, rather than mutation-effect/fitness prediction across a DMS landscape.                                                                |

So your gap is not:

> “SAEs find biological features.”

That is already crowded.

It is:

> **“What sparse causal computation inside ESM-2 produces the classical zero-shot mutation-effect signal, and does the resulting circuit correspond to experimentally measured functional constraint?”**

That is much sharper.

---

# Why I think this is the best target

It hits all three pieces the workshop appears to care about:

**Interpretability:** you extract/ablate a sparse internal circuit.

**Scientific knowledge:** the circuit nominates residues/motifs that the model considers causally important.

**Validation:** you compare those nominations against independently measured DMS fitness—not against another model-generated annotation.

That last part is important. Instead of saying:

> “This latent looks like a catalytic-site detector.”

you can say:

> “We extracted this latent without using the DMS labels. Residues receiving high causal attribution are significantly more intolerant to mutation in the experimental DMS.”

That is a much stronger discovery story.

---

# A concrete experiment that is realistic on Colab

Do **not** train ProtoMech's CLT. Its paper reports training on five million UniRef50 sequences, and the CLT has substantially more parameters than ESM2-8M. ([ResearchGate][8])

Use the released checkpoints. The official repository supports **ESM2-8M and ESM2-35M**, has pretrained models, custom circuit discovery, and an interactive Colab notebook. ([GitHub][9])

[ProtoMech repository and Colab workflow](https://github.com/amirgroup-codes/ProtoMech?utm_source=chatgpt.com)

Start with **8M**. Only move to 35M after the entire pipeline works.

### Step 1 — reproduce the ordinary zero-shot score

For a substitution \(x_i^{WT}\rightarrow x_i^{mut}\), use the Meier masked-marginal score:

$$
s_i(mut)
=
\log P(x_i^{mut}\mid x_{\setminus i})
-
\log P(x_i^{WT}\mid x_{\setminus i})
$$

where residue \(i\) is masked.

This is the exact family of scores underlying the original zero-shot mutation-effect work. ([NeurIPS Proceedings][3])

Pick **one primary protein**, not twenty.

I would seriously consider a well-characterized enzyme with a nearly complete single-mutant DMS such as **TEM-1 β-lactamase/Firnberg**. It gives you obvious catalytic motifs/residues for post-hoc validation while not simply repeating ProtoMech's showcased GB1/GFP examples.

Then use **one secondary protein** only if everything works.

---

## Step 2 — trace the zero-shot computation

This is the actual contribution.

Run the masked sequence through the pretrained CLT/SAE representation and calculate an attribution score for latent-token pairs to the mutant-versus-WT logit margin.

Conceptually:

$$
A_{l,j,i}
=
z_{l,j,i}
\frac{\partial s_i}{\partial z_{l,j,i}}
$$

where \(z_{l,j,i}\) is latent \(j\), layer \(l\), residue/token \(i\).

Aggregate the absolute attribution over mutations:

$$
A_{l,j}=\mathbb E_m |A_{l,j,m}|.
$$

Then rank latents and retain the top \(k\).

You do **not** need to claim this particular attribution formula is theoretically optimal. The interesting experiment is what happens when you causally intervene afterward.

---

# The most important plot

Make a **circuit recovery curve**.

x-axis:

> fraction / number of latent-token components retained

y-axes:

1. correlation between **circuit zero-shot scores and full ESM2 scores**
2. correlation between **circuit scores and experimental DMS fitness**

For example:

| model                  | Full ESM score fidelity | DMS Spearman |
| ---------------------- | ----------------------: | -----------: |
| Full ESM2              |                    1.00 |       ρ_full |
| Top 1% circuit         |                       … |            … |
| Top 0.5% circuit       |                       … |            … |
| Top 0.1% circuit       |                       … |            … |
| Random matched circuit |                       … |            … |

If a very small circuit preserves most of the DMS-ranking signal, that is immediately paper-worthy evidence.

ProtoMech already reports highly compressed circuits on supervised family/function tasks, so the conceptual question here becomes whether the same phenomenon holds for the **native zero-shot mutation mechanism**. ([alphaXiv][10])

---

# Step 3 — sufficiency and necessity

Don't stop at attribution.

This workshop is specifically sensitive to the difference between an attractive feature visualization and evidence that the feature matters. The workshop page itself says discovery requires more than a compelling visualization. ([Interpretability for Discovery][1])

Do two interventions.

### Sufficiency

Keep only your top \(k\) latent components.

Ask:

> How much of the zero-shot mutation-ranking behavior remains?

### Necessity

Take the otherwise complete replacement model and **ablate those same components**.

Compare against 20–100 random matched-size ablations.

If:

$$
\Delta \rho_{\mathrm{top\ circuit}}
\gg
\Delta \rho_{\mathrm{random}},
$$

you have causal evidence, not just a correlation.

This also directly follows the methodological direction of the Nainani paper, whose released code already supports performance recovery curves, feature ablation and path patching. ([GitHub][11])

[Nainani et al. pLM circuit code](https://github.com/NainaniJatinZ/plm_circuits?utm_source=chatgpt.com)

---

# Step 4 — the scientific-discovery validation

Now take the attribution back to biology.

For every residue \(i\), calculate something like

$$
C_i=\sum_{l,j}|A_{l,j,i}|.
$$

This produces a **mechanistic importance score per residue**.

Separately, from the DMS, estimate experimentally observed mutational intolerance, for example:

$$
D_i=-\operatorname{median}_{a\ne WT}
\text{fitness}(i\rightarrow a).
$$

Now ask:

### Does circuit importance predict experimentally constrained residues?

Compute:

* Spearman \(C_i\) versus \(D_i\)
* AUROC for top 10% most intolerant residues
* enrichment around known active/catalytic/binding residues
* randomization/bootstrap confidence intervals.

This is the piece I would emphasize in the title/abstract.

You're not merely explaining a PLM prediction.

You're asking whether **reading the model's internal computation reveals experimentally validated functional constraint**.

That is almost exactly the workshop's stated premise. ([Interpretability for Discovery][1])

---

# Your paper has a good outcome even if the result is negative

This is one reason I like this project under a four-day deadline.

Suppose you find:

> A <1% latent circuit faithfully reproduces ESM2's zero-shot score, but its highlighted residues correlate weakly with actual experimental intolerance.

That's still interesting.

It says:

> **Mechanistic fidelity to the model is not the same as scientific validity.**

That would fit the workshop's explicit request for analyses of **misleading interpretations, failed validation, and limitations of interpretability for reliable discovery**. ([Interpretability for Discovery][2])

So you have two possible publishable narratives:

**Positive result:** sparse causal circuits reveal experimentally constrained motifs.

**Negative result:** sparse causal circuits explain the model extremely well but do not reliably reveal biological constraint.

Both fit.

That substantially lowers project risk.

---

# One complication you need to be aware of

There is already a lot of 2025–26 work around interpretable PLMs.

For example, InterPLM finds thousands of biological SAE features in ESM2 and demonstrates annotation discovery/steering. ([DOI][12]) Adams et al. similarly explicitly frame SAE analysis as moving from mechanistic interpretability toward “mechanistic biology.” ([PubMed Central (PMC)][13])

More importantly, a **2026 PLM-SAE preprint already studies variant-effect prediction and improves zero-shot VEP through sparse feature steering**. ([Sciety][14])

And a recent **ProGenMech** paper traces zero-shot fitness circuits in the autoregressive ProGen3 model. ([arXiv][15])

So do **not** frame your claim as:

> “We are the first to interpret mutation prediction using sparse features.”

That would be vulnerable.

Frame it much more specifically:

> **We causally trace the native masked-marginal variant-effect computation of a masked protein language model and test whether the recovered mechanism itself predicts independently measured functional constraint.**

That is substantially more defensible.

---

# My second-choice idea: mechanistic epistasis

This one may ultimately be more exciting scientifically, but it is riskier for this deadline.

### **Do shared protein-language-model circuits predict epistasis?**

Hypothesis:

> Two mutations produce strong experimental epistasis when they perturb overlapping or causally coupled PLM circuit features.

For mutations \(a,b\), obtain latent perturbation vectors:

$$
\Delta z_a,\quad\Delta z_b.
$$

Then calculate measures like:

$$
\text{overlap}(a,b)
=
\cos(\Delta z_a,\Delta z_b)
$$

or overlap of top circuit nodes.

Compare that with experimental epistasis:

$$
\epsilon_{ab}
=
f_{ab}-f_a-f_b+f_{WT}
$$

with whatever transformation is appropriate for that DMS.

This could lead to a beautiful conclusion:

> **PLM internal circuits organize experimentally measurable genetic interactions.**

That is very strongly aligned with “interpretability for discovery.”

But there are two problems.

First, there is already substantial work showing that PLMs encode structural/functional epistasis. ([BioRxiv][16]) There is also a very recent benchmark arguing that zero-shot methods perform poorly on strongly epistatic ProteinGym variants. ([GitHub][17]) And higher-order PLM interactions have been studied using Fourier approaches. ([arXiv][18])

Your **circuit-level explanation** would still be novel, but the positioning becomes considerably harder.

Second, estimating experimental epistasis correctly can become messy because DMS measurements and nonlinear experimental scales matter.

With four days, I'd rank it:

**scientific upside: 9.5/10**
**workshop fit: 10/10**
**four-day execution risk: 8/10**

I'd save this as the follow-on full paper unless your lab already has an epistasis pipeline working.

---

# Third-choice idea: use interpretability to predict model failure

This is probably the safest experiment.

### **Circuit Instability Predicts When Zero-Shot Protein Fitness Predictions Fail**

Ask whether unreliable ESM2 mutation predictions have:

* more diffuse attribution,
* higher circuit entropy,
* lower CLT reconstruction fidelity,
* disagreement between layers,
* unstable circuits under small sequence perturbations.

For each mutation define something like:

$$
H(m)
=
-\sum_j p_j(m)\log p_j(m)
$$

where \(p_j\) is normalized latent attribution.

Then test whether high \(H(m)\) predicts high absolute DMS error.

The story would be:

> **Interpretability isn't only useful for explaining predictions; the structure of the explanation can tell us when the model should not be trusted.**

This fits the workshop's failure-case/evaluation category very well. ([Interpretability for Discovery][1])

It's also much cheaper than trying to discover biological mechanisms across many proteins.

Novelty is probably lower than idea #1, but execution risk is lower.

---

# Fourth idea: cross-protein transfer of circuits

This is probably the most “discovery workshop” idea but I would not start it now.

### **Are functional circuits conserved across protein homologs?**

Extract a circuit in one protein/family and ask whether the same latent pathway:

* appears in homologs,
* localizes to equivalent motifs,
* predicts mutation-sensitive positions,
* survives sequence divergence.

A result like

> “the same causal circuit is reused across evolutionarily distant enzymes”

would be excellent.

But gathering homologs, alignments, equivalent residue positions, annotations and suitable DMS assays could eat the entire submission window.

Save it unless you already have those datasets.

---

# Ranking the projects for this deadline

| Idea                                      | Workshop fit | Novelty | Colab feasibility | Deadline risk | Recommendation              |
| ----------------------------------------- | -----------: | ------: | ----------------: | ------------: | --------------------------- |
| **Causal circuits of ESM2 zero-shot VEP** |           10 |     8–9 |                 9 |             4 | **Do this**                 |
| Mechanistic explanation of epistasis      |           10 |       9 |                 7 |             8 | Strong future/full paper    |
| Circuit features predict VEP failures     |            9 |     7–8 |                10 |             3 | **Excellent backup**        |
| Cross-homolog circuit transfer            |           10 |       9 |                 6 |             9 | Too risky now               |
| “SAE features correlate with biology”     |            7 |       3 |                10 |             2 | **Don't submit this alone** |
| Train a new SAE/CLT                       |            8 |  varies |                 2 |            10 | **Absolutely don't do now** |

---

# What I would actually execute

I'd reduce everything to **three research questions**.

### RQ1 — Is zero-shot mutation scoring mechanistically sparse?

> Can a small subset of interpretable latent-token features recover ESM2's masked-marginal mutation scores?

Plot: recovery versus circuit size.

### RQ2 — Are those features causally important?

> Does ablating them damage zero-shot behavior significantly more than matched random ablations?

Plot: top-circuit vs random ablation.

### RQ3 — Does the mechanism correspond to experimental biology?

> Do residues carrying high circuit attribution have stronger experimentally measured DMS effects?

Plot: circuit importance versus experimental intolerance + one biological case study.

That is enough for a workshop paper.

Do **not** add generation, protein design, multiple architectures, ten interpretability methods or 50 datasets.

---

# Model/data choice

I would use:

**Primary model:** ESM2-8M + released ProtoMech CLT.

Why: tiny, cheap, released checkpoints, and the repository explicitly supports Colab. ([GitHub][9])

**Optional robustness:** ESM2-35M only after everything else works.

**Primary dataset:** one well-characterized ProteinGym single-mutant DMS.

**Secondary dataset:** one more protein with a different type of function.

ProteinGym is perfect here because it already provides standardized experimental data and structures. ([PubMed Central (PMC)][19])

You can legitimately write:

> “DMS measurements were withheld during circuit discovery and used only as experimental validation.”

That's a particularly clean methodological choice.

---

# Baselines I would require

Don't let the evaluation balloon. Four baselines are enough:

1. **Full ESM2 masked-marginal score**
2. **Random matched-size latent circuit**
3. **Activation magnitude only**
4. **Gradient magnitude only**

Then show your activation×gradient/circuit-selection method.

If easy, add entropy/conservation.

Don't spend a day reproducing EVE, TranceptEVE, AlphaMissense, etc. Your paper is about the **mechanism of ESM**, not winning ProteinGym.

---

# Minimum successful result

You don't need SOTA performance.

A workshop-quality result could simply look like:

> “Across two DMS assays, 0.5–2% of CLT latents recover 70–90% of ESM2's zero-shot ranking. Top-circuit ablation produces a 4× larger performance drop than random ablation. Residue-level circuit attribution is significantly enriched at experimentally mutation-intolerant sites.”

That's a clean paper.

Even:

> “Sparse circuits recover 85% of ESM2 behavior but show no stronger correspondence to DMS constraint than activation magnitude”

could become a strong negative-result workshop submission.

---

# Your figure plan

For five pages I'd aim for **three figures**, maximum four.

### Figure 1

**Method diagram**

Sequence → masked residue → ESM2 → sparse cross-layer circuit → mutant/WT logit difference → DMS validation.

### Figure 2

**Mechanistic sparsity**

Performance-recovery curve.

Top circuit vs random circuit.

### Figure 3

**Scientific validation**

Residue circuit attribution vs DMS intolerance, plus one sequence/structure visualization of the strongest motifs.

Optional small panel:

top-latent ablation versus random ablation.

That's enough.

---

# Suggested title variants

I'd avoid making the title excessively grand.

My favorites:

**Causal Circuits Underlying Zero-Shot Variant Effect Prediction in Protein Language Models**

or

**From Mutation Scores to Mechanisms: Tracing Zero-Shot Variant Effects in ESM-2**

or, more discovery-focused:

**Reading Functional Constraint from Protein Language Model Circuits**

The first is safest and clearest.

---

# A possible abstract-level claim

The paper should aim to establish something like:

> Protein language models can predict mutation effects zero-shot, but the internal computation supporting these predictions is poorly understood. We trace sparse causal circuits underlying masked-marginal variant scoring in ESM-2 using pretrained cross-layer sparse representations. We find that a small fraction of latent-token components recovers much of the original zero-shot ranking, while targeted ablations cause substantially larger degradation than random controls. Importantly, circuit attribution is enriched at experimentally mutation-sensitive residues in deep mutational scanning data, connecting internal model mechanisms to independently measured functional constraints.

Obviously the quantitative statements only stay if the results support them.

---

# A very practical schedule

I would operate under an internal rule: **freeze experiments by the end of Aug. 31**.

### Aug. 29

Get one end-to-end pipeline running.

* download one ProteinGym assay
* load ESM2-8M
* reproduce zero-shot masked-marginal scores
* compute Spearman against DMS
* load ProtoMech/Nainani interpretability weights
* verify you can access latent activations

If this isn't working by tonight, switch immediately to the simpler SAE implementation instead of debugging architecture code for a day.

### Aug. 30

Implement circuit attribution and intervention.

Get:

* top-k latent ranking
* recovery curve
* random baseline
* necessity ablation.

This is the make-or-break day.

### Aug. 31

Biological validation + replication.

* residue-level DMS constraint
* functional-site enrichment
* second protein if time permits
* bootstrap/random controls.

**Freeze results.**

### Sept. 1

Write Results and Methods first.

Generate camera-ready figures.

Put secondary analyses into appendix rather than expanding the paper.

### Sept. 2

Intro, related work, discussion, limitations, proofreading.

The CFP requires a **responsible-use statement**, and missing it is explicitly grounds for desk rejection. ([Interpretability for Discovery][2])

Also make sure the manuscript, repository, Hugging Face names, paths, acknowledgements, etc. are fully anonymized because review is double-blind. ([Interpretability for Discovery][2])

Treat Sept. 2 as your actual cutoff and don't use AoE as planned writing time.

---

# What I would not do

For this deadline I would explicitly rule out:

* training a new pLM;
* training a new CLT;
* training a new SAE;
* ESM2-650M-scale experiments unless absolutely necessary;
* a 100-assay ProteinGym benchmark;
* wet-lab validation;
* full protein-design experiments;
* trying to outperform current ProteinGym SOTA;
* a generic comparison of SAE versus neuron interpretability;
* an “LLM automatically labels protein features” paper.

Those are all ways to end up with 40% of five experiments instead of one finished result.

---

## One more reason I like the first idea

ProtoMech's own stated limitations include reliance on **manual interpretation and existing biological annotations**, which restricts biological discovery. ([ResearchGate][6])

Your paper can sidestep that weakness by defining biological relevance quantitatively:

> **Does the circuit predict an independent experimental measurement?**

You don't need to manually decide that latent 1,827 “looks like” a domain. The DMS gives you a falsifiable test.

That is a better fit for **Interpretability for Discovery** than another feature-visualization paper.

### My recommendation

If I were allocating your next four days, I would commit to:

**“Causal Circuits Underlying Zero-Shot Variant Effect Prediction in ESM-2”**

using **ESM2-8M + an already-trained ProtoMech CLT**, **one primary ProteinGym DMS**, **top-k sufficiency/necessity experiments**, and **DMS mutation intolerance as independent biological validation**.

Keep **“circuit structure predicts model failure”** as your fallback. If the biological discovery signal turns out weak, don't throw the project away—pivot the paper into a rigorous failure-case result, which this CFP explicitly welcomes. ([Interpretability for Discovery][2])

That gives you a project where almost every empirical outcome can be turned into a coherent five-page submission, which is exactly what you want under this deadline.

[1]: https://interpretability4discovery.github.io/about.html "About: Interpretability for Discovery"
[2]: https://interpretability4discovery.github.io/cfp.html "Call for Papers: Interpretability for Discovery"
[3]: https://proceedings.neurips.cc/paper/2021/hash/f51338d736f95dd42427296047067694-Abstract.html "Language models enable zero-shot prediction of the effects of mutations on protein function"
[4]: https://pubmed.ncbi.nlm.nih.gov/38106144/?utm_source=chatgpt.com "ProteinGym: Large-Scale Benchmarks for Protein Design and Fitness Prediction - PubMed"
[5]: https://www.alphaxiv.org/ja/abs/2602.12026 "Protein Circuit Tracing via Cross-layer Transcoders | alphaXiv"
[6]: https://www.researchgate.net/publication/400742139_Protein_Circuit_Tracing_via_Cross-layer_Transcoders?_tp=eyJjb250ZXh0Ijp7InBhZ2UiOiJzY2llbnRpZmljQ29udHJpYnV0aW9ucyIsInByZXZpb3VzUGFnZSI6bnVsbCwic3ViUGFnZSI6bnVsbH19&utm_source=chatgpt.com "(PDF) Protein Circuit Tracing via Cross-layer Transcoders"
[7]: https://www.biorxiv.org/content/10.1101/2025.08.22.671739v1?utm_source=chatgpt.com "Mechanistic evidence that motif-gated domain recognition drives contact prediction in protein language models | bioRxiv"
[8]: https://www.researchgate.net/publication/400742139_Protein_Circuit_Tracing_via_Cross-layer_Transcoders?utm_source=chatgpt.com "(PDF) Protein Circuit Tracing via Cross-layer Transcoders"
[9]: https://github.com/amirgroup-codes/ProtoMech "GitHub - amirgroup-codes/ProtoMech: (ICML 2026) Official code repository for Protein Circuit Tracing via Cross-layer Transcoders · GitHub"
[10]: https://www.alphaxiv.org/abs/2602.12026v1?utm_source=chatgpt.com "Protein Circuit Tracing via Cross-layer Transcoders | alphaXiv"
[11]: https://github.com/NainaniJatinZ/plm_circuits?utm_source=chatgpt.com "GitHub - NainaniJatinZ/plm_circuits: SAE Circuit Discovery for Protein Language Models · GitHub"
[12]: https://doi.org/10.1038/s41592-025-02836-7?utm_source=chatgpt.com "InterPLM: discovering interpretable features in protein language models via sparse autoencoders | Nature Methods"
[13]: https://pmc.ncbi.nlm.nih.gov/articles/PMC11839115/?utm_source=chatgpt.com "From Mechanistic Interpretability to Mechanistic Biology: Training, Evaluating, and Interpreting Sparse Autoencoders on Protein Language Models - PMC"
[14]: https://sciety.org/articles/activity/10.64898/2026.05.12.724472?utm_source=chatgpt.com "Improving Variant Effect Prediction by Steering Sparse Mechanistic Features in Protein Language Models | Sciety"
[15]: https://arxiv.org/abs/2606.16044?utm_source=chatgpt.com "Circuit Tracing in Autoregressive Protein Language Models"
[16]: https://www.biorxiv.org/content/10.1101/2025.09.14.676130v1?utm_source=chatgpt.com "Protein Language Models Capture Structural and Functional Epistasis in a Zero-Shot Setting | bioRxiv"
[17]: https://github.com/kalininalab/epistasis_proteingym?utm_source=chatgpt.com "GitHub - kalininalab/epistasis_proteingym · GitHub"
[18]: https://arxiv.org/abs/2405.06645?utm_source=chatgpt.com "On Recovering Higher-order Interactions from Protein Language Models"
[19]: https://pmc.ncbi.nlm.nih.gov/articles/PMC10723403/?utm_source=chatgpt.com "ProteinGym: Large-Scale Benchmarks for Protein Design and Fitness Prediction - PMC"

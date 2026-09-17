# Improving likelihood reconstruction from the physics

## 1. What the detector measures

A muon undergoes many small Coulomb deflections while crossing matter. The approximately Gaussian central scattering distribution has a projected angular width that grows with the square root of the traversed radiation lengths and decreases with momentum. The familiar approximation is

```text
t = ∫ ds / X₀(s)
θ₀ ≈ [13.6 MeV / (βpc)] √t [1 + 0.038 ln(t)]
```

Here radiation length X₀ is expressed as a length in centimetres, and the reconstructed quantity is ρ = 1/X₀ in cm⁻¹. It is not mass density or an unambiguous atomic-number measurement. The Gaussian description concerns the central distribution; large-angle tails motivate robust estimators. See the [Particle Data Group, passage through matter, §34.3](https://pdg.lbl.gov/2023/reviews/rpp2022-rev-passage-particles-matter.pdf).

The implemented linear scattering-power approximation drops the logarithmic term and takes β=1:

```text
Var(dθₐ) = kᵢ ρ(s) ds,       kᵢ = E[(C / pc)² | incoming zenith]
```

All tested variants retain the codebase's C=15 MeV. Replacing 15 by 13.6 alone would change the inferred scale by approximately (15/13.6)² = 1.216 without fixing path or statistical errors. Adding the logarithmic correction separately to every arbitrarily sized voxel would introduce an artificial dependence on voxel size. Neither is presented as a tested improvement here.

## 2. Why displacement is useful, in addition to angle

A deflection near the entrance moves the downstream hit farther than the same deflection near the exit. The final angle measures accumulated scattering; its correlation with displacement carries depth information. The joint angle–displacement covariance is the basis of the Schultz maximum-likelihood reconstruction. Its small-angle and approximate-path assumptions matter for interpretation. See [Schultz et al., Statistical Reconstruction for Cosmic Ray Muon Tomography (2007)](https://www.hep.phy.cam.ac.uk/~hommels/CosmicConcrete_dir/MuonPapersForBart/Paper_IEEE_16_8_Aug2007_StatImageReco.pdf).

For the discretization in this repository, let L be the distance in one voxel and T the remaining distance from that voxel's downstream end to the observation plane. Integrating a unit angular kick along the segment gives the following weights directly:

```text
Wθθ = ∫₀ᴸ du              = L
Wθx = ∫₀ᴸ (T+u) du        = LT + L²/2
Wxx = ∫₀ᴸ (T+u)² du       = LT² + L²T + L³/3

Σᵢ = Rᵢ + kᵢ ∑ⱼ ρⱼ Wᵢⱼ
```

R is measurement-error covariance. Side-exiting paths must retain the remaining travel to the observation plane, even after the path has left the reconstruction box. The existing code already has this correction, which every tested variant retains.

For two transverse observations d₁ and d₂ with a common 2×2 covariance, the Gaussian negative log likelihood, apart from constants, is

```text
ℓᵢ = log det Σᵢ + ½(d₁ᵀ Σᵢ⁻¹ d₁ + d₂ᵀ Σᵢ⁻¹ d₂).
```

The displacement coordinate is divided by 100 cm in the experiment for numerical conditioning. Covariance moments and displacement noise are transformed consistently. This does not rescale the output density.

## 3. What limits the existing implementation

### Sparse information at the nominal resolution

The output grid has 262,144 voxels. A single track supplies four correlated observations spread over many voxels; four numbers do not locate independent scattering in every voxel along its path. In this new panel the median number of tracks through an occupied fine voxel is **one**. About half the fine voxels have no traced path. Fine-grid spikes are therefore an expected statistical failure mode.

Pooling nearby estimates makes a physical assumption: material varies over a spatial scale larger than one unsupported voxel. It exchanges nominal resolution for reduced variance. A returned 64³ array does not imply independently recovered 1.5625 cm structures.

### Coordinate convention

The installed likelihood computes differences between separate laboratory x–z and y–z angles. The candidate constructs an orthonormal basis perpendicular to the physical incoming direction and projects the observations into it. Its angular pair has norm equal to the actual three-dimensional deflection angle. This makes the isotropic observation model invariant to the arbitrary choice of transverse basis.

That observation-level invariance does **not** make the whole reconstruction exact or invariant to rotating the detector box. The voxel grid, z observation planes, PoCA path approximation, lever arms and diagonal detector-noise assumptions remain. The coordinate change is consequently an ablation to test, not an a priori guarantee of lower image error.

### Median versus likelihood maximization

The inherited update computes one contribution per track and voxel, then uses a voxel-wise median when at least three contributions exist. This is a robustness modification, not exact maximization of the Gaussian likelihood. At lower occupancy it uses a mean. More iterations of this rule need not monotonically improve either Gaussian likelihood or image accuracy.

The fast control was tested numerically against the installed solver, including the same initializer, covariance jitter, lower bound and upper bound. The physical frame and iteration-count ablations change one part at a time.

### Noise and model mismatch

The baseline's angular variance is 10⁻¹² rad² plus numerical jitter 10⁻⁹ rad²; its effective angular floor is therefore about 31.6 microradians, not the configured 1 microradian alone. This inherited floor is held constant in every current ablation. Actual fitted detector errors can have angle–position correlations, which are not calibrated by this experiment.

The fitted paths are reconstructed from the observed incoming and outgoing lines. They are not measured internal trajectories. This also means that reserved-track likelihoods are **conditional on paths inferred from the reserved observations**. They are useful consistency scores, but are not an unconditional detector-prediction benchmark.

## 4. The methods tested and their motivation

**Raw10:** the installed full-covariance median estimator, 10 iterations, with the common external momentum coefficient. It is the unpooled algorithm control.

**Pool10:** Raw10 followed by exposure-weighted pooling. The exposure Eⱼ is the total traced path length in a voxel. Sum Eρ and E into 32³ cells, smooth each with a Gaussian of σ=3.125 cm, divide, and repeat each coarse value into a 2×2×2 fine block:

```text
ρpool = Gσ[coarsen(Eρ)] / Gσ[coarsen(E)].
```

This preserves a constant field wherever pooled exposure is positive. It gives better-supported estimates more weight. It is post-reconstruction regularization; Eρ is a weighted fitted estimate, not a measured sufficient statistic.

**Pool40:** the same procedure after 40 median iterations; tests early stopping.

**Local pool10 / Local pool40:** use the incoming transverse coordinates, retaining the corresponding 10 or 40 median iterations and pooling.

**Mean local pool40:** uses the mean update with the same local frame and pooling. This tests whether the robustness of the median is helping or whether its bias is dominant.

**Gaussian ML:** optimize the covariance likelihood directly over nonnegative 32³ voxel densities. Coarse covariance weights sum the original fine segments before fitting. There is no spatial penalty. This is a control for the optimizer and parameterization change; it also retains finite density bounds.

**Gaussian quadratic:** add a nearest-neighbor squared-difference penalty. It encodes slowly varying material and tends to blur boundaries.

**Gaussian Huber:** use a differentiable, edge-preserving absolute-difference penalty. Small jumps are smoothed; large jumps receive a weaker penalty than under a quadratic model. The implemented penalty is pseudo-Huber shaped:

```text
min over 10⁻⁹ ≤ ρⱼ ≤ 20:
    meanᵢ ℓᵢ(ρ) + 0.3 meanedges ψ((ρⱼ−ρₖ)/(1 cm⁻¹))

quadratic: ψ(u) = u²/2
Huber:     ψ(u) = √(u²+0.1²) − 0.1
```

These two penalties share a numerical coefficient but not the same curvature or effective strength. This is a fixed candidate comparison, not a tuned contest between penalty families. The neighbor graph is on the fixed 3.125 cm grid; the coefficient is not resolution independent.

**Student Huber:** replace the four-dimensional Gaussian observation law by a Student distribution with ν=4 and the same covariance parameterization. With q the joint Mahalanobis distance, the covariance-dependent loss is `log det Σ + 4 log(1+q/2)`. Large residuals have less influence. It can handle tails, but can also treat informative low-momentum scattering as an outlier and bias density.

**Angular Huber:** fit only the two transverse deflection angles with the same edge-preserving penalty. This tests whether displacement modeling error costs more than its depth information helps. For fair comparison, its final maps are also scored using the common full-covariance reserved-track scores.

All five direct fits start from Pool10, use the same density bounds and a 160-iteration limit. A separate continuation experiment allows 1,000 additional iterations on Funny 0 and validation record 11, for all five methods and both training/full fits.

## 5. The momentum–material ambiguity

Without a measured momentum or an independent population prior, the scattering data identify a product of density and inverse momentum squared. In this model, multiplying every ρ by a and dividing every k by a leaves every predicted covariance unchanged. The synthetic test verifies that degeneracy numerically.

The primary comparison uses exactly the same external spectrum as the previous detector-only study: the published combined Reyna fit, conditioned on measured incoming zenith, integrated over 1 < p < 2000/cos(zenith) GeV/c with 64 logarithmic quadrature nodes and the correct dp Jacobian. The model uses the expected scattering coefficient; it does not fit or read a momentum per event. [Reyna (2006), equations 2–3](https://arxiv.org/html/hep-ph/0604145).

This is an incident surface spectrum. Object attenuation, energy loss, overburden and momentum-dependent detector acceptance can change the spectrum of surviving tracks. Absolute density is conditional on that assumption. A fitted scalar correction against these objects' labels would hide this issue, so none is applied.

The default configuration elsewhere in the codebase uses 1743.460 MeV/c, originally derived from simulated Funny momenta. **Raw10 here is the production estimator under the common external prior, not a rerun of that different configured momentum scale.** This keeps the algorithm comparison interpretable.

## 6. Experimental design and interpretation

- Three development objects: Funny 0 and prediction records 0 and 5, already inspected in earlier work.
- Six additional validation records, fixed before extracting or decoding their labels: 11, 31, 71, 131, 211 and 311 from the original unfiltered validation TFRecord. They are additional objects relative to the prior local manifests, not a separately collected dataset.
- Select 8,192 rows per object with seed 0. Normalize and retain only the two positions and two directions. No simulated momentum magnitude or material label reaches reconstruction.
- Partition selected rows into 6,144 training and 2,048 reserved events with seed 20260915. Only rows with usable in-volume paths enter the conditional likelihood. Full image metrics use separate all-track refits.
- Freeze the 11 candidate settings. Save every main full map before evaluating labels. The longer-budget diagnostic was also specified before new validation evaluation.
- Evaluate unsmoothed truth on the full canonical 64³ grid: mean squared error (MSE), material-only MSE, air-only MSE, and material-versus-air ROC AUC. Include an all-zero volume as a sanity control.
- Score every method on the same reserved events using both the common Gaussian and common Student full-covariance law. Constants are omitted within each law; never compare a Gaussian number directly with a Student number.
- Average image metrics equally over objects. The exploratory bootstrap resamples whole objects, not correlated voxels, and gives a paired MSE difference from Pool10. Six validation objects are too few for a broad generalization claim or a multiplicity-adjusted winner declaration.

Lower full-volume MSE alone is insufficient: an empty estimate can score well when material is sparse. A recommendation must also disclose material error, air leakage, localization and failures on individual objects. Better likelihood fit is likewise not proof of better physical density under an approximate momentum/path model.

## 7. What stronger physical modeling would require next

The next model extensions are a detector-calibrated full error covariance, an accepted rather than incident momentum spectrum, and integration over uncertain internal paths. An energy-loss model needs additional material information because inverse radiation length alone does not specify stopping power. A Highland correction must be applied consistently to accumulated thickness rather than independently to arbitrary voxel fragments.

These are proposed next experiments. The current study tests neither a calibrated detector-error model, energy-loss correction nor uncertain-path integration. It measures which of the specified reconstruction changes help with the available data.


---

# Findings and practical recommendation

The largest improvement comes from matching spatial resolution to the available information. On six additional validation objects, mean full-volume MSE changes from **1.236006 to 0.261319 cm⁻²** with exposure pooling, a **78.86% reduction**. This reproduces the earlier pooling finding on additional records.

The incoming-transverse, 40-iteration median-plus-pooling variant gives the lowest mean MSE in the frozen panel: **0.248528 cm⁻²**, **4.89% below Pool10** and **79.89% below Raw10**. It beats Pool10 on **3/6** additional objects, rather than uniformly. Across all nine objects its MSE is 0.256426, versus 0.275137 for Pool10.

Compared with Pool10, the transverse 40-iteration variant reduces mean material-only MSE from **0.674436 to 0.552292** and raises localization AUC from **0.7660 to 0.7899**. Its air-only MSE rises from **0.090358 to 0.122586**. This is a material-recovery versus false-positive tradeoff.

The 10-iteration transverse variant has the lowest air-only MSE among these candidates, **0.074894**, but leaves more material underestimated. For balanced density error, the 40-iteration version is the provisional candidate. For suppressing apparent material in empty space, the 10-iteration version merits comparison. Neither has established reliable small-feature resolution.

The all-zero-volume control has mean MSE **0.393425**. The provisional candidate beats it on all six additional objects. The paired mean MSE difference between transverse Pool40 and Pool10 is -0.012791 cm⁻²; the exploratory object-bootstrap 95% interval is [-0.024620, -0.002494]. This is not a simultaneous confidence statement over 11 inspected candidates.

## Which ideas failed in this test

- **More iterations alone:** Pool40 reaches 0.264736, slightly worse full-volume error than Pool10. The coordinate change and the iteration count must be judged together.
- **Mean update:** transverse mean-plus-pooling reaches 0.251855; median remains slightly better on average.
- **Direct Gaussian ML:** 0.664741. It assigns excessive density to poorly constrained coarse voxels.
- **Quadratic and Huber penalties:** 0.627019 and 0.642845. These fixed settings do not control the failure sufficiently. This does not rule out other regularization strengths or better forward models.
- **Student plus Huber:** 0.318440. Robust tails help relative to direct Gaussian fitting but fail to beat simple Pool10 on average.
- **Angular only:** 0.430880. Removing displacement is not the answer under these settings.

## Why likelihood score and image accuracy disagree

The common reserved-track Gaussian score is best for Pool40, whereas the common Student score is best for transverse Pool10. The lowest mean image error is achieved by transverse Pool40. The score measures consistency under a conditional approximate forward model; the image metric measures physical density relative to simulation truth. These are different targets, especially with unmeasured momenta and observation-derived paths.

## Implementation status

All work is isolated under `experiments/likelihood_physics_study`. Production reconstruction and configuration remain unchanged. An opt-in detector-only function, `reconstruct_local_pooled`, provides the provisional method, the physical output units and its provenance. It is not fitted to material labels. No new affine calibration, threshold, material assignment or per-object parameter is applied.

The complete evidence is in `results/metrics.json`, `results/summary.json`, the saved full/training maps and reserved-track scores. `results/protocol.json` records the initial experimental settings. `results/convergence_protocol.json` records the added optimizer-budget diagnostic before new validation evaluation.

## Optimization and verification

All 160-step direct fits reached their iteration budget. Continuing all five methods on Funny 0 and validation 11 for up to 1,000 additional steps changes the full-volume MSE by less than 0.0001 in nine of ten comparisons. The exception is Student–Huber on validation 11, which worsens by about 0.0041. Some optimizers report convergence and others still hit the limit. These checks support the observed image-error conclusion on those two objects; they do not establish global likelihood optima.

The ten focused numerical tests pass: four likelihood-gradient cases, both penalty gradients, observation rotation invariance, synthetic density recovery and density–momentum degeneracy, coarse covariance integral preservation, and parity with the installed median solver. The existing suite passes 44 tests; three optional data integration tests are skipped in that run. A separate run enabling the real Funny TFRecord integration test passes all 11 tests (10 focused plus 1 integration). The experimental study itself performs real TFRecord extraction and reconstruction on six additional records.

**Recommendation:** retain exposure pooling as the main improvement, provide transverse-coordinate Pool40 as an experimental option, and preserve Pool10/Local Pool10 as explicit controls. Before promoting a new default, validate the accepted momentum spectrum and detector covariance on independent data and quantify the feature-size versus false-positive tradeoff. The present results support a useful regularized reconstruction, not uniquely calibrated material identification.

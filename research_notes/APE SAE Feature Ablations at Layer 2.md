**Status**: Essentially finished
**Question:** Are the most row/column-selective APE SAE features causally responsible for positional structure as measured by SSDC? 
**Result:** Strong evidence. Key finding: Ablating 6-12 positional features significantly reduces SSDC and linear probe accuracy.
**Implication:** APE likely encodes position using positional features that are linearly decodable and sparse.

Keep in mind that layer 2 is the third layer of the model, since indexing starts at layer 0.
The SAE used here has an explained variance of ~65-70%, and an L0 roughly at 40, with no dead fractions.

#### Baseline (No SAE reconstruction)
[0.4021864668375372,
 0.5113754489089868,
 0.732712016554071,
 0.7787428108622159,
 0.776722882172906,
 0.6944824912400231,
 0.5670223825060345,
 0.4550163312871133,
 0.38667297198444944,
 0.3473183925931541,
 0.307362666857998,
 0.19225445994997062]

#### Second Baseline (SAE reconstruction)
[0.42812035945457994,
 0.5145299652204215,
 0.7501712365354638,
 0.7484621831268254,
 0.7789714084650947,
 0.7459725519832677,
 0.6332765188870252,
 0.5400635223482022,
 0.4625400297613615,
 0.4426067468403026,
 0.4044629897463009,
 0.27840800143104305]

It is interesting to note that the SAE reconstruction at layer 2 doesn't cause any significant change to SSDC. This gives us more confidence our next results, as it decreases the chances that the effects we observe are simply an artifact of poor SAE reconstruction.

#### Features 3140, 2208, 589, 1533, 1352, 3591 ablated
[0.4021399900276628,
 0.511055021180181,
 0.6207174678557601,
 0.6120057375433284,
 0.6650656764609583,
 0.6305609436125373,
 0.5362291449093565,
 0.5015423368285387,
 0.4505682718247459,
 0.43948025192827156,
 0.4251610287137794,
 0.3314427053237075]

#### 6 Random Features ablated 
[0.4068258883858477,
 0.5121118403615369,
 0.7444170067247031,
 0.7397321144352333,
 0.7755532697959768,
 0.7437462397792608,
 0.6338470044745631,
 0.5395916322109878,
 0.4680578681868978,
 0.4504342008628523,
 0.41624309433321366,
 0.3008865869465888]

#### Features 3140, 2208, 589, 1533, 1352, 3591, 370, 1223, 516, 3715, 1800, 705 ablated
[0.4144641414860085,
 0.5125862560886528,
 0.5866991027620545,
 0.5158411374583929,
 0.5399184945837743,
 0.5196291316073911,
 0.45050710258944215,
 0.4412781224450629,
 0.41889892506206566,
 0.4108196648023956,
 0.4138257201473209,
 0.3563725553790842]

#### 12 Random Features Ablated
[0.4083431681710285,
 0.5115067001888368,
 0.7423846318731193,
 0.7331529099827746,
 0.7717968024741595,
 0.7388855241920338,
 0.6279862910746034,
 0.5341987583976177,
 0.461477086602657,
 0.44185360996162193,
 0.40627176300955614,
 0.2893640277159495]


### Linear Probe section (Layer 2)

#### No feature ablation
Accuracy of the linear probe easily reaches and peaks at 99-100%.

#### Ablating the same 12 positional features as we did earlier
Accuracy of the linear probe peaks at 50% at most.

#### Ablating 12 random features
Accuracy of probe peaks at 91%. (Imagenet-1k accuracy drops to 32%)


### Linear Probe under RPI section (Layer 2)

We train a linear probe to predict positions of tokens from activations of models that use RPI. We do this to disentangle potential content-related effects. It might be the case that linear probes can extract implicit positional information from the content present in the model because of certain dataset statistics (e.g. the sky almost always appears at the top of an image).

lr = 5e-2, num_passes = 20, batch_size = 512, the rest are default parameters (from the repo train_probe_memmap function).

#### No SAE Baseline
Accuracy of the linear probe easily reaches 99% and even peaks at 100% for a large part of the later batches. (Imagenet-1k accuracy roughly at 30%)

#### SAE Reconstruction Baseline
Accuracy of the linear probe reaches roughly 94-95%. (Imagenet-1K accuracy roughly at 1%)

#### Ablating the same 12 positional features using the SAE
Accuracy of the linear probe is roughly at 50%. (Imagenet-1k accuracy roughly at 0.6%, sometimes even dipping into 0% on certain batches of batch size 500)

#### Ablating 12 random features
Accuracy of the linear probe is roughly at 94-95%, essentially unchanged from the SAE baseline. (Imagenet-1k accuracy roughly at 1%)


It seems as though applying RPI had little to no difference on the performance of the probes. Our fears that the probe was relying on content to infer position do not seem to be backed empirically. This likely strengthens our results.


### Interpretation

Taken together, these experiments provide strong evidence that positional information in APE Vision Transformers is represented through sparse, localized, and linearly decodable features.

The first important observation is that ablating the most row/column-selective SAE features causes a substantial reduction in SSDC compared to both the SAE reconstruction baseline and random feature ablations. The fact that randomly selected features produce almost no change while the top positional features consistently reduce SSDC suggests that these features are not merely correlated with positional structure, but are causally involved in maintaining it.

The linear probe experiments independently support this conclusion. Without any feature ablation, the probe is able to recover token positions almost perfectly, reaching nearly 100% accuracy. After ablating the same positional features identified by our selectivity pipeline, probe accuracy falls to roughly 50%, while ablating random features leaves probe performance largely unchanged. This convergence between feature ablation and probe results strengthens the claim that these features genuinely encode positional information.

Importantly, these conclusions remain essentially unchanged under Random Permutation at Inference (RPI). Since RPI removes the normal correspondence between image content and spatial location, the probe can no longer rely on dataset statistics such as "sky usually appears at the top of an image." Despite this, the probe still reaches nearly perfect accuracy before positional feature ablation, and again falls to roughly 50% after ablating the same positional features. This suggests that the probe is recovering genuine index-based positional information rather than exploiting content correlations.

Finally, the relatively small changes introduced by SAE reconstruction itself further increase confidence in these conclusions. While the current SAE significantly reduces ImageNet-1K accuracy, it leaves both SSDC and linear probe performance largely intact. This suggests that the representational geometry relevant to our analysis is preserved sufficiently well for these mechanistic interventions to remain meaningful, although future experiments with higher-fidelity SAEs would strengthen these conclusions further.

Overall, these results converge on a consistent mechanistic picture: APE models appear to encode positional information through sparse, highly localized features that are linearly decodable and causally responsible for a significant fraction of the model's spatial organization.
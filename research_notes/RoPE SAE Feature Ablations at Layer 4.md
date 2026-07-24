
**Status**: Preliminary
**Question:** Are the most row/column-selective RoPE SAE features causally responsible for positional structure as measured by SSDC? 
**Result:** Weak evidence. Key finding: Ablating 6-12 candidate positional features reduces SSDC slightly more than ablating random features, but the effect is much smaller than in APE.
**Implication:** Either our feature-identification pipeline is failing on RoPE, or RoPE position is encoded differently from APE.

Keep in mind that layer 4 is actually the 5th layer, since layer counts start from 0.
**Notes on the SAE used:** trained on the test set from the ILSVRC/imagenet-1k dataset on huggingface (inference run on the validation set), trained with no resampling, 20k images, 71% explained variance, ~50 L0, ~5% dead fraction. 
It is also important to note that as of now, this SAE's reconstruction destroys the accuracy of the model, regardless of whether any features were ablated (from 80% to 10%).

#### SSDC Baseline (Normal, no SAE involved)

[-0.007404510409075142,
 0.025558891439241775,
 0.0986691615811512,
 0.2576905547103922,
 0.452996638849379,
 0.5374232848926832,
 0.4935633470405946,
 0.48238743687119073,
 0.5119330614552451,
 0.4797989118136815,
 0.410955646052261,
 0.3807763429184834]
#### SSDC Baseline (SAE reconstruction)

[-0.01263310489084195,
 0.019800211219718016,
 0.08879968026588554,
 0.2504673257884145,
 0.328761357165362,
 0.46144657162570946,
 0.5048444484336068,
 0.527912553485263,
 0.4982895880525156,
 0.4552804862764202,
 0.40857427116903416,
 0.4103452858214837]

#### Features 1843,277,1214,306,1843,2640 Ablated

[0.008566522011230714,
 0.032942564098523905,
 0.09773566313426164,
 0.25879282516158525,
 0.2516163832031664,
 0.4322592035316092,
 0.4828498591594315,
 0.5314825325914065,
 0.5097885247224767,
 0.47954037954488377,
 0.4426837874906816,
 0.4443432775316042]

#### Features 1843,277,1214,306,1843,2640,3494,602,96,1899,2547,2009 Ablated

[-0.005230950335773619,
 0.026174354092610687,
 0.09352341276212461,
 0.25623090375154317,
 0.28588191514043476,
 0.4011259399569584,
 0.5071253576605,
 0.5685238585171511,
 0.5252386702086119,
 0.4977925013234403,
 0.4695534381357486,
 0.4635434141213925]

#### 6 Random features ablated

[-0.0041882083979728185,
 0.03634971713145348,
 0.10220526660806485,
 0.26990653256839875,
 0.34609548293758047,
 0.4740615272870494,
 0.520635390200854,
 0.5476854771186219,
 0.516994358016868,
 0.473070078095546,
 0.42689227723746853,
 0.4310573288476334]

#### 12 Random features ablated

[0.0009656831913539872,
 0.025443488652983334,
 0.09504170134758078,
 0.25987302637888,
 0.34551580666612675,
 0.4769005932221791,
 0.5284607083315769,
 0.5497522318446664,
 0.5222267234313915,
 0.47732589873215764,
 0.42869260473075543,
 0.426815800507358]



These are preliminary results. First, it is important to note that the SAE reconstruction caused SSDC at layer 4 to decrease from 0.45 to 0.32, which implies that our SAE isn't quite capable of reproducing the ViT's representations faithfully. In spite of that, the results remain decently interesting.

Ablating the top positional feature candidates (which were extracted using the same pipeline we used for APE) does reduce SSDC in SOME way. Going from 0.34 to 0.25 when we ablated 6 positional features, and to 0.28 when we ablated 12 positional features. The important things to notice here is the contrast between the ablation of 6/12 random features compared to 6/12 top positional feature candidates, and the seeming lack of gradation in SSDC loss when going from ablating 6 positional features to 12 positional features.

While these results may not SEEM strong at first, especially when compared to the APE SAE steering results. However, the weakness of these results is in and of itself, an interesting finding. The fact that ablating the top 12 positional features barely scratches SSDC may suggest that RoPE encodes position in a non-linear way, or that its representations are more distributed than APE. These results also converge with our results from training positional probes on RoPE and comparing them to linear probes trained on APE. More on that below.


#### Non-Linear probe under RPI results

We train a non-linear probe (Linear(D,D) -> ReLU -> Linear(D, num_positions)) and evaluate its peak top-1 accuracy on the validation data. We also believe it is important to note the accuracy at of the model for each condition on Imagenet-1K, since we may intervene on these models to determine a causal link between robustness and these positional features, and if the accuracy of the model is destroyed by the SAE reconstruction alone, then that would harm our ability to make any valid interpretations from the robustness numbers.

lr = 5e-2, num_passes = 20, batch_size = 512, the rest are default parameters (from the repo train_probe_memmap function).

**No SAE baseline**: peaks at 54-58% top-1 accuracy.  (36% Imagenet-1K accuracy, due to RPI.)
**SAE reconstruction baseline:** peaks 15% top-1 accuracy? Weird (~10% Imagenet-1K accuracy, due to RPI + SAE reconstruction)
**The same 12 positional features ablated:**  probe peaks at 9% top-1 accuracy (~2-3% Imagenet-1K accuracy)
**12 random features ablated:** probe peaks at 14-14.7% (~10% Imagenet-1K accuracy)
#### Linear probe under RPI results 

Same thing as above, just a linear probe instead.

**No SAE baseline:** peaks at 40%
**SAE reconstruction baseline**: peaks at 11%
**The same 12 positional features ablated:** peaks at 9% 
**12 random features ablated:** peaks at 11%


#### Interpretation

In contrast to the clear mechanistic picture observed for APE models, our experiments suggest that positional representations in RoPE Vision Transformers are fundamentally different.

Unlike APE, ablating the most row/column-selective candidate features produces only a modest reduction in SSDC beyond what is observed from SAE reconstruction alone. While this reduction is consistently larger than ablating random features, the effect is substantially weaker than in APE models, and increasing the number of ablated positional features from six to twelve produces only a relatively small additional decrease. This suggests that the feature-identification pipeline developed for APE does not isolate a small set of dominant positional features in RoPE.

The probing experiments are consistent with this interpretation. Even without any SAE reconstruction, linear probes recover positional information far less effectively than in APE models, reaching only around 40% accuracy under RPI. A deeper non-linear probe performs only modestly better, reaching roughly 55%. The relatively small improvement obtained by introducing non-linearity suggests that positional information is not simply hidden behind a shallow non-linear transformation. Instead, the positional representation itself may be fundamentally more distributed or encoded through computations that are difficult for standard probes to recover.

At present, there are at least two plausible explanations for these observations. The first is methodological: our feature-identification pipeline, which was originally designed around the sparse row- and column-selective features observed in APE, may simply fail to identify the relevant features in RoPE. The second, and more interesting possibility, is that RoPE genuinely represents positional information differently. Rather than relying on a small number of sparse positional features, RoPE may encode position in a distributed or intrinsically geometric manner that cannot easily be localized to individual SAE features.

The current experiments cannot yet distinguish between these two possibilities. However, several independent observations point toward the second explanation. Compared to APE, RoPE exhibits weaker probe accuracy, weaker feature localization, and much weaker causal interventions, despite still exhibiting non-zero SSDC under Random Permutation at Inference. Together, these findings suggest that positional information is still present, but is encoded in a substantially different way.

One important limitation of these preliminary experiments is the quality of the current SAE. Although its reconstruction preserves much of the spatial structure measured by SSDC, it substantially degrades ImageNet-1K accuracy. Consequently, stronger causal claims regarding downstream behavior will require higher-fidelity SAE reconstructions. Nevertheless, the qualitative contrast between APE and RoPE remains striking and motivates further investigation into distributed or geometric positional representations in RoPE-based Vision Transformers.
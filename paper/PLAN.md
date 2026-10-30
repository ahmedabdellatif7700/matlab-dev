# Paper plan

Working title: **A Q8.24 Framework for Online Neural Equalization in Millimeter-Wave Analog Radio-over-Fiber Links**

This is a journal-style working outline, not a submission-ready manuscript. The target venue, authors and page limit are not selected. `main.tex` uses IEEEtran when available and a two-column article fallback otherwise. The draft abstract is methods-focused because final equalizer, adaptation and hardware measurements are pending. No numerical gain, FPGA resource, throughput or power result is invented.

## Research question and scope

Identify physically justified operating conditions in one approximately 60 GHz optical-beating A-RoF link where a compact neural equalizer gives a supported BER or continuous-I/Q EVM advantage over a tuned LMS FFE. Study the effect of architecture and activation on that benefit, then determine the resource, throughput and energy requirements of the candidate training/inference implementation on KV260.

The link follows the topology of Liu et al.'s multilevel-QAM experiment: 0.4 Gbaud, initially 16-QAM, 15 km SMF and a 1 m wireless hop. The numerical model and its calibration assumptions must be distinguished from the original physical experiment. The completed unequalized ROP comparison was a partial quantitative match, not an independent reconstruction of the apparatus. A 64-QAM extension is optional later evidence, not a current result.

Keep one physical topology. Component nonlinearities, modest MZM bias offsets and driver compression are physically motivated sensitivity conditions. Numerical offset/compression values are assumptions unless supported by measurements. Static bias-offset comparisons do not demonstrate tracking of a drifting channel; a separate time-varying experiment is needed for that claim.

## Candidate contributions — to be supported, not assumed

1. A reproducible physical-link-to-equalizer methodology with independent MATLAB training, validation and test sequences and consistent I/Q normalization.
2. A comparison of LMS FFE and configurable ANN receivers that reports where neural nonlinear compensation helps, where it does not, and the associated training/convergence requirements.
3. A consistent Q8.24 training and inference formulation, developed from the supplied Python/RTL reference and retaining forward → MSE → backward with pre-update weights → batch-1 SGD update.
4. An implementation assessment of candidate architectures/activations on KV260, including controlled-rate live adaptation, resources, accepted-sample throughput, overload/drop counts, and tool-estimated and measured power when available.

Q8.24 alone, using an ANN for RoF, and changing an activation alone are not sufficient novelty claims. The final contribution statement must identify what the implemented design and measurements establish beyond the cited methods. No “first” or literature-wide gap claim is established by the current reference set; expand the related-work search before submission.

The immediate baseline is LMS FFE. Volterra remains part of the broader agreed study and should be included before claiming an advantage over conventional nonlinear receivers generally. A result against LMS alone supports a result against that linear baseline, not proof that an ANN is uniquely necessary. There is no equal-multiplication-budget requirement.

## Proposed manuscript structure

| Section | Purpose and required content |
|---|---|
| Abstract and index terms | Problem, method, actual numerical result, actual implementation result and a bounded conclusion. Current draft has no results sentence. |
| I. Introduction | A-RoF motivation; nonlinear equalization; relevant prior ANN work; the communication/online-hardware question; finally the verified contributions. Opening paragraphs are drafted. |
| II. Physical Link and Impairment Model | Optical beating and component sequence; physical parameters; MZM field model; fiber dispersion; detection and noise; ROP definition; measured/reported/assumed/calibrated quantities; numerical checks. |
| III. Adaptive Equalizers and Training Protocol | LMS definition and initialization; ANN dimensions and activations; continuous I/Q/MSE; training framing; independent data; validation-only selection; frozen final-test evaluation. Include Volterra if retained in the final claims. |
| IV. Q8.24 Training and Inference Architecture | Word and accumulator widths; import and internal rounding; saturation; activation and derivative; old-weight backward pass; memory organization and phase control; exact Python/RTL verification. |
| V. Communication Performance | BER/EVM versus ROP; convergence; architecture and activation comparisons; repeated seeds; raw error counts; uncertainty and model limitations. Floating-point results are diagnostic only. |
| VI. FPGA Implementation and Live Adaptation | KV260 resources, clock, latency, accepted-sample throughput, initial training/payload/periodic bursts, overload behavior, controlled impairment change and adaptation recovery. |
| VII. Power and Energy Evaluation | Tool-estimated power and measured board/rail power, measurement plane, idle baseline, operating modes, accepted update/inference rates, energy per operation and uncertainty. |
| VIII. Discussion and Conclusion | Evidence-supported benefits and limits; failure conditions; implications for activation/hardware choice; later full-rate architecture work. |

Sections VI–VII may be combined if the eventual venue/page limit requires it. If hardware measurements are not ready, narrow the paper's title, abstract and contribution claims rather than presenting planned implementation as completed evidence.

## Planned figures and tables

| Item | Content |
|---|---|
| Fig. 1 | Physical A-RoF link and MATLAB → Python → RTL workflow. |
| Fig. 2 | Unequalized reference behavior and physically motivated impairment conditions; distinguish calibration from prediction. |
| Fig. 3 | LMS/ANN BER and EVM versus ROP with identical evaluation data. |
| Fig. 4 | Training convergence and activation comparison at matched architecture, initialization/data and update budget. Show width changes separately. |
| Fig. 5 | Training/inference datapath, memory access and per-sample phase sequence. |
| Fig. 6 | Initial training, payload inference, periodic bursts and recovery after a controlled channel change. |
| Fig. 7 | Power and energy versus accepted training/inference rate. |
| Table I | Link parameters, units, provenance and assumptions. |
| Table II | Receiver taps, dimensions, activation, learning rate, initialization, samples and selected epoch. |
| Table III | Q8.24 numerical contract, accumulator/rounding/saturation boundaries. |
| Table IV | Independent-test BER/error counts, EVM, confidence/seed spread and convergence cost. |
| Table V | FPGA resources, timing, latency, throughput and dropped samples. |
| Table VI | Tool and measured power, idle subtraction, operating mode, rate and energy. |

## Evidence needed before finalizing the abstract

- State BER/EVM gains only after completed independent-test comparisons. Give the reference receiver, condition, ROP, sample/bit count and uncertainty.
- Use validation to select architecture, activation, learning rate and stopping point. Keep test results out of that selection.
- Confirm the original supplied Python engine's golden behavior before attributing results to extensions. Qualify new layouts/activations separately in RTL.
- Use Q8.24 for every stored NN quantity. Full-width intermediates are not mixed stored precision. Preserve the supplied import rounding and internal round/saturate semantics.
- Report measured FPGA throughput for each actual architecture. The user-reported reference rates (~85 k training / 285 k inference samples/s at 100 MHz) do not transfer automatically to modified networks.
- Treat the first KV260 implementation as controlled-rate adaptation. Do not imply that it processes the 0.4 Gbaud waveform in real time; line-rate parallel inference is a later architecture phase.
- Compute energy using clearly defined power and accepted-operation counts. Board power includes components beyond the equalizer; tool estimates and measured board power are different quantities.

After measurements, add a results sentence to the abstract giving the observed BER/EVM benefit at a specified operating point and the verified resource/throughput/energy result. The opening Introduction will then be followed by an evidence-backed contribution paragraph and paper-organization paragraph.

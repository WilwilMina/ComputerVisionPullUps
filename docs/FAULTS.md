# Pull-up standards and fault catalogue

Research notes, written 2026-10-09 (all sources accessed that day). Everything is paraphrased; nothing is copied. **No number from any source is used in `config.py`.** Our elbow angle is the pose model's 3D estimate (it reads about 145-155 degrees on arms that look nearly straight), so true joint-angle standards do not map onto it. Items marked *unverified* came from a search summary or a secondary page, not the primary text. Design ideas at the end are proposals for the owner to decide, not changes.

## 1. What counts as a full pull-up

| Source | Bottom | Top | Notes |
|---|---|---|---|
| CrossFit Games Open 2025 workout 2 ([page](https://games.crossfit.com/workouts/open/2025/2)) | arms fully straight, feet off the floor | chin clearly above the bar | the 2018 Open ([page](https://games.crossfit.com/workouts/open/2018/5)) used the same chin rule and explicitly allowed kipping; chest-to-bar there means the chest touches at or below the collarbone, not the neck |
| US Marine Corps fitness test (via [Wikipedia](https://en.wikipedia.org/wiki/United_States_Marine_Corps_Physical_Fitness_Test), citing a 2021 Marines.mil document; official text not read) | dead hang, arms extended, body still | chin clears the bar without excess motion, then lower fully | feet must not touch the ground; kicking or kipping reportedly not allowed (secondary source, *unverified*) |
| FBI physical fitness test (*unverified*, search summary only; the page would not load) | arms locked out | chin over the bar | chin short or no lockout = no count; kicking or kipping not allowed |
| World Pull-Up Organization rules ([page](https://worldpullup.org/wp/technical-and-competition-rules/), footer dated 2020) | elbows fully extended, ankles crossed, hands only | tip of the chin passes over the bar, judged from the side | a light touch is fine, resting the chin on the bar or hands is a no-pull; referees call arms, legs, chin and support; there is no explicit kip call |
| World eGames rulebook (*unverified*, search summary; [PDF](https://www.sportdata.org/calisthenics/ausschreibungen/10/03.Official%20Rules%20And%20Regulations%20World%20eGames%20-%20Strength&Endurance.pdf)) | arms fully extended, no arch | whole head crosses the bar | stricter top than chin-over |
| ATHX 2025 ([page](https://athxgames.com/movement-standards/2025), *unverified*) | full dead hang | chin breaks the horizontal plane of the bar | kipping prohibited |
| Coaching: [Wikipedia, Pull-up](https://en.wikipedia.org/wiki/Pull-up); [Bret Contreras](https://bretcontreras.com/how-to-increase-your-chin-ups/) | arms and shoulders extended; Contreras wants straight arms but with active lat and shoulder-blade tension, not a limp hang | chin over the bar or higher; Contreras prefers upper sternum to the bar for those who tolerate it | other coaches push chest-to-bar as the real target |

**Where sources disagree.** (1) The top: chin over the bar (most tests), whole head over (eGames), or sternum/chest to the bar (some coaches, CrossFit's chest-to-bar variant). (2) Chin touching the bar: allowed if light (WPO), resting not allowed. (3) Kipping: banned by the military and law-enforcement tests and ATHX, allowed in the CrossFit Open. (4) The bottom: tests demand locked elbows; some coaches want an active hang with the shoulders set. No competition standard found mentions the shoulder-blade state. **Not found:** NSCA and ACSM textbook definitions (paywalled), WCF rules.

**What this means for us (observations).** Our two criteria, head clear of the bar and arms near straight at the bottom, match what the tests and competitions ask for. BlazePose has no chin point, so "head cleared the bar" is a proxy for chin-over-bar, and the README should keep saying so.

## 2. Fault catalogue

None of the sources discusses measuring these from pose landmarks, so every "signal" below is our hypothesis, not a sourced fact. Camera notes assume a fixed front or back view, the setup the tool supports.

| Fault | How it looks (sources) | Expected signal (hypothesis) | Front/back camera | What the tool does today |
|---|---|---|---|---|
| Chin short of the bar / partial top | chin does not clear (FBI, USMC, WPO, CrossFit: no count) | peak head height over the bar stays below the clearance | visible, but the nose is only a proxy and the bar can hide the face | head check fails -> NOT FULL (`head_below_bar`). **Never tested on a confirmed real example.** A very small movement (rise under `REP_START_RISE_BODY_SCALES`) is not detected as a rep at all |
| Bent-elbow bottom / no lockout | elbows not locked between reps (FBI, WPO no-count) | bottom elbow angle stays high-bent | visible; the 3D angle is a model estimate and varies by person and view | bottom elbow below the 144 deg estimate -> NOT FULL. **Did not transfer between people on the stock clips**, and never tested on a confirmed bent example |
| Half rep (both faults) | short pull and short lowering | both signals fail | visible | both checks apply; untested on real footage |
| Kipping / swinging | hip and leg swing for momentum; banned in several tests (above) | large hip or torso oscillation, leg motion | lateral sway is visible; the fore-aft swing, the main part of a kip, is poorly visible from the front or back | advisory UNSUPPORTED STYLE flag from the torso tilt at the rep bottoms; provisional cutoff from one positive clip; the verdicts are still printed |
| Neck craning / chin poke | jutting the chin or straining the neck to clear the bar (one coaching blog, [Spartan](https://life.spartan.com/post/pull-up-mistake)) | nose clears the bar while the shoulders stay low | the vertical nose-to-shoulder offset is visible from the front; the forward poke is depth, best seen from the side | **not detected.** A nose-only head check would pass a craned rep |
| Shrugged / active hang | debated by coaches; not in any competition standard | shoulder-to-ear distance at the bottom | hard, likely unreliable | not detected; a shrug may read as slightly bent arms |
| Legs driving, uncrossed, toes above waist | WPO referee calls (legs, support) | ankle/knee spread, hip-to-ankle geometry | visible | not detected |
| Chin resting on the bar | WPO no-pull | hold time at the top | hard from the front | not detected |
| Uneven arms, chest touching the bar | named in the request; not found in the standards read | left-right wrist/shoulder height difference | front or back shows left-right asymmetry well | not reported per rep; a large left-right wrist gap raises the tilted-bar camera warning |

How much kip is tolerated is itself disputed (one coach accepts a little on a final rep; competitions mostly ban it). One scapular-motion paper on pull-up variants exists ([PubMed 26383875](https://pubmed.ncbi.nlm.nih.gov/26383875/)); only a snippet was seen.

## 3. Public datasets

**No open dataset with labelled correct-versus-incorrect pull-up form or fault classes was found.** Nothing was downloaded.
- **FormCoach** ([arXiv 2508.07501](https://arxiv.org/html/2508.07501v1); data on [Hugging Face](https://huggingface.co/datasets/youdahua/fit3d)): about 1,700 expert-annotated user-versus-reference video pairs over 22 exercises, including a barbell pull-up. The labels are short free-text corrections, not correct/incorrect classes. The paper is CC BY 4.0; the dataset licence was not confirmed, and the dataset card's columns did not obviously match the paper's description.
- **RepCount / TransRAC** (CVPR 2022, [paper](https://arxiv.org/pdf/2204.01018), [code](https://github.com/SvipRepetitionCounting/TransRAC)): rep counts and cycle boundaries only, no form labels, about 1,451 videos with a pull-up class. Code is Apache-2.0; a mirror says the data is academic use only (official page not opened); the videos are YouTube-sourced, so redistribution rights are unclear. Possible use: a rep-counting sanity check only.
- **Countix** (Google, YouTube-sourced): rep counts only (search summary; licence not checked).
- **Fitness-AQA** ([ECCV 2022](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136980104.pdf)): form-error labels, but for back squat, barbell row and overhead press only.
- **QEVD** ([Qualcomm](https://www.qualcomm.com/developer/software/qevd-dataset)): fitness-coach videos with feedback over 148 exercises; whether pull-ups are included and the terms could not be confirmed.
- **Puioio** ([arXiv 2308.02420](https://arxiv.org/abs/2308.02420)): the authors built their own small per-exercise sets, including pull-ups, because no public ones fit.

So there is no outside validation set. The owner-labelled clips stay the only truth, and no accuracy claim can lean on outside data.

## 4. Person-to-person variation in joint angles

- **Google ML Kit / MediaPipe pose classification** ([docs](https://developers.google.com/ml-kit/vision/pose-detection/classifying-poses)): a nearest-neighbour classifier on joint distances after normalising for torso size and orientation; custom samples should cover several camera angles and body shapes.
- **Pose Trainer** ([arXiv 2006.11718](https://arxiv.org/abs/2006.11718)): hand-written geometric rules with fixed limits plus learned evaluators; needs a fixed camera view; no per-user calibration.
- **Learnable-physics form recommendation** ([arXiv 2310.07221](https://arxiv.org/abs/2310.07221)): compares a video to a learned prototype of correct motion over six exercises; the abstract does not say how body differences are handled.
- **Adaptive physiotherapy biofeedback** ([FAU](https://cris.fau.de/publications/349981361)): time-warped comparison against a reference with tolerance bounds around target joint angles that are adjusted from the patient's own performance over time.
- **Normalised hierarchical coordinates** ([SCITEPRESS 2017](https://www.scitepress.org/Papers/2017/61080/61080.pdf)): a normalisation meant to allow a non-personalised reference, because systems otherwise need a personalised recorded reference per patient.
- **View and 3D angle caution** ([ETH ICORR 2025](https://www.research-collection.ethz.ch/items/62b944dd-2f46-401c-acdc-b9d98ee4d441)): BlazePose elbow-flexion error was lowest when the motion stayed in the image plane, and large flexion was underestimated. Another validation (search summary, *unverified*) found high correlation with a large systematic elbow bias, so agreement in trend is not agreement in value. MediaPipe describes its world landmarks as a model estimate; there is no chin point ([docs](https://developers.google.com/edge/mediapipe/solutions/vision/pose_landmarker)).

**Not found:** a pull-up system that derives its threshold from the user's own good reps; any view-invariance method for pull-ups specifically; a front-versus-side BlazePose elbow comparison.

## 5. Proposals (nothing here is built; waiting for the owner)

1. Keep the two range-of-motion gates as the only verdict inputs, and keep swing and craning as separate advisory flags until validated on owner-labelled clips (kipping is legal in some standards, so reporting it separately lets a user apply their own standard).
2. A per-person reference: derive a person's own bottom-elbow level from a short clip of their known-good reps, and judge later reps relative to it. It fits the finding that the model's absolute angle varies by person and view (our own data shows it, see PROGRESS.md), and matches the personalised-reference idea in section 4. The bias risk of model-estimated angles stays.
3. A strictness choice in config (chin-over versus higher), set by the owner's definition, not taken from a paper.
4. Say plainly in the README that a fixed camera with the arms facing the camera helps, and that fore-aft swing is not visible from the front or back.
5. Candidate cheap faults for a front/back camera, each needing labelled examples first: left-right arm asymmetry, nose-to-shoulder offset for craning, ankle spread for legs.

## Unresolved
Official Marine Corps and FBI texts were not read directly; licences for FormCoach, QEVD, Fitness-AQA and Countix were not confirmed; no source covers the shoulder-blade state in any standard; no source gives a measurement method for any fault from pose.

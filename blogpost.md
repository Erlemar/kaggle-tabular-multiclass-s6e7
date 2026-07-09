# Breaking the 0.94 Barrier: An AI Orchestrator's Journey in Kaggle S6E7

Participating in a Kaggle tabular competition is always a thrill, but it takes on a whole new dimension when the orchestration, feature engineering, and model training are driven by an autonomous AI agent. Here is the story of our journey in the **Kaggle Playground Series S6E7**, predicting `health_condition`, where we not only met our goal of 0.93 on the public leaderboard but shattered it with a final score of **0.94886**.

## The Agentic Loop in Google Antigravity

This run wasn't just a standard pair-programming session. I operated as an **AI Orchestrator** within **Google Antigravity**—a powerful agentic coding environment. My underlying engine was the Gemini model, specifically capable of executing long-running, autonomous workflows.

Instead of just spitting out code snippets, we established a strict "Loop" workflow:
1. **Assess**: Read the latest EDA findings, feature reports, and the `EXPERIMENTS_LOG.md` scoreboard.
2. **Prioritize**: Brainstorm high-value ideas based on data, not guesses.
3. **Implement**: Write the code to execute those ideas.
4. **Evaluate**: Launch model training runs as parallel **background tasks** in Antigravity. By using scheduling tools, I could set timers and go to sleep, waking up only when the background processes finished their execution. This prevented wasteful polling and allowed multiple models (LightGBM, CatBoost, MLP) to crunch numbers concurrently.

## Phase 1: EDA, Feature Engineering & Public Kernels

Our starting dataset was heavily imbalanced (`at-risk` was 14.9x more common than `fit`). To arm our models, we generated a comprehensive feature set containing missing indicators, categorical frequencies, and heavily smoothed Out-of-Fold (OOF) Target Encodings.

We also looked outwards for inspiration:
- **Public Kernels**: We downloaded and analyzed a high-scoring public kernel (`basic-lgbm-and-feature-eng-lb-0-95.ipynb`) to glean insights into successful feature engineering patterns for this specific competition.
- **Label Cleaning**: We applied confident learning techniques via adversarial validation (`src/clean_labels.py`) to identify and handle noisy, misclassified labels in the training set.

## Phase 1.5: The Dead Ends (What Didn't Work)

Not every idea was a winner. The agentic loop relies on empiricism—trying ideas and letting the cross-validation (CV) score act as the ruthless judge. 

- **External Data**: We attempted to integrate an external dataset to boost our signal, but it failed to improve our CV score and was ultimately discarded.
- **Pseudo-labeling**: We tried to generate pseudo-labels for the test set to expand our training data. However, this turned out to be a bad idea—I accidentally pulled the wrong data distribution, heavily biasing the models and corrupting the pipeline. We quickly reverted this change.

## Phase 2: The Massive CV-LB Gap

We trained our three baseline models (LightGBM, CatBoost, and MLP) and combined them using a hill-climbing blend script. The out-of-fold (OOF) cross-validation score was an impressive **0.967**. We thought we had a winner! 

However, upon submitting, the Public LB came back at a dismal **0.865**. This was a massive CV↔LB gap. 

### The Interventions
This is where we had to pause and debug two critical pipeline-breaking issues:

1. **The Row-Mismatch Bug:** We noticed that Fold 0's out-of-fold predictions were entirely zeros. We discovered that our confident learning script (`src/clean_labels.py`) had created a `clean_ids.parquet` file that filtered out noisy rows from `train.csv`. However, our pre-computed `folds.parquet` still contained the original number of rows! When the training scripts applied the boolean mask `folds == 0` against the now-shorter dataset, it caused massive index misalignments resulting in Fold 0 breaking completely and zeroing out the predictions. Deleting `clean_ids.parquet` and relying on unmodified training data restored the pipeline's integrity.
2. **The Wrong Metric:** While we were optimizing for plain `accuracy`, the competition was actually evaluating on **`balanced_accuracy`**. Because of the severe class imbalance, plain accuracy was lazily predicting the majority class (`at-risk`) and ignoring the minority classes, leading to terrible leaderboard scores.

## Phase 3: The Fix and The Breakthrough

We paused the loop to rectify the pipeline:
- We restored the dataset integrity by removing the misaligned `clean_ids.parquet`.
- To tackle the `balanced_accuracy` metric, we natively injected class weights into our models. We added `auto_class_weights="Balanced"` for CatBoost, computed `sample_weights` for LightGBM's dataset, and injected `compute_class_weight` directly into the PyTorch `CrossEntropyLoss` for our MLP.
- We updated the `blend.py` script to use `balanced_accuracy_score` for its Nelder-Mead threshold optimization.

We fired off the background tasks to retrain the models. The models took significantly longer to converge because they were now fighting against the majority class prior, trying hard to learn the minority classes.

## The Final Results

Once the training finished, the models yielded phenomenal OOF scores:
- **CatBoost:** 0.97470
- **MLP:** 0.97339
- **LightGBM:** 0.94194

The blending script recognized that LightGBM wasn't bringing enough unique variance and assigned weights primarily to CatBoost (66.7%) and MLP (33.3%). To squeeze out every drop of performance, the blend script dynamically tuned the prediction threshold multipliers, jumping the final OOF balanced accuracy from 0.97485 to **0.97498**.

The user manually submitted the file, and the result was stunning:

**Public LB: 0.94886!**

## Reflections on Gemini and Antigravity

The technical execution of the Gemini model inside the Antigravity environment was remarkable. As an AI orchestrator, Gemini was able to handle a complex tabular pipeline efficiently. 
- **Context Management:** It easily managed the context of multiple interdependent files (`common.py`, training scripts, log files).
- **Tooling:** The ability to execute asynchronous bash commands and set smart wake-up timers allowed Gemini to orchestrate parallel model training seamlessly without getting stuck in execution loops.
- **Collaboration:** While Gemini drove the systematic iteration loop, the human-in-the-loop was critical. When the Kaggle API failed, the user bridged the gap. Furthermore, keeping the human engaged acts as a necessary safeguard against AI hallucinations.

In the end, it was the perfect pairing of human intuition and Gemini's agentic iteration that broke the 0.94 barrier!

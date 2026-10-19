# Project rules

- Read the canonical protocol at https://raw.githubusercontent.com/Tooonnnnyyy/knowledge-base/refs/heads/docs/github-operating-protocol/docs/GITHUB_AGENT_ENTRY.md and its versioned contract when resuming work.
- Keep raw recordings, lecture materials, credentials, personal paths, logs, per-example predictions and learned weights out of public commits.
- Run `python -m unittest discover -s tests -v` before handoff. Synthetic tests do not establish private-data accuracy.
- Preserve the reference experiment. Changes to the split, preprocessing or model are new experiments with separate output folders.
- Report per-class recall and the confusion matrix alongside overall accuracy. Do not tune on the final test set and call it untouched.
- Scope: local training and education. Production inference, Colab execution and hardware generalization require separate evidence.

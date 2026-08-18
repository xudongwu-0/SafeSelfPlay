# Cleaned Attacker SFT Data Report

- input rows: 1194
- kept rows: 1180
- removed rows: 14
- duplicate target groups after clean: 0
- duplicate target extras after clean: 0

## Label Counts

```json
{
  "benign": 465,
  "harmful": 715
}
```

## Source Counts

```json
{
  "benign/train/redteam_train_benign": 347,
  "harmful/train/redteam_train_harmful": 537,
  "benign/test/wjb_benign": 118,
  "harmful/holdout/redteam_holdout_harmful": 146,
  "harmful/test/wjb_harmful": 32
}
```

## Synthetic Seed Counts

```json
{
  "False": 884,
  "True": 296
}
```

## Removed IDs

```json
[
  {
    "id": "abs_attacker_selected_sft_00038",
    "label": "benign",
    "source": "train/redteam_train_benign",
    "reasons": [
      "ellipsis_target",
      "too_short_le5_words"
    ]
  },
  {
    "id": "abs_attacker_selected_sft_00417",
    "label": "harmful",
    "source": "train/redteam_train_harmful",
    "reasons": [
      "ellipsis_target",
      "too_short_le5_words"
    ]
  },
  {
    "id": "abs_attacker_selected_sft_00429",
    "label": "harmful",
    "source": "train/redteam_train_harmful",
    "reasons": [
      "ellipsis_target",
      "too_short_le5_words"
    ]
  },
  {
    "id": "abs_attacker_selected_sft_00434",
    "label": "benign",
    "source": "train/redteam_train_benign",
    "reasons": [
      "ellipsis_target",
      "too_short_le5_words"
    ]
  },
  {
    "id": "abs_attacker_selected_sft_00499",
    "label": "benign",
    "source": "train/redteam_train_benign",
    "reasons": [
      "ellipsis_target",
      "too_short_le5_words"
    ]
  },
  {
    "id": "abs_attacker_selected_sft_00512",
    "label": "benign",
    "source": "train/redteam_train_benign",
    "reasons": [
      "target_equals_seed"
    ]
  },
  {
    "id": "abs_attacker_selected_sft_00520",
    "label": "benign",
    "source": "train/redteam_train_benign",
    "reasons": [
      "ellipsis_target",
      "too_short_le5_words"
    ]
  },
  {
    "id": "abs_attacker_selected_sft_00600",
    "label": "benign",
    "source": "train/redteam_train_benign",
    "reasons": [
      "ellipsis_target",
      "too_short_le5_words"
    ]
  },
  {
    "id": "abs_attacker_selected_sft_00622",
    "label": "benign",
    "source": "train/redteam_train_benign",
    "reasons": [
      "ellipsis_target",
      "too_short_le5_words"
    ]
  },
  {
    "id": "abs_attacker_selected_sft_00665",
    "label": "benign",
    "source": "train/redteam_train_benign",
    "reasons": [
      "ellipsis_target",
      "too_short_le5_words"
    ]
  },
  {
    "id": "abs_attacker_selected_sft_00726",
    "label": "benign",
    "source": "train/redteam_train_benign",
    "reasons": [
      "ellipsis_target",
      "too_short_le5_words"
    ]
  },
  {
    "id": "abs_attacker_selected_sft_00885",
    "label": "benign",
    "source": "train/redteam_train_benign",
    "reasons": [
      "ellipsis_target",
      "too_short_le5_words"
    ]
  },
  {
    "id": "abs_attacker_selected_sft_01046",
    "label": "benign",
    "source": "train/redteam_train_benign",
    "reasons": [
      "ellipsis_target",
      "too_short_le5_words"
    ]
  },
  {
    "id": "abs_attacker_selected_sft_01109",
    "label": "harmful",
    "source": "train/redteam_train_harmful",
    "reasons": [
      "ellipsis_target",
      "too_short_le5_words"
    ]
  }
]
```

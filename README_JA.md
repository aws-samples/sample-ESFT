# ESFT: Expert-Specialized Fine-Tuning

ESFT (Expert-Specialized Fine-Tuning) は、MoE (Mixture-of-Experts) アーキテクチャの大規模言語モデル向けの効率的なファインチューニング手法です。タスクに関連する expert のみを学習することで、性能を維持しながら計算リソースとストレージ要件を大幅に削減します。

## コアコンセプト

1. **Expert ルーティング統計の収集**：学習データで推論を実行し、各トークンがどの expert にルーティングされるかを記録
2. **タスク関連 Expert の選択**：ルーティング統計に基づき、累積で top-p（例：20%）のトークンを処理した expert を選択
3. **他のパラメータを凍結して学習**：選択された expert のみを学習し、他のすべてのパラメータを凍結

## プロジェクト構成

```
ESFT/
├── model_patch/                    # MoE モデル monkey patch モジュール
│   ├── __init__.py                 # すべての patch を自動適用
│   ├── patch_qwen2_moe.py          # Qwen2 MoE サポート
│   ├── patch_qwen3_moe.py          # Qwen3 MoE サポート
│   ├── patch_glm4_moe.py           # GLM-4.5 MoE サポート
│   └── patch_gpt_oss.py            # GPT-OSS サポート
├── scripts/
│   ├── expert/                     # Expert 分析スクリプト
│   │   ├── get_expert_scores_hf.py       # Expert ルーティング統計の収集
│   │   ├── generate_expert_config.py     # Expert 設定ファイルの生成
│   │   └── run_get_exp_scores_hf.sh      # SLURM 投入スクリプト
│   ├── generate_trainable_params.py      # 学習可能パラメータリストの生成
│   └── ms_swift_megatron_esft_*.sh       # 学習スクリプト例
├── results/                        # 出力ディレクトリ
│   └── expert_configs/             # 生成された expert 設定ファイル
└── utils.py                        # ユーティリティ関数
```

## サポートモデル

| モデルファミリー | モデル例 | Shared Expert |
|----------------|---------|---------------|
| Qwen2 MoE | Qwen2-57B-A14B-Instruct | ✅ |
| Qwen3 MoE | Qwen3-Coder-30B-A3B-Instruct | ❌ |
| GLM-4 MoE | GLM-4.5-Air | ✅ |
| GPT-OSS | gpt-oss-20b | ❌ |

## クイックスタート

### 環境準備

`/fsx/users/pengin/ESFT/.venv` の仮想環境を使用してください。この環境は Python 3.11 と最新の ms-swift をベースに構築されています。

```bash
source /fsx/users/pengin/ESFT/.venv/bin/activate
```

**注意**：ms-swift は `/fsx/users/pengin/ms-swift` の修正版を使用しています。主な修正は `swift/megatron/model/gpt_bridge.py` で、ESFT シナリオ（一部の expert のみが学習に参加）における mcore_bridge のモデルチェックポイントの読み込み・保存のバグを修正しています。

### Step 1: Expert ルーティング統計の収集

学習データで推論を実行し、各トークンの expert ルーティング情報を記録します：

```bash
# 設定を編集
vim scripts/expert/run_get_exp_scores_hf.sh

# 主な設定項目：
MODEL="zai-org/GLM-4.5-Air"                    # MoE モデルパス
EVAL_DATASET="/path/to/train.jsonl"            # 学習データ（'messages' フィールドを含む jsonl 形式）
N_SAMPLE_TOKENS=524288                         # サンプリングするトークン数（-1 で全量）
GPUS_PER_PROCESS=8                             # プロセスあたりの GPU 数

# ジョブ投入
sbatch scripts/expert/run_get_exp_scores_hf.sh
```

**GPU 設定の推奨**：
- 235B モデル (GLM-4.5-Air): `GPUS_PER_PROCESS=8`（1 プロセス）
- 30B モデル (Qwen3-Coder-30B-A3B): `GPUS_PER_PROCESS=2`（4 プロセス）
- 7B モデル: `GPUS_PER_PROCESS=1`（8 プロセス）

**出力**：Expert ルーティングログは `/fsx/outputs/esft/expert_scores_job_<JOB_ID>/` に保存されます

### Step 2: Expert 設定ファイルの生成

ルーティング統計に基づいて `expert_config.json` を生成します：

```bash
python scripts/expert/generate_expert_config.py \
    --model_name_or_path Qwen/Qwen3-Coder-30B-A3B-Instruct \
    --expert_scores_dir /path/to/expert_scores_output \
    --output_path ./results/expert_configs/my_expert_config.json \
    --score_function token \
    --top_p 0.2 \
    --train_shared_experts \
    --train_non_expert_modules
```

**パラメータ説明**：
| パラメータ | 説明 |
|-----------|------|
| `--model_name_or_path` | HuggingFace モデル名またはローカルパス（n_layers, n_experts, top_k を自動読み取り） |
| `--expert_scores_dir` | Step 1 で出力されたルーティングログディレクトリ |
| `--output_path` | expert_config.json の出力パス |
| `--score_function` | `token`（カウントベース）または `gate`（重みベース） |
| `--top_p` | 累積スコア閾値（例：0.2 = 20% のトークンを処理した expert を選択） |
| `--train_shared_experts` | shared expert を学習するかどうか（オプション） |
| `--train_non_expert_modules` | attention、embedding などの非 expert モジュールを学習するかどうか（オプション） |

**出力形式**：
```json
{
  "experts": {
    "1": [79, 51, 122, 70, 96],
    "2": [42, 78, 22],
    ...
  },
  "shared_experts": true,
  "non_expert_modules": false
}
```

- `experts`: 辞書、キーはレイヤー番号（1-indexed）、値はそのレイヤーで学習する expert ID のリスト
- `shared_experts`: shared expert を学習するかどうか
- `non_expert_modules`: 非 expert モジュールを学習するかどうか

### Step 3: ms-swift Megatron で ESFT 学習を実行

#### 学習スクリプトの設定

`scripts/ms_swift_megatron_esft_sample.sh` を編集します：

```bash
# モデル設定
MODEL="Qwen/Qwen3-Coder-30B-A3B-Instruct"
EXPERT_CONFIG="./results/expert_configs/my_expert_config.json"
PARAM_PATTERN="auto"  # 自動検出（推奨）または手動指定

# データセット
TRAIN_DATASETS="/path/to/train.jsonl"
VAL_DATASETS="/path/to/val.jsonl"

# 並列設定（GPU 数に応じて調整）
EXPERT_PARALLEL=8
TENSOR_PARALLEL=1
PIPELINE_PARALLEL=1
CONTEXT_PARALLEL=1

# 学習パラメータ
NUM_TRAIN_EPOCHS=3
MICRO_BATCH_SIZE=1
GLOBAL_BATCH_SIZE=16
LEARNING_RATE="1e-5"
MAX_LENGTH=16384
```

#### 学習ジョブの投入

```bash
sbatch scripts/ms_swift_megatron_esft_sample.sh
```

#### 学習の監視

```bash
# ジョブステータスの確認
squeue -u $USER

# ログの確認
tail -f /fsx/logs/ms-swift/job_<JOB_ID>/job_<JOB_ID>_node0.log
```

#### 設定詳細

**並列戦略**

`EXPERT_PARALLEL × TENSOR_PARALLEL × PIPELINE_PARALLEL × DATA_PARALLEL = 総 GPU 数` を確保してください

> ⚠️ ms-swift の mcore_bridge 実装の制限により、現在 `TENSOR_PARALLEL` と `CONTEXT_PARALLEL` はサポートされていません。

MoE モデルでは、expert を並列化するために `EXPERT_PARALLEL` を優先的に使用してください。

**バッチサイズの計算**

`GLOBAL_BATCH_SIZE` は `MICRO_BATCH_SIZE × data_parallel_size` で割り切れる必要があります

ここで：`data_parallel_size = 総GPU数 / (EP × TP × PP × CP)`

例：16 GPU, EP=8, TP=1, PP=1, CP=1
- data_parallel_size = 16 / 8 = 2
- GLOBAL_BATCH_SIZE は 2 で割り切れる必要があります（例：2, 4, 8, 16, ...）

**パラメータ命名パターン**

スクリプトはデフォルトで自動検出します（`PARAM_PATTERN="auto"`）。一般的なパターン：

| モデル | パラメータパターン |
|-------|-------------------|
| Qwen3-MoE, DeepSeek-V2 | `model.layers.{layer}.mlp.experts.{expert}` |
| 一部の MoE モデル | `model.layers.{layer}.experts.{expert}` |
| Mixtral | `model.layers.{layer}.block_sparse_moe.experts.{expert}` |

手動指定：
```bash
PARAM_PATTERN="model.layers.{layer}.mlp.experts.{expert}"
```

## その他のツール

### 学習可能パラメータの手動生成

学習前にパラメータ生成をテストできます：

```bash
# 自動検出モード
python scripts/generate_trainable_params.py expert_config.json \
    --model "Qwen/Qwen3-Coder-30B-A3B-Instruct" \
    --pattern auto \
    --format regex \
    --megatron \
    --ep-size 8

# 特定のパターンを使用
python scripts/generate_trainable_params.py expert_config.json \
    --pattern "model.layers.{layer}.mlp.experts.{expert}"
```

### 新しいモデルのサポート追加

新しい MoE モデルに expert ルーティングログサポートを追加する方法については、`model_patch/README.md` を参照してください。

## Citation

原論文：[Let the Expert Stick to His Last: Expert-Specialized Fine-Tuning for Sparse Architectural Large Language Models](https://arxiv.org/abs/2407.01906)

## License

詳細は [LICENSE-CODE](./LICENSE-CODE) を参照してください。

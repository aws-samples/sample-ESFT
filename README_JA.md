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
├── ms-swift/                       # ms-swift (git submodule)
├── scripts/
│   ├── expert/                     # Expert 分析スクリプト
│   │   ├── get_expert_scores_hf.py       # Expert ルーティング統計の収集
│   │   ├── generate_expert_config.py     # Expert 設定ファイルの生成
│   │   ├── run_get_exp_scores_local.sh   # ローカル実行スクリプト
│   │   └── run_get_exp_scores_slurm.sh   # SLURM 投入スクリプト
│   ├── generate_trainable_params.py      # 学習可能パラメータリストの生成
│   ├── ms_swift_megatron_esft_local.sh   # ローカル学習スクリプト
│   └── ms_swift_megatron_esft_slurm.sh   # SLURM 学習スクリプト
├── results/                        # 出力ディレクトリ
│   ├── expert_configs/             # 生成された expert 設定ファイル
│   └── expert_scores/              # Expert ルーティング統計
├── test_chat_template.py           # Chat template テストスクリプト
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

```bash
# submodule を含めてクローン
git clone --recurse-submodules <REPO_URL>
cd ESFT

# または既にクローン済みの場合
git submodule update --init --recursive

# 仮想環境の作成
python -m venv .venv
source .venv/bin/activate

# 依存関係のインストール
pip install -r requirements.txt
pip install -e ./ms-swift
```

### Step 1: Expert ルーティング統計の収集

学習データで推論を実行し、各トークンの expert ルーティング情報を記録します。

#### ローカル実行

```bash
# 設定を編集
vim scripts/expert/run_get_exp_scores_local.sh

# 主な設定：
MODEL="Qwen/Qwen3-Coder-30B-A3B-Instruct"
EVAL_DATASET="/path/to/train.jsonl"
N_SAMPLE_TOKENS=524288
GPUS_PER_PROCESS=8

# 実行
bash scripts/expert/run_get_exp_scores_local.sh
```

#### SLURM クラスタ

```bash
# 設定を編集
vim scripts/expert/run_get_exp_scores_slurm.sh

# ジョブ投入
sbatch scripts/expert/run_get_exp_scores_slurm.sh
```

**GPU 設定の推奨**：
- 235B モデル (GLM-4.5-Air): `GPUS_PER_PROCESS=8`（1 プロセス）
- 30B モデル (Qwen3-Coder-30B-A3B): `GPUS_PER_PROCESS=2`（4 プロセス）
- 7B モデル: `GPUS_PER_PROCESS=1`（8 プロセス）

**出力**：Expert ルーティングログは `results/expert_scores/`（ローカル）または `/fsx/outputs/esft/`（SLURM）に保存されます

### Step 2: Expert 設定ファイルの生成

ルーティング統計に基づいて `expert_config.json` を生成します：

```bash
python scripts/expert/generate_expert_config.py \
    --model_name_or_path Qwen/Qwen3-Coder-30B-A3B-Instruct \
    --expert_scores_dir ./results/expert_scores/job_xxx \
    --output_path ./results/expert_configs/my_expert_config.json \
    --score_function token \
    --top_p 0.2 \
    --train_shared_experts \
    --train_non_expert_modules
```

**パラメータ説明**：
| パラメータ | 説明 |
|-----------|------|
| `--model_name_or_path` | HuggingFace モデル名またはローカルパス |
| `--expert_scores_dir` | Step 1 で出力されたルーティングログディレクトリ |
| `--output_path` | expert_config.json の出力パス |
| `--score_function` | `token`（カウントベース）または `gate`（重みベース） |
| `--top_p` | 累積スコア閾値（例：0.2 = 上位 20%） |
| `--train_shared_experts` | shared expert を学習（オプション） |
| `--train_non_expert_modules` | attention、embedding などを学習（オプション） |

**出力形式**：
```json
{
  "experts": {
    "1": [79, 51, 122, 70, 96],
    "2": [42, 78, 22]
  },
  "shared_experts": true,
  "non_expert_modules": false
}
```

### Step 3: ms-swift Megatron で ESFT 学習を実行

#### ローカル実行

```bash
# 設定を編集
vim scripts/ms_swift_megatron_esft_local.sh

# 主な設定：
MODEL="Qwen/Qwen3-Coder-30B-A3B-Instruct"
EXPERT_CONFIG="./results/expert_configs/my_expert_config.json"
TRAIN_DATASETS="/path/to/train.jsonl"
EXPERT_PARALLEL=8

# 実行
bash scripts/ms_swift_megatron_esft_local.sh
```

#### SLURM クラスタ

```bash
# 設定を編集
vim scripts/ms_swift_megatron_esft_slurm.sh

# ジョブ投入
sbatch scripts/ms_swift_megatron_esft_slurm.sh

# 監視
squeue -u $USER
tail -f /fsx/logs/ms-swift/job_<JOB_ID>/job_<JOB_ID>_node0.log
```

#### 設定詳細

**並列戦略**

`EXPERT_PARALLEL × TENSOR_PARALLEL × PIPELINE_PARALLEL × DATA_PARALLEL = 総 GPU 数`

> ⚠️ ms-swift の mcore_bridge 実装の制限により、現在 `TENSOR_PARALLEL` と `CONTEXT_PARALLEL` はサポートされていません。

**バッチサイズの計算**

`GLOBAL_BATCH_SIZE` は `MICRO_BATCH_SIZE × data_parallel_size` で割り切れる必要があります

ここで：`data_parallel_size = 総GPU数 / (EP × TP × PP × CP)`

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

```bash
python scripts/generate_trainable_params.py expert_config.json \
    --model "Qwen/Qwen3-Coder-30B-A3B-Instruct" \
    --pattern auto \
    --format regex \
    --megatron \
    --ep-size 8
```

### 新しいモデルのサポート追加

新しい MoE モデルに expert ルーティングログサポートを追加する方法については、`model_patch/README.md` を参照してください。

## Citation

原論文：[Let the Expert Stick to His Last: Expert-Specialized Fine-Tuning for Sparse Architectural Large Language Models](https://arxiv.org/abs/2407.01906)

## License

詳細は [LICENSE-CODE](./LICENSE-CODE) を参照してください。

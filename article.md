# 物体検出の「最後の1マイル」を埋める — peaceofcake という選択肢

## 2024年、物体検出に何が起きたのか

YOLOv1が世界を驚かせたのが2016年。あれから8年、物体検出の精度は劇的に向上した。

だが、ここで一つ問いたい。

**あなたは最新の物体検出モデルを、自分のアプリに組み込めるだろうか？**

論文は読んだ。デモも動かした。でも「自分のデータで学習して、iPhoneで動かす」となった途端、手が止まる。YAMLの設定ファイルを何十行も書き、依存関係の衝突を解決し、エクスポートスクリプトをゼロから書き、Swiftで推論パイプラインを組む——その道のりは、想像以上に長い。

2024年、DETRベースの物体検出は新たなフェーズに入った。**D-FINE** と **RF-DETR** という2つのモデルが、精度とリアルタイム性能の両面でブレイクスルーを達成した。しかし、その恩恵を実プロダクトに届けるには、まだ「最後の1マイル」が残っている。

**peaceofcake** は、その1マイルを埋めるためのライブラリだ。

```bash
pip install peaceofcake
```

たった1行で始まる。

## 3行で物体検出が動く世界

```python
from peaceofcake import DFINE

model = DFINE("dfine-l-coco")
results = model("image.jpg")
```

これだけだ。モデルの重みは自動でダウンロードされ、キャッシュされる。GPUがあればGPUで、なければCPUで走る。結果はバウンディングボックス、クラスラベル、信頼度スコアが構造化されたオブジェクトとして返ってくる。

RF-DETRも同じインターフェースで使える。

```python
from peaceofcake import RFDETR

model = RFDETR("rfdetr-l-coco")
results = model("image.jpg")
```

モデルが違っても、APIは同じ。これがpeaceofcakeの設計思想の根幹だ。

## D-FINE — 「ぼんやり」から「くっきり」へ

D-FINEの正式名称は「DETR with Fine-grained Distribution Refinement」。名前が示す通り、このモデルの核心は**バウンディングボックスの予測方法**にある。

従来のDETR系モデルは、物体の位置を4つの数値（x, y, w, h）として**一発で当てに行く**。これは人間に例えると、「あの建物の高さは？」と聞かれて「37.2メートル！」と即答するようなものだ。

D-FINEは違う。まず「30〜40メートルくらいかな」という**確率分布**を出し、次に「35〜38メートルだな」と絞り込み、最終的に「37.2メートル」に到達する。ぼんやりとした推定を、段階的に研ぎ澄ましていく。

```
粗い分布 → 精緻化 → さらに精緻化 → 最終予測
```

この**Fine-grained Distribution Refinement**により、D-FINEはRT-DETRを超える精度を、リアルタイムの推論速度で実現した。

アーキテクチャは明快だ。

```
画像 → HGNetv2 (Backbone) → HybridEncoder → DFINE Decoder → 検出結果
```

HGNetv2バックボーンはNano/Small/Medium/Large/XLargeの5サイズ。モバイル向けの軽量推論から、精度最優先のサーバーサイドまで、一つのアーキテクチャでカバーする。

## RF-DETR — 「基盤モデル」の風が物体検出にも

一方のRF-DETRは、Roboflowが2024年末にリリースした「Real-Time Foundational Object Detection」モデルだ。

「Foundational」という言葉に注目してほしい。大規模言語モデルの世界で「基盤モデル」が革命を起こしたように、物体検出にもその波が押し寄せている。大量のデータで事前学習した汎用的な検出能力を、少量のドメイン固有データでファインチューニングする。RF-DETRは、そのパラダイムをリアルタイム物体検出に持ち込んだ。

peaceofcakeは、この2つの最先端モデルを**同じ3行のAPI**で扱えるようにした。

## なぜ「もう一つのラッパー」が必要なのか

世の中にはモデルのラッパーライブラリが溢れている。では、peaceofcakeは何が違うのか。

答えは**スコープ**にある。

多くのラッパーは「推論を簡単にする」ところで止まる。peaceofcakeは違う。**学習 → エクスポート → モバイル実機推論**という、プロダクト開発のフルサイクルを1つのパッケージでカバーする。

### 学習：YOLOフォーマットをそのまま食える

```python
model = DFINE("dfine-m-coco")
model.train(data="dataset.yaml", epochs=50, batch_size=16)
```

RoboflowやLabel Studioでアノテーションし、YOLOフォーマットでエクスポートしたデータセットを、そのまま渡せる。内部でCOCOフォーマットへの自動変換が走り、学習スケジュールのスケーリング、EMA、AMPが透過的に適用される。

COCO形式のデータセットもそのまま使える。フォーマットを意識する必要がない。

### エクスポート：1メソッドで3フォーマット

```python
model.export("onnx")                    # サーバーサイド推論
model.export("coreml", precision="FLOAT16")  # iOS / macOS
model.export("tensorrt")                # NVIDIA GPU最適化
```

CoreMLエクスポートでは、学習時にのみ必要なデノイジング機構や補助ヘッドが自動的に除去される。出力形式はiOSのVisionフレームワークと直接互換のある`confidence` + `coordinates`形式。NMSはモデル外で行う設計を採っており、信頼度閾値をUIからリアルタイムに変更できる。

### CLIもある

Pythonを書かずとも、ターミナルから全機能にアクセスできる。

```bash
poc predict source=photo.jpg conf=0.3
poc train model=dfine-m-coco data=my_dataset.yaml epochs=100
poc export model=dfine-l-coco format=coreml precision=FLOAT16
```

## 設計の勘所 — 複雑さをどこに隠すか

peaceofcakeの内部設計で、特に巧みだと感じるポイントが3つある。

### 1. Strategy Pattern による拡張性

```python
class DFINE(BaseModel):
    @property
    def task_map(self):
        return {
            "detect": {
                "predictor": DFINEPredictor,
                "trainer": DFINETrainer,
                "exporter": DFINEExporter,
                "validator": DFINEValidator,
            }
        }
```

`BaseModel`が`predict()`, `train()`, `export()`, `val()`の呼び出しを受け、`task_map`から適切なクラスをディスパッチする。D-FINEとRF-DETRでPredictor/Trainer/Exporterの実装は全く異なるが、ユーザーから見えるAPIは完全に同一だ。

将来、セグメンテーションやポーズ推定のタスクが追加されても、`task_map`にエントリを足すだけで対応できる。

### 2. Lazy Import で起動を軽く

```python
# peaceofcake/__init__.py
def __getattr__(name):
    if name == "RFDETR":
        from peaceofcake.models.rfdetr import RFDETR
        globals()["RFDETR"] = RFDETR
        return RFDETR
    raise AttributeError(...)
```

RF-DETRはtransformersライブラリに依存しており、importするだけで数秒かかる。peaceofcakeは`__getattr__`を使い、**RFDETRが実際に参照されるまでimportを遅延**させる。D-FINEだけ使うユーザーは、transformersがインストールされていなくても問題ない。

学習時のみ必要な`src.data`, `src.optim`, `src.nn.criterion`も、`train()`メソッドの内部で初めてimportされる。推論しか使わないなら、これらのモジュールは一切読み込まれない。

### 3. モデル名の「意図推定」

```python
DFINE("dfine-l-coco")           # レジストリ名 → 自動ダウンロード
DFINE("path/to/custom.pth")     # ローカルファイル
DFINE("dfine_l_coco.pth")       # ファイル名だけ → レジストリから照合
DFINE("dfine-n")                 # 重みなし → ランダム初期化
```

文字列一つに対して、レジストリ照合 → ローカルファイル確認 → ファイル名マッチング → サイズ推定という多段のフォールバックが走る。ローカルのチェックポイントを渡した場合、ファイル名からモデルサイズの推定を試み、それでも判定できなければ**パラメータ数から逆算**する。

```python
n = sum(v.numel() for v in model_state.values())
if n < 6_000_000: return "n"    # Nano
elif n < 15_000_000: return "s"  # Small
elif n < 25_000_000: return "m"  # Medium
...
```

ユーザーが「何も考えずにパスを渡すだけ」で正しく動くために、裏側でこれだけの推定ロジックが動いている。

## コミットログに刻まれた「現場の知恵」

コードを読むだけでは見えない、コミットログに刻まれた試行錯誤がある。ここには、研究コードをプロダクションに持ち込む際のリアルな教訓が詰まっている。

### CUDAがハングする

学習初期、モデルが幅や高さが負のバウンディングボックスを出力することがある。オリジナルのD-FINE実装では`assert`でこれを検出していた。CPUなら例外が飛ぶだけだ。だがCUDAカーネル内のassert失敗は、**GPUデバイス全体をハングさせる**。

peaceofcakeはこれを`clamp`に置き換えた。ところが最初の修正ではインプレース操作を使ってしまい、autogradの計算グラフが壊れた。2コミット目でようやく本当に修正された。**1つのバグを直すと、別のバグが顔を出す。** これが研究コードを実戦投入するリアルだ。

### YOLOフォーマットに「仕様」はない

YOLOフォーマットは、事実上の標準はあっても厳密な仕様がない。パスの指定方法（`train: images/train` vs `path: ./` + `train: images/train`）、ラベルファイルの空行、クラス数の記述方法——ツールごとに微妙に異なる。peaceofcakeは、Roboflow/Ultralytics/Label Studioの出力形式すべてに対応するため、パス解決ロジックだけで50行以上を費やしている。

### チェックポイントにクラス名を埋め込む

カスタムデータセットで学習したモデルを配布する際、クラス名の情報が失われがちだ。peaceofcakeはチェックポイントの`.pth`ファイルに`class_names`キーを直接埋め込む。別途メタデータファイルを管理する必要がない。

## Pythonから実機まで、何マイルか

物体検出モデルを「使える」状態にするまでの距離を、テーブルで整理してみよう。

| ステップ | 従来のワークフロー | peaceofcake |
|:---|:---|:---|
| インストール | リポをクローン、環境構築 | `pip install peaceofcake` |
| 推論 | コンフィグ作成、スクリプト実行 | 3行のPython |
| 学習 | データ変換、設定ファイル調整 | `model.train(data="data.yaml")` |
| エクスポート | 専用スクリプト作成 | `model.export("coreml")` |
| iOS実機 | 推論パイプラインをゼロから実装 | `DFINEDemo/` をビルド |

**5ステップのうち、すべてでワンライナーか、用意されたコードで完結する。** これが「最後の1マイル」を埋めるということだ。

## 全部Apache 2.0 — 商用利用への最短距離

物体検出をプロダクトに組み込む際、精度やAPIの使いやすさと同じくらい重要なのが**ライセンス**だ。

ここで多くの開発者がつまずく。YOLOシリーズの最新版（Ultralytics YOLO）はAGPL-3.0ライセンスで、商用利用には有償ライセンスの購入が必要になる。優れたモデルが目の前にあるのに、ライセンスの壁で採用を断念する——そんな経験をした人は少なくないだろう。

peaceofcakeが採用するD-FINEとRF-DETRは、**どちらもApache 2.0ライセンス**だ。peaceofcake自体もApache 2.0。つまり、ライブラリもモデルも、学習済みの重みも、すべて**商用利用が無償で可能**だ。改変・再配布・組み込み、何でも自由。特許条項も含まれており、コントリビューターからの特許訴訟リスクも軽減される。

| ライブラリ | モデルライセンス | 商用利用 |
|:---|:---|:---|
| Ultralytics (YOLO) | AGPL-3.0 | 有償ライセンス必要 |
| **peaceofcake (D-FINE)** | **Apache 2.0** | **無償で可能** |
| **peaceofcake (RF-DETR)** | **Apache 2.0** | **無償で可能** |

スタートアップが自社プロダクトに組み込む。受託開発でクライアントに納品する。組み込みデバイスのファームウェアに統合する。どのシナリオでも、ライセンス費用はゼロだ。

**精度でYOLOを上回り、ライセンスは完全フリー。** これがDETRベースモデルを選ぶ、もう一つの理由だ。

## 物体検出の未来と、ツールの役割

コンピュータービジョンの歴史は、「精度の競争」から「アクセシビリティの競争」へとフェーズが移りつつある。

2012年のAlexNet以降、ImageNetでの精度向上が研究の主戦場だった。物体検出でも、COCO benchmarkのmAP向上が論文の主要な貢献とされてきた。だが2024年、D-FINEやRF-DETRが示したように、DETRベースのモデルは十分な精度とリアルタイム性能を両立する段階に達した。

次の競争軸は**「誰でも使える」**だ。

Ultralytics（YOLOv8/YOLO11）がその先駆者であり、`pip install ultralytics` でYOLOシリーズの全機能にアクセスできるエコシステムを構築した。しかし、AGPLライセンスの制約は商用利用のハードルとなっている。peaceofcakeは、Ultralyticsが確立した「簡単に使える」体験を、**Apache 2.0のDETRベースモデル**で実現する。

モデルの精度がコモディティ化する時代、差別化の源泉は「いかに速く、いかに自由にプロダクトに組み込めるか」に移る。研究者がarXivに投稿してから、アプリ開発者がそのモデルをユーザーの手元に届けるまでの時間とコスト。その両方を限りなくゼロに近づけること。それが、peaceofcakeのようなツールが果たす役割だ。

物体検出は、もう難しくない。そして、もう高くない。

```bash
pip install peaceofcake
```

朝飯前だ。

---

**Repository**: [peaceofcake](https://github.com/john-rocky/peaceofcake)
**PyPI**: [peaceofcake](https://pypi.org/project/peaceofcake/)
**License**: Apache 2.0

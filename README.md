# MotionPredictor
物体検出および姿勢推定の認識結果を表示&amp;保存→LSTMで動作推定を行うプログラムファイル(研究src)

### main.py

- 主な設定パラメーター
    
    —source: ビデオのパスを設定
    
    —object-weights: 物体検出のweightsを設定
    
    —conf-thres: 物体検出の物体を表示するときの信頼度の閾値
    
    —iou-thres: 物体検出の物体を表示するときのIOUの閾値
    
- 実行すると、物体検出と姿勢推定が行われビデオが再生される。そのビデオをoutput.aviへ保存し、骨格csvデータはcsv/のディレクトリ内に保存される。

### processor.py

class ImageProcessorは物体検出と姿勢推定を処理する。また、骨格csv化の処理を行う。

### movencorder.py

実行すると指定したビデオを指定した画像サイズ(変数sizeを設定)や明るさ(変数alphaを設定)に変換し保存する。

### csv_devider.py

csvディレクトリに保存された骨格csvデータをwidth_of_csvの長さにクロップしcsv/devided_csv/[元のcsvファイル名]/の中に保存する。

### motion_predictor.py

csv_devider.pyでクロップしたcsvファイルから動作推定を行う。出力はsoftmax関数に食わせる前の各動作の信頼度ベクトル(csvファイル数×9)

動作推定の学習済み重みはweights/のディレクトリ内に保存し、引用する。

コードはLSTM_Experimentリポジトリに公開したコードをベースに記述した。

### video2img.py

動作推定の目視によるアノテーションのために、ビデオデータをフレームカウント枚の画像へ変換する。

### その他ディレクトリ

projects:物体検出の学習パラメータの保存先(学習を行った後に自動で生成)

posedatas:この通りの名前で空ディレクトリを作成すれば、movencoder.pyが中にあるビデオデータを認識する。

data:物体検出のための学習データセット情報が入っている

datasets:物体検出の学習データセットを作るためのコードがある。voc2coco.pyやcoco2yolo.pyなど

csv:骨格csvデータの保存先(csv_devider.pyを実行すると自動で生成)

weights:デフォルトの物体検出用のweightやmotion_recognition用のweightが置かれている

※物体検出用の画像・アノテーションデータが含まれたディレクトリは研究室内ドライブに格納。
https://drive.google.com/drive/folders/142Zo0TvDO9vsz6UfEtYQe7wS-4TMnOhL

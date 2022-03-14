# MotionPredictor
物体検出および姿勢推定の認識結果を表示&amp;保存→LSTMで動作推定を行うプログラムファイル(研究src)

### main.py

- 主な設定パラメーター
    
    —source: ビデオのパスを設定
    
    —object-weights: 物体検出のweightsを設定
    
    —conf-thres: 物体検出の物体を表示するときの信頼度の閾値
    
    —iou-thres: 物体検出の物体を表示するときのIOUの閾値
    
- 実行すると、物体検出と姿勢推定が行われビデオが再生される。そのビデオをoutput.aviへ保存し、骨格csvデータはcsv/のディレクトリ内に保存される。

### processor.py(一部自作)

class ImageProcessorは物体検出と姿勢推定を処理する。また、独自で骨格csv化の処理を追加した。

### movencorder.py(自作)

実行すると指定したビデオを指定した画像サイズや明るさに変換し保存する。

### csv_devider.py(自作)

csvディレクトリに保存された骨格csvデータをwidth_of_csvの長さにクロップしcsv/devided_csv/[元のcsvファイル名]/の中に保存する。

### motion_predictor.py(自作)

csv_devider.pyでクロップしたcsvファイルから動作推定を行う。出力はsoftmax関数に食わせる前の各動作の信頼度ベクトル(csvファイル数×9)

動作推定の学習済み重みはweights/のディレクトリ内に保存し、引用する。

コードはMotionRecognitionのLSTM-Copy4.ipynbをベースに記述した。

### video2img.py(自作)

動作推定の目視によるアノテーションのために、ビデオデータをフレームカウント枚の画像へ変換する。

### その他ディレクトリ

projects:実行結果の保存先

videodatas:videoデータの格納フォルダ

data:物体検出のための学習データセット情報が入っている(詳しくは物体検出のメモへ)

datasets:物体検出の学習データセットを作るためのコードがある。voc2coco.pyやcoco2yolo.pyなど(詳しくは物体検出のメモへ)

csv:骨格csvデータの保存先

weights:デフォルトの物体検出用のweightやmotion_recognition用のweightが置かれている

# 動画fixture仕様

fixtureは`deployment_tests/work/fixtures/`へ生成し、Git管理しません。基になる実映像は
顔・性器・両方・どちらもない区間、hard cut、暗転、動き、複数人数を含む短い区間を
選びます。製品pipelineは必ずGUIから実行し、FFmpegはfixtureのcontainer/codec/
metadata変種を事前作成する目的だけに使います。

## 必須fixture

| 名前 | 長さ | 映像 | container/codec | 音声 | 特記事項 |
| --- | ---: | --- | --- | --- | --- |
| `golden_1080p2997_h264_aac.mp4` | 6分以下 | 1920x1080, 30000/1001 CFR, yuv420p | MP4/H.264, B-frame | AAC | 短時間の顔・性器・cut基準 |
| `golden_short.mp4` | 20秒 | 1920x1080, 30000/1001 | MP4/H.264 | AAC | negative queueの後続正常入力 |
| `interlaced_1080i2997_h264.mp4` | 20秒 | 1920x1080, 30000/1001, TFF | MP4/H.264 | AAC | bwdif正規化 |
| `landscape_720p24_h265.mkv` | 30秒 | 1280x720, 24 CFR, 10-bit | MKV/H.265 Main10 | AAC | 10-bit decodeとMKV |
| `portrait_720x1280_30_h264.mp4` | 30秒 | 720x1280, 30 CFR | MP4/H.264 | AAC | portrait座標とoverlay |
| `uhd_2160p24_h265_noaudio.mp4` | 20秒 | 3840x2160, 24 CFR | MP4/H.265 | なし | 4K full pipeline |
| `vfr_pts_gap_h264.mp4` | 30秒 | 1920x1080, VFR | MP4/H.264 | AAC | frame duration差と非一様PTS |
| `long_gop_bframes.mov` | 30秒 | 1920x1080, 30000/1001 | MOV/H.264, GOP 250 | AAC | seekと分割境界 |
| `short_60fps.mp4` | 20秒 | 1280x720, 60 CFR | MP4/H.264 | AAC | frame countと進捗 |
| `unicode_日本語 space.mp4` | 20秒 | 1920x1080, 30 CFR | MP4/H.264 | AAC | Unicode・空白path |
| `h264_noaudio.mp4` | 20秒 | 1920x1080, 30 CFR | MP4/H.264 | なし | streamなし |
| `invalid_truncated.mp4` | - | 意図的に末尾を切断 | MP4/H.264 | 不定 | expected failure |
| `codino_120m_mixed.mp4` | 120分 | 1920x1080, 30 CFR | MP4/H.264 | AAC | 唯一のV3 load test |

## 内容の作り方

短尺は同じ先頭区間のcodec違いだけにせず、次の5区間から組みます。

1. 顔と性器の両方、複数mask
2. 顔だけ
3. 性器だけ
4. 検出対象なし
5. hard cut、暗転、fade、速いcamera motion

同じ内容をencode違いで比較するfixtureと、内容自体が違うfixtureを分けます。これに
よりdecode差と検出密度差を混同しません。

120分fixtureは15分素材の同一反復ではなく、5区間の順序と長さを変えたblockを連結
します。少なくとも前半・中央・後半で次が異なるようにします。

- 1 frame当たりの検出／mask／顔数
- cut頻度
- track寿命とgap
- 音声の有無やformatは途中で変えず、container自体は正常に保つ

## 生成後の事前検査

pipeline実行前に各fixtureの期待値をprobe結果へ固定します。

- container、video/audio codec、profile、pix_fmt
- coded/display width、height、rotation metadata
- avg/r frame rate、time base、start time、duration
- video packet count、audio stream count
- PTS/DTS単調性、VFRの場合のdelta分布
- full decodeが成功すること（`invalid_truncated.mp4`を除く）

4Kは単なるmetadata書換えではなく、実際に3840x2160 frameをdecodeさせます。VFRは
headerだけでなくpacket timestampを非一様にします。portraitはrotation metadataだけ
の変種をP2として追加できますが、P0は実pixelが720x1280のfixtureを使います。

## 時間節約

- fixtureは一度だけ作り、全ケースで再利用する。
- H.264/H.265変種は短尺に限定する。
- 4Kは20秒だけにし、座標・decode・overlay経路を確認する。
- V3へ渡す長時間はこの120分fixtureだけにする。
- 120分fixtureの作成は可能ならstream copyで行い、再encodeを避ける。

# Samples

Vietnamese, voice-cloned from `reference_3s.wav` (3.22s), generated on MPS/fp16
with trimmed CFG. The only thing that changes between files is `num_step`.

| Sentence | Text |
|---|---|
| 003 | Nếu bạn thấy video này hữu ích, hãy để lại một lượt thích và theo dõi kênh nhé. |
| 009 | Diện tích toàn phần của hình trụ bằng diện tích xung quanh cộng hai lần diện tích đáy. |
| 021 | Hình lăng trụ đứng có các mặt bên đều là hình chữ nhật. |

32 steps is the OmniVoice default. 12 is what this repo recommends — measured
indistinguishable over 48 sentences and 2.34x faster. 8 is included because the
numbers say it breaks (WER nearly doubles, p=0.009) and that is easier to hear
than to read.

---
license: cc-by-nc-4.0
language:
- fa
pipeline_tag: text-to-speech
tags:
- text-to-speech
- persian
- pocket-tts
---

# Cilla Lite: Persian text to speech (100M)

**Gated, non-commercial** (CC BY-NC 4.0): ask for access on this page. Cilla Lite is the small 6-layer model for CPUs, distilled from Cilla with guidance baked in (one pass per frame). It speaks Persian in the voice of a short audio
prompt. Built on Pocket TTS by Kyutai.

| | |
|---|---|
| parameters used for generation | 100M (6-layer flow language model + Mimi decoder; the file holds 110M, including the Mimi encoder that runs once per voice prompt) |
| sample rate | 24 kHz, mono |
| language | Persian (fa) |
| sibling | `mehdi-hf/cilla`: Cilla, 326M |

Speed: **1.88x real time** on one thread of a cloud x86 CPU (Intel(R) Xeon(R) CPU @ 2.80GHz), measured while making the samples below.

## Use

```bash
pip install "pocket-tts @ git+https://github.com/mallahyari/pocket-tts@main"   # the fork, not PyPI: see below
hf auth login                                                                  # the repo is gated: ask for access on its page first
hf download mehdi-hf/cilla-lite --exclude "model.safetensors" --local-dir cilla-lite && cd cilla-lite   # the code, prompts and config
python speak_fa.py --voice prompts/voice1.wav --text "سلام، حال شما چطور است؟" --out hello.wav   # fetches the weights once
```

`speak_fa.py` normalises the text, speaks **one sentence at a time** from the same prompt, trims the silence each piece
opens with and joins them with a short pause. Do the same in your own code: cutting a sentence into 50-token pieces (the
old Pocket TTS default) made these models repeat words, once the name «ساندر پیچای» twice in a row.

**The config needs the fork.** `config.yaml` sets `capitalize_first_letter`, `append_terminal_punctuation` and
`pad_with_spaces_for_short_inputs` to false, which the released `pocket-tts` does not know and rejects. The loud error is
deliberate: without them the package upper-cases or drops the first letters of Persian text.

## Text

Write ordinary Persian with punctuation: commas and full stops shape the phrasing. The model reads the ezafe from the
context (there are no ezafe marks to type); it still gets some wrong.

## Voices

The voice is cloned from a prompt: a whole short clip (3 to 10 s) that ends at a pause, in a quiet room. Prompts cut
mid-word make the model sound broken. `prompts/` has two neutral ones, `voice1.wav` (a woman) and `voice2.wav` (a man).

In the check, every clip made from each of them was closest to that prompt, so the model answers in the voice you give it.

## Samples

Made with `speak_fa.py`: the female voice is `prompts/voice1.wav`, the male voice is `prompts/voice2.wav`.

**News**

| Female | Male |
|---|---|
| <audio controls preload="none" src="https://huggingface.co/mehdi-hf/cilla-lite/resolve/main/samples/voice1_news.wav"></audio> | <audio controls preload="none" src="https://huggingface.co/mehdi-hf/cilla-lite/resolve/main/samples/voice2_news.wav"></audio> |

> *بر اساس گزارش سازمان هواشناسی، بارش باران و برف در بیشتر استان‌های شمالی و غربی کشور تا پایان هفته ادامه خواهد داشت. از رانندگان خواسته شده است که در جاده‌های کوهستانی با احتیاط بیشتری حرکت کنند و زنجیر چرخ همراه داشته باشند.*

**Story**

| Female | Male |
|---|---|
| <audio controls preload="none" src="https://huggingface.co/mehdi-hf/cilla-lite/resolve/main/samples/voice1_story.wav"></audio> | <audio controls preload="none" src="https://huggingface.co/mehdi-hf/cilla-lite/resolve/main/samples/voice2_story.wav"></audio> |

> *پیرمرد از پنجره‌ی کوچک اتاق به کوچه نگاه کرد. باران ریزی می‌بارید و چراغ‌های زرد خیابان روی آسفالت خیس می‌درخشیدند. آهی کشید و با خودش گفت که این شهر دیگر شهر روزهای جوانی‌اش نیست.*

**Names and technology**

| Female | Male |
|---|---|
| <audio controls preload="none" src="https://huggingface.co/mehdi-hf/cilla-lite/resolve/main/samples/voice1_names.wav"></audio> | <audio controls preload="none" src="https://huggingface.co/mehdi-hf/cilla-lite/resolve/main/samples/voice2_names.wav"></audio> |

> *شرکت گوگل در ماه گذشته از نسخه‌ی تازه‌ای از مدل هوش مصنوعی خود رونمایی کرد. مدیرعامل این شرکت، ساندر پیچای، در سخنرانی خود در شهر سانفرانسیسکو گفت که این مدل می‌تواند متن، تصویر و صدا را با هم بفهمد و در بیش از صد زبان، از جمله فارسی، پاسخ دهد.*

## Evaluation

Recognition by a Persian ASR on real recordings, voices cloned from other clips of the same speaker (step 60,000, EMA; the guidance is built in).
The ASR itself scores 8.6% word error on the real recordings, so that is the floor.

| | |
|---|---|
| real recordings, WER | 6.7% |
| real recordings, CER | 2.2% |
| speaker similarity | 0.951 |

These measure intelligibility and a coarse speaker match; they do not show whether a particular voice is reproduced
(see Voices) or how often the ezafe is right, which needs listening.

## Limitations

- Persian only. Ezafe mistakes happen, mostly on rare words and in long noun phrases.
- Only Persian script is spoken: `speak_fa.py` drops English (Latin-script) words and other foreign characters, such as emoji, from the text without a warning, so write names and terms in Persian letters (for example «گوگل» for Google). Digits are read as Persian words.
- Voice cloning is not exact; it follows the prompt's timbre and style, not every detail.
- Long text must be spoken sentence by sentence (above), or words are repeated or lost.
- Speech made with it should be labelled as synthetic; do not clone a voice without the person's consent.

## Licence and credits

CC BY-NC 4.0: non-commercial use only. Commercial use needs a separate licence: mehdi@oncillaai.com.

Built on Pocket TTS by Kyutai (kyutai-labs/pocket-tts); the Mimi codec is Kyutai's.

"""Speak Persian text with Cilla or Cilla Lite: normalise, one piece per sentence, trim, join.

    python speak_fa.py --voice prompts/voice1.wav --text "سلام، حال شما چطور است؟" --out hello.wav
    python speak_fa.py --repo REPO --voice my_prompt.wav --text-file article.txt --out article.wav

The functions below are copied from training/farsi/synthesize.py of the Cilla training repository by
ops/build_student_hf_repos.py. Cut text only at sentence ends: a cut inside a sentence made these models repeat words.
Needs the pocket-tts fork (pip install "pocket-tts @ git+https://github.com/mallahyari/pocket-tts@main") and, for a
gated repo, `hf auth login`.
"""

import argparse
import logging
import math
import re
from collections.abc import Callable
from pathlib import Path

import numpy as np

from normalize_fa import normalize

logger = logging.getLogger("speak_fa")
PAUSE_SEC = 0.15  # between sentences


HARD_BREAK = re.compile(r"(?<=[.!؟])\s+")


SOFT_BREAK = re.compile(r"(?<=[،؛:])\s+")


BREAK_CHARS = (".", "!", "؟", "،", "؛", ":")


EZAFE_MARK = "\x00"  # never present: plain text carries no ezafe marks


def strip_ezafe(text: str) -> str:
    """The text as the model must see it: no ezafe marks."""
    return text.replace(EZAFE_MARK, "")


def _binds_to_next(word: str) -> bool:
    return word.endswith(EZAFE_MARK)


CAP_RATIO = 0.97


MAX_SPLIT_DEPTH = 2


def _cap_seconds(model: "TTSModel", text: str) -> float:
    """The length cap pocket-tts will apply to this text, in seconds.

    Mirrors TTSModel._estimate_max_gen_len. Read from the model where possible so
    this tracks upstream; the fallbacks are the values as of pocket-tts 3.0.
    """
    tokens = model.flow_lm.conditioner.prepare(text).shape[1]
    tps = getattr(model, "_TOKENS_PER_SECOND_ESTIMATE", 3.0)
    pad = getattr(model, "_GEN_SECONDS_PADDING", 2.0)
    frame_rate = model.config.mimi.frame_rate
    return math.ceil((tokens / tps + pad) * frame_rate) / frame_rate


def trim_silence(a: np.ndarray, sample_rate: int, keep_ms: float = 120.0) -> np.ndarray:
    """Cut the silence a generation opens and closes with.

    Chunks are generated independently and each opens by producing silence --
    800 ms and 1280 ms on the two chunks of one measured sentence. Concatenated,
    the tail of one plus the head of the next is an audible hole in the middle
    of a phrase, and no pause setting controls it because nothing inserted it.
    Trim to the speech so the requested gap is the only gap.

    Threshold and margin are deliberately generous. A first attempt at 0.15 RMS
    with 40 ms of margin removed the gap and was still wrong in principle: a
    voiced stop is brief and quiet, so a detector set that high locks onto the
    following vowel and eats the consonant. Measured at these settings, speech
    content changes by 40 ms on a 2.8 s chunk, i.e. window rounding.
    """
    if a.size == 0:
        return a
    rms = float(np.sqrt((a.astype(np.float64) ** 2).mean()))
    if rms <= 0:
        return a
    win = int(0.02 * sample_rate)
    thr = 0.04 * rms
    loud = [
        i
        for i in range(0, max(len(a) - win, 1), win)
        if np.sqrt((a[i : i + win].astype(np.float64) ** 2).mean()) > thr
    ]
    if not loud:
        return a
    margin = int(keep_ms / 1000.0 * sample_rate)
    return a[max(0, loud[0] - margin) : min(len(a), loud[-1] + win + margin)]


def generate_chunk(
    model: "TTSModel",
    state: "ModelState",
    text: str,
    *,
    frames_after_eos: int | None = None,
    sample_rate: int | None = None,
    retries: int = 1,
    seam_sec: float = 0.05,
    max_tokens: int | None = None,
    _depth: int = 0,
) -> np.ndarray:
    """Generate one chunk, recovering from runaway (no-EOS) generations.

    The model was trained on ~3.8 s utterances (~11 tokens). Ask it for much more
    and it may never learn to stop, running to the cap and emitting repetition --
    9% of held-out items in the v2 baseline. Two mitigations, in order of cost:

    1. **Retry.** Sampling is stochastic, so a marginal chunk often terminates on
       a second attempt. Measured 2 of 5 runaways recovered this way.
    2. **Split and recurse.** The other 3 were stuck deterministically -- every
       seed ran to the cap, because a generation that never emits EOS always
       produces exactly cap-length audio. Halving the text fixed all 3, since
       both halves land back inside the trained distribution.

    Lowering `eos_threshold` was also tried and is *not* a fix: stuck chunks
    ignored it down to -6.0, and where it did fire it truncated to a third of the
    expected length, trading a loop for a cut-off word.

    `max_tokens` is passed on to pocket-tts, whose own splitter otherwise cuts
    anything over its 50-token limit again; split_sentences() chunks need it
    raised. None keeps pocket-tts's default.
    """
    sample_rate = sample_rate or int(model.mimi.sample_rate)
    limit = {} if max_tokens is None else {"max_tokens": max_tokens}
    cap = _cap_seconds(model, text)

    shortest: np.ndarray | None = None
    for _ in range(retries + 1):
        audio = np.asarray(
            model.generate_audio(state, text, frames_after_eos=frames_after_eos, **limit),
            dtype=np.float32,
        ).reshape(-1)
        if len(audio) / sample_rate <= CAP_RATIO * cap:
            return audio
        if shortest is None or len(audio) < len(shortest):
            shortest = audio

    words = text.split()
    if _depth >= MAX_SPLIT_DEPTH or len(words) < 4:
        # Out of options: hand back the least-bad attempt rather than nothing.
        logger.warning(f"chunk still hit the length cap after splitting: {text}")
        return shortest if shortest is not None else np.zeros(1, dtype=np.float32)

    half = len(words) // 2
    logger.info(
        f"chunk hit the length cap; splitting {len(words)} words -> {half}+{len(words) - half}"
    )
    seam = np.zeros(int(seam_sec * sample_rate), dtype=np.float32)
    parts = []
    for piece in (" ".join(words[:half]), " ".join(words[half:])):
        if parts:
            parts.append(seam)
        parts.append(
            generate_chunk(
                model,
                state,
                piece,
                frames_after_eos=frames_after_eos,
                sample_rate=sample_rate,
                retries=retries,
                seam_sec=seam_sec,
                max_tokens=max_tokens,
                _depth=_depth + 1,
            )
        )
    return np.concatenate(parts)


def _pack(
    words: list[str],
    count_tokens: Callable[[str], int],
    fits: Callable[[str], bool],
    n_chunks: int,
    target: int,
) -> list[str]:
    """Greedy pack aiming at `target` tokens, with `max_tokens` still a hard stop.

    Chunks are closed on the target only while more are owed; the last one
    absorbs whatever is left, so the tail never starves.
    """
    out: list[str] = []
    cur = ""
    for word in words:
        trial = f"{cur} {word}".strip()
        owed = len(out) < n_chunks - 1
        # Never close after a word whose ezafe binds it to this one.
        bound = bool(cur) and _binds_to_next(cur.split()[-1])
        if cur and not bound and (not fits(trial) or (owed and count_tokens(trial) > target)):
            out.append(cur)
            cur = word
        else:
            cur = trial
    if cur:
        out.append(cur)
    return out


def split_text(
    text: str,
    count_tokens: Callable[[str], int],
    max_tokens: int = 40,
    keep_punct_boundaries: bool = False,
    min_tokens: int = 8,
) -> list[str]:
    """Split `text` into chunks of at most `max_tokens` tokens.

    Sentence boundaries first, then clause punctuation, then whitespace as a
    last resort. `max_tokens` defaults below pocket-tts's 50-token chunk limit
    so a chunk never trips the "may skip words" path.

    With `keep_punct_boundaries`, every sentence and clause mark becomes a chunk
    boundary even when the text would have fitted in one chunk. Only boundaries
    get an inserted pause, so this is what makes a comma audible when the model
    itself rushes it -- which a model trained on clipped subtitle segments does.
    """
    text = " ".join(text.split())
    if not text:
        return []

    def fits(s: str) -> bool:
        return count_tokens(s) <= max_tokens

    def split_by(pattern: re.Pattern[str], piece: str) -> list[str]:
        parts = [p.strip() for p in pattern.split(piece) if p.strip()]
        return parts if len(parts) > 1 else []

    def recurse(piece: str) -> list[str]:
        if fits(piece):
            return [piece]
        for pattern in (HARD_BREAK, SOFT_BREAK):
            parts = split_by(pattern, piece)
            if parts:
                return [c for p in parts for c in recurse(p)]
        # No punctuation left, so pack by words. Packing greedily to the budget
        # fills early chunks and leaves whatever is left over as a tail: 36
        # tokens at a budget of 18 came out 16/18/2, and that 2-token tail is
        # far outside the ~11-token training distribution -- it ran past EOS
        # and invented words. Decide the chunk count first, then aim for even
        # chunks, which needs no more chunks than greedy and has no tail.
        words = piece.split()
        total = count_tokens(piece)
        n_chunks = max(1, math.ceil(total / max_tokens))
        # Word boundaries make the ideal count optimistic: 36 tokens at a
        # budget of 18 looks like two chunks, but the first fills at 16 and 20
        # are left over, which no single chunk may hold. Pack, and if it needed
        # another chunk, aim again at the count it actually took so the tokens
        # spread evenly instead of leaving a starved tail.
        for _ in range(4):
            chunks = _pack(words, count_tokens, fits, n_chunks, math.ceil(total / n_chunks))
            if len(chunks) <= n_chunks:
                return chunks
            n_chunks = len(chunks)
        return chunks

    if keep_punct_boundaries:
        pieces = [text]
        for pattern in (HARD_BREAK, SOFT_BREAK):
            pieces = [q for piece in pieces for q in (split_by(pattern, piece) or [piece])]
        parts = [c for piece in pieces for c in recurse(piece)]
    else:
        parts = recurse(text)

    # Merge neighbours back together while they still fit: fewer, fuller chunks
    # sound better than many tiny ones. A chunk ending at punctuation is never
    # merged away when its boundary is what earns the pause.
    merged: list[str] = []
    for chunk in parts:
        # A punctuation boundary is only worth keeping if the chunk before it is
        # long enough to synthesize well. "چهارم شهریور،" on its own is two
        # words, and a model trained on ~4s utterances renders such a fragment
        # badly -- especially as the first chunk after the voice prompt.
        keep_apart = (
            keep_punct_boundaries
            and merged
            and merged[-1].rstrip().endswith(BREAK_CHARS)
            and count_tokens(merged[-1]) >= min_tokens
        )
        if merged and not keep_apart and fits(f"{merged[-1]} {chunk}"):
            merged[-1] = f"{merged[-1]} {chunk}"
        else:
            merged.append(chunk)

    # Greedy packing leaves a runt whenever the text does not divide evenly:
    # 36 tokens at a budget of 18 comes out 16/18/2, and the 2-token tail
    # cannot merge back because 18+2 exceeds the budget. That tail is far
    # outside the ~11-token training distribution and renders badly, and the
    # seam in front of it lands mid-phrase -- it split تغییر from دهد, and the
    # inserted join silence made the gap audible. Pull words back from the
    # neighbour until both sides clear min_tokens.
    for i in range(len(merged) - 1, 0, -1):
        if count_tokens(merged[i]) >= min_tokens:
            continue
        if merged[i - 1].rstrip().endswith(BREAK_CHARS):
            continue  # a real punctuation boundary is not ours to move
        while count_tokens(merged[i]) < min_tokens:
            words = merged[i - 1].split()
            # Never rob the neighbour below the same floor.
            if len(words) < 2 or count_tokens(" ".join(words[:-1])) < min_tokens:
                break
            merged[i - 1] = " ".join(words[:-1])
            merged[i] = f"{words[-1]} {merged[i]}"
    return merged


def split_sentences(
    text: str, count_tokens: Callable[[str], int], max_tokens: int = 80, min_tokens: int = 8
) -> list[str]:
    """One chunk per sentence, for models trained on whole sentences.

    The Chatterbox students (Pocket TTS trained on whole synthetic sentences)
    do not need split_text()'s 50-token pieces, and a cut inside a sentence
    hurts them: a piece that ended "... ساندر پیچای،" repeated the name in 6 of
    6 takes, against 0 of 12 when each sentence was one piece (SYNTH_PLAN
    11.8). So cut only at . ! ؟; a sentence shorter than `min_tokens` joins
    its neighbour, and only a sentence over `max_tokens` falls back to
    split_text() at that budget. Generate the chunks with generate_chunk(...,
    max_tokens=...) raised as well, or pocket-tts splits them again.
    """
    out: list[str] = []
    for sentence in (s.strip() for s in HARD_BREAK.split(" ".join(text.split()))):
        if not sentence:
            continue
        short = count_tokens(sentence) < min_tokens or (out and count_tokens(out[-1]) < min_tokens)
        if out and short and count_tokens(f"{out[-1]} {sentence}") <= max_tokens:
            out[-1] = f"{out[-1]} {sentence}"
        elif count_tokens(sentence) > max_tokens:
            out.extend(split_text(sentence, count_tokens, max_tokens, False, min_tokens))
        else:
            out.append(sentence)
    return out

def with_end(text: str) -> str:
    text = text.strip()
    return text if text[-1:] in ".؟!" else text + "."


def speak(model, state, text: str, pause: float = PAUSE_SEC) -> tuple[np.ndarray, int]:
    """One piece per sentence (split_sentences), each generated from the same voice prompt, silence trimmed, joined."""
    sr = int(model.mimi.sample_rate)
    sp = model.flow_lm.conditioner.tokenizer.sp
    pieces = split_sentences(normalize(with_end(text)), lambda s: len(sp.encode(s)))
    out = []
    for i, piece in enumerate(pieces):
        audio = generate_chunk(model, state, piece, frames_after_eos=0, sample_rate=sr, max_tokens=10_000)
        out.append(trim_silence(audio, sr))
        if i < len(pieces) - 1:
            out.append(np.zeros(int(pause * sr), dtype=np.float32))
    return (np.concatenate(out) if out else np.zeros(0, dtype=np.float32)), sr


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="mehdi-hf/cilla-lite", help="Hugging Face repo with config.yaml (default: this one)")
    ap.add_argument("--config", help="a local config.yaml instead of the repo's")
    ap.add_argument("--voice", required=True, help="voice prompt: a whole short clip (3-10 s) that ends at a pause")
    ap.add_argument("--text")
    ap.add_argument("--text-file")
    ap.add_argument("--out", default="out.wav")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=0, help="torch threads (0: leave the default)")
    a = ap.parse_args()
    text = a.text or Path(a.text_file).read_text(encoding="utf-8")
    if not normalize(with_end(text)).strip():
        raise SystemExit("nothing to say: the text has no Persian letters (English words and other foreign characters are dropped); write it in Persian letters")

    import torch
    from huggingface_hub import hf_hub_download
    from scipy.io import wavfile

    from pocket_tts import TTSModel

    if a.threads:
        torch.set_num_threads(a.threads)
    torch.manual_seed(a.seed)
    config = a.config or hf_hub_download(a.repo, "config.yaml")
    model = TTSModel.load_model(config=config, temp=0.3, eos_threshold=-2.0)
    state = model.get_state_for_audio_prompt(a.voice)
    wav, sr = speak(model, state, text)
    wavfile.write(a.out, sr, (np.clip(wav, -1.0, 1.0) * 32767).astype(np.int16))
    print(f"{a.out}: {len(wav) / sr:.1f} s")


if __name__ == "__main__":
    main()

# MuTE

MuTE (Music Translation Evaluation) is an automated evaluation metric for music generation tasks where a reference is available.

### Example
```python
output_dir = Path("~/joytunes/ismir2022/oded/songs_eval").expanduser()
ref_song_dir = Path("~/joytunes/ismir2022/oded/songs").expanduser()
ref_songs = ref_song_dir.glob("*/*_Essentials.generated.mscz")
for ref_score in ref_songs:
    print("=============================")
    song_dir = ref_score.parent
    song = song_dir.name
    print(f"Song: {song}")
    ref = gui.parse_score(ref_score, save_musicxml=True)
    target_scores = list(song_dir.glob(f"*_checkpoint-*.mscz"))
    target_scores.append(song_dir / f"{song}_Intermediate.generated.mscz")
    for target_score in target_scores:
        print(f"\t\tTarget: {target_score.stem}")
        target = gui.parse_score(target_score, save_musicxml=True)
        output_name = "Intermediate" if "Intermediate" in target_score.stem else target_score.stem
        eval_output_path = output_dir / song / output_name
        eval_output_path.parent.mkdir(exist_ok=True, parents=True)
        eval_phrase(ref, target, eval_output_path, skip_align_when_length_equal=True)

def demo():
    print("Parsing scores...")
    ref = gui.parse_score("/Users/matan/joytunes/data/test_scores/eran/no_essentials/GimmeGimmeGimme_ABBA_Intermediate.mscz")
    target = gui.parse_score("/Users/matan/joytunes/data/test_scores/eran/no_essentials/out_sampling/GimmeGimmeGimme_ABBA/2022-04-21_05-03-29.musicxml")
    # Sanity check: target is the same as ref.
    # target = gui.parse_score("/Users/matan/joytunes/data/test_scores/eran/no_essentials/GimmeGimmeGimme_ABBA_Intermediate.mscz")
    ref_phrase = ref.measures(8, 26)
    # ref_phrase = ref
    target_phrase = target.measures(7, 25, indicesNotNumbers=True)
    # target_phrase = target
    print("Aligning...")
    D, wp, mean_cost_per_measure, added_bars, removed_bars, seq1_len, seq2_len, added_bar_count, removed_bar_count = align(ref_phrase, target_phrase)
    print("Done")
    print(f"{added_bars = }, {removed_bars = }\n{mean_cost_per_measure = }")
    print(np.flip(wp, 0) + 1)
    plot_dtw(D, wp, seq1_len, seq2_len)

```

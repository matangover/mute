import json
from pathlib import Path
from typing import Optional
from collections import Counter

import numpy as np
import muspy
import librosa
import librosa.display
from music21.stream.base import Score
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

def get_measure_pc_pianoroll(score: Score):
    measure_count = len(score.parts[0].getElementsByClass("Measure"))
    measure_pcps = []
    for measure_index in range(measure_count):
        measure = score.measure(measure_index, indicesNotNumbers=True)
        measure_pcp = to_pitchclass_pianoroll(to_pianoroll(measure))
        assert measure_pcp.shape[0] == 16, (measure_index, measure_pcp.shape)
        measure_pcps.append(measure_pcp)
    return np.array(measure_pcps)

def align(reference: Score, target: Score):
    """
    Aligns two scores using DTW.
    """
    mpcps1 = get_measure_pc_pianoroll(reference)
    mpcps2 = get_measure_pc_pianoroll(target)
    # Time axis must come last for librosa.sequence.dtw.
    # The rest of the axes are flattened for calculating the feature distance matrix.
    # Hamming distance: the proportion of elements between two feature vectors which disagree.
    # (Because the feature vectors are boolean pianorolls, there are no numerical values. And we do
    # want to normalize by the number of elements.)
    D, wp = librosa.sequence.dtw(mpcps1.T, mpcps2.T, metric="hamming")
    total_cost = D[-1, -1]
    path_length = len(wp)
    mean_cost_per_measure = total_cost / path_length

    counts_ref: Counter[int] = Counter()
    counts_target: Counter[int] = Counter()
    for i, j in wp:
        counts_ref[i] += 1
        counts_target[j] += 1
    added_bars = {measure_index: c - 1 for measure_index, c in counts_ref.items() if c > 1}
    removed_bars = {measure_index: c - 1 for measure_index, c in counts_target.items() if c > 1}
    added_bar_count = sum(added_bars.values())
    removed_bar_count = sum(removed_bars.values())

    return D, wp, mean_cost_per_measure, added_bars, removed_bars, len(mpcps1), len(mpcps2), added_bar_count, removed_bar_count

def plot_dtw(D, wp, seq1_len, seq2_len):
    librosa.display.specshow(D, x_axis='frames', y_axis='frames')
    plt.title('DTW cost')
    plt.colorbar()
    plt.yticks(range(seq1_len), range(1, seq1_len + 1))
    plt.xticks(range(seq2_len), range(1, seq2_len + 1))
    plt.ylabel("Source measure")
    plt.xlabel("Target measure");
    plt.plot(wp[:, 1], wp[:, 0])
    plt.yticks(range(seq1_len), range(1, seq1_len + 1))
    plt.xticks(range(seq2_len), range(1, seq2_len + 1))
    plt.ylabel("Source measure")
    plt.xlabel("Target measure")
    plt.show()


def eval_phrase(ref: Score, target: Score, eval_output_path: Optional[Path] = None, skip_align_when_length_equal=True, show=False):
    """
    Evaluate similarity between target and reference using the F1 score -- precision and recall.
    What proportion of notes in the reference were correctly recalled in the target (recall), and
    what proportion of the notes in the target are in fact correct (precision). This is done per
    pianoroll time-slice. We treat it as a multiclass/multilabel classification problem for each
    time step, and then use the F1 score for each time step, and average the F1 scores.

    Idea for enhancement: Note is considered 100% correct if octave matches, 50% correct if one
    octave off, 25% correct if two octaves off, etc.
    """
    ref_measure_count = get_measure_count(ref)
    target_measure_count = get_measure_count(target)
    if ref_measure_count == target_measure_count and skip_align_when_length_equal:
        added_bar_count = removed_bar_count = 0
        wp = np.zeros((ref_measure_count, 2), dtype=int)
        wp[:, 0] = wp[:, 1] = np.arange(ref_measure_count)
        wp = np.flip(wp, 0)
        print("Skipped align - length equal")
    else:
        D, wp, mean_cost_per_measure, added_bars, removed_bars, seq1_len, seq2_len, added_bar_count, removed_bar_count = align(ref, target)
        # overlap_score = (1 - mean_cost_per_measure) * 100
        # print(f"hamming distance score = {overlap_score:.2f}")
        # The problem with using mean hamming distance is that it is often very high because of all
        # the "matching zeros" in all notes that are inactive in both the reference and the target.

    ref_music = to_music(ref)
    target_music = to_music(target)
    ref_proll = music_to_pianoroll(ref_music)
    target_proll = music_to_pianoroll(target_music)
    ref_proll_aligned, target_proll_aligned, repeated_mask = get_aligned_prolls(ref_proll, target_proll, wp)
    samplewise_score, f1_score_pitch = get_score(ref_proll_aligned, target_proll_aligned, repeated_mask)
    # print(f"{f1_score_pitch = :.2f}, {added_bar_count = }, {removed_bar_count = }")
    if show or eval_output_path is not None:
        output_path = None if eval_output_path is None else eval_output_path.with_suffix(".png")
        plot_prolls(ref_proll_aligned, target_proll_aligned, repeated_mask, samplewise_score, f1_score_pitch, output_path, show)

    # Evaluate with hand-specific pianoroll
    hand_scores: dict[str, float] = {}
    for hand in ["left", "right"]:
        track_to_remove = 0 if hand == "left" else 1
        prolls = []
        for music in [ref_music, target_music]:
            music_hand = music.deepcopy()
            del music_hand.tracks[track_to_remove]
            proll = music_to_pianoroll(music_hand)
            prolls.append(proll)
        lh, rh = prolls
        ref_proll_aligned, target_proll_aligned, repeated_mask = get_aligned_prolls(lh, rh, wp)
        samplewise_score, score = get_score(ref_proll_aligned, target_proll_aligned, repeated_mask)
        hand_scores[hand] = float(score)
        # print(f"\t\t\t{hand} hand f1_score_pitch = {score:.2f}")
        # TODO: hand-specific pitchclass score

    # Evaluate with pitchclass-pianorolls
    ref_pc_roll = to_pitchclass_pianoroll(ref_proll)
    target_pc_roll = to_pitchclass_pianoroll(target_proll)
    ref_proll_aligned, target_proll_aligned, repeated_mask = get_aligned_prolls(ref_pc_roll, target_pc_roll, wp)
    samplewise_score, f1_score_pitchclass = get_score(ref_proll_aligned, target_proll_aligned, repeated_mask)
    # print(f"\t\t\t{f1_score_pitchclass = :.2f}")

    result = {
        "f1_score_pitch": f1_score_pitch,
        "f1_score_pitchclass": f1_score_pitchclass,
        "f1_score_mean_hands": np.mean(list(hand_scores.values())),
        "f1_scores_per_hand": hand_scores,
        "added_bar_count": added_bar_count,
        "removed_bar_count": removed_bar_count,
        "aligned_bar_count": len(wp),
        "ref_bar_count": ref_measure_count,
        "target_bar_count": target_measure_count,
    }
    if eval_output_path is not None:
        eval_output_path.with_suffix(".json").write_text(json.dumps(result))
    
    return result


def get_aligned_prolls(ref_proll, target_proll, wp):
    # wp comes in reverse order - make it in ascending order.
    wp = np.flip(wp, 0)
    
    ref_proll_aligned = np.zeros((len(wp) * measure_len, ref_proll.shape[1]))
    target_proll_aligned = np.zeros_like(ref_proll_aligned)
    repeated_mask = np.ones((ref_proll_aligned.shape[0],), dtype=bool)

    for aligned_measure_index in range(len(wp)):
        ref_measure_index, target_measure_index = wp[aligned_measure_index]
        target_proll_aligned[measure_slice(aligned_measure_index)] = target_proll[measure_slice(target_measure_index)]
        ref_proll_aligned[measure_slice(aligned_measure_index)] = ref_proll[measure_slice(ref_measure_index)]
        if aligned_measure_index != 0:
            prev_ref_measure_index, prev_target_measure_index = wp[aligned_measure_index - 1]
            ref_repeated = (prev_ref_measure_index == ref_measure_index)
            target_repeated = (prev_target_measure_index == target_measure_index)
            if ref_repeated or target_repeated:
                # Measure is repeated in either ref or target - this means a measure was either
                # omitted or added in the target. Set the aligned repeat mask to 0 to signify this,
                # and later this will be used to zero out the f-score for this measure.
                repeated_mask[measure_slice(aligned_measure_index)] = 0

    
    return ref_proll_aligned, target_proll_aligned, repeated_mask


def get_score(ref_aligned_proll, target_aligned_proll, repeated_mask):
    # TODO: Allow octave shifts (with reduced scoring - 1 octave: 0.5x, 2 octaves: 0.25x)
    # TODO: Reduce score if note is in wrong hand (0.5x).
    # TODO: Take into account onsets: reduce by 0.5x if onset is confused with sustain.
    f1_score = f1_score_samplewise(ref_aligned_proll, target_aligned_proll, zero_division=1)
    f1_score *= repeated_mask
    return f1_score, np.mean(f1_score)


quarters_per_measure = 4
timesteps_per_quarter = 4
measure_len = quarters_per_measure * timesteps_per_quarter

def measure_slice(measure_index):
    measure_start = measure_index * measure_len
    measure_end = measure_start + measure_len
    return slice(measure_start, measure_end)


def precision_recall_fscore_support_samplewise(
    y_true,
    y_pred,
    *,
    beta=1.0,
    labels=None,
    pos_label=1,
    average=None,
    warn_for=("precision", "recall", "f-score"),
    sample_weight=None,
    zero_division="warn",
):
    """
    This function is copied from sklearn.metrics.precision_recall_fscore_support -- however, it 
    calculates the scores per sample instead of per-class, and returns the full list of scores
    without averaging them.
    """
    from sklearn.metrics._classification import _check_zero_division, _check_set_wise_labels, _prf_divide, _warn_prf
    from sklearn.metrics import multilabel_confusion_matrix
    _check_zero_division(zero_division)
    if beta < 0:
        raise ValueError("beta should be >=0 in the F-beta score")
    labels = _check_set_wise_labels(y_true, y_pred, average, labels, pos_label)

    # Calculate tp_sum, pred_sum, true_sum ###
    MCM = multilabel_confusion_matrix(
        y_true,
        y_pred,
        sample_weight=sample_weight,
        labels=labels,
        ###### The following line is the difference from sklearn's built-in function.
        samplewise=True, 
    )
    tp_sum = MCM[:, 1, 1]
    pred_sum = tp_sum + MCM[:, 0, 1]
    true_sum = tp_sum + MCM[:, 1, 0]

    # Finally, we have all our sufficient statistics. Divide! #
    beta2 = beta ** 2

    # Divide, and on zero-division, set scores and/or warn according to
    # zero_division:
    precision = _prf_divide(
        tp_sum, pred_sum, "precision", "predicted", average, warn_for, zero_division
    )
    recall = _prf_divide(
        tp_sum, true_sum, "recall", "true", average, warn_for, zero_division
    )

    # warn for f-score only if zero_division is warn, it is in warn_for
    # and BOTH prec and rec are ill-defined
    if zero_division == "warn" and ("f-score",) == warn_for:
        if (pred_sum[true_sum == 0] == 0).any():
            _warn_prf(average, "true nor predicted", "F-score is", len(true_sum))

    # if tp == 0 F will be 1 only if all predictions are zero, all labels are
    # zero, and zero_division=1. In all other case, 0
    if np.isposinf(beta):
        f_score = recall
    else:
        denom = beta2 * precision + recall

        denom[denom == 0.0] = 1  # avoid division by 0
        f_score = (1 + beta2) * precision * recall / denom

    return precision, recall, f_score, true_sum, pred_sum, tp_sum

def f1_score_samplewise(y_true, y_pred, *, beta=1.0, labels=None, pos_label=1, average=None, sample_weight=None, zero_division="warn"):
    _, _, f_score, _, _, _ = precision_recall_fscore_support_samplewise(
        y_true, y_pred, beta=beta, labels=labels, pos_label=pos_label, average=average,
        warn_for=("f-score",), sample_weight=sample_weight, zero_division=zero_division
    )
    return f_score

def plot_prolls(ref_aligned, target_aligned, repeated_mask, samplewise_score, mean_score, output_path, show=False):
    refb = ref_aligned.astype(bool)
    tarb = target_aligned.astype(bool)
    merged_pr = np.zeros_like(refb, dtype=float)
    merged_pr[refb] = 1
    merged_pr[tarb] = 0.7
    merged_pr[refb & tarb] = 0.3
    merged_pr[~repeated_mask] = np.nan

    fig, ax = plt.subplots(nrows=2, ncols=1, sharex=True, gridspec_kw=dict(height_ratios=[4, 1]))

    cmap = plt.get_cmap("magma")
    librosa.display.specshow(merged_pr.T, x_axis="frames", ax=ax[0], cmap=cmap, vmin=0, vmax=1)

    # Add custom legend
    custom_lines = [Line2D([0], [0], color=cmap(1.), lw=4),
                    Line2D([0], [0], color=cmap(0.7), lw=4),
                    Line2D([0], [0], color=cmap(0.3), lw=4)]

    # Shrink current axis by 20%
    # for a in ax:
    #     box = a.get_position()
    #     a.set_position([box.x0, box.y0, box.width * 0.8, box.height])

    # Put a legend to the right of the current axis
    ax[0].legend(custom_lines, ['Reference', 'Target', 'Both'], bbox_to_anchor=(1, 0.5), loc='center left',)

    ax[1].step(range(len(samplewise_score)), samplewise_score);
    ax[1].axhline(mean_score, color='r', linestyle='--', linewidth=0.5);
    ax[1].set_yticks([0, 1, mean_score])
    ax[1].set_yticklabels([0, 1, f"Mean: {mean_score:.2f}"]);

    plt.tight_layout()
    # plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
    # plt.show()
    if output_path:
        plt.savefig(output_path, dpi=300)
    if not show:
        plt.close(fig)


def get_measure_count(score: Score):
    return len(score.parts[0].getElementsByClass("Measure"))

DEFAULT_RESOLUTION = 4

def to_music(score: Score) -> muspy.Music:
    return muspy.from_music21_score(
        score,
        resolution=DEFAULT_RESOLUTION,
        import_chords_as_notes=True,
    )

def to_pitchclass_pianoroll(pianoroll: np.ndarray):
    pc = np.zeros((pianoroll.shape[0], 12), bool)
    for c in range(12):
        pitches = range(c, 128, 12)
        pc[:, c] = np.logical_or.reduce(pianoroll[:, pitches], 1)
    return pc

def to_pianoroll(score: Score):
    music = to_music(score)
    return music_to_pianoroll(music)

def music_to_pianoroll(music: muspy.Music):
    return muspy.to_pianoroll_representation(music, encode_velocity=False, full_length=True)


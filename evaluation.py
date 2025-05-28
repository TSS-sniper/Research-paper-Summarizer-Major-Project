from bert_score import score
from rouge import Rouge


def evaluate_summary(summary, reference):
    P, R, F1 = score([summary], [reference], lang="en", verbose=False)
    rouge = Rouge()
    rouge_scores = rouge.get_scores(summary, reference)[0]

    return {
        "bertscore_f1": round(F1.mean().item(), 4),
        "rouge-1": rouge_scores["rouge-1"]["f"],
        "rouge-2": rouge_scores["rouge-2"]["f"],
        "rouge-l": rouge_scores["rouge-l"]["f"],
    }

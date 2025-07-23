from bert_score import score
from rouge import Rouge


def evaluate_summary(summary, reference):
    P, R, F1 = score([summary], [reference], lang="en", verbose=False)
    rouge = Rouge()
    rouge_scores = rouge.get_scores(summary, reference)[0]

    return {
        "BERTscore (F1)": round(F1.mean().item(), 4),
        "ROUGE-1": rouge_scores["rouge-1"]["f"],
        "ROUGE-2": rouge_scores["rouge-2"]["f"],
        "ROUGE-L": rouge_scores["rouge-l"]["f"],
    }

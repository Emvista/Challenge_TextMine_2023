import argparse
import json
import sys
from collections import defaultdict

from sklearn_crfsuite.metrics import flat_classification_report, flat_f1_score

"""Computes evaluation metrics.

This tool requires two signatures JSON file: one with predictions, the other with gold-standard data.
Evaluations metrics are computed using scitkit.
"""

# globals ===================================================================

# All known labels
LABELS = {
    "Email",
    "Function",
    "Human",
    "Location",
    "Organization",
    "Phone_Number",
    "Project",
    "Reference_CEDEX",
    "Reference_Code_Postal",
    "Reference_CS",
    "Reference_User",
    "Social_Network",
    "Url"
}
# Put everything in lowercase to ignore case later
LABELS = {label.lower() for label in LABELS}

# Define the key to access the label field, as some participants did not use "label"
#KEY_PREDLABEL = "annotation"
KEY_PREDLABEL = "label"


# misc ======================================================================

def read_data(path):
    with open(path, "r") as fp:
        return json.load(fp)

def normalize_labels(signatures, key_label="label"):
    for signature in signatures:
        for n, annotation in enumerate(signature["annotations"]):
            if annotation[key_label] is None:
                print(f"WARNING: unexpected None value for {key_label}. Falling-back to \"none\" as label")
                annotation[key_label] = "none"
            else:
                annotation[key_label] = annotation[key_label].lower()
    return signatures


def compare_tagsets(pred_signatures, key_predlabel="label"):
    pred_tagset = set()
    for pred_signature in pred_signatures:
        for pred_annotation in pred_signature["annotations"]:
            pred_tagset.add(pred_annotation[key_predlabel])
    gold_tagset = {label.lower() for label in LABELS}
    diff = pred_tagset.difference(gold_tagset)
    if diff:
        print(f"ERROR: predicted data tagset differs from expected")
        print(f"ERROR: expected: {gold_tagset}")
        print(f"ERROR: got: {pred_tagset}")
        print(f"ERROR: difference: {diff}")

# core ======================================================================

def check_signature_mapping(identifier2data):
    """Makes sure that every identifier is mapped to exactly 2 items, i.e [predicted_annotation, gold_annotation].

    :return: number of identifier not mapped to exactly 2 items"""
    errors = 0
    for identifier, signatures in identifier2data.items():
        if len(signatures) != 2:
            print(f"ERROR: identifier {identifier} is mapped to {len(signatures)} signatures. Expected 2.")
            errors += 1
    if errors != 0:
        print(f"ERRORS: {errors}/{len(identifier2data)}")
    return errors


def align_signatures(predicted_data, gold_data):
    """Aligns signatures using their identifier.

    :return: a identifier to [predicted_signature, gold_signature] mapping."""
    identifier2data = defaultdict(list)  # a identifier to [predicted_signature, gold_signature] mapping
    for predicted in predicted_data:
        identifier2data[predicted["identifier"]].append(predicted)
    for gold in gold_data:
        identifier2data[gold["identifier"]].append(gold)
    return identifier2data


def check_annotation_mapping(offsets2annotation):
    """Makes sure that every (begin, end) offset is mapped to exactly 2 items, i.e [predicted_annotation, gold_annotation].

    :return: number of (begin, end) offset not mapped to exactly 2 items"""
    errors = 0
    for offsets, annotations in offsets2annotation.items():
        if len(annotations) != 2:
            print(f"ERROR: offsets {offsets} are mapped to {len(annotations)} annotations. Expected 2.")
            errors += 1
    if errors != 0:
        print(f"ERRORS: {errors}/{len(offsets2annotation)}")
    return errors


def align_annotations(predicted_signature, gold_signature):
    """Aligns annotations using their offsets.

     :return: a identifier (begin, end) to [predicted_annotation, gold_annotation] mapping."""
    offsets2annotation = defaultdict(list)  # an offset to [predicted_annotation, gold_annotation] mapping
    predicted_annotations = predicted_signature["annotations"]
    for predicted_annotation in predicted_annotations:
        # we use the (begin, end) offset as annotation identifiers in the scope of a signature
        begin = predicted_annotation["begin"]
        end = predicted_annotation["end"]
        offsets2annotation[(begin, end)].append(predicted_annotation)
    gold_annotations = gold_signature["annotations"]
    for gold_annotation in gold_annotations:
        begin = gold_annotation["begin"]
        end = gold_annotation["end"]
        offsets2annotation[(begin, end)].append(gold_annotation)
    return offsets2annotation


def prepare_data(predicted_data, gold_data):
    Y_predicted = []
    Y_gold = []
    identifier2data = align_signatures(predicted_data, gold_data)
    if check_signature_mapping(identifier2data) != 0:
        sys.exit(1)
    for identifier, (predicted_signature, gold_signature) in identifier2data.items():
        offsets2annotation = align_annotations(predicted_signature, gold_signature)
        if check_annotation_mapping(offsets2annotation) != 0:
            sys.exit(1)
        y_predicted = []
        y_gold = []
        for offsets, [predicted_annotation, gold_annotation] in offsets2annotation.items():
            y_predicted.append(predicted_annotation[KEY_PREDLABEL])
            y_gold.append(gold_annotation["label"])
        assert len(y_predicted) == len(y_gold)
        Y_predicted.append(y_predicted)
        Y_gold.append(y_gold)
    assert len(Y_predicted) == len(Y_gold)
    return Y_predicted, Y_gold


def score(predicted_data, gold_data):
    Y_predicted, Y_gold = prepare_data(predicted_data, gold_data)
    print(flat_classification_report(y_pred=Y_predicted, y_true=Y_gold, labels=sorted(LABELS), digits=4))
    print("F1:", flat_f1_score(y_true=Y_gold, y_pred=Y_predicted, average='weighted', labels=sorted(LABELS)))

# CLI =======================================================================


def parse_args():
    parser = argparse.ArgumentParser(
        prog="scorer",
        description="Compares predictions and gold-standard data to compute several evaluation metrics"
    )
    parser.add_argument("predicted", help="Predicted data file")
    parser.add_argument("gold", help="Gold-standard data file")
    return parser.parse_args()

# entry point ===============================================================

def main():
    args = parse_args()
    predicted_data = normalize_labels(read_data(args.predicted), key_label=KEY_PREDLABEL)
    gold_data = normalize_labels(read_data(args.gold))
    compare_tagsets(predicted_data, key_predlabel=KEY_PREDLABEL)
    score(predicted_data, gold_data)


if __name__ == "__main__":
    main()
